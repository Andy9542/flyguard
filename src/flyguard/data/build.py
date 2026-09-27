"""Two-stage, idempotent data build (docs/design.md §2): ``python -m flyguard.data.build [--without-traces] [--smoke]``.

load raw -> normalise -> language -> E1 split -> (smoke subset) -> windows -> dedup -> contamination (ТЗ 3.2) ->
splits -> pools -> audit -> ``data/processed/{documents,windows,episodes}.parquet`` +
``data/manifests/{splits,pools,dedup,contamination}.json`` + ``audit.md``. A smoke build skips the contamination
audit when ``smoke.contamination_audit`` is false and writes a skip note instead (:func:`smoke_contamination_note`).
Stage 1 (``--without-traces``) uses deepset, BIPIA, NotInject; the full stage adds dojo/dyn/para when their files
and the extraction module exist. Both stages rewrite every output from scratch, so re-running is safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flyguard.config import ROOT, Configs, load_configs, seeds_for
from flyguard.data import audit as audit_mod
from flyguard.data import loaders, pools, splits
from flyguard.data.dedup import dedup_windows
from flyguard.data.normalize import language_columns
from flyguard.data.windows import build_windows
from flyguard.io import atomic_write_bytes, atomic_write_json, atomic_write_text

SPAN_TYPE = pa.list_(pa.struct([("start", pa.int32()), ("end", pa.int32())]))
DOCUMENTS_SCHEMA = pa.schema([("doc_id", pa.string()), ("source", pa.string()), ("split", pa.string()),
                              ("label", pa.int8()), ("text", pa.string()), ("text_orig", pa.string()),
                              ("lang", pa.string()), ("lang_stratum", pa.string()), ("cluster_id", pa.string()),
                              ("spans", SPAN_TYPE), ("meta_json", pa.string()), ("dedup_dropped", pa.bool_())])
WINDOWS_SCHEMA = pa.schema([("window_id", pa.string()), ("doc_id", pa.string()), ("source", pa.string()),
                            ("split", pa.string()), ("start", pa.int32()), ("end", pa.int32()), ("text", pa.string()),
                            ("label", pa.int8()), ("cluster_id", pa.string()), ("text_hash", pa.string()),
                            ("dedup_excluded", pa.bool_()), ("dup_of", pa.string())])


def output_dirs(root: Path, smoke: bool) -> tuple[Path, Path]:
    """Smoke mode writes beside the real outputs (design §2): ``data/processed/smoke``, ``data/manifests/smoke``."""
    if smoke:
        return root / "data" / "processed" / "smoke", root / "data" / "manifests" / "smoke"
    return root / "data" / "processed", root / "data" / "manifests"


# ------------------------------------------------------------------------------------------- parquet IO

def write_parquet(df: pd.DataFrame, path: Path, schema: pa.Schema | None) -> None:
    """Explicit-schema parquet through ``flyguard.io`` (atomic): pandas 3 would otherwise infer ``list<null>`` for
    all-empty span columns and lose the design's types."""
    if schema is None:
        table = pa.Table.from_pandas(df, preserve_index=False)
    else:
        arrays = []
        for field in schema:
            col = df[field.name].tolist() if field.name in df.columns else [None] * len(df)
            if field.name == "spans":
                col = [[{"start": int(s), "end": int(e)} for s, e in (sp or [])] for sp in col]
            elif field.name == "dup_of":
                col = [None if (x is None or (isinstance(x, float) and x != x)) else str(x) for x in col]
            arrays.append(pa.array(col, type=field.type))
        table = pa.Table.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    atomic_write_bytes(path, sink.getvalue().to_pybytes())


def documents_to_frame(documents: pd.DataFrame) -> pd.DataFrame:
    """In-memory documents (dict ``meta``) -> the on-disk columns of design §2 (``meta_json``, ``dedup_dropped``)."""
    df = documents.copy()
    if "meta_json" not in df.columns:
        df["meta_json"] = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in df["meta"]]
    if "dedup_dropped" not in df.columns:
        df["dedup_dropped"] = False
    return df


def read_documents(path: Path) -> pd.DataFrame:
    """Read ``documents.parquet`` back with ``spans`` as ``(start, end)`` pairs and ``meta`` as dicts."""
    df = pq.read_table(path).to_pandas()
    df["spans"] = [[(int(s["start"]), int(s["end"])) for s in (sp if sp is not None else [])] for sp in df["spans"]]
    df["meta"] = [json.loads(m) if m else {} for m in df["meta_json"]]
    return df


def read_windows(path: Path) -> pd.DataFrame:
    """Read ``windows.parquet`` (design §2) as a DataFrame; ``dup_of`` is None where a window is not excluded."""
    return pq.read_table(path).to_pandas()


# ----------------------------------------------------------------------------------------------- smoke

SMOKE_RULE = ("per source, the deterministic head after sorting by id, taken in whole units (cluster; episode for "
              "dojo/dyn; base for para) per stratum (E1 split x BIPIA task / trace suite / paraphrase kind) in "
              "proportion to the stratum's share of the source, at least one unit per stratum; the caps are "
              "smoke.docs_per_source documents, smoke.episodes episodes, smoke.paraphrase_bases bases")


def proportional_head(items: pd.DataFrame, stratum_col: str, unit_col: str, cap: int) -> set[str]:
    """Units (whole clusters) to keep so that about ``cap`` rows survive: each stratum gets ``round(cap * share)``
    rows, at least its first unit, filled with units in sorted order until the quota is reached (the unit that
    crosses the quota is kept whole). Deterministic and idempotent: depends only on ids and sizes."""
    keep: set[str] = set()
    total = len(items)
    if total == 0:
        return keep
    for stratum, grp in items.groupby(stratum_col, sort=True):
        quota = max(1, int(round(cap * len(grp) / total)))
        taken = 0
        for unit, size in grp.groupby(unit_col, sort=True).size().items():
            if taken >= quota:
                break
            keep.add(unit)
            taken += int(size)
    return keep


def smoke_subset(documents: pd.DataFrame, cfg: Configs) -> pd.DataFrame:
    """Smoke limit (design §2, ТЗ "Бюджет времени"): ``SMOKE_RULE``.

    Review finding: a plain sorted head gave a smoke deepset of 116 test / 66 train / 18 val documents and a smoke
    BIPIA of seven ``code`` contexts of one split. Strata now keep every split (and BIPIA task / trace suite) present
    in proportion to the full data; a BIPIA pair or a trace episode is never cut in half. dojo/dyn are first limited
    to ``smoke.episodes`` episodes (proportionally over split x suite) and paraphrases to ``smoke.paraphrase_bases``.
    """
    sm = cfg.default["smoke"]
    cap = int(sm["docs_per_source"])
    metas = splits._meta(documents)
    df = documents.assign(_episode=[m.get("episode_id") for m in metas],
                          _task=[m.get("task") or m.get("suite") or m.get("kind") or "" for m in metas])
    df["_stratum"] = df["split"].astype(str) + "/" + df["_task"].astype(str)
    keep: list[str] = []
    for src, grp in df.groupby("source", sort=True):
        grp = grp.sort_values("doc_id", kind="stable")
        if src in ("dojo", "dyn"):
            eps = grp.dropna(subset=["_episode"]).drop_duplicates("_episode")
            chosen = proportional_head(eps, "_stratum", "_episode", int(sm["episodes"]))
            grp = grp[grp["_episode"].isin(chosen)]
        if src == "para":
            bases = grp.drop_duplicates("cluster_id")
            chosen = proportional_head(bases, "_stratum", "cluster_id", int(sm["paraphrase_bases"]))
            grp = grp[grp["cluster_id"].isin(chosen)]
        clusters = proportional_head(grp, "_stratum", "cluster_id", cap)
        keep.extend(grp[grp["cluster_id"].isin(clusters)]["doc_id"])
    return documents[documents["doc_id"].isin(set(keep))].reset_index(drop=True)


# ------------------------------------------------------------------------------------- smoke contamination

def smoke_contamination_audit(cfg: Configs) -> bool:
    """Whether a smoke build repeats the ТЗ 3.2 contamination audit (``smoke.contamination_audit``, default true)."""
    return bool(cfg.default.get("smoke", {}).get("contamination_audit", True))


def smoke_contamination_note(cfg: Configs) -> dict[str, Any]:
    """The ``contamination.json`` of a smoke build that skips the audit (ASSUMPTIONS A54): the counts-only overlap
    with the PIGuard training set is a property of the full tables and is reported from the full build's
    ``data/manifests/contamination.json``; re-measuring it on the 200-document smoke subset only costs time (it
    streams the whole PIGuard set). ``audit.md`` renders the note as a skipped section."""
    return {"skipped": True, "smoke": True, "train_file": str(audit_mod.PIGUARD_TRAIN),
            "note": ("smoke: the ТЗ 3.2 contamination audit is not repeated on the smoke subset "
                     "(smoke.contamination_audit false, ASSUMPTIONS A54); the numbers are those of the full build, "
                     "data/manifests/contamination.json"),
            "model_cards": audit_mod.MODEL_CARDS, "threats_to_validity": audit_mod.THREATS_TO_VALIDITY}


# ------------------------------------------------------------------------------------------------ build

def load_sources(cfg: Configs, root: Path, without_traces: bool, access_log=None, smoke: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every loader of design §4; returns the concatenated raw documents and the loader info/notes. ``smoke`` routes
    the trace-extraction manifest to ``data/manifests/smoke`` (design §2: a smoke build never rewrites the real one)."""
    seed_sub = seeds_for(cfg, 0)["subsample"]
    frames = [loaders.load_deepset(cfg, "train", root, access_log), loaders.load_deepset(cfg, "test", root, access_log)]
    bip, bip_info = loaders.load_bipia(cfg, seed_sub, root, access_log)
    frames.append(bip)
    frames.append(loaders.load_notinject(cfg, root, access_log))
    notes: list[str] = []
    episodes: dict[str, pd.DataFrame] = {}
    if without_traces:
        notes.append("dojo, dyn, para skipped: stage --without-traces")
    else:
        manifest_dir = output_dirs(root, smoke)[1]
        for benchmark in ("agentdojo", "agentdyn"):
            ep, docs, note = loaders.load_trace_documents(cfg, benchmark, root, smoke=smoke, access_log=access_log,
                                                          manifest_dir=manifest_dir)
            notes.append(note)
            if docs is not None:
                frames.append(docs)
                if ep is not None:
                    episodes[benchmark] = ep
        para = loaders.load_paraphrases(cfg, root, access_log)
        if para is None:
            notes.append("para skipped: data/paraphrases/paraphrases.csv missing")
        else:
            frames.append(para)
            notes.append(f"para: {len(para)} documents")
    documents = pd.concat(frames, ignore_index=True, sort=False)
    documents["label"] = documents["label"].astype(int)
    return documents, {"bipia": bip_info, "notes": notes, "episodes": episodes, "seed_subsample": int(seed_sub)}


def add_languages(documents: pd.DataFrame, cfg: Configs) -> pd.DataFrame:
    """ТЗ 1.2 language + stratum. BIPIA attacked variants inherit the language of their context's clean document:
    the attack strings are short English snippets and detecting 45 near-copies per context would be wasted time."""
    df = documents.copy()
    own = ~((df["source"] == "bipia") & (df["label"] == 1))
    langs, strata = language_columns(df.loc[own, "text"].tolist(), cfg)
    df["lang"] = None
    df["lang_stratum"] = None
    df.loc[own, "lang"] = langs
    df.loc[own, "lang_stratum"] = strata
    clean = df[(df["source"] == "bipia") & (df["label"] == 0)].drop_duplicates("cluster_id").set_index("cluster_id")
    inherit = ~own
    df.loc[inherit, "lang"] = df.loc[inherit, "cluster_id"].map(clean["lang"]).to_numpy()
    df.loc[inherit, "lang_stratum"] = df.loc[inherit, "cluster_id"].map(clean["lang_stratum"]).to_numpy()
    df["lang"] = df["lang"].fillna("unk")
    df["lang_stratum"] = df["lang_stratum"].fillna(f"non-{cfg.default.get('language', {}).get('main', 'en')}")
    return df


def build_all(cfg: Configs | None = None, without_traces: bool = False, smoke: bool = False, root: Path = ROOT,
              access_log=None, write: bool = True) -> dict[str, Any]:
    """Run the whole pipeline; returns the in-memory artefacts (and writes them when ``write``)."""
    cfg = cfg or load_configs()
    documents, info = load_sources(cfg, root, without_traces, access_log, smoke=smoke)
    documents["split"] = splits.assign_e1_split(documents, cfg, info["seed_subsample"]).to_numpy()
    documents = add_languages(documents, cfg)
    if smoke:
        documents = smoke_subset(documents, cfg)
    documents = documents.sort_values("doc_id", kind="stable").reset_index(drop=True)
    windows = build_windows(documents, cfg)
    windows, dropped, dedup_report = dedup_windows(documents, windows, cfg)
    documents["dedup_dropped"] = documents["doc_id"].isin(dropped)
    if smoke and not smoke_contamination_audit(cfg):
        contamination = smoke_contamination_note(cfg)
    else:
        contamination = audit_mod.contamination_audit(cfg, documents, windows, root, access_log)   # ТЗ 3.2, after dedup
    if contamination.get("skipped"):
        info["notes"].append(f"contamination audit skipped: {contamination.get('note')}")
    episodes_dojo = info["episodes"].get("agentdojo")
    split_manifest = splits.build_splits(documents, cfg, dropped, episodes_dojo)
    if smoke:
        split_manifest["smoke"] = {"rule": SMOKE_RULE, "caps": dict(cfg.default["smoke"])}
    pool_manifest = pools.build_pools(documents, cfg, dropped)
    audit_info = {"stage": "without-traces" if without_traces else "full", "smoke": smoke, "notes": info["notes"],
                  "bipia": info["bipia"], "dedup": dedup_report, "pools": pool_manifest, "contamination": contamination,
                  "smoke_rule": SMOKE_RULE if smoke else None}
    audit_md = audit_mod.build_audit(documents, windows, cfg, audit_info)
    episodes = pd.concat(list(info["episodes"].values()), ignore_index=True) if info["episodes"] else None
    result = {"documents": documents, "windows": windows, "episodes": episodes, "splits": split_manifest,
              "pools": pool_manifest, "dedup": dedup_report, "contamination": contamination, "audit": audit_md,
              "notes": info["notes"]}
    if write:
        processed, manifests = output_dirs(root, smoke)
        write_parquet(documents_to_frame(documents), processed / "documents.parquet", DOCUMENTS_SCHEMA)
        write_parquet(windows, processed / "windows.parquet", WINDOWS_SCHEMA)
        ep_path = processed / "episodes.parquet"
        if episodes is not None:
            write_parquet(episodes, ep_path, None)
        elif ep_path.exists():
            ep_path.unlink()               # a stage-1 rebuild must not leave a stale trace artefact behind
        atomic_write_json(manifests / "splits.json", split_manifest)
        atomic_write_json(manifests / "pools.json", pool_manifest)
        atomic_write_json(manifests / "dedup.json", dedup_report)
        atomic_write_json(manifests / "contamination.json", contamination)
        atomic_write_text(manifests / "audit.md", audit_md)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI of design §2: ``--without-traces`` (stage 1) and ``--smoke`` (smoke subset and directories)."""
    ap = argparse.ArgumentParser(description="FlyGuard data build (design §2)")
    ap.add_argument("--without-traces", action="store_true", help="stage 1: deepset, BIPIA, NotInject only")
    ap.add_argument("--smoke", action="store_true", help="smoke subset -> data/processed/smoke, data/manifests/smoke")
    args = ap.parse_args(argv)
    res = build_all(load_configs(), without_traces=args.without_traces, smoke=args.smoke)
    for line in summary_lines(res):
        print(line)
    return 0


def summary_lines(res: dict[str, Any]) -> list[str]:
    """Counts only (data-safety rule): documents/windows per source and split, dedup, pools, contamination."""
    docs, win = res["documents"], res["windows"]
    dd = res["dedup"]
    out = [f"documents={len(docs)} windows={len(win)} dropped_documents={dd['documents_dropped_total']} "
           f"excluded_test_windows={dd['test_windows_excluded']}"]
    if len(docs):
        by = docs.groupby(["source", "split"]).size()
        wby = win.groupby(["source", "split"]).size() if len(win) else {}
        for (src, split), n in by.items():
            lab = docs[(docs["source"] == src) & (docs["split"] == split)]["label"]
            out.append(f"  {src}/{split}: documents={n} (pos={int((lab == 1).sum())}, neg={int((lab == 0).sum())}) "
                       f"windows={int(wby.get((src, split), 0))}")
    out.append(f"  dedup: excluded_by_source_pair={dd.get('test_windows_excluded_by_source_pair')} "
               f"dropped_by_source_variant={dd.get('documents_dropped_by_source_variant')}")
    for name in ("p_val", "p_test"):
        p = res.get("pools", {}).get(name, {})
        out.append(f"  {name}: n={p.get('n')} by_source={p.get('by_source')} meets_target={p.get('meets_target')}")
    c = res.get("contamination") or {}
    if c.get("skipped"):
        out.append(f"  contamination: skipped ({c.get('note')})")
    elif c:
        for sk, v in c["document_level"].items():
            t = v["total"]
            w = c["window_level"].get(sk, {}).get("total", {})
            out.append(f"  contamination {sk}: documents {t['documents_matched']}/{t['documents']} (share {t['share']}), "
                       f"windows {w.get('windows_matched')}/{w.get('windows')} (share {w.get('share')})")
    for note in res.get("notes", []):
        out.append(f"  note: {note}")
    return out


if __name__ == "__main__":
    sys.exit(main())
