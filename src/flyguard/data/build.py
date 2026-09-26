"""Two-stage, idempotent data build (docs/design.md §2): ``python -m flyguard.data.build [--without-traces] [--smoke]``.

load raw -> normalise -> language -> E1 split -> (smoke subset) -> windows -> dedup -> splits -> pools -> audit ->
``data/processed/{documents,windows,episodes}.parquet`` + ``data/manifests/{splits,pools,dedup}.json`` + ``audit.md``.
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

def smoke_subset(documents: pd.DataFrame, cfg: Configs) -> pd.DataFrame:
    """Smoke limit (design §2, ТЗ "Бюджет времени"): per source the deterministic head after sorting by doc_id,
    taken in whole clusters (a BIPIA pair or a trace episode is never cut in half) until ``docs_per_source`` is
    reached; dojo/dyn are first limited to ``smoke.episodes`` episodes and paraphrases to ``smoke.paraphrase_bases``.
    Before filling, the first cluster of every E1 split present in the source is taken, so a smoke run always has
    val and test material of each source (a pure sorted head of BIPIA would be a few ``code`` contexts of one split)."""
    sm = cfg.default["smoke"]
    cap = int(sm["docs_per_source"])
    metas = splits._meta(documents)
    df = documents.assign(_episode=[m.get("episode_id") for m in metas])
    keep: list[str] = []
    for src, grp in df.groupby("source", sort=True):
        grp = grp.sort_values("doc_id", kind="stable")
        if src in ("dojo", "dyn"):
            eps = sorted(set(e for e in grp["_episode"] if e is not None))[: int(sm["episodes"])]
            grp = grp[grp["_episode"].isin(eps)]
        if src == "para":
            bases = sorted(set(grp["cluster_id"]))[: int(sm["paraphrase_bases"])]
            grp = grp[grp["cluster_id"].isin(bases)]
        total = 0
        taken: set[str] = set()
        first_per_split = grp.drop_duplicates("split")["cluster_id"].tolist()
        ordered = first_per_split + [c for c in grp["cluster_id"].drop_duplicates() if c not in first_per_split]
        by_cluster = {c: cg for c, cg in grp.groupby("cluster_id", sort=False)}
        for cluster in ordered:
            if total >= cap and cluster not in first_per_split:
                break
            if cluster in taken:
                continue
            cg = by_cluster[cluster]
            keep.extend(cg["doc_id"])
            taken.add(cluster)
            total += len(cg)
    return documents[documents["doc_id"].isin(set(keep))].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ build

def load_sources(cfg: Configs, root: Path, without_traces: bool, access_log=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every loader of design §4; returns the concatenated raw documents and the loader info/notes."""
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
        for benchmark in ("agentdojo", "agentdyn"):
            ep, docs, note = loaders.load_trace_documents(cfg, benchmark, root)
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
    documents, info = load_sources(cfg, root, without_traces, access_log)
    documents["split"] = splits.assign_e1_split(documents, cfg, info["seed_subsample"]).to_numpy()
    documents = add_languages(documents, cfg)
    if smoke:
        documents = smoke_subset(documents, cfg)
    documents = documents.sort_values("doc_id", kind="stable").reset_index(drop=True)
    windows = build_windows(documents, cfg)
    windows, dropped, dedup_report = dedup_windows(documents, windows, cfg)
    documents["dedup_dropped"] = documents["doc_id"].isin(dropped)
    episodes_dojo = info["episodes"].get("agentdojo")
    split_manifest = splits.build_splits(documents, cfg, dropped, episodes_dojo)
    pool_manifest = pools.build_pools(documents, cfg, dropped)
    audit_info = {"stage": "without-traces" if without_traces else "full", "smoke": smoke, "notes": info["notes"],
                  "bipia": info["bipia"], "dedup": dedup_report, "pools": pool_manifest}
    audit_md = audit_mod.build_audit(documents, windows, cfg, audit_info)
    episodes = pd.concat(list(info["episodes"].values()), ignore_index=True) if info["episodes"] else None
    result = {"documents": documents, "windows": windows, "episodes": episodes, "splits": split_manifest,
              "pools": pool_manifest, "dedup": dedup_report, "audit": audit_md, "notes": info["notes"]}
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
        atomic_write_text(manifests / "audit.md", audit_md)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI of design §2: ``--without-traces`` (stage 1) and ``--smoke`` (smoke subset and directories)."""
    ap = argparse.ArgumentParser(description="FlyGuard data build (design §2)")
    ap.add_argument("--without-traces", action="store_true", help="stage 1: deepset, BIPIA, NotInject only")
    ap.add_argument("--smoke", action="store_true", help="smoke subset -> data/processed/smoke, data/manifests/smoke")
    args = ap.parse_args(argv)
    res = build_all(load_configs(), without_traces=args.without_traces, smoke=args.smoke)
    docs, win = res["documents"], res["windows"]
    print(f"documents={len(docs)} windows={len(win)} dropped={res['dedup']['documents_dropped_total']} "
          f"excluded_windows={res['dedup']['test_windows_excluded']}")
    for src, n in docs["source"].value_counts().sort_index().items():
        print(f"  {src}: {n}")
    for note in res["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
