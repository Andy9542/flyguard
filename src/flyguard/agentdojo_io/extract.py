"""Trace logs -> `episodes` and `documents` frames of design §2 plus `data/manifests/traces_extraction.json`
(ТЗ 1.5 "Извлечение", 1.8, 1.10; contract §2–§6) — design §3 `extract.py`.

`build_episode_documents(cfg, benchmark)` is the only entry point the data layer (`flyguard.data.build`) needs:
it returns one row per episode and one row per tool step (a step document), with the labels of `labels.py`.
Tool outputs are real prompt injections: this module never prints them, only counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from flyguard.agentdojo_io import labels as L
from flyguard.agentdojo_io.contract import split_manifest, validate_split_manifest, write_split_manifest
from flyguard.agentdojo_io.parse import (NONE_TOKEN, TraceLog, episode_id, iter_log_paths, read_log, tool_steps,
                                          unanswered_tool_calls)
from flyguard.config import ROOT, Configs, load_configs, seeds_for
from flyguard.io import atomic_write_json, read_json, sha256_file
from flyguard.netlog import DATA_ACCESS_LOG, log_data_access

SOURCE_PREFIX = {"agentdojo": "dojo", "agentdyn": "dyn"}
BENCHMARKS = ("agentdojo", "agentdyn")

EPISODE_COLUMNS = ["episode_id", "benchmark", "suite", "user_task", "injection_task", "attack", "model", "episode_class",
                   "utility", "security", "n_steps", "injection_step", "first_harmful_step", "match", "contract_split",
                   "e1_val_task", "log_path", "sha256"]
"""`episodes.parquet` columns, design §2."""

DOCUMENT_COLUMNS = ["doc_id", "source", "split", "label", "text", "text_orig", "lang", "lang_stratum", "cluster_id",
                    "spans", "meta_json"]
"""`documents.parquet` columns for the dojo/dyn sources, design §2."""

FILL_STRINGS = {"user": "Emma Johnson", "model": "DeepSeek"}
"""Fallback copy of `configs/default.yaml` `traces.fill_strings` for callers without a config: the values the
harness substitutes for `{user}` / `{model}` in the attack templates (`Emma Johnson` is agentdojo's
`ImportantInstructionsAttack.user_name`, `DeepSeek` the prose name registered for both candidate agent models in
`gen/harness_run.py`, ASSUMPTIONS A4). `build_episode_documents` reads the frozen config value, records it in
the manifest (design §3) and hands it to the span matcher; the log's `injections` are already filled, so on real
logs the substitution is a no-op and `unfilled_placeholders` counts what survived in the raw log."""

MANIFEST_PATH = ROOT / "data" / "manifests" / "traces_extraction.json"
SPLIT_MANIFEST_PATH = ROOT / "results" / "shared" / "split_manifest.json"


def extraction_settings(cfg: Configs) -> dict[str, Any]:
    """The frozen extraction constants of `configs/default.yaml`, read here and nowhere hard-coded (CLAUDE.md):
    `extraction.decode_yaml_quoted_scalars` (the second span-matching pass, DEVIATIONS D7), `traces.fill_strings`
    (ASSUMPTIONS A4) and `extraction.harm_matching.target_args` (key arguments of meta-derived references)."""
    extraction = cfg.default.get("extraction") or {}
    fill = (cfg.default.get("traces") or {}).get("fill_strings") or FILL_STRINGS
    return {"decode_yaml_quoted_scalars": bool(extraction.get("decode_yaml_quoted_scalars", True)),
            "fill_strings": {str(k): str(v) for k, v in dict(fill).items()},
            "target_args": list(L.target_args_from_cfg(cfg))}


def traces_root(cfg: Configs, benchmark: str) -> Path:
    """`data/traces/<benchmark>` (or the operator's `shared.traces_dir` when traces were handed over)."""
    shared = (cfg.operator.get("shared") or {}).get("traces_dir")
    base = Path(shared) if shared else ROOT / cfg.default["traces"]["logdir"]
    if not base.is_absolute():
        base = ROOT / base
    return base / benchmark


PILOT_PATH = ROOT / "results" / "pilot.json"
TRACES_MANIFEST_PATH = ROOT / "results" / "shared" / "traces_manifest.json"


def choose_model(cfg: Configs, benchmark: str, root: Path | None = None, pilot_path: Path = PILOT_PATH,
                 manifest_path: Path = TRACES_MANIFEST_PATH) -> str | None:
    """The agent model whose logs form the dataset — one model for both benchmarks (contract §1): the pilot's
    `chosen_model` (`results/pilot.json`), else the frozen manifest's `agent_model`; only when neither record
    exists, the only model directory under the traces root. The pilot's choice is binding even before that
    model's logs of a benchmark arrive: pilot runs of a rejected candidate sit next to the chosen one (ТЗ 1.5),
    and a leftover directory must not turn into another model's episodes (the benchmark then yields empty
    frames). None when nothing decides."""
    for record, key in ((Path(pilot_path), "chosen_model"), (Path(manifest_path), "agent_model")):
        if record.exists():
            model = read_json(record).get(key)
            if model:
                return str(model)
    root = root or traces_root(cfg, benchmark)
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return dirs[0] if len(dirs) == 1 else None


def _detect_lang(text: str, seed: int) -> str:
    """langdetect code, `unk` on failure (ТЗ 1.2); seeded once so that the detector is deterministic."""
    try:
        from langdetect import DetectorFactory, detect  # local import: keeps parse/labels free of langdetect
        DetectorFactory.seed = int(seed)
        return str(detect(text)) if text.strip() else "unk"
    except Exception:  # noqa: BLE001 - langdetect raises on short/odd texts; those are `unk` by design
        return "unk"


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _episode_record(log: TraceLog, benchmark: str, model: str, steps, cls: str, inj_step: int | None,
                    harmful: tuple[int | None, str | None], split: str, path: Path, e1_rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode_id(log, model), "benchmark": benchmark, "suite": log.suite_name,
        "user_task": log.user_task_id, "injection_task": log.injection_task_id, "attack": log.attack_type,
        "model": model, "episode_class": cls,
        "utility": (None if log.utility is None else bool(log.utility)),
        "security": (None if log.security is None else bool(log.security)),
        "n_steps": len(steps), "injection_step": inj_step, "first_harmful_step": harmful[0], "match": harmful[1],
        "contract_split": split, "e1_val_task": L.is_e1_val_task(log.user_task_id, e1_rule),
        "log_path": _relpath(path), "sha256": sha256_file(path),
    }


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _typed_episodes(pd.DataFrame(columns=EPISODE_COLUMNS)), _typed_documents(pd.DataFrame(columns=DOCUMENT_COLUMNS))


def _typed_episodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=EPISODE_COLUMNS)
    for col in ("injection_step", "first_harmful_step", "n_steps"):
        df[col] = pd.array(df[col].tolist(), dtype="Int64")
    for col in ("utility", "security"):
        df[col] = pd.array(df[col].tolist(), dtype="boolean")
    df["e1_val_task"] = df["e1_val_task"].astype(bool)
    return df.reset_index(drop=True)


def _typed_documents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=DOCUMENT_COLUMNS)
    df["label"] = df["label"].astype("int8")
    return df.reset_index(drop=True)


def build_episode_documents(cfg: Configs, benchmark: str, model: str | None = None, traces_dir: str | Path | None = None,
                            harm_refs_path: str | Path = L.HARM_REFERENCES_PATH, meta_dir: str | Path = L.META_DIR,
                            manifest_path: str | Path | None = MANIFEST_PATH, data_access_log: Path = DATA_ACCESS_LOG,
                            global_seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every episode log of `benchmark` into the `episodes` and `documents` frames of design §2 and write
    the benchmark's section of `data/manifests/traces_extraction.json` (design §3).

    Rules implemented: step numbering from 0 (contract §3, `parse.tool_steps`); step label and spans by exact
    match after normalisation, with the YAML-quoted second pass only when `extraction.decode_yaml_quoted_scalars`
    is true (D7), `injection_step` = first labelled step (ТЗ 1.5); `episode_class` (contract §2 + `error`);
    `first_harmful_step`/`match` (contract §4, `configs/harm_references.yaml` with the meta ground truth as
    fallback, key arguments `extraction.harm_matching.target_args`); contract split (§6: test tasks by crc32
    mod 3 == 2; validation = `labels.validation_tasks`: for AgentDojo `val_fraction` of the non-test tasks of
    each suite from the `subsample` child of `global_seed`, for AgentDyn every non-test task when
    `splits.contract.agentdyn_clean_non_test == "validation"` (A19); `important_instructions` excluded from
    training/validation); E1 role (ТЗ 1.8/1.10: AgentDojo tasks with crc32 mod 5 == 0 -> `val`, the rest and
    all AgentDyn -> `test`); `doc_id = <src>:<episode_id>#<step>`, `cluster_id = <suite>/<user_task>`.

    Excluded from `documents` but kept in `episodes`: `error` episodes and attacked episodes without a
    recovered span (`injection_step` null) — their tool outputs carry the injection in a form the exact match did
    not find, so they can be neither positives nor negatives. Steps with empty output produce no document.
    Test material is journalled through `flyguard.netlog.log_data_access` *before* any file is opened (one line
    per suite directory with its file count), so an aborted run still leaves the read on record.
    """
    if benchmark not in SOURCE_PREFIX:
        raise ValueError(f"unknown benchmark {benchmark!r}; expected one of {sorted(SOURCE_PREFIX)}")
    src = SOURCE_PREFIX[benchmark]
    root = Path(traces_dir) if traces_dir is not None else traces_root(cfg, benchmark)
    model = model or choose_model(cfg, benchmark, root)
    settings = extraction_settings(cfg)
    rule = L.contract_rule(cfg)
    seed = seeds_for(cfg, global_seed)["subsample"]
    stats: dict[str, Any] = {
        "benchmark": benchmark, "model": model, "traces_dir": _relpath(root), "n_logs": 0,
        "episodes_by_class": {}, "episodes_by_contract_split": {}, "episodes_by_attack": {}, "e1_val_task_episodes": 0,
        "steps_total": 0, "steps_labelled": 0, "steps_labelled_by_mode": {}, "documents": 0, "documents_positive": 0,
        "empty_steps": 0, "escaped_pass_skipped_map_mismatch": 0,
        "unanswered_tool_calls": 0, "unfilled_placeholders": 0,
        "attacked_without_span": {"count": 0, "episode_ids": []}, "errors": {"count": 0, "episode_ids": []},
        "match_counts": {}, "match_counts_by_class": {}, "unmatched": {"count": 0, "episode_ids": []},
        "unmatched_hijacked": {"count": 0, "episode_ids": []}, "reference_sources": {},
        "fill_strings": settings["fill_strings"], "decode_yaml_quoted_scalars": settings["decode_yaml_quoted_scalars"],
        "target_args": settings["target_args"], "normalizer": L.normalizer_source(),
        "validation_rule": L.validation_rule_text(benchmark, rule), "val_seed": int(seed), "val_tasks": {},
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if model is None or not (root / model).exists():
        stats["note"] = "no trace logs found"
        _write_manifest(manifest_path, benchmark, stats)
        return _empty_frames()

    paths = list(iter_log_paths(root, model))
    files_per_suite: dict[Path, int] = {}
    for p in paths:
        files_per_suite[p.parent.parent.parent] = files_per_suite.get(p.parent.parent.parent, 0) + 1
    for suite_dir in sorted(files_per_suite):  # journal the test read before the first file is opened
        log_data_access(suite_dir, split="test",
                        purpose=f"agentdojo_io.extract {benchmark}: {files_per_suite[suite_dir]} trace logs parsed into "
                                "step documents (test tasks included; tool outputs never printed)",
                        path=Path(data_access_log))
    logs: list[tuple[Path, TraceLog]] = [(p, read_log(p)) for p in paths]

    refs = L.load_harm_references(harm_refs_path) if Path(harm_refs_path).exists() else {}
    e1_rule = L.e1_val_rule(cfg)
    lang_seed = int(cfg.default.get("language", {}).get("seed", 0))
    decode = settings["decode_yaml_quoted_scalars"]
    fill = settings["fill_strings"]
    target_args = settings["target_args"]
    metas: dict[str, dict[str, Any] | None] = {}
    val_tasks: dict[str, set[str]] = {}
    for _, log in logs:
        suite = log.suite_name
        if suite not in metas:
            metas[suite] = L.load_meta(benchmark, suite, meta_dir)
    for suite, meta in metas.items():
        universe = set((meta or {}).get("user_tasks") or {}) | {lg.user_task_id for _, lg in logs if lg.suite_name == suite}
        non_test = [t for t in universe if not L.is_test_task(t, rule)]
        val_tasks[suite] = L.validation_tasks(benchmark, suite, non_test, seed, rule)
        stats["val_tasks"][suite] = sorted(val_tasks[suite])

    episodes: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for path, log in logs:
        stats["n_logs"] += 1
        steps = tool_steps(log)
        cls = L.episode_class(log)
        attacked = log.injection_task_id is not None
        reports = ([L.injection_spans_report(s.output_text, log.injections, fill, decode) for s in steps] if attacked
                   else [L.SpanReport() for _ in steps])
        spans = [r.spans for r in reports]
        span_modes = [r.mode for r in reports]
        stats["escaped_pass_skipped_map_mismatch"] += sum(1 for r in reports if not r.map_ok)
        for mode in span_modes:
            if mode is not None:
                stats["steps_labelled_by_mode"][mode] = stats["steps_labelled_by_mode"].get(mode, 0) + 1
        inj_step = next((s.index for s, sp in zip(steps, spans) if sp), None)
        ref_calls, ref_source = L.reference_calls(benchmark, log.suite_name, log.injection_task_id, refs,
                                                  metas.get(log.suite_name), target_args)
        harmful = L.first_harmful_step(steps, ref_calls) if attacked else (None, None)
        split = L.contract_split(log.user_task_id, log.attack_type, val_tasks.get(log.suite_name, set()), rule)
        rec = _episode_record(log, benchmark, model, steps, cls, inj_step, harmful, split, path, e1_rule)
        episodes.append(rec)

        stats["episodes_by_class"][cls] = stats["episodes_by_class"].get(cls, 0) + 1
        stats["episodes_by_contract_split"][split] = stats["episodes_by_contract_split"].get(split, 0) + 1
        atk = log.attack_type or NONE_TOKEN
        stats["episodes_by_attack"][atk] = stats["episodes_by_attack"].get(atk, 0) + 1
        stats["e1_val_task_episodes"] += int(rec["e1_val_task"])
        stats["steps_total"] += len(steps)
        stats["steps_labelled"] += sum(1 for sp in spans if sp)
        stats["unanswered_tool_calls"] += unanswered_tool_calls(log)
        stats["unfilled_placeholders"] += sum(1 for v in log.injections.values() if any("{" + p + "}" in v for p in L.PLACEHOLDERS))
        if attacked:
            stats["reference_sources"][ref_source] = stats["reference_sources"].get(ref_source, 0) + 1
            stats["match_counts"][harmful[1]] = stats["match_counts"].get(harmful[1], 0) + 1
            by_class = stats["match_counts_by_class"].setdefault(cls, {})
            by_class[harmful[1]] = by_class.get(harmful[1], 0) + 1
            if harmful[1] == L.MATCH_UNMATCHED and cls != L.CLASS_ERROR:
                stats["unmatched"]["count"] += 1
                stats["unmatched"]["episode_ids"].append(rec["episode_id"])
                if cls == L.CLASS_HIJACKED:  # the ones contract §4 drops from the stopped-before-harm metric (A20)
                    stats["unmatched_hijacked"]["count"] += 1
                    stats["unmatched_hijacked"]["episode_ids"].append(rec["episode_id"])
        if cls == L.CLASS_ERROR:
            stats["errors"]["count"] += 1
            stats["errors"]["episode_ids"].append(rec["episode_id"])
            continue
        if attacked and inj_step is None:
            stats["attacked_without_span"]["count"] += 1
            stats["attacked_without_span"]["episode_ids"].append(rec["episode_id"])
            continue

        # E1 role (ТЗ 1.8/1.10, ASSUMPTIONS A25): AgentDojo validation tasks (crc32 % 5 == 0) contribute only their
        # clean outputs (role "val": P_val, C_unl, dedup reference); their attacked episodes have no role in E1
        # ("unused": not val, not test, not a dedup reference), otherwise the static environments would make every
        # identical test window a duplicate of a validation window. Everything else is test material.
        if benchmark == "agentdojo" and rec["e1_val_task"]:
            e1_split = L.SPLIT_VAL if not attacked else L.SPLIT_UNUSED
        else:
            e1_split = L.SPLIT_TEST
        if e1_split == L.SPLIT_UNUSED:
            stats["e1_unused_episodes"] = stats.get("e1_unused_episodes", 0) + 1
        cluster_id = f"{log.suite_name}/{log.user_task_id}"
        for step, step_spans, span_mode in zip(steps, spans, span_modes):
            text = L.normalize_text(step.output_text)
            if not text:
                stats["empty_steps"] += 1
                continue
            label = L.step_label(step_spans)
            lang = _detect_lang(text, lang_seed)
            meta_json = {"benchmark": benchmark, "suite": log.suite_name, "user_task": log.user_task_id,
                         "injection_task": log.injection_task_id, "attack": log.attack_type, "episode_id": rec["episode_id"],
                         "step": step.index, "tool": step.tool, "tool_error": step.error is not None,
                         "episode_class": cls, "contract_split": split, "model": model, "span_mode": span_mode}
            documents.append({
                "doc_id": f"{src}:{rec['episode_id']}#{step.index}", "source": src, "split": e1_split, "label": label,
                "text": text, "text_orig": step.output_text, "lang": lang,
                "lang_stratum": "en" if lang == cfg.default.get("language", {}).get("main", "en") else "non-en",
                "cluster_id": cluster_id, "spans": [{"start": int(a), "end": int(b)} for a, b in step_spans],
                "meta_json": json.dumps(meta_json, sort_keys=True, ensure_ascii=False),
            })
            stats["documents"] += 1
            stats["documents_positive"] += label

    for key in ("attacked_without_span", "errors", "unmatched", "unmatched_hijacked"):
        stats[key]["episode_ids"].sort()
    _write_manifest(manifest_path, benchmark, stats)
    episodes_df = _typed_episodes(pd.DataFrame(episodes, columns=EPISODE_COLUMNS)) if episodes else _empty_frames()[0]
    documents_df = _typed_documents(pd.DataFrame(documents, columns=DOCUMENT_COLUMNS)) if documents else _empty_frames()[1]
    return episodes_df, documents_df


def _write_manifest(path: str | Path | None, benchmark: str, stats: dict[str, Any]) -> None:
    """Merge this benchmark's section into `traces_extraction.json` (the other benchmark's section is kept)."""
    if path is None:
        return
    path = Path(path)
    current = read_json(path) if path.exists() else {}
    if not isinstance(current, dict):
        current = {}
    current[benchmark] = stats
    atomic_write_json(path, current)


def build_all(cfg: Configs, benchmarks: Sequence[str] = BENCHMARKS,
              split_manifest_path: str | Path | None = SPLIT_MANIFEST_PATH, global_seed: int = 0,
              traces_dirs: Mapping[str, str | Path] | None = None,
              **kwargs: Any) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict[str, Any]]:
    """Extract every benchmark and write the *joint* `results/shared/split_manifest.json` (contract §1, §6: one
    manifest naming the test episodes of both benchmarks, handed to the second team unchanged). Returns
    `{benchmark: (episodes, documents)}` and the manifest; the manifest must pass `validate_split_manifest`
    (design §3 shape) or the call fails. `traces_dirs` maps a benchmark to its log root (tests); other keyword
    arguments go to `build_episode_documents` for every benchmark."""
    frames = {b: build_episode_documents(cfg, b, traces_dir=(traces_dirs or {}).get(b), global_seed=global_seed, **kwargs)
              for b in benchmarks}
    episodes = [rec for eps, _ in frames.values() for rec in eps.to_dict("records")]
    rule = L.contract_rule(cfg)
    val_seed = seeds_for(cfg, global_seed)["subsample"]
    if split_manifest_path is not None:
        manifest = write_split_manifest(episodes, split_manifest_path, rule=rule, val_seed=val_seed)
    else:
        manifest = split_manifest(episodes, rule=rule, val_seed=val_seed)
    problems = validate_split_manifest(manifest)
    if problems:
        raise ValueError("split manifest failed its shape check: " + "; ".join(problems[:5]))
    return frames, manifest


def _summary(benchmark: str, episodes: pd.DataFrame, documents: pd.DataFrame) -> dict[str, Any]:
    return {"benchmark": benchmark, "episodes": int(len(episodes)), "documents": int(len(documents)),
            "classes": episodes["episode_class"].value_counts().to_dict() if len(episodes) else {},
            "contract_split": episodes["contract_split"].value_counts().to_dict() if len(episodes) else {},
            "positive_documents": int(documents["label"].sum()) if len(documents) else 0}


def main(argv: list[str] | None = None) -> int:
    """`python -m flyguard.agentdojo_io.extract --benchmark {agentdojo,agentdyn,all} [--model M] [--traces-dir D]
    [--episodes-out P] [--documents-out P] [--split-manifest P]`: prints counts only (never document text).
    `all` extracts both benchmarks and writes the joint `results/shared/split_manifest.json` (or
    `--split-manifest`); for a single benchmark the split manifest is written only when a path is given."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=sorted(SOURCE_PREFIX) + ["all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--traces-dir", default=None, help="log root of the one benchmark (not with `all`)")
    ap.add_argument("--episodes-out", default=None, help="optional parquet path for the episodes frame")
    ap.add_argument("--documents-out", default=None, help="optional parquet path for the documents frame")
    ap.add_argument("--split-manifest", default=None, help="path for the contract split_manifest.json")
    args = ap.parse_args(argv)
    cfg = load_configs()
    if args.benchmark == "all":
        if args.traces_dir:
            ap.error("--traces-dir applies to one benchmark; with `all` the roots come from the config")
        frames, manifest = build_all(cfg, split_manifest_path=args.split_manifest or SPLIT_MANIFEST_PATH, model=args.model)
        episodes = pd.concat([e for e, _ in frames.values()], ignore_index=True)
        documents = pd.concat([d for _, d in frames.values()], ignore_index=True)
        summary = {**_summary("all", episodes, documents),
                   "per_benchmark": {b: _summary(b, e, d) for b, (e, d) in frames.items()},
                   "split_manifest": {k: manifest["counts"].get(k) for k in ("test", "observation", "validation_clean",
                                                                             "validation_attacks", "train_attacks",
                                                                             "excluded", "error")}}
    else:
        episodes, documents = build_episode_documents(cfg, args.benchmark, model=args.model, traces_dir=args.traces_dir)
        if args.split_manifest:
            write_split_manifest(episodes.to_dict("records"), args.split_manifest, rule=L.contract_rule(cfg),
                                 val_seed=seeds_for(cfg, 0)["subsample"])
        summary = _summary(args.benchmark, episodes, documents)
    if args.episodes_out:
        Path(args.episodes_out).parent.mkdir(parents=True, exist_ok=True)
        episodes.to_parquet(args.episodes_out, index=False)
    if args.documents_out:
        Path(args.documents_out).parent.mkdir(parents=True, exist_ok=True)
        documents.to_parquet(args.documents_out, index=False)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
