"""The comparison contract run (ТЗ Этап 5; docs/spec/Контракт_сравнения_v3.md §3, §5–§9; design_experiments §3
"Contract"): train the submitted variants, set the contract threshold, score every test episode step by step,
write ``results/shared/flyguard.csv`` and the §9 metrics to ``results/contract.json``.

Protocol (one global seed, 0 by default -- the contract exchanges one CSV, not ten):

1. **Split** (contract §6, ``episodes.parquet: contract_split``): test = clean runs of test tasks
   (``crc32(user_task_id) mod 3 == 2``) plus ``important_instructions`` attacks on them; validation = 20 % of the
   non-test AgentDojo tasks and every clean non-test AgentDyn run (A19); ``important_instructions`` on non-test
   tasks is excluded. The lists are recomputed from the episodes and compared with the manifest already handed
   to the second team (``results/shared/split_manifest.json``); a disagreement raises.
2. **Training** (ТЗ Этап 5 with the D6/D11 fallback): the ТЗ trains the variants on windows of non-test tasks and
   non-``important_instructions`` templates of AgentDojo, but only ``important_instructions`` was generated
   (DEVIATIONS D6), so the contract training set contains no attacked AgentDojo window. Per D11 the variants
   are trained on the deepset-train labels with γ / C chosen on deepset validation, and C_unl (SVD, idf,
   centring) is the E1 C_unl plus the clean windows of the contract train + validation AgentDojo episodes
   (``fc.register_set("c_unl", ...)`` before any nose is fitted). ``dojo_negatives=True`` additionally adds those
   clean windows as label-0 training / validation rows (the design_experiments §3 wording); the mode is written
   to ``notes``. AgentDyn teaches nothing (contract §6).
3. **Threshold** (contract §7): per variant, the maximum score of each validation ``benign`` episode over its
   step documents -> ``eval.thresholds.contract_threshold`` (one false alarm per 100 episodes, per 20 below 200
   validation episodes, flagged); ``threshold_n_benign`` = number of validation benign episodes, episodes without
   any document included (they produced no alarm).
4. **Steps** (contract §3, §5): the document of step *i* is the tool output of call *i*; its score is the max over
   *all* its windows (dedup exclusions are an E1 device; contract §10 already accepts the train/test overlap of the
   static environments, and dropping a window would move ``alarm_step``); ``alarm_step`` = first step with
   ``score >= threshold``, ``max_score`` = max over steps. An episode without documents (no tool call; tool outputs
   that produced no document in the tables -- empty outputs, or in smoke mode steps outside the per-source subset;
   or an attacked episode whose injection text was not recovered by the extractor) gets ``max_score 0.0`` and no
   alarm; its id and reason are recorded (``episodes_without_documents``: ``nothing_to_scan`` / ``no_step_document``
   / ``not_extracted``), because for the last group a 0.0 is "not scanned", not "clean".
5. **Output**: ``results/shared/flyguard.csv`` (16 columns, ``agentdojo_io.contract.write_csv``, validated),
   ``results/contract_steps.csv`` (per-step scores for audit) and ``results/contract.json`` with the §9 metrics of
   ``eval.metrics.contract_metrics_by_variant`` overall (``contract/<metric>/<detector>``) and per benchmark
   (``contract/<benchmark>/<metric>/<detector>``; AgentDyn carries the honest FPR, contract §10; intervals for the
   two headline rates only, :func:`benchmark_metrics`). Smoke runs write under ``results/smoke/`` and never touch
   the shared CSV or manifest.

Test windows are opened only through the context door, under the memo keys ``<source>#contract_trainval``
(before fitting), ``<source>#contract_val`` and ``<source>#contract_test`` (after the thresholds are set).
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from flyguard.agentdojo_io import labels as L
from flyguard.agentdojo_io.contract import (MANIFEST_LISTS, VARIANTS as CSV_VARIANTS, ContractRow, split_manifest,
                                            validate_csv, write_csv, write_split_manifest)
from flyguard.config import ROOT, Configs, config_hash, git_commit, load_configs, seeds_for
from flyguard.eval.bootstrap import cluster_bootstrap
from flyguard.eval.metrics import contract_frame, contract_metrics_by_variant, contract_point_metrics, doc_scores
from flyguard.eval.thresholds import contract_threshold
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import DETECTORS, FeatureContext, FittedDetector
from flyguard.io import atomic_write_json, atomic_write_text, read_json

EXPERIMENT = "contract"
VARIANTS: dict[str, str] = {"real_fly/bloom": "real_fly_bloom", "real_fly/linear": "real_fly_linear",
                            "flyhash/linear": "flyhash_linear", "tfidf_lr": "tfidf_lr"}
"""Contract §8 variant -> detector of :data:`flyguard.experiments.engine.DETECTORS` (results keys use the latter)."""
BENCHMARK_SOURCE = {"agentdojo": "dojo", "agentdyn": "dyn"}
METRIC_KEYS = ("stopped_before_harm", "false_alarms_per_100_benign", "detection_delay_mean", "detection_delay_median",
               "alarms_on_injection_ignored_share", "n_hijacked", "n_hijacked_eligible", "n_unmatched", "n_stopped",
               "n_benign", "n_false_alarms_benign", "n_detections", "n_early_alarms", "n_injection_ignored",
               "alarms_on_injection_ignored", "n_episodes", "n_clusters")
OVERALL_CI_KEYS = ("stopped_before_harm", "false_alarms_per_100_benign", "detection_delay_mean",
                   "alarms_on_injection_ignored_share")
"""The keys ``eval.metrics.contract_metrics`` bootstraps (the contract's §9 table)."""
BENCHMARK_CI_KEYS = ("stopped_before_harm", "false_alarms_per_100_benign")
"""Per-benchmark metrics carry intervals for the two headline rates only (see :func:`benchmark_metrics`)."""
assert set(VARIANTS) == set(CSV_VARIANTS)


# ----------------------------------------------------------------------------------------------------------------
# Paths and currency
# ----------------------------------------------------------------------------------------------------------------
def shared_dir(root: Path = ROOT, smoke: bool = False) -> Path:
    return results_mod.results_dir(root, smoke) / "shared"


def contract_result_path(root: Path = ROOT, smoke: bool = False) -> Path:
    return results_mod.results_dir(root, smoke) / "contract.json"


def contract_csv_path(root: Path = ROOT, smoke: bool = False) -> Path:
    return shared_dir(root, smoke) / "flyguard.csv"


def split_manifest_path(root: Path = ROOT, smoke: bool = False) -> Path:
    return shared_dir(root, smoke) / "split_manifest.json"


def step_scores_path(root: Path = ROOT, smoke: bool = False) -> Path:
    return results_mod.results_dir(root, smoke) / "contract_steps.csv"


def is_current(root: Path = ROOT, smoke: bool = False) -> bool:
    """``results/contract.json`` exists with the current ``config_hash`` and its CSV is present (skip rule)."""
    path = contract_result_path(root, smoke)
    if not path.exists() or not contract_csv_path(root, smoke).exists():
        return False
    try:
        return read_json(path).get("config_hash") == config_hash(Path(root))
    except (OSError, ValueError):
        return False


# ----------------------------------------------------------------------------------------------------------------
# Episodes and their documents
# ----------------------------------------------------------------------------------------------------------------
def _missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def episode_table(ctx: Context) -> pd.DataFrame:
    """Non-``error`` episodes (contract §2: errors are excluded from every metric) with plain-Python fields."""
    if ctx.episodes is None or len(ctx.episodes) == 0:
        raise FileNotFoundError("the contract run needs episodes.parquet (trace extraction, design §3) -- none found")
    ep = ctx.episodes.drop_duplicates("episode_id").reset_index(drop=True)
    ep = ep[ep["episode_class"].astype(str) != L.CLASS_ERROR].reset_index(drop=True)
    # object dtype on purpose: pandas 3 would infer its string dtype and turn None into NaN
    ep["attack"] = pd.Series([None if _missing(a) or str(a).lower() in ("", "none") else str(a) for a in ep["attack"]],
                             index=ep.index, dtype=object)
    ep["n_steps"] = [0 if _missing(n) else int(n) for n in ep["n_steps"]]
    return ep


def parse_doc_id(doc_id: str, source: str) -> tuple[str, int]:
    """``<source>:<episode_id>#<step>`` (design §2) -> ``(episode_id, step)``."""
    prefix = f"{source}:"
    if not doc_id.startswith(prefix) or "#" not in doc_id:
        raise ValueError(f"not an episode document id of {source!r}: {doc_id!r}")
    eid, step = doc_id[len(prefix):].rsplit("#", 1)
    return eid, int(step)


def source_doc_index(ctx: Context, source: str, episode_ids: Iterable[str]) -> pd.DataFrame:
    """Every step document of ``source`` without opening a test row: ids of the non-test frame, of the E1 test
    list and of the documents dedup dropped (``dedup.json``), parsed into ``episode_id`` / ``step`` with the flag
    ``nontest`` (row available in ``ctx.windows``). Ids of unknown episodes raise (id-format drift)."""
    known = set(str(e) for e in episode_ids)
    nontest = ctx.documents.loc[ctx.documents["source"] == source, "doc_id"].astype(str).tolist()
    test: set[str] = set(ctx.test_doc_ids(source)) if source in ctx.splits["e1"]["test"] else set()
    for ids in ((ctx.dedup or {}).get("documents_dropped_ids") or {}).values():
        test.update(d for d in ids if str(d).startswith(f"{source}:"))
    rows = []
    for doc_id, flag in [(d, True) for d in nontest] + [(d, False) for d in sorted(test - set(nontest))]:
        eid, step = parse_doc_id(doc_id, source)
        rows.append({"doc_id": doc_id, "episode_id": eid, "step": step, "nontest": flag})
    idx = pd.DataFrame(rows, columns=["doc_id", "episode_id", "step", "nontest"])
    unknown = sorted(set(idx["episode_id"]) - known)
    if unknown:
        raise ValueError(f"{len(unknown)} {source} documents belong to episodes missing from episodes.parquet")
    return idx


def episode_windows(ctx: Context, source: str, index: pd.DataFrame, episode_ids: Iterable[str], name: str,
                    purpose: str) -> pd.DataFrame:
    """All windows of the given episodes: non-test rows from ``ctx.windows`` plus the test rows through the door
    (memo key ``<source>#<name>``; one call per name, so pass every episode of that role at once). Columns of
    ``windows.parquet`` plus ``episode_id`` and ``step``; dedup flags are kept but not applied (module docstring)."""
    sel = index[index["episode_id"].isin(set(episode_ids))]
    frames = [ctx.windows[ctx.windows["doc_id"].isin(set(sel.loc[sel["nontest"], "doc_id"]))]]
    test_ids = sel.loc[~sel["nontest"], "doc_id"].tolist()
    if test_ids:
        frames.append(ctx.load_test_windows(source, purpose, doc_ids=test_ids, name=name))
    w = pd.concat(frames, ignore_index=True).drop_duplicates("window_id")
    w = w.merge(sel[["doc_id", "episode_id", "step"]], on="doc_id", how="left")
    return w.sort_values(["episode_id", "step", "start"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# Split manifest
# ----------------------------------------------------------------------------------------------------------------
def check_split_manifest(episodes: pd.DataFrame, manifest: Mapping[str, Any], cfg: Configs, val_seed: int) -> dict[str, Any]:
    """Compare the manifest handed to the second team with the lists recomputed from ``episodes.parquet`` and the
    contract rules: the five id lists, ``rule`` (mod / rem / test attack / val_seed), the crc32 rule on every test
    episode and the absence of ``important_instructions`` from every training / validation list."""
    rule = L.contract_rule(cfg)
    test_attack = rule.get("test_attack", "important_instructions")
    computed = split_manifest(episodes.to_dict("records"), rule=rule, val_seed=val_seed)
    problems: list[str] = []
    for k in MANIFEST_LISTS:
        mine, theirs = set(computed[k]), set(manifest.get(k) or [])
        if mine != theirs:
            problems.append(f"{k}: {len(theirs - mine)} ids only in the manifest, {len(mine - theirs)} only in episodes.parquet")
    mrule = manifest.get("rule") or {}
    for key in ("mod", "rem", "test_attack"):
        if key in mrule and str(mrule[key]) != str(rule[key]):
            problems.append(f"rule.{key}: manifest {mrule[key]!r} != config {rule[key]!r}")
    if "val_seed" in mrule and int(mrule["val_seed"]) != int(val_seed):
        problems.append(f"rule.val_seed: manifest {mrule['val_seed']} != subsample child of global seed 0 ({val_seed})")
    attack_of = dict(zip(episodes["episode_id"], episodes["attack"]))
    task_of = dict(zip(episodes["episode_id"], episodes["user_task"].astype(str)))
    bad_test = [e for e in manifest.get("test") or [] if e in task_of and (not L.is_test_task(task_of[e], rule)
                                                                        or not (_missing(attack_of.get(e)) or attack_of.get(e) == test_attack))]
    if bad_test:
        problems.append(f"{len(bad_test)} test episodes violate the crc32 mod {rule['mod']} == {rule['rem']} / {test_attack} rule")
    for k in MANIFEST_LISTS[1:]:
        leaked = [e for e in manifest.get(k) or [] if attack_of.get(e) == test_attack]
        if leaked:
            problems.append(f"{k}: {len(leaked)} `{test_attack}` episodes in a training/validation list")
    return {"ok": not problems, "problems": problems, "counts": computed["counts"],
            "n_test": len(computed["test"]), "n_validation_clean": len(computed["validation_clean"])}


# ----------------------------------------------------------------------------------------------------------------
# Scores per step and per episode
# ----------------------------------------------------------------------------------------------------------------
def step_scores(fc: FeatureContext, fitted: Mapping[str, FittedDetector], ws_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-document (= per-step) max window score of every variant on ``frame`` (all windows, contract §3)."""
    ws = fc.register_set(ws_name, frame.drop(columns=["episode_id", "step"]))
    scores = fc.score_many(fitted, ws)
    plain = ws.frame.drop(columns=[c for c in ("dedup_excluded", "dup_of") if c in ws.frame.columns])
    table = frame[["doc_id", "episode_id", "step"]].drop_duplicates("doc_id").reset_index(drop=True)
    for variant, det in VARIANTS.items():
        if det not in scores:
            raise RuntimeError(f"variant {variant!r} ({det}) produced no scores")
        ds = doc_scores(pd.DataFrame({"window_id": scores[det].index.to_numpy(), "score": scores[det].to_numpy(dtype=float)}), plain)
        table = table.merge(ds[["doc_id", "score"]].rename(columns={"score": variant}), on="doc_id", how="left")
    return table.sort_values(["episode_id", "step"], kind="stable").reset_index(drop=True)


def episode_scores(steps: pd.DataFrame, episode_ids: Sequence[str], variant: str,
                   tau: float | None = None) -> pd.DataFrame:
    """``max_score`` (max over steps; 0.0 without documents) and, given ``tau``, ``alarm_step`` = the first step
    with ``score >= tau`` (``score >= tau`` is the alarm rule of the whole project, ``eval.metrics``)."""
    out = pd.DataFrame({"episode_id": list(episode_ids)})
    if len(steps):
        mx = steps.groupby("episode_id")[variant].max()
        out["max_score"] = out["episode_id"].map(mx)
    else:
        out["max_score"] = np.nan
    out["n_documents"] = out["episode_id"].map(steps.groupby("episode_id").size()).fillna(0).astype(int) if len(steps) else 0
    out["max_score"] = out["max_score"].fillna(0.0).astype(float)
    out["alarm_step"] = pd.array([None] * len(out), dtype="Int64")
    if tau is not None and len(steps):
        hit = steps[steps[variant] >= tau].groupby("episode_id")["step"].min()
        out["alarm_step"] = pd.array([None if _missing(v) else int(v) for v in out["episode_id"].map(hit)], dtype="Int64")
    return out


def benchmark_metrics(rows: Sequence[Mapping[str, Any]], n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    """Contract §9 metrics of one benchmark per variant: every point value of ``eval.metrics.contract_point_metrics``
    and cluster-bootstrap intervals for :data:`BENCHMARK_CI_KEYS` only, on the same draws (same seed, same
    ``cluster_bootstrap``) as the overall metrics. The overall table is the contract's; the per-benchmark split is
    supplementary (AgentDyn carries the honest false-alarm rate, contract §10), and ``contract_metrics`` runs one
    bootstrap pass per interval, so the split is limited to the two headline rates to keep the run inside the
    smoke budget (the bootstrap, not the detectors, dominated the first smoke run)."""
    df = contract_frame(list(rows))
    out: dict[str, dict[str, Any]] = {}
    for v in sorted(df["variant"].astype(str).unique()):
        sub = df[df["variant"].astype(str) == v]
        m: dict[str, Any] = dict(contract_point_metrics(sub))
        m["n_episodes"] = int(len(sub))
        m["n_clusters"] = int(sub["cluster_id"].nunique())
        m["ci"] = {}
        for key in BENCHMARK_CI_KEYS:
            if m.get(key) is None:
                m["ci"][key] = None
                continue

            def _stat(s: pd.DataFrame, _k: str = key) -> float:
                val = contract_point_metrics(s)[_k]
                return float("nan") if val is None else float(val)

            m["ci"][key] = cluster_bootstrap(sub, _stat, n=n_boot, seed=seed).to_dict()
        out[v] = m
    return out


# ----------------------------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------------------------
def _composition(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    src = frame["source"].value_counts().to_dict() if len(frame) else {}
    return {"set": name, "n_windows": int(len(frame)), "n_pos": int((frame["label"] == 1).sum()) if len(frame) else 0,
            "n_neg": int((frame["label"] == 0).sum()) if len(frame) else 0,
            "n_docs": int(frame["doc_id"].nunique()) if len(frame) else 0,
            **{f"n_{k}": int(v) for k, v in sorted(src.items())}}


def contract_body(fc: FeatureContext, root: Path, smoke: bool, dojo_negatives: bool = False) -> dict[str, Any]:
    """Everything of the module docstring for one seed; returns the payload of ``results/contract.json`` without
    the header fields. Order: split check -> training sets (trainval door) -> fit -> validation thresholds
    (dyn val door) -> test episodes (test doors) -> CSV -> metrics."""
    ctx, cfg = fc.ctx, fc.cfg
    notes: list[str] = []
    episodes = episode_table(ctx)
    rule = L.contract_rule(cfg)
    val_seed = seeds_for(cfg, 0)["subsample"]
    mpath = split_manifest_path(root, smoke)
    if not mpath.exists():
        write_split_manifest(ctx.episodes.to_dict("records"), mpath, rule=rule, val_seed=val_seed)
        notes.append(f"split manifest was missing and has been written from episodes.parquet: {mpath}")
    check = check_split_manifest(episodes, read_json(mpath), cfg, val_seed)
    if not check["ok"]:
        raise ValueError("split_manifest.json disagrees with episodes.parquet / the contract rules: " + "; ".join(check["problems"]))
    if not (shared_dir(root, smoke) / "traces_manifest.json").exists():
        notes.append("results/shared/traces_manifest.json is missing (owned by the trace generation stage)")

    by_split = {(b, s): set(g["episode_id"]) for (b, s), g in episodes.groupby(["benchmark", "contract_split"])}
    benign = set(episodes.loc[episodes["episode_class"] == L.CLASS_BENIGN, "episode_id"])
    dojo_train = by_split.get(("agentdojo", L.SPLIT_TRAIN), set()) & benign
    dojo_val = by_split.get(("agentdojo", L.SPLIT_VAL), set()) & benign
    dyn_val = by_split.get(("agentdyn", L.SPLIT_VAL), set()) & benign
    test_ids = sorted(set(episodes.loc[episodes["contract_split"] == L.SPLIT_TEST, "episode_id"]))
    attacked_trainval = (by_split.get(("agentdojo", L.SPLIT_TRAIN), set()) | by_split.get(("agentdojo", L.SPLIT_VAL), set())) - benign
    if attacked_trainval:
        notes.append(f"{len(attacked_trainval)} attacked AgentDojo train/val episodes exist (other templates); their windows "
                     "are NOT used: this run implements the D11 fallback (deepset labels)")

    # -- training sets (D11) ---------------------------------------------------------------------------------------
    dojo_index = source_doc_index(ctx, "dojo", episodes["episode_id"])
    dyn_index = source_doc_index(ctx, "dyn", episodes["episode_id"])
    with_docs = set(dojo_index["episode_id"]) | set(dyn_index["episode_id"])
    trainval = episode_windows(ctx, "dojo", dojo_index, dojo_train | dojo_val, "contract_trainval", fc.purpose)
    clean = trainval[trainval["label"] == 0]
    clean_train = clean[clean["episode_id"].isin(dojo_train)]
    clean_val = clean[clean["episode_id"].isin(dojo_val)]
    win_cols = list(ctx.windows.columns)
    c_unl = pd.concat([ctx.c_unl_windows, clean[win_cols]], ignore_index=True).drop_duplicates("window_id").reset_index(drop=True)
    fc.register_set("c_unl", c_unl)  # before any nose: SVD / idf / centring see the augmented C_unl (D11)
    train, val = ctx.train_windows, ctx.val_windows_deep
    if dojo_negatives:
        train = pd.concat([train, clean_train[win_cols]], ignore_index=True)
        val = pd.concat([val, clean_val[win_cols]], ignore_index=True)
    fc.register_set("contract_train", train)
    fc.register_set("contract_val", val)
    composition = [_composition("c_unl", c_unl), _composition("train", train), _composition("val", val),
                   _composition("dojo_clean_train_episodes", clean_train), _composition("dojo_clean_val_episodes", clean_val)]
    if (train["source"] != "deep").any() and not dojo_negatives:
        raise RuntimeError("D11 training set must be deepset only")
    notes.append("training (D11 fallback): labels = deepset train windows" + (" + clean AgentDojo train-episode windows as "
                 "negatives (dojo_negatives=True)" if dojo_negatives else "") + "; gamma/C on deepset validation" +
                 (" + clean AgentDojo val-episode windows" if dojo_negatives else "") +
                 f"; C_unl = E1 C_unl + clean windows of {len(dojo_train)} train and {len(dojo_val)} val AgentDojo episodes "
                 f"({int(len(clean))} windows); no `important_instructions` window in any training/validation set")

    # -- fit --------------------------------------------------------------------------------------------------------
    fitted = {v: fc.fit(DETECTORS[det], train="contract_train", val="contract_val") for v, det in VARIANTS.items()}

    # -- thresholds on validation benign episodes (contract §7) ----------------------------------------------------
    val_frames = [trainval[trainval["episode_id"].isin(dojo_val)]]
    if dyn_val:
        val_frames.append(episode_windows(ctx, "dyn", dyn_index, dyn_val, "contract_val", fc.purpose))
    val_episode_ids = sorted(dojo_val | dyn_val)
    val_frame = pd.concat(val_frames, ignore_index=True)
    val_steps = step_scores(fc, fitted, "contract_val_episodes", val_frame) if len(val_frame) else None
    thresholds: dict[str, dict[str, Any]] = {}
    val_table: list[dict[str, Any]] = []
    bench_of = dict(zip(episodes["episode_id"], episodes["benchmark"]))
    for v in VARIANTS:
        es = episode_scores(val_steps, val_episode_ids, v) if val_steps is not None else episode_scores(
            pd.DataFrame(columns=["episode_id", "step", v]), val_episode_ids, v)
        if len(es) == 0:
            raise ValueError("no validation benign episodes: the contract threshold cannot be set (contract §7)")
        rec = contract_threshold(es["max_score"].to_numpy(), cfg)
        rec["n_by_benchmark"] = {b: int(sum(1 for e in val_episode_ids if bench_of[e] == b)) for b in sorted(set(bench_of.values()))}
        rec["n_without_documents"] = int((es["n_documents"] == 0).sum())
        thresholds[f"contract/{VARIANTS[v]}"] = rec
        for r in es.itertuples(index=False):
            val_table.append({"episode_id": r.episode_id, "benchmark": bench_of[r.episode_id], "variant": v,
                              "max_score": float(r.max_score), "n_documents": int(r.n_documents)})
    n_benign_val = len(val_episode_ids)
    if n_benign_val < int(cfg.default["stats"]["contract"]["threshold_min_val_episodes"]):
        notes.append(f"contract §7: {n_benign_val} validation benign episodes < 200 -> one false alarm per 20 (flagged in the records)")

    # -- test episodes ----------------------------------------------------------------------------------------------
    test_frames = []
    dojo_test = by_split.get(("agentdojo", L.SPLIT_TEST), set())
    dyn_test = by_split.get(("agentdyn", L.SPLIT_TEST), set())
    if dojo_test:
        test_frames.append(episode_windows(ctx, "dojo", dojo_index, dojo_test, "contract_test", fc.purpose))
    if dyn_test:
        test_frames.append(episode_windows(ctx, "dyn", dyn_index, dyn_test, "contract_test", fc.purpose))
    test_frame = pd.concat(test_frames, ignore_index=True) if test_frames else trainval.iloc[0:0]
    if not len(test_frame):
        raise ValueError("no test episodes with documents (contract §6)")
    steps = step_scores(fc, fitted, "contract_test", test_frame)
    records = {r["episode_id"]: r for r in episodes[episodes["episode_id"].isin(test_ids)].to_dict("records")}
    rows: list[ContractRow] = []
    per_variant: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        tau = float(thresholds[f"contract/{VARIANTS[v]}"]["value"])
        es = episode_scores(steps, test_ids, v, tau)
        per_variant[v] = es
        for r in es.itertuples(index=False):
            rows.append(ContractRow.from_episode(records[r.episode_id], v, None if _missing(r.alarm_step) else int(r.alarm_step),
                                                 float(r.max_score), tau, n_benign_val))
    csv_path = contract_csv_path(root, smoke)
    write_csv(rows, csv_path)
    problems = validate_csv(csv_path)
    if problems:
        raise ValueError(f"{csv_path} fails the contract §8 validator: {problems[:5]}")
    buf = io.StringIO()
    steps.assign(benchmark=steps["episode_id"].map(bench_of)).to_csv(buf, index=False, lineterminator="\n")
    atomic_write_text(step_scores_path(root, smoke), buf.getvalue())

    # -- episodes without documents ---------------------------------------------------------------------------------
    es0 = per_variant[next(iter(VARIANTS))]
    without = es0.loc[es0["n_documents"] == 0, "episode_id"].tolist()
    nothing_to_scan = [e for e in without if records[e]["n_steps"] == 0]
    not_extracted = [e for e in without if records[e]["n_steps"] > 0 and not _missing(records[e]["attack"])
                     and _missing(records[e].get("injection_step"))]
    no_step_document = [e for e in without if e not in nothing_to_scan and e not in not_extracted]
    if without:
        notes.append(f"{len(without)} test episodes have no step document (max_score 0.0, no alarm): {len(nothing_to_scan)} without "
                     f"tool calls, {len(no_step_document)} whose tool outputs produced no document in the tables (empty outputs"
                     + ("; in smoke mode also steps outside the documents-per-source subset" if ctx.smoke else "")
                     + f"), {len(not_extracted)} attacked episodes whose injection text the extractor did not recover "
                     "(for these 0.0 means 'not scanned', not 'clean')")

    # -- metrics (contract §9) --------------------------------------------------------------------------------------
    n_boot, seed_boot = ctx.bootstrap_n(), fc.seeds["bootstrap"]
    dict_rows = [asdict(r) for r in rows]
    metrics = contract_metrics_by_variant(dict_rows, n_boot=n_boot, seed=seed_boot)
    by_bench: dict[str, dict[str, Any]] = {}
    for b in sorted(set(bench_of[e] for e in test_ids)):
        sub = [r for r in dict_rows if bench_of[r["episode_id"]] == b]
        if sub:
            by_bench[b] = benchmark_metrics(sub, n_boot=n_boot, seed=seed_boot)
    numbers: dict[str, Any] = {}

    def add_numbers(prefix: str, table: Mapping[str, Mapping[str, Any]], point_note: str | None = None) -> None:
        for v, m in table.items():
            det = VARIANTS.get(v, v.replace("/", "_"))
            for key in METRIC_KEYS:
                if key in m:
                    ci = (m.get("ci") or {}).get(key)
                    # overall rates that carry an interval there but only a point value in the per-benchmark split
                    note = point_note if (point_note and ci is None and key in OVERALL_CI_KEYS) else None
                    numbers[results_mod.check_key(f"{prefix}/{key}/{det}")] = results_mod.number(m[key], ci, note=note)

    add_numbers("contract", metrics)
    for b, table in by_bench.items():
        add_numbers(f"contract/{b}", table, point_note="per-benchmark split: point estimate without interval")
    if by_bench:
        notes.append(f"per-benchmark metrics ({', '.join(sorted(by_bench))}) carry intervals for {list(BENCHMARK_CI_KEYS)} only; "
                     "the contract's §9 table is the overall one (contract/<metric>/<detector>)")
    for v, det in VARIANTS.items():
        numbers[f"contract/threshold/{det}"] = results_mod.number(thresholds[f"contract/{det}"]["value"], n=n_benign_val)
        for hp in ("gamma", "C"):
            if hp in fitted[v].choices:
                numbers[f"hyper/{det}/{hp}"] = results_mod.number(fitted[v].choices[hp], note=str(fitted[v].choices.get(f"{hp}_source")))
    numbers["contract/n_test_episodes"] = results_mod.number(len(test_ids))
    numbers["contract/n_validation_benign_episodes"] = results_mod.number(n_benign_val)
    numbers["contract/n_test_episodes_without_documents"] = results_mod.number(len(without))

    variants_table = [{"variant": v, "detector": VARIANTS[v], "threshold": thresholds[f"contract/{VARIANTS[v]}"]["value"],
                       "threshold_n_benign": n_benign_val, "threshold_target": thresholds[f"contract/{VARIANTS[v]}"]["target"],
                       "fit_seconds": fitted[v].fit_seconds, "n_train_windows": fitted[v].n_train,
                       **{k: val for k, val in fitted[v].choices.items() if not isinstance(val, (dict, list))}}
                      for v in VARIANTS]
    ep_table = [{"benchmark": b, "contract_split": s, "episode_class": c, "n": int(len(g)),
                 "n_with_documents": int(g["episode_id"].isin(with_docs).sum())}
                for (b, s, c), g in episodes.groupby(["benchmark", "contract_split", "episode_class"], sort=True)]
    notes.append("contract §3: every window of a step document is scored (E1 dedup exclusions not applied); alarm when score >= threshold")
    return {
        "numbers": numbers, "thresholds": thresholds, "notes": notes,
        "tables": {"variants": variants_table, "training": composition, "episodes": ep_table,
                   "split_check": [{"check": "split_manifest", **{k: v for k, v in check.items() if k != "counts"}}]},
        "validation_episodes": val_table,
        "metrics": metrics, "metrics_by_benchmark": by_bench,
        "csv": str(csv_path.relative_to(root)) if csv_path.is_relative_to(root) else str(csv_path), "csv_valid": True,
        "n_rows": len(rows), "step_scores": str(step_scores_path(root, smoke)),
        "split_manifest": {"path": str(mpath), **check},
        "episodes_without_documents": {"nothing_to_scan": sorted(nothing_to_scan), "no_step_document": sorted(no_step_document),
                                       "not_extracted": sorted(not_extracted)},
        "training_mode": "D11 + dojo_negatives" if dojo_negatives else "D11",
    }


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path = ROOT, cfg: Configs | None = None,
        keep_cache: bool = False, force: bool = False, guard_factory: Any = None, access_log: Any = None,
        cache: bool = True, dojo_negatives: bool = False) -> Path:
    """Run the contract for one global seed; writes the CSV, the step scores and ``results/contract.json``
    (``results/smoke/...`` in smoke mode). Skipped when the result is current unless ``force``."""
    root = Path(root)
    out = contract_result_path(root, smoke)
    if not force and is_current(root, smoke):
        return out
    if ctx is None:
        ctx = Context(cfg, smoke=smoke, root=root, access_log=access_log)
    fc = FeatureContext(ctx, seed, purpose=f"{EXPERIMENT} seed={seed}", cache=cache, guard_factory=guard_factory)
    t0 = time.perf_counter()
    n_before = len(ctx.test_reads)
    try:
        payload = contract_body(fc, root, smoke, dojo_negatives=dojo_negatives)
        payload.update({
            "experiment": EXPERIMENT, "seed": int(seed), "config_hash": config_hash(root), "git_commit": git_commit(root),
            "seeds": dict(fc.seeds), "smoke": bool(smoke),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timing": {"seconds": time.perf_counter() - t0, "cache_bytes": fc.cache_bytes(),
                       "test_reads": [s for s, _ in ctx.test_reads[n_before:]]},
        })
        payload["notes"].append(f"seed children: {fc.seeds}")
        atomic_write_json(out, payload)
        return out
    finally:
        if not keep_cache:
            fc.cleanup()


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Plug-in hook of :mod:`flyguard.experiments.run` (``run contract --dojo-negatives``)."""
    parser.add_argument("--dojo-negatives", dest="dojo_negatives", action="store_true",
                        help="also use clean AgentDojo train/val episode windows as label-0 rows (default: D11, C_unl only)")


def configs_for(args: argparse.Namespace) -> Configs | None:
    """``args.cfg`` when the caller passed one, else the configs of ``args.root`` when that tree has its own
    ``configs/`` (a checkout), else ``None`` so that :class:`Context` falls back to this checkout's configs (test
    roots hold data only). Nothing is loaded before the currency check of :func:`run` needs it."""
    cfg = getattr(args, "cfg", None)
    if cfg is not None:
        return cfg
    root = Path(getattr(args, "root", ROOT))
    return load_configs(root) if (root / "configs" / "default.yaml").exists() else None


def run_cli(args: argparse.Namespace) -> Path:
    """Plug-in hook of :mod:`flyguard.experiments.run`: one CSV, the first of ``args.seeds`` (contract §8 exchanges
    one file per detector, not one per seed)."""
    root = Path(getattr(args, "root", ROOT))
    seeds = list(getattr(args, "seeds", None) or [0])
    return run(None, seeds[0], args.smoke, root=root, cfg=configs_for(args), keep_cache=args.keep_cache,
               force=args.force, dojo_negatives=bool(getattr(args, "dojo_negatives", False)))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Contract run (ТЗ Этап 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    add_cli_arguments(p)
    a = p.parse_args(argv)
    a.seeds = [a.seed]
    print(f"contract: {run_cli(a)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
