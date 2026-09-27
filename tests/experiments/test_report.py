"""scripts/make_report.py and scripts/check_acceptance.py on a synthetic results tree (ТЗ "Критерии приёмки":
каждое число REPORT.md прослеживается до results/; the report renders with missing experiments; the acceptance
helpers flip on the right inputs), and the scripts/run_all.sh driver on a fake root with a stub interpreter (a
failing command inside a stage fails the stage). Deterministic, synthetic, no network, no real data."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from flyguard.config import ROOT
from flyguard.experiments.results import number, write_result
from flyguard.io import atomic_write_json

SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mr = load_script("make_report")
ca = load_script("check_acceptance")

DETS = ("tfidf_lr", "lr_svd", "real_fly_bloom", "real_fly_linear", "flyhash_bloom", "knn1", "protectai_v2", "piguard")
SOURCES = ("deep", "bipia", "dojo", "dyn", "para", "para_deep")


def ci(point: float, half: float, level: float = 0.95, n: int = 100) -> dict:
    return {"point": point, "low": point - half, "high": point + half, "level": level, "n_boot": 60, "n": n}


def e1_numbers(seed: int) -> dict:
    """Keys shaped like flyguard.experiments.engine.standard_evaluation (results.py naming rules)."""
    nums: dict = {}
    for i, d in enumerate(DETS):
        for j, s in enumerate(SOURCES):
            v = round(0.6 + 0.04 * i + 0.01 * j + 0.005 * seed, 4)
            nums[f"auc/{s}/{d}"] = number(None, ci(v, 0.03, n=50 + j))
        nums[f"macro_auc/{d}"] = number(None, ci(0.7 + 0.02 * i + 0.003 * seed, 0.02), note="sources=bipia,deep,dojo,dyn,para")
        nums[f"tpr_at_fpr/deep/{d}"] = number(None, ci(0.5 + 0.03 * i, 0.05))
        nums[f"fpr_ptest/{d}"] = number(None, ci(0.05, 0.01))
        for tau in ("tau_fpr", "tau90_deep"):
            nums[f"fpr_notinject/{tau}/{d}"] = number(None, ci(0.2 + 0.02 * i, 0.04))
            for sub in ("one", "two", "three", "en", "non-en"):
                nums[f"fpr_notinject/{tau}/{d}/{sub}"] = number(None, ci(0.15 + 0.02 * i, 0.05))
        nums[f"val_auc/deep/{d}"] = number(0.8 + 0.01 * i, n=110)
        nums[f"val_macro_auc/{d}"] = number(0.79 + 0.01 * i)
        nums[f"latency_ms/{d}"] = number(1.25 * (i + 1), n=200)
        nums[f"state_bytes/{d}"] = number(1024 * (i + 1))
    nums["hyper/real_fly_bloom/gamma"] = number(0.9, note="validation")
    nums["hyper/tfidf_lr/C"] = number(10.0, note="validation")
    nums["diff/macro_auc/real_fly_linear-lr_svd"] = number(None, ci(-0.01, 0.02), p=0.31)
    nums["diff90/macro_auc/real_fly_linear-lr_svd"] = number(None, ci(-0.01, 0.015, level=0.9))
    nums["diff/auc/deep/tfidf_lr-protectai_v2"] = number(None, ci(0.02, 0.03), p=0.2)
    nums["diff/fpr_notinject/tau90_deep/real_fly_bloom-protectai_v2"] = number(None, ci(0.03, 0.05), p=0.4)
    return nums


def write_results(root: Path) -> None:
    thresholds = {f"tau_fpr/{d}": {"value": 0.42 + 0.01 * i, "source": "P_val", "target": "FPR=0.05", "n": 107}
                  for i, d in enumerate(DETS)}
    thresholds.update({f"tau90_deep/{d}": {"value": 0.3, "source": "deep test positives", "target": "TPR=0.9", "n": 40} for d in DETS})
    tables = {"detectors": [{"detector": d, "kind": "fly" if "fly" in d else "lexical", "fit_seconds": 0.5 + i} for i, d in enumerate(DETS)],
              "sources": [{"source": s, "n_docs": 100 + i, "n_pos": 40, "n_neg": 60 + i, "n_clusters": 90} for i, s in enumerate(SOURCES)]}
    grid = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)     # e1.roc_tables layout: TPR on a common FPR grid; regex as points
    tables["roc"] = [{"source": s, "detector": d, "fpr": f, "tpr": min(1.0, f + 0.3 + 0.05 * i)}
                     for i, d in enumerate(("tfidf_lr", "real_fly_bloom", "protectai_v2", "regex")) for s in ("deep", "bipia") for f in grid]
    tables["roc_points"] = [{"source": s, "detector": "regex", "fpr": 0.1, "tpr": 0.4, "threshold": 1.0} for s in ("deep", "bipia")]
    for seed in (0, 1):
        write_result("E1", seed, e1_numbers(seed), tables, thresholds, ["test reads: 6; bootstrap n=60, alpha=0.05", f"seed children: {{'nose': {seed}}}"],
                     root=root, seeds={"nose": seed}, timing={"seconds": 12.5 + seed, "test_reads": ["deep", "bipia"]})
    # E2: fewshot/<metric>/<level>/<detector> (e2.level_key -> shots<k> | full), paired diffs per level
    e2 = {f"fewshot/macro_auc/{lvl}/{d}": number(None, ci(0.5 + 0.05 * i + 0.1 * j, 0.03), n_reps=10,
                                                  band_low=0.4 + 0.05 * i + 0.1 * j, band_high=0.6 + 0.05 * i + 0.1 * j)
          for i, d in enumerate(("real_fly_bloom", "knn1", "tfidf_lr")) for j, lvl in enumerate(("shots1", "shots10", "shots100", "full"))}
    e2["diff/macro_auc/shots1/real_fly_bloom-knn1"] = number(None, ci(0.01, 0.04), p=0.6)
    e2["diff90/macro_auc/shots1/real_fly_bloom-knn1"] = number(None, ci(0.01, 0.03, level=0.9), n_reps=10)
    e2_tables = {"fewshot_curve": [{"level": 1, "rep": 0, "detector": "real_fly_bloom", "macro_auc": 0.52,
                                    "macro_ci_low": 0.49, "macro_ci_high": 0.55, "auc_deep": 0.51}]}
    write_result("E2", 0, e2, e2_tables, {}, ["subsamples=10"], root=root, timing={"seconds": 3.0, "test_reads": []})
    # E4: the H3 record diff90/macro_auc/<det>-curveball_mean (reference + p_randomization extras), the null family
    # numbers and the per-null histogram table "curveball" (matrix column curveball:<j>, metric column macro_auc)
    e4 = {"macro_auc/real_fly_bloom": number(None, ci(0.81, 0.02)),
          "diff90/macro_auc/real_fly_bloom-curveball_mean": number(None, ci(0.004, 0.02, level=0.9), reference=0.806,
                                                                   p_randomization=0.43, delta=0.0403),
          "diff/macro_auc/real_fly_bloom-curveball_mean": number(None, ci(0.004, 0.025), p=0.43, reference=0.806),
          "macro_auc/curveball_mean/real_fly_bloom": number(None, ci(0.806, 0.01)),
          "macro_auc/pi_mean/real_fly_bloom": number(None, ci(0.81, 0.01)),
          "p_randomization/macro_auc/real_fly_bloom": number(0.43, n=20),
          "tost_equivalent/macro_auc/real_fly_bloom-curveball_mean": number(1.0, note="1 = 90% CI inside +/-delta"),
          "contrib/wiring/real_fly_bloom/macro_auc": number(None, ci(0.004, 0.02), p=0.43),
          "curveball_sd/macro_auc/real_fly_bloom": number(0.003, n=20), "pi_sd/macro_auc/real_fly_bloom": number(0.002, n=10)}
    e4_tables = {"curveball": [{"readout": "bloom", "detector": "real_fly_bloom", "j": j, "matrix": f"curveball:{j}",
                                "macro_auc": 0.80 + 0.001 * (j % 7)} for j in range(20)],
                 "h3": [{"detector": "real_fly_bloom", "measured": 0.81, "null_mean": 0.806, "diff": 0.004, "p_randomization": 0.43}]}
    write_result("E4", 0, e4, e4_tables, {}, ["perm seeds: 10", "two_stage bootstrap n=60; n_perms=10"], root=root,
                 timing={"seconds": 40.0, "test_reads": ["deep"]})
    # E5: cells <nose>__<expansion>__<readout>, contrib/<kind>/.../<metric> numbers and the contributions table
    cells5 = ("n51_svd__none__linear", "n51_svd__measured__bloom", "n51_svd__measured__linear", "n16k__random_1886__linear")
    e5 = {f"auc/{s}/{n}": number(None, ci(0.6 + 0.02 * i, 0.03)) for i, n in enumerate(cells5) for s in ("deep", "bipia")}
    e5.update({f"macro_auc/{n}": number(None, ci(0.65 + 0.01 * i, 0.02)) for i, n in enumerate(cells5)})
    e5["contrib/nose/n51_svd/macro_auc"] = number(None, ci(0.65, 0.02))
    e5["contrib/expansion/n51_svd/measured/macro_auc"] = number(None, ci(0.02, 0.01), p=0.1)
    e5["contrib/rule/n51_svd/measured/macro_auc"] = number(None, ci(-0.01, 0.01), p=0.4)
    e5["contrib/wiring/n51_svd/bloom/macro_auc"] = number(None, ci(0.003, 0.004), p=0.2, note="AUC(measured) - mean AUC over 20 curveball nulls")
    e5_tables = {"cells": [{"cell": n, "nose": n.split("__")[0], "expansion": n.split("__")[1], "readout": n.split("__")[2],
                            "m": 1886, "k": 94, "gamma": 0.9 if "bloom" in n else None, "C": None if "bloom" in n else 1.0,
                            "macro_auc": 0.65 + 0.01 * i} for i, n in enumerate(cells5)],
                 "contributions": [{"kind": "expansion", "nose": "n51_svd", "expansion": "measured", "readout": "linear",
                                    "metric": "macro_auc", "value": 0.02, "ci_low": 0.01, "ci_high": 0.03, "level": 0.95, "p": 0.1,
                                    "cell": "n51_svd__measured__linear"},
                                   {"kind": "wiring", "nose": "n51_svd", "expansion": "measured", "readout": "bloom",
                                    "metric": "macro_auc", "value": 0.003, "ci_low": -0.001, "ci_high": 0.007, "level": 0.95,
                                    "p": 0.2, "n_null": 20, "cell": "n51_svd__measured__bloom"}]}
    write_result("E5", 0, e5, e5_tables, {}, [], root=root, timing={"seconds": 9.0, "test_reads": ["deep", "bipia"]})


def write_side_files(root: Path, current_hash: str) -> None:
    now = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)
    stamp = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    atomic_write_json(root / "results/power.json", {
        "stage": 2, "frozen": True, "created_at": stamp(now - timedelta(hours=1)), "config_hash": current_hash,
        "delta_rel": 0.05, "alpha": 0.05, "fpr_target": 0.05,
        "sizes": {"deep": {"n_pos": 55, "n_neg": 60, "n_clusters": 115}, "macro": {"n_pos": 300, "n_neg": 400, "n_clusters": 500}},
        "carriers": {"deep": {"auc": "несёт", "auc_diff": "несёт", "tpr_at_fpr": "только 5%"}, "macro": {"auc": "несёт", "auc_diff": "не хватило данных", "tpr_at_fpr": "только 5%"}},
        "planning_level": {"deep": 0.85, "macro": 0.85},
        "cells": {"deep": {"0.85": {"mdd": 0.061, "delta": 0.0425, "power": 0.83, "status": "несёт"}}, "macro": {"0.85": {"mdd": 0.03, "delta": 0.0425, "power": 0.6, "status": "не хватило данных"}}},
        "notinject": {"n": 339, "width_pp": [{"fpr": 0.1, "width": 0.064}]}, "spread": {"curveball": {"sd": 0.004, "n": 20}, "perm": {"sd": 0.002, "n": 10}},
        "hypotheses": {"H2": {"notinject_n": 339}}})
    atomic_write_json(root / "results/verdicts.json", {
        "config_hash": current_hash, "created_at": stamp(now),
        "H1a": {"status": "не хватило данных", "effect": {"template": {"deep": 0.02}, "semantic": {}}, "ci": {"template": {"deep": ci(0.02, 0.03, 0.9)}},
                "reason": "нет семантических источников", "inputs": {"template": {"deep": {"carrier": True, "delta": 0.045}}}, "hypothesis": "H1a"},
        "H1b": {"real_fly": {"status": "подтверждена", "effect": {"linear_equivalence": -0.01}, "ci": {"linear_equivalence": ci(-0.01, 0.015, 0.9)},
                             "reason": "90 % ДИ внутри ±0.045", "inputs": {"macro_auc": {"carrier": True}}, "hypothesis": "H1b"}},
        "H2": {"status": "предусловие не выполнено", "effect": {"fly_minus_protectai": None}, "ci": {}, "reason": "AUC мухи на валидации deepset 0.712 < 0.75",
               "inputs": {"val_auc_deep": 0.712, "precondition": 0.75, "margin": 0.1}, "hypothesis": "H2"},
        "H3": {"status": "опровергнута", "effect": 0.03, "ci": ci(0.03, 0.01, 0.9), "reason": "интервал вне коридора",
               "inputs": {"precondition": 0.75, "val_macro_auc": 0.8, "macro_auc_measured": 0.81, "carrier": True}, "hypothesis": "H3"}})
    atomic_write_json(root / "results/contract.json", {"config_hash": current_hash, "variants": {"real_fly/bloom": {"stopped_before_harm": {"value": 0.4, "low": 0.2, "high": 0.6}, "fa_per_100": 1.5, "n_hijacked": 20}}})
    atomic_write_json(root / "results/spend.json", {"budget_usd": 7.0, "spent_usd": 4.64, "n_calls": 6195, "within_budget": True,
                                                    "breakdown": {"stream:traces": {"calls": 6195, "cost_usd": 4.12}, "model:deepseek-v4-pro": {"calls": 6030, "cost_usd": 4.08}}})
    atomic_write_json(root / "results/pilot.json", {"chosen_model": "m-pro", "chosen_by_rule": True,
                                                    "candidates": [{"model": "m-flash", "targeted_asr": 0.0, "utility_clean": 1.0, "pilot_cost_usd": 0.034, "accepted": False},
                                                                   {"model": "m-pro", "targeted_asr": 0.35, "utility_clean": 0.9, "pilot_cost_usd": 0.169, "accepted": True}]})
    atomic_write_json(root / "results/paraphrases.json", {"counts": {"bases": {"template": 140, "deepset_benign": 56}, "selected": 300},
                                                          "rates": {"generation_refusal": {"by_kind": {"template": {"n": 280, "refusals": 14, "rate": 0.05}}},
                                                                    "acceptance": {"by_stratum": {"deep": {"n": 200, "accepted": 120, "rate": 0.6}}}}})
    atomic_write_json(root / "data/manifests/sources.json", {"generated": stamp(now), "pins": {"malecns": {"version": "v1.0", "minconf": 0.5}},
                                                             "files": [{"path": "data/raw/deepset/train.parquet", "url": "https://huggingface.co/x", "sha256": "a" * 64, "bytes": 1},
                                                                       {"path": "data/processed/connectome/malecns_R.npz", "url": "flypath build (x)", "sha256": "b" * 64, "bytes": 2}]})
    atomic_write_json(root / "results/shared/traces_manifest.json", {"agent_model": "m-pro", "provider": "p", "temperature": 0.0, "thinking": "disabled", "generated": stamp(now),
                                                                    "counts": {"agentdojo": {"workspace": {"important_instructions": {"hijacked": 71, "injection_ignored": 489}, "none": {"benign": 40}}}}, "files": []})
    atomic_write_json(root / "results/shared/split_manifest.json", {"test": [], "observation": [], "validation_clean": [], "validation_attacks": [], "train_attacks": [],
                                                                   "rule": {"mod": 3, "rem": 2}, "counts": {"test": 591, "observation": 49, "validation_clean": 52, "validation_attacks": 0, "train_attacks": 0}})
    atomic_write_json(root / "data/manifests/traces_extraction.json", {"agentdojo": {"n_logs": 1046, "documents": 3840, "documents_positive": 700, "steps_total": 3900, "steps_labelled": 690,
                                                                                     "attacked_without_span": {"count": 7}, "errors": 0}})
    write_report_inputs(root, stamp(now - timedelta(hours=3)))
    (root / "DEVIATIONS.md").write_text("# Отклонения\n\n- **D1 (2026-09-26).** Один провайдер.\n- **D10 (2026-09-26).** Журнал задним числом (retroactive).\n", encoding="utf-8")
    (root / "ASSUMPTIONS.md").write_text("- **A1.** x\n", encoding="utf-8")
    (root / "BLOCKERS.md").write_text("- **B1.** x\n", encoding="utf-8")


def write_report_inputs(root: Path, created: str) -> None:
    """Inputs of report sections 3, 4, 5 and 9: traces_stats.json (schema of flyguard.gen.trace_stats), the counts of
    contamination.json, an audit.md with the section headings of flyguard.data.audit, the E0 stage-1 table."""
    pub = {m: {"workspace": {"targeted_asr": a, "utility_clean": u, "n_attacked": 560, "n_clean": 40}}
           for m, a, u in (("model-a", 0.1, 0.8), ("model-b", 0.25, 0.7), ("model-c", 0.4, 0.6))}
    atomic_write_json(root / "results/traces_stats.json", {
        "ours": {"agentdojo": {"workspace": {"n_attacked": 560, "n_hijacked": 71, "targeted_asr": 0.1268, "n_clean": 40,
                                             "utility_clean": 0.9, "utility_under_attack": 0.72}}},
        "published": {"agentdojo": pub}, "published_source": "https://github.com/x/agentdojo@abc123, data/ext/runs",
        "same_model_published": False, "notes": ["agentdojo: 3 undefended models, 1680 logs read"]})
    lvl = lambda n, m: {"documents": n, "documents_matched": m, "share": round(m / n, 4)}  # noqa: E731
    atomic_write_json(root / "data/manifests/contamination.json", {
        "skipped": False, "document_level": {"deep": {"train": {"0": lvl(274, 274), "1": lvl(162, 162)}, "test": {"0": lvl(56, 1), "1": lvl(60, 3)}},
                                             "bipia": {"test": {"0": lvl(140, 14), "1": lvl(140, 5)}}},
        "window_level": {"deep": {"test": {"0": {"windows": 58, "windows_matched": 1, "share": 0.0172}}}},
        "targeted_containment": {"deep": {"tags": ["prompt-injections"], "by_split": {"test": {"documents": 116, "ours_in_piguard": 8, "piguard_in_ours": 7}}}},
        "model_cards": {"protectai_v2": {"model": "protectai/deberta-v3-base-prompt-injection-v2", "training_datasets_on_card": ["VMware/open-instruct"],
                                         "named_overlap_with_our_sources": "none of the listed datasets"}},
        "threats_to_validity": ["NotInject, PIGuard and AgentDyn share a first author", "BIPIA is part of the PIGuard authors' test set"]})
    (root / "data/manifests/audit.md").write_text(
        "# Аудит данных (ТЗ 1.2)\n\n## Документы по источникам\n| источник | документов | test |\n|---|---|---|\n| deep | 662 | 116 |\n\n"
        "## Окна (ТЗ 1.3)\n| источник | окон | окон на документ |\n|---|---|---|\n| deep | 777 | 1.17 |\n\n"
        "## Языки (langdetect, сид из конфига)\n| источник | en | non-en |\n|---|---|---|\n| deep | 355 | 307 |\n\n"
        "Доля немецкого в deepset:\n\n| часть | de | доля de |\n|---|---|---|\n| all | 265 | 40.0 % |\n", encoding="utf-8")
    atomic_write_json(root / "results/E0/power_stage1.json", {
        "stage": 1, "frozen": False, "created_at": created, "sizes": {"deep": {"n_pos": 41, "n_neg": 55, "n_clusters": 96}},
        "carriers": {"deep": {"auc": "несёт"}}, "planning_level": {"deep": 0.75},
        "cells": {"deep": {"0.75": {"mdd": 0.09, "delta": 0.0375, "tost_power": 0.4, "status": "не хватило данных"}}}})


@pytest.fixture(scope="module")
def report_root(tmp_path_factory, toy_root) -> Path:
    """A copy of the toy tree (tables, manifests, connectome) plus synthetic results and side files."""
    root = tmp_path_factory.mktemp("report_root") / "repo"
    shutil.copytree(toy_root, root)
    # other test modules run experiments inside the session-scoped toy tree (results with the toy config's hash,
    # feature caches, score caches): the report tree starts from the tables and manifests only
    for stale in ("results", "logs", "data/processed/features", "data/processed/scores_cache"):
        shutil.rmtree(root / stale, ignore_errors=True)
    shutil.copytree(ROOT / "configs", root / "configs")
    from flyguard.config import config_hash

    write_results(root)
    write_side_files(root, config_hash(root))
    return root


# ------------------------------------------------------------------------------------------------ independent resolver
NUM_REF = re.compile(r"(?P<v>-?\d+(?:\.\d+)?)(?: \[(?P<lo>-?\d+(?:\.\d+)?), (?P<hi>-?\d+(?:\.\d+)?)\])? \((?P<path>[\w./-]+)#(?P<key>[^\s()]+)\)")


def resolve(root: Path, path: str, key: str):
    p = root / path
    node = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else yaml.safe_load(p.read_text(encoding="utf-8"))
    segs = key.split("/")
    i = 0
    while i < len(segs):
        if isinstance(node, list):
            node, i = node[int(segs[i])], i + 1
            continue
        for j in range(len(segs), i, -1):
            if "/".join(segs[i:j]) in node:
                node, i = node["/".join(segs[i:j])], j
                break
        else:
            raise KeyError(key)
    return node


def leaf_values(node) -> list:
    """Numbers and strings under a node (a number quoted inside a string, e.g. a verdict reason, traces by substring)."""
    if isinstance(node, bool) or node is None:
        return []
    if isinstance(node, (int, float, str)):
        return [node]
    if isinstance(node, dict):
        return [x for v in node.values() for x in leaf_values(v)]
    return [x for v in node for x in leaf_values(v)]


def close(tok: str, values: list) -> bool:
    """Exact after the rendering format (no tolerance): an integer token needs an integral value, a token with d
    decimals a value that formats to it; strings contain it as a whole number."""
    dec = len(tok.split(".")[1]) if "." in tok else 0
    whole = re.compile(rf"(?<![\d.]){re.escape(tok)}(?!\d|\.\d)")
    return any(bool(whole.search(v)) if isinstance(v, str) else
               ((float(v).is_integer() and int(v) == int(tok)) if dec == 0 else float(f"{v:.{dec}f}") == float(tok))
               for v in values)


# ------------------------------------------------------------------------------------------------ tests
def test_every_number_is_found_in_the_referenced_json(report_root):
    text, path = mr.render(report_root, smoke=False, figures=True)
    assert path == report_root / "REPORT.md" and path.exists()
    hits = list(NUM_REF.finditer(text))
    assert len(hits) > 200
    for m in hits:
        node = resolve(report_root, m.group("path"), m.group("key"))
        values = leaf_values(node)
        assert values, m.group(0)
        for g in ("v", "lo", "hi"):
            if m.group(g) is not None:
                assert close(m.group(g), values), m.group(0)
    checked, problems = mr.trace_numbers(text, report_root)
    assert checked > 300 and problems == []
    # the sections of ТЗ Этап 6 and the pivot tables of E1
    assert all(re.search(rf"^## {i}\. ", text, re.M) for i in range(1, 11))
    assert "| tfidf_lr |" in text and "results/E1/summary.json#numbers/macro_auc/tfidf_lr" in text
    assert "подтверждена" in text and "results/verdicts.json#H1b/real_fly/status" in text
    # two seeds (A55): the mean over seeds with the envelope of the per-seed bootstrap intervals, then sd and n_seeds
    s = json.loads((report_root / "results/E1/summary.json").read_text(encoding="utf-8"))
    rec = s["numbers"]["macro_auc/tfidf_lr"]
    lo, hi = min(r["ci_low"] for r in rec["per_seed"].values()), max(r["ci_high"] for r in rec["per_seed"].values())
    assert (f"{rec['mean']:.3f} [{lo:.3f}, {hi:.3f}] (results/E1/summary.json#numbers/macro_auc/tfidf_lr); "
            f"sd {rec['sd']:.3f}, сидов 2") in text
    assert "seed_ci" not in text and "t-интервал" not in text
    figs = {p.name for p in (report_root / "results/figures").glob("*.png")}
    assert {"e1_roc_by_source.png", "e1_auc_by_source.png", "e2_learning_curves.png", "e4_curveball_hist.png",
            "e1_notinject_fpr_tau90.png", "para_strata_auc.png"} <= figs
    assert "![ROC-кривые E1 по источникам" in text and "| `roc` |" not in text     # the curves are a figure, not a dump
    # section 2 lists the commits the results were computed on, apart from the commit the report is rendered on
    assert "**Коммиты, на которых посчитаны результаты**" in text
    assert "(results/E1/summary.json#git_commits) | 0, 1 (results/E1/summary.json#seeds) |" in text
    assert "(results/verdicts.json#git_commit)" in text and "(results/E0/power_stage1.json#git_commit)" in text
    assert "Коммит, на котором собран отчёт:" in text


def test_report_renders_without_any_experiment(tmp_path):
    root = tmp_path / "empty"
    shutil.copytree(ROOT / "configs", root / "configs")
    text, path = mr.render(root, smoke=False, figures=False)
    assert all(re.search(rf"^## {i}\. ", text, re.M) for i in range(1, 11))
    assert text.count("не выполнено") >= 10 and "results/traces_stats.json нет" in text
    assert mr.trace_numbers(text, root)[1] == []
    text_s, path_s = mr.render(root, smoke=True, figures=False)
    assert path_s == root / "results/smoke/REPORT.md" and (root / "results/smoke/setup.json").exists()


def test_verifier_catches_wrong_and_unreferenced_numbers(report_root):
    good = "| a | 0.700 [0.680, 0.720] (results/E1/summary.json#numbers/macro_auc/tfidf_lr) |"
    text = "| k | v |\n|---|---|\n" + good + "\n| b | 0.123 (results/E1/summary.json#numbers/macro_auc/tfidf_lr) |\n| c | 42 |\n"
    _, problems = mr.trace_numbers(text, report_root)
    assert len(problems) == 2 and any("0.123" in p for p in problems) and any("without reference" in p for p in problems)
    assert mr.trace_numbers("Смотри ТЗ 1.7, §6 и Этап 4; H1a; N16k; `x = 512`\n", report_root)[1] == []


def test_seed_summary_is_mean_with_bootstrap_envelope_and_spread():
    """ASSUMPTIONS A55: several seeds -> the mean, the envelope [min ci_low, max ci_high] of the per-seed cluster
    bootstrap intervals, then sd and n_seeds; one seed -> its own value and interval. A seed-independent detector keeps
    its bootstrap width instead of the zero-width t-interval of the seed mean. ROC rows are averaged per FPR point."""
    from types import SimpleNamespace

    per = {"0": {"value": 0.8, "ci_low": 0.7, "ci_high": 0.9}, "1": {"value": 0.8, "ci_low": 0.72, "ci_high": 0.88},
           "2": {"value": 0.8, "ci_low": 0.69, "ci_high": 0.91}}
    rec = {"mean": 0.8, "sd": 0.0, "n_seeds": 3, "seed_ci_low": 0.8, "seed_ci_high": 0.8, "per_seed": per}
    assert mr.rec_triple(rec) == (0.8, 0.69, 0.91)
    assert mr.rec_num({"numbers": {"k": rec}}, "results/E1/summary.json", "k") == \
        "0.800 [0.690, 0.910] (results/E1/summary.json#numbers/k); sd 0.000, сидов 3"
    one = {"mean": 0.8, "n_seeds": 1, "per_seed": {"4": {"value": 0.81, "ci_low": 0.7, "ci_high": 0.9}}}
    assert mr.rec_triple(one) == (0.81, 0.7, 0.9) and "сидов" not in mr.rec_num({"numbers": {"k": one}}, "p.json", "k")
    roc = [{"source": "deep", "detector": "tfidf_lr", "fpr": 0.1, "tpr": t, "seed": s} for s, t in ((0, 0.2), (1, 0.4))]
    roc.append({"source": "deep", "detector": "knn5", "fpr": 0.1, "tpr": 0.9, "seed": 0})          # not a figure detector
    pts = [{"source": "deep", "detector": "regex", "fpr": f, "tpr": 0.5, "threshold": 1.0, "seed": s} for s, f in ((0, 0.1), (1, 0.3))]
    curves, points = mr.roc_data(SimpleNamespace(summaries={"E1": {"tables": {"roc": roc, "roc_points": pts}}}))
    assert curves == {"deep": {"tfidf_lr": [(0.1, pytest.approx(0.3))]}} and points == {"deep": {"regex": [(pytest.approx(0.2), 0.5)]}}


def test_verifier_is_exact(tmp_path):
    """No tolerance: an integer token needs an integral value, a decimal token a value that renders to it; strings
    contain the token as a whole number; dotted versions are not numbers of the report."""
    atomic_write_json(tmp_path / "r.json", {"a": 0.9, "b": 2.6, "c": 3, "d": 0.1234, "s": "reason 0.4 and 12 items"})
    ok = lambda line: mr.trace_numbers(line + "\n", tmp_path)[1] == []  # noqa: E731
    assert not ok("x 1 (r.json#a)") and ok("x 0.9 (r.json#a)") and ok("x 0.900 (r.json#a)")
    assert not ok("x 3 (r.json#b)") and ok("x 3 (r.json#c)") and ok("x 3.0 (r.json#c)")
    assert ok("x 0.123 (r.json#d)") and not ok("x 0.124 (r.json#d)")
    assert ok("x 0.4 and 12 (r.json#s)") and not ok("x 4 (r.json#s)") and not ok("x 1 (r.json#s)")
    assert ok("Python 3.11.14 без ссылки") and not ok("x 7 без ссылки")


def test_report_sections_render_manifests_traces_and_threats(report_root):
    text, _ = mr.render(report_root, smoke=False, figures=False)
    assert mr.trace_numbers(text, report_root)[1] == []
    sec = lambda i: text.split(f"\n## {i}. ")[1].split("\n## ")[0]  # noqa: E731
    s2, s3, s4, s5, s9, s10 = sec(2), sec(3), sec(4), sec(5), sec(9), sec(10)
    snapshot = json.loads((report_root / "results/report_derived.json").read_text(encoding="utf-8"))
    derived = snapshot["entries"]
    from flyguard.config import config_hash

    assert snapshot["config_hash"] == config_hash(report_root) and "git_commit" in snapshot and snapshot["smoke"] is False
    # section 2: harness pins, threads of the result files
    assert "| AgentDojo |" in s2 and "(results/E1/summary.json#timing/0/threads/cpu_count)" in s2
    malecns = next(ln for ln in s2.splitlines() if ln.startswith("| MaleCNS |")).split(" | ")
    assert malecns[2] == mr.NA and "(results/setup.json#pins/malecns/version)" in malecns[3]   # a data release: no commit
    # section 3: window scheme, audit tables, per split/label counts, pools and the FPR point, C_unl composition
    assert "256 (configs/default.yaml#windows/size)" in s3 and "64 (configs/default.yaml#windows/min_span_chars)" in s3
    assert "| deep | 355 | 307 | (data/manifests/audit.md) |" in s3 and "| all | 265 | 40.0 % | (data/manifests/audit.md) |" in s3
    assert "(data/manifests/contamination.json#document_level/deep/test/1/documents)" in s3
    assert "Рабочая точка TPR" in s3 and "(results/power.json#fpr_target)" in s3
    splits = json.loads((report_root / "data/manifests/splits.json").read_text(encoding="utf-8"))
    assert derived["data/c_unl/n"]["value"] == len(splits["c_unl"]) and "#entries/data/c_unl/n/value)" in s3
    # section 4: ТЗ 1.5 «Сверка» next to the published min / median / max, the pilot, every paraphrase rate row
    assert "0.127 (results/traces_stats.json#ours/agentdojo/workspace/targeted_asr)" in s4
    assert "0.100 / 0.250 / 0.400 (моделей 3)" in s4 and "опубликованных прогонов: нет (results/traces_stats.json#same_model_published)" in s4
    assert derived["traces/published/agentdojo/workspace/targeted_asr"]["value"]["median"] == 0.25
    assert "(results/pilot.json#candidates/1/targeted_asr)" in s4
    assert "(results/paraphrases.json#rates/generation_refusal/by_kind/template/rate)" in s4
    # section 5: the stage-1 table next to the frozen one
    assert "(results/E0/power_stage1.json#sizes/deep/n_pos)" in s5 and "(results/power.json#sizes/deep/n_pos)" in s5
    # section 9: contamination counts with derived totals, authorship notes and model cards from the file
    assert "(data/manifests/contamination.json#document_level/deep/test/1/documents_matched)" in s9
    assert derived["contamination/deep/test/documents_matched"]["value"] == 4 and derived["contamination/bipia/test/documents"]["value"] == 280
    assert "share a first author (data/manifests/contamination.json#threats_to_validity/0)" in s9 and "`VMware/open-instruct`" in s9
    # section 10: runnable on a clean PATH, the final run with --jobs
    assert ".venv/bin/python scripts/check_acceptance.py" in s10 and "scripts/run_all.sh --jobs 4" in s10 and "scripts/smoke.sh" in s10
    band = mr.learning_series(mr.Inputs(report_root, False))["knn1"]["shots1"]          # E2 band = min / max over subsamples
    assert band[1:] == (pytest.approx(0.45), pytest.approx(0.65))


def test_smoke_skipped_parts_are_marked(tmp_path):
    """ASSUMPTIONS A54: guard latency, the guard-heavy E6 parts and the contamination re-audit are marked
    «не выполнено в смоуке», and the report still traces."""
    root = tmp_path / "smoke"
    shutil.copytree(ROOT / "configs", root / "configs")
    nums = {"auc/deep/protectai_v2": number(None, ci(0.8, 0.05)), "latency_ms/tfidf_lr": number(1.5, n=200)}
    write_result("E1", 0, nums, {}, {}, [], smoke=True, root=root)
    write_result("E6", 0, {"auc/deep/real_fly_bloom_k2.5": number(None, ci(0.7, 0.05))}, {}, {}, ["E6 parts: ['k', 'gamma']"],
                 smoke=True, root=root)
    atomic_write_json(root / "data/manifests/smoke/contamination.json", {"skipped": True, "note": "smoke profile", "threats_to_validity": ["x"]})
    text, _ = mr.render(root, smoke=True, figures=False)
    assert mr.trace_numbers(text, root)[1] == []
    assert re.search(r"\| protectai_v2 \|[^\n]*не выполнено в смоуке", text)
    assert re.search(r"\| `bipia_all` \|[^\n]*\| не выполнено в смоуке \|", text) and re.search(r"\| `k` \|[^\n]*\| выполнено \|", text)
    assert "Пересечение с обучающим набором PIGuard: не выполнено в смоуке" in text
    parts = [{"part": "k", "status": "выполнено", "flag": "k_frac", "reason": None},      # the table E6 writes itself
             {"part": "bipia_all", "status": "не выполнено в смоуке", "flag": "bipia_all_attacks_positions", "reason": "not in smoke.e6_parts"}]
    write_result("E6", 0, {"auc/deep/real_fly_bloom_k2.5": number(None, ci(0.7, 0.05))}, {"parts": parts}, {}, [], smoke=True, root=root)
    text, _ = mr.render(root, smoke=True, figures=False)
    assert mr.trace_numbers(text, root)[1] == [] and "(results/smoke/E6/summary.json#tables/parts/1/status)" in text
    assert re.search(r"\| `bipia_all` \|[^\n]*\| не выполнено в смоуке \(results/smoke/E6/summary.json#tables/parts/1/status\) \|", text)


def test_commit_provenance_warnings(tmp_path, monkeypatch):
    """A52: one line when every result shares one commit; a warning when the code of two commits differs or cannot
    be compared, when a commit is missing or the code was dirty; no git repository is never a crash."""
    from types import SimpleNamespace

    inp, a, b = SimpleNamespace(root=tmp_path), "a" * 40, "b" * 40
    assert "одном коммите" in " ".join(mr.commit_warnings(inp, [a, a], []))
    for verdict, word in ((True, "отличается"), (None, "не удалось сравнить"), (False, "только вне")):
        monkeypatch.setattr(mr, "code_differs", lambda root, x, y, v=verdict: v)
        lines = " ".join(mr.commit_warnings(inp, [a, a, b], []))
        assert word in lines and (("⚠" in lines) == (verdict is not False))
    lines = " ".join(mr.commit_warnings(inp, [a, None], ["E1, сиды 3 (results/E1/summary.json#seeds)"]))
    assert "не записан" in lines and "незакоммиченными" in lines
    monkeypatch.undo()
    assert mr.code_differs(tmp_path, a, b) is None


def test_acceptance_helpers():
    t0 = datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc)
    entries = [{"ts": t0 + timedelta(minutes=30), "split": "test", "path": "data/raw/x", "purpose": "read; logged after the fact", "retro": True},
               {"ts": t0 + timedelta(hours=1), "split": "test", "path": "data/raw/y", "purpose": "build", "retro": False}]
    assert ca.regex_order(t0, False, entries, "**D10** задним числом").ok is True
    assert ca.regex_order(t0, False, entries, "").ok is False                       # retroactive reads without a journal entry
    assert ca.regex_order(t0 + timedelta(hours=2), False, entries, "D10").ok is False
    assert ca.regex_order(t0, True, entries, "D10").ok is False and ca.regex_order(None, False, entries, "").ok is False
    assert ca.regex_order(t0, False, [], "").ok is True
    good = {"r": {"thresholds": {"tau_fpr/a": {"value": 0.4, "source": "P_val", "target": "FPR=0.05", "n": 107}}}}
    bad = {"r": {"thresholds": {"tau_fpr/a": {"value": 0.4, "source": "P_val", "target": None, "n": 107}}}}
    assert ca.thresholds_complete(good).ok and not ca.thresholds_complete(bad).ok
    lines = ["t\tpypi.org\tGET\tu\tp", "t\tevil.example\tPOST\tu\tp", "t\tapi.deepseek.com\tPOST\tu\tp"]
    assert ca.hosts_outside(lines, {"pypi.org", "api.deepseek.com"}) == {"evil.example": 1}
    import zlib

    test_task = next(f"user_task_{i}" for i in range(50) if zlib.crc32(f"user_task_{i}".encode()) % 3 == 2)
    other = next(f"user_task_{i}" for i in range(50) if zlib.crc32(f"user_task_{i}".encode()) % 3 != 2)
    m = {"test": [f"workspace/{test_task}/injection_task_1/important_instructions/M", f"workspace/{test_task}/none/none/M"],
         "observation": [f"workspace/{other}/none/none/M"], "validation_clean": [], "validation_attacks": [],
         "train_attacks": [f"workspace/{other}/injection_task_1/tool_knowledge/M"]}
    assert ca.split_rule_violations(m, 3, 2, "important_instructions") == []
    m["train_attacks"].append(f"workspace/{other}/injection_task_2/important_instructions/M")
    m["test"].append(f"workspace/{other}/none/none/M")
    assert len(ca.split_rule_violations(m, 3, 2, "important_instructions")) == 2
    power = {"stage": 2, "created_at": "2026-09-26T09:00:00Z", "config_hash": "h"}
    e1 = {"results/E1/0.json": {"created_at": "2026-09-26T10:00:00Z"}}
    assert ca.power_before_e1(power, e1, "h").ok and not ca.power_before_e1({**power, "stage": 1}, e1, "h").ok
    assert not ca.power_before_e1({**power, "created_at": "2026-09-26T11:00:00Z"}, e1, "h").ok
    assert ca.config_hash_mismatches({"a": {"config_hash": "h"}, "b": {"config_hash": "x"}, "c": None}, "h") == ["b", "c"]


def test_smoke_timing_sums_the_latest_measured_stage_durations():
    """The 15-minute criterion is the time of a clean smoke run: idempotent re-runs skip stages, so the estimate is
    the sum of each stage's most recent smoke 'done' duration; skipped, interrupted and failed stages measure
    nothing, real-mode lines are ignored."""
    def line(ts, mode, stage, status, secs="0", note=""):
        return "\t".join([ts, mode, stage, status, str(secs), note])

    log = [line("t1", "smoke", "RUN", "start"), line("t1", "smoke", "build_stage1", "done", 160),
           line("t2", "smoke", "e0_stage1", "done", 30), line("t3", "smoke", "build_full", "done", 150),
           line("t4", "smoke", "e0_stage2", "done", 35), line("t5", "smoke", "e1", "start"),          # interrupted
           line("t6", "smoke", "RUN", "start"), line("t6", "smoke", "build_stage1", "skip", 0, "done"),
           line("t7", "smoke", "e1", "done", 200), line("t8", "smoke", "e4", "fail", 50, "exit 1"),
           line("t9", "smoke", "RUN", "TOTAL", 260, "failed at e4"), line("t9", "real", "e5", "done", 9999)]
    t = ca.smoke_timing(log)
    assert t["stages"]["build_stage1"] == ("t1", 160.0) and t["stages"]["e1"] == ("t7", 200.0)
    assert t["sum"] == 160 + 30 + 150 + 35 + 200 and t["failed"] == ["e4"]
    assert t["missing"] == ["prescore", "e4", "e5", "e3", "e2", "e6", "contract", "verdicts", "report"]
    assert t["last_total"] == ("t9", 260.0, "failed at e4") and t["prescore_cache_rows"] is None
    # the guard prescoring is inside the smoke range (ТЗ step 5): its duration counts, with the cache state of its start
    assert ca.SMOKE_STAGES.index("prescore") == ca.SMOKE_STAGES.index("e0_stage2") + 1
    log += [line("u0", "smoke", "prescore", "start", 0, "scores_cache_rows=1190"), line("u0", "smoke", "prescore", "done", 512)]
    log += [line("u1", "smoke", s, "done", 20) for s in ("e4", "e5", "e3", "e2", "e6", "contract", "verdicts", "report")]
    t = ca.smoke_timing(log)
    assert t["missing"] == [] and t["failed"] == [] and t["sum"] == 575 + 512 + 8 * 20 and t["stages"]["e4"] == ("u1", 20.0)
    assert t["prescore_cache_rows"] == 1190
    # the latest prescore measurement decides; "unknown" and a start without a note are not a clean measurement
    t = ca.smoke_timing(log + [line("v0", "smoke", "prescore", "start", 0, "scores_cache_rows=0"), line("v1", "smoke", "prescore", "done", 700)])
    assert t["prescore_cache_rows"] == 0 and t["stages"]["prescore"] == ("v1", 700.0)
    assert ca.smoke_timing(log + [line("v0", "smoke", "prescore", "start", 0, "scores_cache_rows=unknown"),
                                  line("v1", "smoke", "prescore", "done", 9)])["prescore_cache_rows"] is None
    assert ca.smoke_timing(log + [line("v0", "smoke", "prescore", "start"), line("v1", "smoke", "prescore", "done", 9)])["prescore_cache_rows"] is None
    assert ca.smoke_timing([])["stages"] == {} and ca.smoke_timing(["garbage line"])["missing"] == list(ca.SMOKE_STAGES)


def write_smoke_log(root: Path, prescore_note: str, seconds: dict | None = None) -> None:
    """A run_all.log with one smoke measurement of every stage and a REPORT.md with the ten section headings."""
    secs = {s: 20 for s in ca.SMOKE_STAGES} | dict(seconds or {})
    lines = ["2026-09-27T01:00:00Z\tsmoke\tRUN\tstart\t0\targs"]
    for s in ca.SMOKE_STAGES:
        lines += [f"2026-09-27T01:00:01Z\tsmoke\t{s}\tstart\t0\t{prescore_note if s == 'prescore' else ''}",
                  f"2026-09-27T01:00:02Z\tsmoke\t{s}\tdone\t{secs[s]}\t"]
    lines.append("2026-09-27T01:10:00Z\tsmoke\tRUN\tTOTAL\t600\tok")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/run_all.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "results/smoke").mkdir(parents=True, exist_ok=True)
    (root / "results/smoke/REPORT.md").write_text("".join(f"## {i}. x\n\n" for i in range(1, 11)), encoding="utf-8")


def test_smoke_criterion_needs_prescore_from_an_empty_score_cache(tmp_path):
    """A warm guard-score cache makes prescore/E1/E6 look fast: the 15-minute criterion passes only on a prescore
    measured from an empty cache, and the prescore duration counts toward the limit."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    name = "smoke.sh проходит за 15 минут со всеми разделами отчёта"

    def smoke_check():
        return next(c for c in ca.run_checks(root, smoke=True, pytest_mode="skip", only=["c_smoke"]) if c.name == name)

    write_smoke_log(root, "scores_cache_rows=0")
    c = smoke_check()
    assert c.ok is True and "prescore 20" in c.detail and "пустого кеша" in c.detail, c.detail
    write_smoke_log(root, "scores_cache_rows=5377")
    c = smoke_check()
    assert c.ok is False and "5377" in c.detail and "перенесите" in c.detail
    write_smoke_log(root, "")                                        # a log written before the note existed
    assert smoke_check().ok is False
    write_smoke_log(root, "scores_cache_rows=0", {"prescore": 900 - 20 * (len(ca.SMOKE_STAGES) - 1) + 1})
    c = smoke_check()
    assert c.ok is False and "901" in c.detail                        # over the limit only because of prescore


def test_seed_coverage_and_code_provenance_helpers():
    files = {f"results/{e}/{s}.json": {} for e in ("E1", "E4") for s in (0, 1, 2)}
    assert ca.seed_coverage(files, [0, 1, 2]).ok is True
    c = ca.seed_coverage({k: v for k, v in files.items() if k != "results/E1/1.json"}, [0, 1, 2])
    assert c.ok is False and "E1: нет сидов [1]" in c.detail                  # a failed or killed seed process
    c = ca.seed_coverage({**files, "results/E4/7.json": {}}, [0, 1, 2])
    assert c.ok is False and "лишние сиды [7]" in c.detail
    c = ca.seed_coverage({k: v for k, v in files.items() if "/E4/" not in k}, [0, 1, 2])
    assert c.ok is False and "E4" in c.detail                                 # E4 is never cut (ТЗ cutting order)
    c = ca.seed_coverage({**files, "results/smoke/E6/0.json": {}}, [0, 1, 2])
    assert c.ok is False and "E6: нет сидов [1, 2]" in c.detail
    assert ca.seed_coverage(files, [0, 1, 2], required=()).ok and "не выполнены: ['E2', 'E3', 'E5', 'E6']" in ca.seed_coverage(files, [0, 1, 2]).detail

    a, b, x = "a" * 40, "b" * 40, "c" * 40
    same = {frozenset((a, b)): True, frozenset((a, x)): False}
    diff = lambda p, q: same.get(frozenset((p, q)))  # noqa: E731 - None: commit unknown to git
    rec = lambda c, **kw: {"git_commit": c, "git_dirty": False, **kw}  # noqa: E731
    assert ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(a)}, diff).ok is True
    assert ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(a), "r/E4/0.json": rec(b)}, diff).ok is True   # docs-only commit
    c = ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(a), "r/E1/2.json": rec(x)}, diff)
    assert c.ok is False and x[:12] in c.detail and "отличается" in c.detail
    c = ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(a), "r/E1/2.json": rec("d" * 40)}, diff)
    assert c.ok is False and "не найдены" in c.detail
    assert ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(a, git_dirty=True)}, diff).ok is False
    assert ca.code_provenance({"r/E1/0.json": rec(a), "r/E1/1.json": rec(None)}, diff).ok is False
    c = ca.code_provenance({"r/E1/0.json": {"git_commit": a}}, diff, head=x)
    assert c.ok is True and "git_dirty не записан в 1" in c.detail and "HEAD отличается" in c.detail
    assert ca.code_provenance({}, diff).ok is False and ca.code_provenance({"r": None}, diff).ok is False


STUB_PY = r"""#!/usr/bin/env bash
# stub interpreter for the run_all.sh driver test: records calls, fails where STUB_FAIL says
echo "$*" | cut -c1-120 >> "$STUB_LOG"
case "$1" in
  -) body="$(cat)"
     if [[ "$body" == *read_metadata* ]]; then echo "${STUB_ROWS:-0}"
     elif [[ "$body" == *'seeds = list'* ]]; then echo "0,1,2"
     else echo todo; fi ;;
  -c) [[ "$*" == *summarize* ]] && echo "summarize $3" >> "$STUB_LOG"; echo "" ;;
  -m) if [[ "$2" == flyguard.experiments.run && " $STUB_FAIL " == *" seed$5 "* ]]; then exit 7; fi ;;
esac
exit 0
"""


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("xargs") is None, reason="needs bash and xargs")
def test_run_all_driver_fails_the_stage_on_a_failing_command(tmp_path):
    """Stages run as `( ... )` in an || list ignore errexit (bash rule), so a failing seed process of --jobs or a
    failed `gen_traces.sh run` used to be followed by summarize/freeze and the stage logged "done". The driver now
    re-arms errexit inside every stage body and run_experiment returns the xargs status after the summary."""
    root = tmp_path / "fake"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    shutil.copy(SCRIPTS / "run_all.sh", root / "scripts/run_all.sh")
    py = root / ".venv/bin/python"
    py.write_text(STUB_PY, encoding="utf-8")
    py.chmod(0o755)
    gen = root / "scripts/gen_traces.sh"
    gen.write_text('#!/usr/bin/env bash\necho "gen_traces $1" >> "$STUB_LOG"\n[[ "$1" == run ]] && exit 5\nexit 0\n', encoding="utf-8")
    gen.chmod(0o755)
    (root / "data/processed/smoke").mkdir(parents=True)
    (root / "data/processed/smoke/windows.parquet").write_bytes(b"x")   # run_prescore hashes it into its done marker
    stub_log = tmp_path / "stub.log"

    def run(*args, fail="", rows="0", out=False):
        stub_log.write_text("", encoding="utf-8")
        env = {**os.environ, "STUB_LOG": str(stub_log), "STUB_FAIL": fail, "STUB_ROWS": rows}
        proc = subprocess.run(["bash", str(root / "scripts/run_all.sh"), *args], cwd=root, env=env,
                              capture_output=True, text=True, timeout=120)
        if out:
            return proc.stdout
        return proc.returncode, stub_log.read_text(encoding="utf-8"), (root / "logs/run_all.log").read_text(encoding="utf-8")

    rc, calls, log = run("--only", "e1", "--seeds", "0,1,2", "--jobs", "2", fail="seed1")
    assert rc == 123 and "summarize E1" in calls                    # the summary of the finished seeds is rebuilt
    assert "\te1\tfail\t" in log and "\te1\tdone\t" not in log
    rc, calls, log = run("--only", "e1", "--seeds", "0,1,2", "--jobs", "2")
    assert rc == 0 and log.rstrip().splitlines()[-2].split("\t")[2:4] == ["e1", "done"]
    rc, calls, log = run("--only", "gen_traces")
    assert rc == 5 and "gen_traces run" in calls and "gen_traces freeze" not in calls
    for mode in ((), ("--smoke",)):                                    # report: trace_stats first, then make_report
        rc, calls, log = run(*mode, "--only", "report")
        assert rc == 0 and calls.index("-m flyguard.gen.trace_stats") < calls.index("scripts/make_report.py")
    rc, calls, log = run("--smoke", "--only", "prescore")                # the cache state is noted at the start
    assert rc == 0 and "\tprescore\tstart\t0\tscores_cache_rows=0" in log
    # the done marker matches windows.parquet; a cache moved aside (0 rows) makes prescore run again
    assert "would run prescore" in run("--smoke", "--dry-run", "--from", "prescore", out=True)
    assert "skip prescore (done)" in run("--smoke", "--dry-run", "--from", "prescore", rows="42", out=True)


def test_acceptance_checklist_on_synthetic_root(report_root, recorder):
    mr.render(report_root, smoke=False, figures=False)
    checks = ca.run_checks(report_root, smoke=False, pytest_mode="skip", access_log=recorder)
    by = {c.name: c for c in checks}
    assert by["у каждого порога записаны значение, источник, цель и n"].ok
    assert by["хеш конфига в результатах совпадает с текущим"].ok
    assert by["каждое число REPORT.md прослеживается до results/; разделы 1–10 на месте"].ok, by["каждое число REPORT.md прослеживается до results/; разделы 1–10 на месте"].detail
    dedup = next(c for c in checks if c.name.startswith("после дедупликации"))
    assert dedup.ok, dedup.detail
    assert any("windows.parquet" in p and split == "test" for p, split, _ in recorder.calls)   # journaled parquet read
    assert by["power.json заморожен (стадия 2) с текущим хешем и записан раньше результатов E1"].ok
    assert by["вердикты используют таблицу носителей E0"].ok
    assert by["одна схема окон для всех детекторов; 512-токенные окна только в E6"].ok
    assert by["pytest проходит"].ok is None
    assert by["smoke.sh проходит за 15 минут со всеми разделами отчёта"].ok is False      # no run_all.log in the toy tree
    assert by["regex_patterns.txt закоммичен до первого чтения теста"].ok is False          # no git history in the toy tree
    assert by["у каждого выполненного эксперимента E1–E6 все сиды конфига"].ok is False     # seeds 0, 1 of seeds.global
    assert by["результаты посчитаны одним кодом (коммит без незакоммиченных изменений src/scripts/configs)"].ok is False  # no git
    assert {c.mark for c in checks} <= {"✅", "❌", "⚠"} and len(checks) >= 20
