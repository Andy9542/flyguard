"""scripts/make_report.py and scripts/check_acceptance.py on a synthetic results tree (ТЗ "Критерии приёмки":
каждое число REPORT.md прослеживается до results/; the report renders with missing experiments; the acceptance
helpers flip on the right inputs). Deterministic, synthetic, no network, no real data."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
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
    for seed in (0, 1):
        write_result("E1", seed, e1_numbers(seed), tables, thresholds, ["test reads: 6; bootstrap n=60, alpha=0.05", f"seed children: {{'nose': {seed}}}"],
                     root=root, seeds={"nose": seed}, timing={"seconds": 12.5 + seed, "test_reads": ["deep", "bipia"]})
    # E2: fewshot/<metric>/<level>/<detector> (e2.level_key -> shots<k> | full), paired diffs per level
    e2 = {f"fewshot/macro_auc/{lvl}/{d}": number(None, ci(0.5 + 0.05 * i + 0.1 * j, 0.03), n_reps=10)
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
    (root / "DEVIATIONS.md").write_text("# Отклонения\n\n- **D1 (2026-09-26).** Один провайдер.\n- **D10 (2026-09-26).** Журнал задним числом (retroactive).\n", encoding="utf-8")
    (root / "ASSUMPTIONS.md").write_text("- **A1.** x\n", encoding="utf-8")
    (root / "BLOCKERS.md").write_text("- **B1.** x\n", encoding="utf-8")


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
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return any((isinstance(v, str) and tok in v) or (not isinstance(v, str) and abs(float(tok) - v) <= 0.5 * 10 ** (-dec) + 1e-9)
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
    # two seeds: the seed t-interval is rendered, not an average of bootstrap bounds
    s = json.loads((report_root / "results/E1/summary.json").read_text(encoding="utf-8"))
    rec = s["numbers"]["macro_auc/tfidf_lr"]
    assert f"{rec['mean']:.3f} [{rec['seed_ci_low']:.3f}, {rec['seed_ci_high']:.3f}] (results/E1/summary.json#numbers/macro_auc/tfidf_lr)" in text
    figs = {p.name for p in (report_root / "results/figures").glob("*.png")}
    assert {"e1_auc_by_source.png", "e2_learning_curves.png", "e4_curveball_hist.png", "e1_notinject_fpr_tau90.png", "para_strata_auc.png"} <= figs


def test_report_renders_without_any_experiment(tmp_path):
    root = tmp_path / "empty"
    shutil.copytree(ROOT / "configs", root / "configs")
    text, path = mr.render(root, smoke=False, figures=False)
    assert all(re.search(rf"^## {i}\. ", text, re.M) for i in range(1, 11))
    assert text.count("не выполнено") >= 10
    assert mr.trace_numbers(text, root)[1] == []
    text_s, path_s = mr.render(root, smoke=True, figures=False)
    assert path_s == root / "results/smoke/REPORT.md" and (root / "results/smoke/setup.json").exists()


def test_verifier_catches_wrong_and_unreferenced_numbers(report_root):
    good = "| a | 0.700 [0.680, 0.720] (results/E1/summary.json#numbers/macro_auc/tfidf_lr) |"
    text = "| k | v |\n|---|---|\n" + good + "\n| b | 0.123 (results/E1/summary.json#numbers/macro_auc/tfidf_lr) |\n| c | 42 |\n"
    _, problems = mr.trace_numbers(text, report_root)
    assert len(problems) == 2 and any("0.123" in p for p in problems) and any("without reference" in p for p in problems)
    assert mr.trace_numbers("Смотри ТЗ 1.7, §6 и Этап 4; H1a; N16k; `x = 512`\n", report_root)[1] == []


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
    assert t["missing"] == ["e4", "e5", "e3", "e2", "e6", "contract", "verdicts", "report"]
    assert t["last_total"] == ("t9", 260.0, "failed at e4")
    log += [line("u1", "smoke", s, "done", 20) for s in ("e4", "e5", "e3", "e2", "e6", "contract", "verdicts", "report")]
    t = ca.smoke_timing(log)
    assert t["missing"] == [] and t["failed"] == [] and t["sum"] == 575 + 8 * 20 and t["stages"]["e4"] == ("u1", 20.0)
    assert ca.smoke_timing([])["stages"] == {} and ca.smoke_timing(["garbage line"])["missing"] == list(ca.SMOKE_STAGES)


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
    assert {c.mark for c in checks} <= {"✅", "❌", "⚠"} and len(checks) >= 20
