"""E0 (power table, both stages, freeze), E1 (full detector table + verdict inputs), the shared CLI and
``verdicts_run`` on a private synthetic root built with the helpers of tests/experiments/conftest.py (the shared
``toy_root`` is left alone: a power.json there would change test_engine's thresholds). Deterministic, no network."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import config_hash
from flyguard.eval.thresholds import fpr_target_for_pool
from flyguard.eval.verdicts import CONFIRMED, INSUFFICIENT, PRECONDITION, REFUTED, STATUSES
from flyguard.experiments import Context, read_result
from flyguard.experiments.e0 import power_copy_path, run_e0, sizes_by_source
from flyguard.experiments.e1 import hypothesis_inputs, run_e1
from flyguard.experiments.results import number, power_path, result_path, summary_path, write_result
from flyguard.experiments.run import build_parser, main as run_main, parse_seeds
from flyguard.experiments.verdicts_run import (aggregate, ci_from_record, find_pair_key, verdicts_path,
                                               write_verdicts)
from flyguard.io import atomic_write_json

_spec = importlib.util.spec_from_file_location("e0e1_conftest", Path(__file__).with_name("conftest.py"))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
POWER_SMALL = {"n_rep": 2, "n_boot": 20, "delta_grid": (0.05, 0.15)}


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def troot(tmp_path_factory, toy_cfg) -> Path:
    root = tmp_path_factory.mktemp("flyguard_e0e1")
    helpers.make_synthetic_root(root, toy_cfg)
    return root


@pytest.fixture(scope="module")
def rec():
    return helpers.Recorder()


@pytest.fixture(scope="module")
def gf(toy_cfg, troot):
    return lambda name, **kw: GuardModel(name, toy_cfg, root=troot, loader=helpers.fake_guard_loader, **kw)


@pytest.fixture(scope="module")
def mctx(toy_cfg, troot, rec):
    return Context(toy_cfg, root=troot, access_log=rec)


@pytest.fixture(scope="module")
def e0_out(toy_cfg, troot, mctx, rec, gf):
    return run_e0(1, 0, root=troot, cfg=toy_cfg, ctx=mctx, access_log=rec, guard_factory=gf, power_overrides=POWER_SMALL)


@pytest.fixture(scope="module")
def e1_paths(toy_cfg, troot, mctx, rec, gf, e0_out):
    return run_e1([0], root=troot, cfg=toy_cfg, ctx=mctx, access_log=rec, guard_factory=gf)


# ---------------------------------------------------------------------------------------------- E0
def test_e0_sizes_are_aggregate_counts_through_the_journaled_door(toy_cfg, troot):
    fresh_rec = helpers.Recorder()                                  # own context: mctx's door stays cold for E0
    fresh = Context(toy_cfg, root=troot, access_log=fresh_rec)
    sizes, extra = sizes_by_source(fresh, "unit test: counts only")
    docs, _ = helpers.make_documents()
    test = docs[docs["split"] == "test"]
    for s, sz in sizes.items():
        sub = test[test["source"] == s]
        assert (sz["n_pos"], sz["n_neg"]) == (int((sub["label"] == 1).sum()), int((sub["label"] == 0).sum())), s
        assert sum(p + q for p, q in sz["cluster_label_sizes"]) == len(sub)
    assert sizes["bipia"]["cluster_label_sizes"] == [[1, 1]] * 4 and sizes["notinject"]["n_pos"] == 0
    assert extra["para_deep"]["n_pos"] + extra["para_deep"]["n_neg"] < sizes["para"]["n_pos"] + sizes["para"]["n_neg"]
    assert len(fresh_rec.calls) == 2 * len(fresh.test_sources) and all("counts only" in c[2] for c in fresh_rec.calls)


def test_e0_stage1_power_json_and_seed_file(toy_cfg, troot, mctx, e0_out):
    p = e0_out["power"]
    assert e0_out["power_path"] == power_path(troot) and not e0_out["frozen"] and not e0_out["skipped"]
    assert p["stage"] == 1 and p["frozen"] is False and p["config_hash"] == config_hash(troot) and p["created_at"]
    assert p["fpr_target"] == fpr_target_for_pool(mctx.p_test_size, toy_cfg) and p["pools"]["P_test"] == mctx.p_test_size
    assert set(p["carriers"]) == set(mctx.test_sources) | {"macro"} and set(p["cells"]["macro"]) == {"0.75", "0.85", "0.95"}
    assert p["notinject"]["n"] == toy_cfg.exp("E0")["notinject_pairs"] and p["sources"] == mctx.test_sources
    vr = p["val_runs"]
    assert p["spread"]["curveball"]["n"] == toy_cfg.default["expansion"]["curveball"]["n_null"] == 3
    assert p["spread"]["perm"]["n"] == toy_cfg.exp("E0")["perm_seeds"] == 10 and "sd_over_delta" in p["spread"]["perm"]
    assert set(vr["val_auc_by_detector"]) == {"real_fly_bloom", "tfidf_lr"} and set(vr["val_sources"]) == {"deep", "bipia", "dojo"}
    for s in vr["val_sources"]:  # planning level input = the lower of the two detectors
        assert vr["val_auc"][s] == pytest.approx(min(vr["val_auc_by_detector"][d][s] for d in vr["val_auc_by_detector"]))
    # the first perm child is the E0 seed's own π: the manual code path reproduces the engine's fly exactly
    assert vr["perm_val_macro_auc"][0] == pytest.approx(vr["val_macro_auc"]["real_fly_bloom"])
    assert len(set(vr["perm_val_macro_auc"])) > 1 and all(0 <= v <= 1 for v in vr["curveball_val_macro_auc"])
    assert p["provenance"]["test_reads"] == mctx.test_sources
    r = read_result(e0_out["result_path"])
    assert r["stage"] == 1 and r["frozen"] is False and r["timing"]["test_reads"] == mctx.test_sources
    for key in ("size/deep/n_pos", "size/macro/n_clusters", "size/para_deep/n_pos", "pool/p_test", "fpr_target",
                "power/macro/0.85/mdd", "power/deep/0.75/tost_power", "planning_level/macro", "spread/perm/sd",
                "spread/curveball/half_width", "val_auc/deep/tfidf_lr", "val_macro_auc/real_fly_bloom",
                "notinject/ci_width/0.05", "notinject/worst_case_width"):
        assert key in r["numbers"], key
    assert r["numbers"]["size/deep/n_pos"]["value"] == p["sizes"]["deep"]["n_pos"]
    assert {row["source"] for row in r["tables"]["carriers"]} == set(p["carriers"])
    assert {row["kind"] for row in r["tables"]["spread_values"]} == {"curveball", "perm"}
    assert len([row for row in r["tables"]["cells"] if row["planning"]]) == len(p["planning_level"])
    assert power_copy_path(1, troot).exists() and summary_path("E0", root=troot).exists()
    assert not any(k.startswith("auc/") for k in r["numbers"])  # E0 never scores a test document


# ---------------------------------------------------------------------------------------------- E1
def test_e1_table_thresholds_and_verdict_inputs(toy_cfg, troot, mctx, e1_paths):
    r = read_result(e1_paths[0])
    nums, th = r["numbers"], r["thresholds"]
    assert e1_paths[0] == result_path("E1", 0, root=troot) and r["e1"]["missing_sources"] == []
    assert set(r["e1"]["detectors"]) == set(toy_cfg.exp("E1")["detectors"]) and "piguard" not in r["e1"]["available"]
    for key in ("auc/deep/tfidf_lr", "auc/para_deep/protectai_v2", "macro_auc/real_fly_bloom", "tpr_at_fpr/dojo/knn1",
                "fpr_ptest/regex", "fpr_notinject/tau_fpr/tfidf_lr/one", "fpr_notinject/tau90_deep/real_fly_bloom/non-en",
                "diff/auc/deep/tfidf_lr-protectai_v2", "diff90/auc/dojo/tfidf_lr-protectai_v2",
                "diff/auc/para_deep/protectai_v2-tfidf_lr", "diff90/macro_auc/real_fly_linear-lr_svd",
                "diff/macro_auc/flyhash_bloom-tfidf_lr", "diff/fpr_notinject/tau90_deep/real_fly_bloom-protectai_v2",
                "val_auc/deep/real_fly_bloom", "val_macro_auc/real_fly_bloom", "latency_ms/protectai_v2",
                "state_bytes/real_fly_bloom", "hyper/real_fly_bloom/gamma",
                "tost/macro_auc/real_fly_linear-lr_svd", "tost/auc/deep/tfidf_lr-protectai_v2",
                "p_holm/auc/para_deep/protectai_v2-tfidf_lr", "p_holm/auc/dyn/protectai_v2-tfidf_lr"):
        assert key in nums, key
    t = nums["tost/macro_auc/real_fly_linear-lr_svd"]
    assert t["value"] in (0.0, 1.0) and t["reference"] == nums["macro_auc/lr_svd"]["value"] and t["delta"] == pytest.approx(0.05 * t["reference"])
    h = nums["p_holm/auc/bipia/protectai_v2-tfidf_lr"]
    assert 0 <= h["p_raw"] <= h["value"] <= 1 and h["note"] == "family=bipia,dyn,para_deep" and isinstance(h["passes"], bool)
    assert all(set(v) >= {"value", "source", "target", "n"} for v in th.values()) and len(th) >= 3 * 11
    assert th["tau_fpr/real_fly_bloom"]["source"] == "P_val" and th["tau90_dojo/regex"]["target"] == "tpr>=0.9"
    assert any("latency_guards=True" in n for n in r["notes"]) and r["timing"]["test_reads"] == []  # E0 read them already


def test_hypothesis_inputs_tost_and_holm(toy_cfg):
    nums = {"macro_auc/b": number(0.90), "diff90/macro_auc/a-b": number(0.001, {"low": -0.02, "high": 0.02, "level": 0.9}),
            "auc/deep/b": number(0.80), "diff90/auc/deep/a-b": number(0.06, {"low": 0.05, "high": 0.07, "level": 0.9}),
            "diff90/auc/dojo/a-b": number(0.0, {"low": -0.1, "high": 0.1, "level": 0.9}),   # no reference -> skipped
            "diff/auc/para_deep/protectai_v2-tfidf_lr": number(0.1, {"low": 0.02, "high": 0.2}, p=0.01),
            "diff/auc/bipia/protectai_v2-tfidf_lr": number(0.1, {"low": 0.01, "high": 0.2}, p=0.03),
            "diff/auc/dyn/protectai_v2-tfidf_lr": number(0.1, {"low": -0.01, "high": 0.2}, p=0.04)}
    out = hypothesis_inputs(nums, toy_cfg)
    assert out["tost/macro_auc/a-b"]["value"] == 1.0 and out["tost/macro_auc/a-b"]["delta"] == pytest.approx(0.045)
    assert out["tost/auc/deep/a-b"]["value"] == 0.0 and out["tost/auc/deep/a-b"]["outside_corridor"] is True
    assert "tost/auc/dojo/a-b" not in out
    holm = {s: out[f"p_holm/auc/{s}/protectai_v2-tfidf_lr"] for s in ("para_deep", "bipia", "dyn")}
    assert holm["para_deep"]["value"] == pytest.approx(0.03) and holm["bipia"]["value"] == pytest.approx(0.06)
    assert holm["dyn"]["value"] == pytest.approx(0.06) and holm["para_deep"]["passes"] and not holm["bipia"]["passes"]
    assert not holm["dyn"]["lower_bound_positive"]


def test_e1_second_seed_skips_current_and_summarizes(toy_cfg, troot, mctx, rec, gf, e1_paths):
    mtime = e1_paths[0].stat().st_mtime_ns
    paths = run_e1([0, 1], root=troot, cfg=toy_cfg, ctx=mctx, access_log=rec, guard_factory=gf, latency_guards="never")
    assert paths[0] == e1_paths[0] and paths[0].stat().st_mtime_ns == mtime and paths[1].exists()
    r1 = read_result(paths[1])
    assert "latency_ms/protectai_v2" not in r1["numbers"] and "latency_ms/tfidf_lr" in r1["numbers"]
    summ = json.loads(summary_path("E1", root=troot).read_text())
    assert summ["n_seeds"] == 2 and set(summ["numbers"]["macro_auc/real_fly_bloom"]["per_seed"]) == {"0", "1"}
    assert not (troot / "data" / "processed" / "features" / "seed1").exists()


# ---------------------------------------------------------------------------------------------- verdicts
def test_verdicts_from_real_results(toy_cfg, troot, e1_paths):
    path = write_verdicts(troot, False, toy_cfg)
    v = json.loads(path.read_text())
    assert path == verdicts_path(troot) and v["config_hash"] == config_hash(troot) and v["seeds"] == [0, 1]
    ov = v["overview"]
    assert ov["H1a"] in STATUSES and ov["H2"] in STATUSES and ov["H3"] in STATUSES and set(ov["H1b"]) == {"real_fly", "flyhash"}
    assert ov["H2"] == INSUFFICIENT and "PIGuard" in v["H2"]["reason"]        # piguard is unavailable
    assert ov["H1b"]["real_fly"] == INSUFFICIENT and ov["H3"] in (INSUFFICIENT, PRECONDITION)  # no E2 / E4 yet
    assert set(v["H1a"]["per_seed"]) == {"0", "1"} and v["H1a"]["n_seeds"] == 2
    assert v["inputs"]["H1a"]["semantic/para"] == "diff/auc/para_deep/protectai_v2-tfidf_lr"
    assert v["inputs"]["H1b"]["real_fly/fewshot/1"].startswith("E2:") and v["inputs"]["H3"]["primary"].startswith("E4:")
    assert any("E2" in w for w in v["warnings"]) and any("не заморожен" in w for w in v["warnings"])
    assert v["sources"]["power"]["frozen"] is False and v["sources"]["E1"]["seeds"] == [0, 1]
    top = {k for k, val in v.items() if isinstance(val, dict) and "status" in val}
    nested = {f"{k}/{s}" for k, val in v.items() if isinstance(val, dict) and "status" not in val
              for s, sv in val.items() if isinstance(sv, dict) and "status" in sv}
    assert top == {"H1a", "H2", "H3"} and nested == {"H1b/real_fly", "H1b/flyhash", "H2_secondary/real_fly_linear"}
    assert "precondition" in json.dumps(v["H2"], ensure_ascii=False) and "carrier" in json.dumps(v["H1a"], ensure_ascii=False)


def _ci(point, low, high, level=0.95):
    return {"low": low, "high": high, "level": level}


def test_verdict_logic_and_seed_aggregation(tmp_path, toy_cfg):
    root = tmp_path / "r"
    carriers = {s: {"auc": "несёт", "auc_diff": "несёт", "tpr_at_fpr": "несёт"} for s in ("deep", "dojo", "bipia", "para", "dyn", "macro")}
    atomic_write_json(power_path(root), {"carriers": carriers, "frozen": True, "stage": 2, "fpr_target": 0.05,
                                         "config_hash": config_hash(root)})
    good = {"auc/deep/protectai_v2": number(0.9), "auc/dojo/protectai_v2": number(0.9),
            "diff90/auc/deep/tfidf_lr-protectai_v2": number(0.0, _ci(0, -0.02, 0.02, 0.9)),
            "diff90/auc/dojo/tfidf_lr-protectai_v2": number(0.0, _ci(0, -0.02, 0.02, 0.9)),
            "diff/auc/para_deep/protectai_v2-tfidf_lr": number(0.1, _ci(0.1, 0.05, 0.15), p=0.001),
            "diff/auc/bipia/protectai_v2-tfidf_lr": number(0.1, _ci(0.1, 0.05, 0.15), p=0.001),
            "macro_auc/lr_svd": number(0.9), "macro_auc/tfidf_lr": number(0.9),
            "diff90/macro_auc/real_fly_linear-lr_svd": number(0.0, _ci(0, -0.02, 0.02, 0.9)),
            "diff90/macro_auc/flyhash_linear-tfidf_lr": number(0.0, _ci(0, -0.02, 0.02, 0.9)),
            "diff/macro_auc/real_fly_bloom-tfidf_lr": number(-0.1, _ci(-0.1, -0.15, -0.05)),
            "diff/macro_auc/flyhash_bloom-tfidf_lr": number(-0.1, _ci(-0.1, -0.15, -0.05)),
            "val_auc/deep/real_fly_bloom": number(0.9), "val_macro_auc/real_fly_bloom": number(0.85),
            "diff/fpr_notinject/tau90_deep/real_fly_bloom-protectai_v2": number(0.0, _ci(0, -0.05, 0.05)),
            "diff/fpr_notinject/tau90_deep/protectai_v2-piguard": number(0.2, _ci(0.2, 0.1, 0.3))}
    bad = dict(good, **{"diff90/macro_auc/real_fly_linear-lr_svd": number(-0.2, _ci(-0.2, -0.3, -0.1, 0.9)),
                        "val_auc/deep/real_fly_bloom": number(0.5)})
    for seed, nums in ((0, good), (1, bad)):
        write_result("E1", seed, nums, {}, {}, [], root=root)
        write_result("E2", seed, {"diff/macro_auc/shots1/real_fly_bloom-knn1": number(0.05, _ci(0.05, -0.01, 0.1)),
                                  "diff/macro_auc/real_fly_bloom-knn1/shots10": number(0.05, _ci(0.05, 0.01, 0.1)),
                                  "diff/macro_auc/shots1/flyhash_bloom-knn1": number(0.05, _ci(0.05, -0.01, 0.1))},
                     {}, {}, [], root=root)
        write_result("E4", seed, {"diff90/macro_auc/real_fly_bloom-curveball_mean": number(0.01, _ci(0.01, -0.02, 0.03, 0.9),
                                                                                           reference=0.85, p_randomization=0.3),
                                  "diff90/macro_auc/real_fly_linear-curveball": number(0.1, _ci(0.1, 0.05, 0.15, 0.9)),
                                  "macro_auc/curveball_mean/real_fly_linear": number(0.8)}, {}, {}, [], root=root)
    from flyguard.experiments.results import summarize
    for e in ("E1", "E2", "E4"):
        summarize(e, root=root)
    v = json.loads(write_verdicts(root, False, toy_cfg).read_text())
    per = {h: v[h]["per_seed"] for h in ("H1a", "H2", "H3")}
    assert per["H1a"]["0"]["status"] == CONFIRMED == per["H1a"]["1"]["status"] and v["overview"]["H1a"] == CONFIRMED
    assert per["H2"]["0"]["status"] == CONFIRMED and per["H2"]["1"]["status"] == PRECONDITION
    assert v["overview"]["H2"] == PRECONDITION and v["H2"]["n_seeds_by_status"] == {CONFIRMED: 1, PRECONDITION: 1}
    assert per["H3"]["0"]["status"] == CONFIRMED and "не отличается" in per["H3"]["0"]["reason"]
    assert v["H3"]["per_seed"]["0"]["inputs"]["secondary"]["linear"]["reference"] == 0.8
    h1b = v["H1b"]
    assert h1b["real_fly"]["per_seed"]["0"]["status"] == CONFIRMED and h1b["real_fly"]["per_seed"]["1"]["status"] == REFUTED
    assert h1b["real_fly"]["status"] == REFUTED                       # tie -> the more conservative status
    assert h1b["flyhash"]["status"] == INSUFFICIENT and "10-shot" in h1b["flyhash"]["reason"]
    assert v["inputs"]["H1b"]["real_fly/fewshot/10"] == "diff/macro_auc/real_fly_bloom-knn1/shots10"
    env = v["H1a"]["ci_envelope"]["semantic"]["para"]
    assert env["low"] == 0.05 and env["high"] == 0.15 and env["descriptive_only"] and v["warnings"] == []
    assert v["H1b"]["real_fly"]["effect"]["equiv"] == pytest.approx(-0.1)   # seed mean of 0 and -0.2


def test_verdict_helpers():
    keys = ["diff/macro_auc/shots1/real_fly_bloom-knn1", "diff/macro_auc/10shot/real_fly_bloom-knn1",
            "diff90/macro_auc/real_fly_bloom-curveball_mean", "diff/macro_auc/real_fly_bloom-knn5"]
    assert find_pair_key(keys, "diff", "macro_auc", "real_fly_bloom", "knn1", ["shots1|1shot|1"]) == keys[0]
    assert find_pair_key(keys, "diff", "macro_auc", "real_fly_bloom", "knn1", ["shots10|10shot"]) == keys[1]
    assert find_pair_key(keys, "diff90", "macro_auc", "real_fly_bloom", "~curveball") == keys[2]
    assert find_pair_key(keys, "diff", "macro_auc", "flyhash_bloom", "knn1") is None
    assert ci_from_record({"value": 0.1, "ci_low": 0.0, "ci_high": 0.2, "n": 5}, 0.9)["level"] == 0.9
    assert ci_from_record({"value": 0.1, "ci_low": None, "ci_high": None}) is None and ci_from_record(None) is None
    assert aggregate({}, "H2")["status"] == INSUFFICIENT


# ---------------------------------------------------------------------------------------------- freeze and CLI
def test_e0_idempotency_and_freeze_rules(toy_cfg, troot, mctx, rec, gf, e0_out, e1_paths):
    again = run_e0(1, 0, root=troot, cfg=toy_cfg, ctx=mctx, access_log=rec, guard_factory=gf, power_overrides=POWER_SMALL)
    assert again["skipped"] and again["power"]["created_at"] == e0_out["power"]["created_at"]
    two = run_e0(2, 0, root=troot, cfg=toy_cfg, ctx=mctx, access_log=rec, guard_factory=gf, power_overrides=POWER_SMALL)
    assert two["frozen"] and not two["skipped"] and power_copy_path(2, troot).exists() and power_copy_path(1, troot).exists()
    frozen = json.loads(power_path(troot).read_text())
    assert frozen["frozen"] is True and frozen["stage"] == 2 and read_result(two["result_path"])["frozen"] is True
    assert run_e0(2, 0, root=troot, cfg=toy_cfg, ctx=mctx, guard_factory=gf, power_overrides=POWER_SMALL)["skipped"]
    with pytest.raises(RuntimeError):
        run_e0(1, 0, root=troot, cfg=toy_cfg, ctx=mctx, guard_factory=gf, power_overrides=POWER_SMALL)
    v = json.loads(write_verdicts(troot, False, toy_cfg).read_text())
    assert v["sources"]["power"]["frozen"] is True and not any("заморожен" in w for w in v["warnings"])
    with pytest.raises(ValueError):
        run_e0(3, 0, root=troot, cfg=toy_cfg)


def test_cli_parsing_and_dispatch(toy_cfg, monkeypatch):
    assert parse_seeds("0,2-4,2", toy_cfg, False) == [0, 2, 3, 4] and parse_seeds("all", toy_cfg, True) == [0]
    assert parse_seeds(None, toy_cfg, False) == list(toy_cfg.default["seeds"]["global"])
    a = build_parser().parse_args(["E0", "--stage", "2", "--smoke"])
    assert (a.experiment, a.stage, a.smoke, a.seed, a.force) == ("E0", 2, True, None, False)
    a = build_parser().parse_args(["E1", "--seeds", "0-2", "--latency-guards", "never", "--keep-cache"])
    assert (a.seeds, a.latency_guards, a.keep_cache) == ("0-2", "never", True)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["E0"])            # --stage is mandatory
    calls = []
    monkeypatch.setattr("flyguard.experiments.run.load_configs", lambda root=None: toy_cfg)
    monkeypatch.setattr("flyguard.experiments.e0.run_e0",
                        lambda **kw: calls.append(("E0", kw)) or {"power_path": "p", "frozen": kw["stage"] == 2, "skipped": False})
    monkeypatch.setattr("flyguard.experiments.e1.run_e1", lambda seeds, **kw: calls.append(("E1", seeds, kw)) or [])
    assert run_main(["E0", "--stage", "2", "--smoke"], log=lambda s: None) == 0
    assert run_main(["E1", "--seeds", "1,3", "--latency-guards", "always"], log=lambda s: None) == 0
    assert calls[0][0] == "E0" and calls[0][1]["stage"] == 2 and calls[0][1]["seed"] == 0 and calls[0][1]["smoke"]
    assert calls[1][1] == [1, 3] and calls[1][2]["latency_guards"] == "always" and not calls[1][2]["smoke"]
