"""E4 (wiring against null models, H3 inputs) and E5 (ablation grid, contributions) on the toy fixture of
tests/experiments/conftest.py: one run of each through the Runner, then assertions on keys, intervals, the
randomisation / Holm family, the two-stage bootstrap, the π-grid consistency and the E4 <-> E5 cross-checks.
Deterministic, synthetic, no network."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from flyguard.config import ROOT, Configs, seeds_for
from flyguard.experiments import Context, read_result, result_path
from flyguard.experiments import e4, e5
from flyguard.experiments.e4 import (Readout, e4_config, fixed_matrix_diff_ci, grid_perms, parse_readout, perm_for_seed,
                                     permute_columns)
from flyguard.experiments.e5 import Cell, cells_from_config, contribution_pairs, matrix_kind

READOUTS = ("bloom_full", "linear", "bloom_10shot")
POS = ("bipia", "deep", "dojo", "dyn", "para")


def _state(path):
    return (path.exists(), path.stat().st_mtime_ns if path.exists() else None)


@pytest.fixture(scope="module")
def e4_cfg(toy_cfg) -> Configs:
    """E4 config whose ``n_curveball`` equals the toy engine's ``expansion.curveball.n_null`` (3): a real run refuses
    to start when the two disagree (``e4.check_n_curveball``)."""
    exps = copy.deepcopy(toy_cfg.experiments)
    exps["E4"]["n_curveball"] = int(toy_cfg.default["expansion"]["curveball"]["n_null"])
    return Configs(toy_cfg.operator, toy_cfg.default, exps)


@pytest.fixture(scope="module")
def e4_run(e4_cfg, toy_root, recorder_class):
    rec = recorder_class()
    ctx = Context(e4_cfg, root=toy_root, access_log=rec)
    repo_file = result_path("E4", 0, root=ROOT)
    before = _state(repo_file)
    path = e4.run(ctx, 0, force=True)
    assert _state(repo_file) == before                      # a run with a ctx never writes into the real repo
    return read_result(path), rec, ctx


@pytest.fixture(scope="module")
def e5_cfg(toy_cfg) -> Configs:
    """E5 config on the 12-cell toy connectome: ``random_12`` exercises the same-density mapping, ``random_1886`` the
    ``random:<m>`` template path; the 327 680-cell expansion is left to the engine tests."""
    exps = copy.deepcopy(toy_cfg.experiments)
    exps["E5"]["expansions"] = ["none", "measured", "random_12", "random_1886"]
    return Configs(toy_cfg.operator, toy_cfg.default, exps)


@pytest.fixture(scope="module")
def e5_run(e5_cfg, toy_root, recorder_class):
    rec = recorder_class()
    ctx = Context(e5_cfg, root=toy_root, access_log=rec)
    return read_result(e5.run(ctx, 0, force=True)), rec


# ---------------------------------------------------------------------------------------------- pure helpers
def test_readout_and_config_parsing(toy_cfg):
    assert parse_readout("bloom_10shot") == Readout("bloom_10shot", "bloom", 10)
    assert parse_readout("linear").suffix == "linear" and parse_readout("bloom_full").suffix == "bloom"
    with pytest.raises(ValueError):
        parse_readout("knn")
    readouts, nulls = e4_config(toy_cfg)
    assert [r.id for r in readouts] == list(READOUTS) and nulls == ["curveball", "random", "dense_sign"]
    perms = grid_perms(toy_cfg, seeds_for(toy_cfg, 0)["perm"])
    assert len(perms) == 10 and sum(p["own"] for p in perms) == 1 and perms[0]["own"]
    assert len(grid_perms(toy_cfg, 12345)) == 11 and grid_perms(toy_cfg, 12345)[-1]["own"]


def test_permute_columns_reproduces_the_other_permutation():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(20, 7)).astype(np.float32)
    pi_own, pi_p = perm_for_seed(11, 7), perm_for_seed(12, 7)
    assert not np.array_equal(pi_own, pi_p)
    assert np.array_equal(permute_columns(S[:, pi_own], pi_own, pi_p), S[:, pi_p])
    assert np.array_equal(permute_columns(S[:, pi_own], pi_own, pi_own), S[:, pi_own])


def test_fixed_matrix_diff_ci_is_zero_for_identical_tables(three_docs):
    frames, scores = three_docs
    nulls = {s: np.stack([scores[s]] * 4) for s in frames}
    cis = fixed_matrix_diff_ci(frames, scores, nulls, n=30, seed=1, alpha=0.05)
    assert set(cis) == {"macro_auc"} | {f"auc/{s}" for s in frames}
    for ci in cis.values():
        assert ci.point == pytest.approx(0.0, abs=1e-12) and ci.low == pytest.approx(0.0, abs=1e-12)


@pytest.fixture(scope="module")
def three_docs():
    import pandas as pd

    rng = np.random.default_rng(3)
    frames, scores = {}, {}
    for s, n in (("a", 30), ("b", 24)):
        y = rng.integers(0, 2, n)
        frames[s] = pd.DataFrame({"doc_id": [f"{s}{i}" for i in range(n)], "label": y,
                                  "cluster_id": [f"{s}c{i // 3}" for i in range(n)]})
        scores[s] = rng.normal(size=n) + y
    return frames, scores


def test_e5_grid_rules(toy_cfg):
    dims = {"n51_svd": 51, "n51_hash": 51, "n16k": 16384}
    cells = cells_from_config(toy_cfg, dims, 51, 1886)
    assert len(cells) == 19
    assert Cell("n16k", "measured", "linear", "measured") not in cells and not any(
        c.expansion == "none" and c.readout == "bloom" for c in cells)
    kinds = {c.name: c.matrix for c in cells}
    assert kinds["n51_svd__random_1886__bloom"] == "random" and kinds["n16k__random_1886__linear"] == "random:1886"
    assert kinds["n16k__random_327680__bloom"] == "random:327680" and kinds["n51_hash__none__linear"] is None
    assert matrix_kind("random_1886", same_channels=False, n_cells=1886) == "random:1886"
    with pytest.raises(ValueError):
        matrix_kind("flyhash", True, 1886)
    pairs = contribution_pairs(cells, ["deep"])
    assert ("macro_auc", "n51_svd__measured__linear", "n51_svd__none__linear") in pairs
    assert ("auc/deep", "n51_svd__measured__bloom", "n51_svd__measured__linear") in pairs
    assert ("macro_auc", "n51_hash__measured__bloom", "n51_hash__random_1886__bloom") in pairs
    assert len(pairs) == 2 * (8 + 8 + 4)


# ---------------------------------------------------------------------------------------------- E4
DETS = {"bloom_full": "real_fly_bloom", "linear": "real_fly_linear", "bloom_10shot": "real_fly_bloom_10shot"}
METRICS = ("macro_auc",) + tuple(f"auc/{s}" for s in POS)


def test_e4_result_keys_and_intervals(e4_run, toy_root):
    r, rec, ctx = e4_run
    nums = r["numbers"]
    assert r["experiment"] == "E4" and r["seed"] == 0 and r["smoke"] is False
    for det in ("real_fly_bloom", "real_fly_linear", "real_fly_bloom_10shot", "random_fly_bloom", "random_fly_linear",
                "random_fly_bloom_10shot", "dense_sign_linear"):
        assert f"macro_auc/{det}" in nums and f"auc/deep/{det}" in nums
    assert "val_macro_auc/real_fly_bloom" in nums and "diff90/macro_auc/real_fly_bloom-random_fly_bloom" in nums
    assert "diff/macro_auc/real_fly_linear-dense_sign_linear" in nums and "diff/auc/bipia/real_fly_bloom_10shot-random_fly_bloom_10shot" in nums
    assert not any("dense_sign_bloom" in k for k in nums)
    for rid, det in DETS.items():
        pair = f"macro_auc/{det}-curveball_mean"
        d90, d95 = nums[f"diff90/{pair}"], nums[f"diff/{pair}"]
        assert d90["level"] == pytest.approx(0.9) and d95["level"] == pytest.approx(0.95)
        assert d95["ci_low"] <= d90["ci_low"] <= d90["value"] <= d90["ci_high"] <= d95["ci_high"]
        assert d90["n_perms"] == 10 and d90["n_null"] == 3 and d90["tost_equivalent"] in (True, False)
        assert d90["reference"] == pytest.approx(nums[f"macro_auc/curveball_mean/{det}"]["value"])
        assert d90["p_randomization"] == pytest.approx(nums[f"p_randomization/macro_auc/{det}"]["value"]) == d95["p"]
        assert d90["delta"] == pytest.approx(0.05 * d90["reference"]) == nums[f"tost_delta/{pair}"]["value"]
        assert nums[f"tost_equivalent/{pair}"]["value"] == float(d90["tost_equivalent"])
        assert d90["value"] == pytest.approx(nums[f"macro_auc/pi_mean/{det}"]["value"] - d90["reference"])
        w = nums[f"contrib/wiring/{det}/macro_auc"]
        assert w["ci_low"] - 1e-12 <= w["value"] <= w["ci_high"] + 1e-12 and 0 < w["p"] <= 1
        for m in METRICS:
            assert f"contrib/wiring/{det}/{m}" in nums and f"curveball_mean_ownperm/{m}/{det}" in nums
            assert nums[f"curveball_sd/{m}/{det}"]["n"] == 3
        assert nums[f"pi_sd/macro_auc/{det}"]["n"] == 10 and f"pi_sd/macro_auc/curveball_mean/{det}" in nums
    h3 = r["tables"]["h3"]
    assert len(h3) == 3 and h3[0]["readout"] == "bloom_full" and h3[0]["primary"] is True and "outside_corridor" in h3[0]
    assert h3[0]["detector"] == "real_fly_bloom" and set(h3[0]["per_perm"]) == {f"p{i:02d}" for i in range(10)}
    curve, sec = r["tables"]["curveball"], r["tables"]["nulls_secondary"]
    assert len(curve) == 3 and {c["readout"] for c in curve} == {"bloom_full"} and curve[0]["matrix"] == "curveball:0"
    assert {"j", "macro_auc", "auc_deep", "auc_para"} <= set(curve[0]) and len(sec) == 6 and "matrix" not in sec[0]
    assert len(r["tables"]["perm_grid"]) == 10 * 3 * 6
    own = [row for row in r["tables"]["perm_grid"] if row["own"]]
    assert len(own) == 18 and all("random" in row and "dense_sign" not in row for row in own if row["readout"] != "linear")
    assert all(row["param"] in ("gamma", "C") for row in r["tables"]["grid_hyper"])
    assert (toy_root / "results" / "E4" / "0.json").exists()
    assert any("two_stage" in n for n in r["notes"]) and any("permutations" in n for n in r["notes"])


def test_e4_randomisation_and_holm_family(e4_run):
    r, _, _ = e4_run
    nums = r["numbers"]
    family = [(det, m) for det in DETS.values() for m in METRICS if (det, m) != ("real_fly_bloom", "macro_auc")]
    assert "p_randomization_holm/macro_auc/real_fly_bloom" not in nums
    for det in DETS.values():
        for m in METRICS:
            p, p_own = nums[f"p_randomization/{m}/{det}"], nums[f"p_randomization_ownperm/{m}/{det}"]
            assert 1 / 4 - 1e-12 <= p["value"] <= 1.0 and 1 / 4 - 1e-12 <= p_own["value"] <= 1.0 and p["n"] == 3
    for det, m in family:
        assert nums[f"p_randomization_holm/{m}/{det}"]["value"] >= nums[f"p_randomization/{m}/{det}"]["value"] - 1e-12
        assert nums[f"p_randomization_holm/{m}/{det}"]["n"] == len(family) == 17
    assert any("Holm family of 17" in n for n in r["notes"])


def test_e4_journal_and_own_perm_consistency(e4_run):
    r, rec, ctx = e4_run
    purposes = {c[2] for c in rec.calls}
    assert all("E4 seed=0" in p for p in purposes) and len(purposes) == 5
    assert {p.split("(")[1].split(")")[0] for p in purposes} == set(POS)   # positive sources only, no NotInject
    assert r["timing"]["test_reads"] == list(ctx.test_sources[:5])
    grid = {(row["readout"], row["metric"]): row for row in r["tables"]["perm_grid"] if row["own"]}
    for rid, det in DETS.items():
        assert grid[(rid, "macro_auc")]["measured"] == pytest.approx(r["numbers"][f"macro_auc/{det}"]["value"])
        assert grid[(rid, "auc/deep")]["measured"] == pytest.approx(r["numbers"][f"auc/deep/{det}"]["value"])
    assert grid[("linear", "macro_auc")]["dense_sign"] == pytest.approx(r["numbers"]["macro_auc/dense_sign_linear"]["value"])
    assert grid[("bloom_full", "macro_auc")]["random"] == pytest.approx(r["numbers"]["macro_auc/random_fly_bloom"]["value"])
    # the wiring point at the own π equals measured − mean(curveball) of the histogram table
    hist = [row["macro_auc"] for row in r["tables"]["curveball"]]
    expect = r["numbers"]["macro_auc/real_fly_bloom"]["value"] - float(np.mean(hist))
    assert r["numbers"]["contrib/wiring/real_fly_bloom/macro_auc"]["value"] == pytest.approx(expect)
    assert r["numbers"]["curveball_mean_ownperm/macro_auc/real_fly_bloom"]["value"] == pytest.approx(float(np.mean(hist)))


def test_e4_smoke_routing_skip_and_config_check(toy_cfg, toy_root, e4_run):
    _, _, ctx = e4_run
    from flyguard.experiments.engine import Runner

    runner = Runner("E4", 0, smoke=True, root=toy_root, cfg=toy_cfg)
    runner._ctx = ctx
    path = runner.run(lambda fc, rb: rb.add_number("smoke/probe", 1.0))   # routing only: the body ran above
    assert path == toy_root / "results" / "smoke" / "E4" / "0.json" and read_result(path)["smoke"] is True
    assert e4.run(ctx, 0) == result_path("E4", 0, root=toy_root) and e4.seeds_to_run(toy_cfg, smoke=True) == [0]
    # E4.yaml's n_curveball must be the J the engine draws (exempt in smoke mode)
    e4.check_n_curveball(toy_cfg, int(toy_cfg.exp("E4")["n_curveball"]), smoke=False)
    e4.check_n_curveball(toy_cfg, 3, smoke=True)
    with pytest.raises(ValueError, match="n_curveball"):
        e4.check_n_curveball(toy_cfg, 3, smoke=False)


# ---------------------------------------------------------------------------------------------- E5
def test_e5_cells_and_contributions(e5_run):
    r, rec = e5_run
    nums, cells = r["numbers"], r["tables"]["cells"]
    assert len(cells) == 19 and {c["cell"] for c in cells} >= {"n51_svd__measured__bloom", "n16k__none__linear",
                                                               "n51_hash__random_12__linear", "n16k__random_1886__bloom"}
    by = {c["cell"]: c for c in cells}
    assert by["n51_svd__random_12__bloom"]["matrix"] == "random" and by["n16k__random_1886__linear"]["matrix"] == "random:1886"
    assert by["n51_svd__measured__bloom"]["gamma"] is not None and by["n51_svd__none__linear"]["m"] is None
    assert by["n51_svd__measured__bloom"]["macro_auc"] == pytest.approx(nums["macro_auc/n51_svd__measured__bloom"]["value"])
    for nose in ("n51_svd", "n51_hash", "n16k"):
        assert nums[f"contrib/nose/{nose}/macro_auc"]["value"] == pytest.approx(nums[f"macro_auc/{nose}__none__linear"]["value"])
    exp = nums["contrib/expansion/n51_svd/measured/macro_auc"]
    assert exp["value"] == pytest.approx(nums["macro_auc/n51_svd__measured__linear"]["value"]
                                         - nums["macro_auc/n51_svd__none__linear"]["value"])
    assert exp["ci_low"] <= exp["value"] <= exp["ci_high"] and "p" in exp
    rule = nums["contrib/rule/n16k/random_12/auc/deep"]
    assert rule["value"] == pytest.approx(nums["auc/deep/n16k__random_12__bloom"]["value"] - nums["auc/deep/n16k__random_12__linear"]["value"])
    assert "contrib/rule/n51_svd/none/macro_auc" not in nums and "contrib/expansion/n16k/measured/macro_auc" not in nums
    for nose in ("n51_svd", "n51_hash"):
        for ro in ("linear", "bloom"):
            w = nums[f"contrib/wiring/{nose}/{ro}/macro_auc"]
            assert w["ci_low"] - 1e-12 <= w["value"] <= w["ci_high"] + 1e-12 and 0.25 - 1e-12 <= w["p"] <= 1.0
            assert f"contrib/wiring_random/{nose}/{ro}/auc/deep" in nums
    kinds = {row["kind"] for row in r["tables"]["contributions"]}
    assert kinds == {"nose", "expansion", "rule", "wiring", "wiring_random"}
    assert len({c[2] for c in rec.calls}) == 5 and all("E5 seed=0" in c[2] for c in rec.calls)


def test_e4_and_e5_agree_on_the_shared_detectors(e4_run, e5_run):
    r4, r5 = e4_run[0]["numbers"], e5_run[0]["numbers"]
    for det, cell in (("real_fly_bloom", "n51_svd__measured__bloom"), ("real_fly_linear", "n51_svd__measured__linear"),
                      ("random_fly_bloom", "n51_svd__random_12__bloom"), ("random_fly_linear", "n51_svd__random_12__linear")):
        assert r4[f"macro_auc/{det}"] == r5[f"macro_auc/{cell}"], det
    w4, w5 = r4["contrib/wiring/real_fly_bloom/macro_auc"], r5["contrib/wiring/n51_svd/bloom/macro_auc"]
    assert (w4["value"], w4["ci_low"], w4["ci_high"], w4["p"]) == (w5["value"], w5["ci_low"], w5["ci_high"], w5["p"])
    assert r4["diff/macro_auc/real_fly_bloom-random_fly_bloom"] == r5["contrib/wiring_random/n51_svd/bloom/macro_auc"]
