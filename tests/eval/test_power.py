"""power.py (E0): table shape, carrier rule, simulation sanity, NotInject width, spread hooks."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from flyguard.config import load_configs
from flyguard.eval.metrics import auc
from flyguard.eval.power import (CARRIES, INSUFFICIENT, ONLY_5, ONLY_AUC, ONLY_FPR, binormal_shift, carrier_rule,
                                 notinject_table, power_cell, power_table, proportion_diff_ci_width, simulate_source,
                                 spread)


def test_carrier_rule_thresholds():
    cfg = load_configs()
    assert carrier_rule(2000, cfg) == (CARRIES, 0.01)
    assert carrier_rule(5000, cfg) == (CARRIES, 0.01)
    assert carrier_rule(1999, cfg) == (ONLY_5, 0.05)
    assert carrier_rule(500, cfg) == (ONLY_5, 0.05)
    assert carrier_rule(499, cfg) == (ONLY_AUC, None)
    assert carrier_rule(0, cfg) == (ONLY_AUC, None)


def test_binormal_simulation_hits_target_auc():
    assert norm.cdf(binormal_shift(0.85) / np.sqrt(2)) == pytest.approx(0.85)
    rng = np.random.default_rng(0)
    df = simulate_source(0.85, 0.05, 4000, 4000, [1, 2, 3, 5], rng, icc=0.2, corr=0.5)
    assert auc(df["ref"], df["label"]) == pytest.approx(0.85, abs=0.015)
    assert auc(df["cand"], df["label"]) == pytest.approx(0.90, abs=0.015)
    # labels are constant inside clusters and cluster sizes follow the given distribution
    assert (df.groupby("cluster_id")["label"].nunique() == 1).all()
    assert set(df.groupby("cluster_id").size().unique()) <= {1, 2, 3, 4, 5}
    solo = simulate_source(0.75, 0.0, 10, 10, None, rng)
    assert solo["cluster_id"].nunique() == 20


def test_power_cell_shape_and_monotone_trend():
    cell = power_cell({"s": {"n_pos": 150, "n_neg": 150, "cluster_sizes": [1, 2, 4]}}, 0.85, 0.05, n_rep=6,
                      n_boot=60, delta_grid=(0.01, 0.05, 0.15), seed=1)
    assert set(cell) >= {"auc", "delta", "mdd", "tost_power", "se_diff", "power_curve", "status", "n_rep",
                         "mdd_normal_approx", "tost_power_normal_approx"}
    assert cell["delta"] == pytest.approx(0.0425) and cell["n_rep"] == 6 and cell["se_diff"] > 0
    assert cell["status"] in (CARRIES, INSUFFICIENT)
    curve = cell["power_curve"]
    assert curve["0.15"] >= curve["0.01"]  # a bigger true difference is detected more often
    assert cell["mdd"] in (None, 0.01, 0.05, 0.15)
    empty = power_cell({"s": {"n_pos": 0, "n_neg": 50, "cluster_sizes": None}}, 0.85, 0.05, n_rep=2, n_boot=10)
    assert empty["status"] == INSUFFICIENT and empty["mdd"] is None


def test_power_table_shape_and_statuses():
    cfg = load_configs()
    sizes = {
        "deep": {"n_pos": 40, "n_neg": 40, "cluster_sizes": [1] * 80},
        "dojo": {"n_pos": 200, "n_neg": 200, "cluster_sizes": [5, 10, 20]},
        "notinject": {"n_pos": 0, "n_neg": 339, "cluster_sizes": [1] * 339},
    }
    val = {"val_auc": {"deep": 0.78, "dojo": 0.93}, "curveball_val_macro_auc": [0.80, 0.81, 0.79, 0.805],
           "perm_val_macro_auc": [0.8, 0.82]}
    tab = power_table(cfg, sizes, val, pools={"P_test": 1200, "P_val": 700}, seed=0, n_rep=3, n_boot=40,
                      delta_grid=(0.02, 0.1))
    assert tab["levels"] == [0.75, 0.85, 0.95] and tab["delta_rel"] == 0.05 and tab["fpr_target"] == 0.05
    assert set(tab["carriers"]) == {"deep", "dojo", "notinject", "macro"}
    for s in ("deep", "dojo", "macro"):
        assert set(tab["carriers"][s]) == {"auc", "auc_diff", "tpr_at_fpr"}
        assert tab["carriers"][s]["tpr_at_fpr"] == ONLY_5
        assert tab["carriers"][s]["auc"] == CARRIES
        assert tab["carriers"][s]["auc_diff"] in (CARRIES, INSUFFICIENT)
        assert set(tab["cells"][s]) == {"0.75", "0.85", "0.95"}
    assert tab["carriers"]["notinject"] == {"auc": ONLY_FPR, "auc_diff": ONLY_FPR, "tpr_at_fpr": ONLY_FPR}
    assert tab["planning_level"] == {"deep": 0.75, "dojo": 0.95, "macro": 0.85}
    assert tab["notinject"]["n"] == 339 and tab["spread"]["curveball"]["n"] == 4 and tab["spread"]["perm"]["n"] == 2
    assert "sd_over_delta" in tab["spread"]["curveball"]
    assert set(tab["hypotheses"]) == {"H1a", "H1b", "H2", "H3"}
    assert set(tab["hypotheses"]["H1a"]["template"]) == {"deep", "dojo"}
    assert tab["assumptions"]["n_boot"] == 40 and tab["sizes"]["dojo"]["n_clusters"] == 3
    # a large pool carries TPR@1 %, a tiny one only AUC
    big = power_table(cfg, {"deep": sizes["deep"]}, None, pools={"P_test": 2500}, n_rep=2, n_boot=20,
                      delta_grid=(0.1,))
    assert big["carriers"]["deep"]["tpr_at_fpr"] == CARRIES and big["fpr_target"] == 0.01
    tiny = power_table(cfg, {"deep": sizes["deep"]}, None, pools={"P_test": 100}, n_rep=2, n_boot=20,
                       delta_grid=(0.1,))
    assert tiny["carriers"]["deep"]["tpr_at_fpr"] == ONLY_AUC and tiny["fpr_target"] is None
    assert tiny["planning_level"]["deep"] == 0.85  # no validation AUC -> middle level
    few = power_table(cfg, {"deep": {"n_pos": 5, "n_neg": 5, "cluster_sizes": None}}, None, pools={"P_test": 3000},
                      n_rep=2, n_boot=20, delta_grid=(0.1,))
    assert few["carriers"]["deep"] == {"auc": INSUFFICIENT, "auc_diff": INSUFFICIENT, "tpr_at_fpr": INSUFFICIENT}


def test_notinject_width_and_spread():
    w = proportion_diff_ci_width(339, 0.5, 0.5)
    assert w == pytest.approx(2 * 1.96 * np.sqrt(0.5 / 339), abs=1e-3)
    assert proportion_diff_ci_width(339, 0.1, 0.1, rho=0.5) < proportion_diff_ci_width(339, 0.1, 0.1)
    tab = notinject_table(339, 10)
    assert tab["worst_case_width"] == pytest.approx(w) and tab["corridor_reachable_worst_case"]
    assert tab["by_fpr"]["0.1"]["max_abs_diff_for_corridor"] == pytest.approx(0.10 - tab["by_fpr"]["0.1"]["ci_width"] / 2)
    s = spread([0.8, 0.82, 0.78], delta=0.04)
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.8) and s["sd_over_delta"] == pytest.approx(s["sd"] / 0.04)
    assert spread(None) is None and spread([]) is None and spread([0.5])["sd"] == 0.0
