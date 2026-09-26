"""power.py (E0): table shape, carrier rule, simulation sanity, config-driven constants, NotInject width, spread."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from flyguard.config import load_configs
from flyguard.eval.metrics import auc
from flyguard.eval.power import (CARRIES, CLUSTER_MODEL_CONSTANT, CLUSTER_MODEL_INDEPENDENT, CLUSTER_MODEL_OBSERVED,
                                 INSUFFICIENT, ONLY_5, ONLY_AUC, ONLY_FPR, binormal_shift, carrier_rule, cell_seed,
                                 cluster_model, notinject_table, power_cell, power_params, power_table,
                                 proportion_diff_ci_width, simulate_source, spread)


def test_carrier_rule_thresholds():
    cfg = load_configs()
    assert carrier_rule(2000, cfg) == (CARRIES, 0.01)
    assert carrier_rule(5000, cfg) == (CARRIES, 0.01)
    assert carrier_rule(1999, cfg) == (ONLY_5, 0.05)
    assert carrier_rule(500, cfg) == (ONLY_5, 0.05)
    assert carrier_rule(499, cfg) == (ONLY_AUC, None)
    assert carrier_rule(0, cfg) == (ONLY_AUC, None)


def test_power_params_come_from_config():
    cfg = load_configs()
    p = power_params(cfg)
    sp = cfg.default["stats"]["power"]
    assert p["icc"] == sp["icc"] and p["corr"] == sp["detector_corr"] and p["n_rep"] == sp["n_rep"]
    assert p["delta_grid"] == tuple(sp["delta_grid"]) and p["power_target"] == sp["power_target"]
    assert p["min_class_docs"] == sp["min_class_docs"] and p["alpha"] == cfg.default["stats"]["bootstrap"]["alpha"]
    assert p["tost_level"] == cfg.default["stats"]["tost"]["ci"] and p["n_boot"] == cfg.exp("E0")["synthetic_bootstrap"]
    smoke = power_params(cfg, smoke=True)
    assert smoke["n_rep"] == cfg.default["smoke"]["power"]["n_rep"]
    assert smoke["n_boot"] == cfg.default["smoke"]["power"]["n_boot"] and smoke["icc"] == p["icc"]
    over = power_params(cfg, n_rep=3, icc=None, delta_grid=[0.1])
    assert over["n_rep"] == 3 and over["icc"] == p["icc"] and over["delta_grid"] == (0.1,)
    with pytest.raises(TypeError):
        power_params(cfg, n_reps=3)


def test_binormal_simulation_hits_target_auc():
    assert norm.cdf(binormal_shift(0.85) / np.sqrt(2)) == pytest.approx(0.85)
    rng = np.random.default_rng(0)
    df = simulate_source(0.85, 0.05, 4000, 4000, [1, 2, 3, 5], rng, icc=0.2, corr=0.5)
    assert auc(df["ref"], df["label"]) == pytest.approx(0.85, abs=0.015)
    assert auc(df["cand"], df["label"]) == pytest.approx(0.90, abs=0.015)
    # label-constant mode (deep, para): labels are constant inside clusters, sizes follow the given distribution
    assert (df.groupby("cluster_id")["label"].nunique() == 1).all()
    assert set(df.groupby("cluster_id").size().unique()) <= {1, 2, 3, 4, 5}
    solo = simulate_source(0.75, 0.0, 10, 10, None, rng)
    assert solo["cluster_id"].nunique() == 20


def test_mixed_cluster_simulation_keeps_observed_composition():
    """bipia pairs and dojo/dyn task clusters hold both labels (ТЗ 1.4, design §2): the synthetic clusters must too."""
    rng = np.random.default_rng(1)
    pairs = simulate_source(0.85, 0.05, 3000, 3000, None, rng, cluster_label_sizes=[[1, 1]] * 400)
    g = pairs.groupby("cluster_id")["label"]
    assert len(pairs) == 6000 and pairs["label"].sum() == 3000 and (g.size() == 2).all() and (g.nunique() == 2).all()
    assert auc(pairs["ref"], pairs["label"]) == pytest.approx(0.85, abs=0.015)  # the marginal AUC is unchanged
    assert auc(pairs["cand"], pairs["label"]) == pytest.approx(0.90, abs=0.015)
    tasks = simulate_source(0.85, 0.0, 61, 240, None, rng, cluster_label_sizes=[[2, 8], [1, 8], [0, 10]])
    assert tasks["label"].sum() == 61 and (tasks["label"] == 0).sum() == 240  # totals exact, surplus trimmed
    comp = tasks.groupby("cluster_id")["label"].agg(["sum", "size"])
    assert set(comp["sum"].unique()) <= {0, 1, 2} and comp["size"].max() <= 10
    with pytest.raises(ValueError):
        simulate_source(0.85, 0.0, 10, 10, None, rng, cluster_label_sizes=[[0, 5]])  # no cluster holds a positive
    assert cluster_model({"cluster_label_sizes": [[1, 1]]}) == CLUSTER_MODEL_OBSERVED
    assert cluster_model({"cluster_sizes": [1]}) == CLUSTER_MODEL_CONSTANT
    assert cluster_model({"cluster_sizes": None}) == CLUSTER_MODEL_INDEPENDENT


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
    # the macro cell does not depend on the order of its sources
    two = {"a": {"n_pos": 30, "n_neg": 30, "cluster_label_sizes": [[1, 1]] * 30},
           "b": {"n_pos": 20, "n_neg": 40, "cluster_sizes": [2, 3]}}
    c1 = power_cell(two, 0.85, 0.05, n_rep=2, n_boot=20, delta_grid=(0.1,), seed=5)
    c2 = power_cell(dict(reversed(list(two.items()))), 0.85, 0.05, n_rep=2, n_boot=20, delta_grid=(0.1,), seed=5)
    assert c1 == c2


def test_power_table_shape_and_statuses():
    cfg = load_configs()
    sizes = {
        "deep": {"n_pos": 40, "n_neg": 40, "cluster_sizes": [1] * 80},
        "dojo": {"n_pos": 200, "n_neg": 200, "cluster_label_sizes": [[5, 10]] * 20 + [[10, 20]] * 10},
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
    # sizes: per source and for macroAUC (ТЗ Этап 0 "по каждому источнику и по macroAUC")
    assert tab["sizes"]["dojo"]["n_clusters"] == 30 and tab["sizes"]["deep"]["n_clusters"] == 80
    assert tab["sizes"]["macro"] == {"n_pos": 240, "n_neg": 240, "n_clusters": 110, "sources": ["deep", "dojo"]}
    # assumptions: overrides where given, configs/default.yaml otherwise, and the cluster model per source
    a = tab["assumptions"]
    sp = cfg.default["stats"]["power"]
    assert a["n_boot"] == 40 and a["n_rep"] == 3 and a["delta_grid"] == [0.02, 0.1]
    assert a["icc"] == sp["icc"] and a["detector_corr"] == sp["detector_corr"]
    assert a["power_target"] == sp["power_target"] and a["min_class_docs"] == sp["min_class_docs"]
    assert a["cluster_model"] == {"deep": CLUSTER_MODEL_CONSTANT, "dojo": CLUSTER_MODEL_OBSERVED,
                                  "notinject": CLUSTER_MODEL_CONSTANT}
    assert tab["power_target"] == sp["power_target"] and tab["tost_level"] == cfg.default["stats"]["tost"]["ci"]


def test_power_table_pool_rule_small_sources_and_order():
    cfg = load_configs()
    deep = {"n_pos": 40, "n_neg": 40, "cluster_sizes": [1] * 80}
    # a large pool carries TPR@1 %, a tiny one only AUC
    big = power_table(cfg, {"deep": deep}, None, pools={"P_test": 2500}, n_rep=2, n_boot=20, delta_grid=(0.1,))
    assert big["carriers"]["deep"]["tpr_at_fpr"] == CARRIES and big["fpr_target"] == 0.01
    tiny = power_table(cfg, {"deep": deep}, None, pools={"P_test": 100}, n_rep=2, n_boot=20, delta_grid=(0.1,))
    assert tiny["carriers"]["deep"]["tpr_at_fpr"] == ONLY_AUC and tiny["fpr_target"] is None
    assert tiny["planning_level"]["deep"] == 0.85  # no validation AUC -> middle level
    # too few documents per class: nothing carries, macroAUC included
    few = power_table(cfg, {"deep": {"n_pos": 5, "n_neg": 5, "cluster_sizes": None}}, None, pools={"P_test": 3000},
                      n_rep=2, n_boot=20, delta_grid=(0.1,))
    assert few["carriers"]["deep"] == {"auc": INSUFFICIENT, "auc_diff": INSUFFICIENT, "tpr_at_fpr": INSUFFICIENT}
    assert few["carriers"]["macro"] == {"auc": INSUFFICIENT, "auc_diff": INSUFFICIENT, "tpr_at_fpr": INSUFFICIENT}
    assert few["sizes"]["macro"]["n_clusters"] is None
    # the caller's dict order does not change any number, and a cell's seed depends on its name and level only
    two = {"deep": deep, "bipia": {"n_pos": 30, "n_neg": 30, "cluster_label_sizes": [[1, 1]] * 30}}
    t1 = power_table(cfg, two, None, pools={"P_test": 3000}, n_rep=1, n_boot=20, delta_grid=(0.1,))
    t2 = power_table(cfg, dict(reversed(list(two.items()))), None, pools={"P_test": 3000}, n_rep=1, n_boot=20,
                     delta_grid=(0.1,))
    assert t1["cells"] == t2["cells"] and t1["carriers"] == t2["carriers"]
    alone = power_table(cfg, {"deep": deep}, None, pools={"P_test": 3000}, n_rep=1, n_boot=20, delta_grid=(0.1,))
    assert t1["cells"]["deep"] == alone["cells"]["deep"]  # adding a source leaves the others' numbers unchanged
    assert cell_seed(0, "deep", 0.85) != cell_seed(0, "dojo", 0.85) != cell_seed(1, "dojo", 0.85)
    assert cell_seed(0, "deep", 0.85) == cell_seed(0, "deep", 0.85) and cell_seed(0, "deep", 0.75) != cell_seed(0, "deep", 0.85)
    # smoke mode takes replicates and draws from smoke.power (other overrides still win)
    sm = power_table(cfg, {"deep": {"n_pos": 12, "n_neg": 12, "cluster_sizes": None}}, None, pools={"P_test": 100},
                     delta_grid=(0.1,), smoke=True)
    assert sm["assumptions"]["n_rep"] == cfg.default["smoke"]["power"]["n_rep"]
    assert sm["assumptions"]["n_boot"] == cfg.default["smoke"]["power"]["n_boot"] and sm["assumptions"]["smoke"]


def test_notinject_width_and_spread():
    w = proportion_diff_ci_width(339, 0.5, 0.5)
    assert w == pytest.approx(2 * 1.96 * np.sqrt(0.5 / 339), abs=1e-3)
    assert proportion_diff_ci_width(339, 0.1, 0.1, rho=0.5) < proportion_diff_ci_width(339, 0.1, 0.1)
    tab = notinject_table(339, 10)
    assert tab["worst_case_width"] == pytest.approx(w) and tab["corridor_reachable_worst_case"]
    assert tab["by_fpr"]["0.1"]["max_abs_diff_for_corridor"] == pytest.approx(0.10 - tab["by_fpr"]["0.1"]["ci_width"] / 2)
    s = spread([0.8, 0.82, 0.78], delta=0.04)
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.8) and s["sd_over_delta"] == pytest.approx(s["sd"] / 0.04)
    assert s["half_width"] == pytest.approx(1.96 * s["sd"], rel=1e-3) and s["level"] == 0.95
    assert spread([0.8, 0.82, 0.78], level=0.90)["half_width"] < s["half_width"]
    assert spread(None) is None and spread([]) is None and spread([0.5])["sd"] == 0.0
