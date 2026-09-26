"""thresholds.py: ТЗ 2.5 records carry value/source/target/n; the |P_test| rule of ТЗ Этап 0 picks the FPR target."""
from __future__ import annotations

import numpy as np
import pytest

from flyguard.config import load_configs
from flyguard.eval.metrics import fpr_at_threshold, tpr_at_threshold
from flyguard.eval.thresholds import (default_threshold_record, fpr_target_for_pool, tau_fpr_record, tau_tpr_record,
                                      threshold_record)


def test_fpr_target_rule_reads_config():
    cfg = load_configs()
    lo = cfg.default["thresholds"]["pool_min_docs"]["fpr_5pct"]
    hi = cfg.default["thresholds"]["pool_min_docs"]["fpr_1pct"]
    assert fpr_target_for_pool(hi, cfg) == pytest.approx(0.01)
    assert fpr_target_for_pool(hi - 1, cfg) == pytest.approx(0.05)
    assert fpr_target_for_pool(lo, cfg) == pytest.approx(0.05)
    assert fpr_target_for_pool(lo - 1, cfg) is None


def test_tau_fpr_target_comes_from_test_pool_not_val_pool():
    """ТЗ Этап 0 / 2.5: the 1 % / 5 % / AUC-only choice is made on |P_test|; τ itself is frozen on P_val."""
    rng = np.random.default_rng(0)
    pool = rng.normal(size=2500)
    with pytest.raises(ValueError):
        tau_fpr_record(pool)  # neither fpr nor |P_test|: the target cannot be inferred from the P_val size
    rec = tau_fpr_record(pool, n_test_pool=2500)
    assert set(rec) >= {"value", "source", "target", "n", "n_test_pool", "achieved_fpr"}
    assert rec["source"] == "P_val" and rec["target"] == "fpr<=0.01" and rec["n"] == 2500 and rec["n_test_pool"] == 2500
    assert fpr_at_threshold(pool, rec["value"]) <= 0.01 and rec["achieved_fpr"] <= 0.01
    # a small validation pool does not change the target when the test pool is large ...
    small_val = tau_fpr_record(pool[:400], n_test_pool=2500)
    assert small_val["target"] == "fpr<=0.01" and small_val["n"] == 400
    # ... and a large validation pool does not rescue a small test pool
    assert tau_fpr_record(pool, n_test_pool=1500)["target"] == "fpr<=0.05"
    assert tau_fpr_record(pool, n_test_pool=400) is None  # AUC only
    explicit = tau_fpr_record(pool[:400], fpr=0.05)  # target taken from results/power.json
    assert explicit["target"] == "fpr<=0.05" and "n_test_pool" not in explicit


def test_tau_tpr_and_default_records():
    rng = np.random.default_rng(1)
    pos = rng.normal(size=300) + 1
    t90 = tau_tpr_record(pos, "deep")
    assert t90["source"] == "deep" and t90["target"] == "tpr>=0.9" and t90["n"] == 300
    assert tpr_at_threshold(pos, t90["value"]) >= 0.9
    assert tau_tpr_record(pos, "dojo", tpr=0.8)["value"] >= t90["value"]
    d = default_threshold_record(0.5, "protectai_v2")
    assert d == {"value": 0.5, "source": "default:protectai_v2", "target": "default", "n": 0}
    assert threshold_record(0.3, "x", "y", 4, note=None) == {"value": 0.3, "source": "x", "target": "y", "n": 4}
