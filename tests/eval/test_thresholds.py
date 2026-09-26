"""thresholds.py: ТЗ 2.5 records carry value/source/target/n; the pool-size rule of ТЗ Этап 0."""
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


def test_records_have_required_fields():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=2500)
    rec = tau_fpr_record(pool)
    assert set(rec) >= {"value", "source", "target", "n"}
    assert rec["source"] == "P_val" and rec["target"] == "fpr<=0.01" and rec["n"] == 2500
    assert fpr_at_threshold(pool, rec["value"]) <= 0.01 and rec["achieved_fpr"] <= 0.01
    assert tau_fpr_record(pool[:400]) is None  # too small: AUC only
    assert tau_fpr_record(pool[:400], fpr=0.05)["target"] == "fpr<=0.05"
    pos = rng.normal(size=300) + 1
    t90 = tau_tpr_record(pos, "deep")
    assert t90["source"] == "deep" and t90["target"] == "tpr>=0.9" and t90["n"] == 300
    assert tpr_at_threshold(pos, t90["value"]) >= 0.9
    assert tau_tpr_record(pos, "dojo", tpr=0.8)["value"] >= t90["value"]
    d = default_threshold_record(0.5, "protectai_v2")
    assert d == {"value": 0.5, "source": "default:protectai_v2", "target": "default", "n": 0}
    assert threshold_record(0.3, "x", "y", 4, note=None) == {"value": 0.3, "source": "x", "target": "y", "n": 4}
