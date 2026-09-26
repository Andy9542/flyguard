"""tost.py: equivalence decisions, Holm, randomisation p-value, bootstrap p."""
from __future__ import annotations

import numpy as np
import pytest

from flyguard.config import load_configs
from flyguard.eval.bootstrap import CI
from flyguard.eval.tost import (bootstrap_p, equivalence_margin, holm, holm_reject, p_from_ci, randomization_p,
                                tost, tost_equivalent)


def test_equivalence_margin_is_relative(ci_dict):
    assert equivalence_margin(0.8, 0.05) == pytest.approx(0.04)
    assert equivalence_margin(0.9) == pytest.approx(0.045)  # delta_rel from configs/default.yaml


def test_tost_decisions(ci_dict):
    delta = 0.04
    assert tost_equivalent(ci_dict(0.0, -0.03, 0.03, 0.9), delta)
    assert tost_equivalent(ci_dict(0.0, -0.04, 0.04, 0.9), delta)  # closed corridor
    assert not tost_equivalent(ci_dict(0.0, -0.05, 0.03, 0.9), delta)
    assert not tost_equivalent(ci_dict(0.06, 0.05, 0.07, 0.9), delta)
    assert not tost_equivalent(ci_dict(0.0, float("nan"), 0.03, 0.9), delta)
    assert tost_equivalent(CI(0.0, -0.01, 0.01, 0.9), delta)
    small_but_real = tost(ci_dict(0.02, 0.01, 0.03, 0.9), delta)
    assert small_but_real["equivalent"] and small_but_real["differs_from_zero"] and not small_but_real["outside_corridor"]
    clearly_worse = tost(ci_dict(-0.08, -0.10, -0.06, 0.9), delta)
    assert not clearly_worse["equivalent"] and clearly_worse["outside_corridor"]
    wide = tost(ci_dict(0.0, -0.1, 0.1, 0.9), delta)
    assert not wide["equivalent"] and not wide["outside_corridor"] and not wide["differs_from_zero"]


def test_holm_known_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    adj = holm(p)
    assert adj["a"] == pytest.approx(0.04)   # 0.01 * 4
    assert adj["c"] == pytest.approx(0.09)   # 0.03 * 3
    assert adj["b"] == pytest.approx(0.09)   # 0.04 * 2 = 0.08 -> monotone -> 0.09
    assert adj["d"] == pytest.approx(0.20)
    rej = holm_reject(p, 0.05)
    assert rej == {"a": True, "b": False, "c": False, "d": False}
    assert holm({"x": 0.3}) == {"x": pytest.approx(0.3)}
    assert holm({}) == {}
    assert holm({"a": 0.5, "b": 0.9}) == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}


def test_randomization_p_on_known_nulls():
    nulls = np.arange(1, 101, dtype=float)  # mean 50.5
    # observed far outside: only the +1 of the observed itself
    assert randomization_p(1000.0, nulls) == pytest.approx(1 / 101)
    # observed equal to the mean: every null is at least as extreme
    assert randomization_p(50.5, nulls) == pytest.approx(1.0)
    # |obs - 50.5| = 40.5 -> nulls with |x - 50.5| >= 40.5 are 1..10 and 91..100 = 20
    assert randomization_p(91.0, nulls) == pytest.approx(21 / 101)
    assert randomization_p(10.0, nulls) == pytest.approx(21 / 101)
    # literal formula with pre-centred statistics (center=0)
    assert randomization_p(0.5, [0.1, 0.2, 0.6, -0.7], center=0.0) == pytest.approx(3 / 5)
    # the default sidedness is stats.randomization.two_sided of configs/default.yaml
    cfg = load_configs()
    assert randomization_p(91.0, nulls) == randomization_p(91.0, nulls,
                                                           two_sided=cfg.default["stats"]["randomization"]["two_sided"])
    # one-sided
    assert randomization_p(91.0, nulls, two_sided=False, alternative="greater") == pytest.approx(11 / 101)
    assert randomization_p(10.0, nulls, two_sided=False, alternative="less") == pytest.approx(11 / 101)
    assert np.isnan(randomization_p(1.0, []))


def test_bootstrap_p_consistent_with_percentile_ci():
    rng = np.random.default_rng(0)
    s = rng.normal(0.03, 0.02, 4000)
    lo, hi = np.percentile(s, [2.5, 97.5])
    p = bootstrap_p(s, 0.0)
    assert (p < 0.05) == (not (lo <= 0.0 <= hi))
    assert bootstrap_p(np.array([1.0, 2.0, 3.0]), 0.0) == pytest.approx(0.0)
    assert bootstrap_p(np.array([-1.0, 1.0]), 0.0) == pytest.approx(1.0)


def test_p_from_ci_normal_approximation(ci_dict):
    ci = ci_dict(0.0392, 0.0, 0.0784, 0.95)  # point = 1.96 se -> p = 0.05
    assert p_from_ci(ci) == pytest.approx(0.05, abs=1e-3)
    assert p_from_ci(ci_dict(0.0, -0.1, 0.1)) == pytest.approx(1.0)
