"""metrics.py: AUC/macroAUC (ТЗ 2.6 property: macroAUC = mean of AUC_S), operating points, document scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from flyguard.eval.metrics import (auc, doc_scores, fpr_at_threshold, macro_auc, tau_for_tpr, threshold_for_fpr,
                                   tpr_at_fpr, tpr_at_threshold)


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.normal(size=400) + 0.8 * y, 1)
    assert auc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_edge_cases():
    assert auc([0.2, 0.9], [0, 1]) == 1.0
    assert auc([0.9, 0.2], [0, 1]) == 0.0
    assert auc([0.5, 0.5], [0, 1]) == 0.5
    assert np.isnan(auc([0.1, 0.2], [1, 1]))
    assert np.isnan(auc([], []))


def test_macro_auc_is_mean_of_present_sources():
    """ТЗ 2.6: macroAUC равен среднему AUC_S; absent/undefined sources are dropped, not zeroed."""
    per = {"deep": 0.9, "bipia": 0.7, "dojo": 0.8}
    assert macro_auc(per) == pytest.approx(0.8)
    assert macro_auc({**per, "dyn": float("nan"), "para": None}) == pytest.approx(0.8)
    assert np.isnan(macro_auc({}))
    # sizes do not matter: pooling by document count is not what macroAUC does
    rng = np.random.default_rng(1)
    big = pd.DataFrame({"y": rng.integers(0, 2, 2000)})
    big["s"] = big["y"] + rng.normal(size=2000)
    small = pd.DataFrame({"y": rng.integers(0, 2, 40)})
    small["s"] = -small["y"] + rng.normal(size=40)
    m = macro_auc({"a": auc(big["s"], big["y"]), "b": auc(small["s"], small["y"])})
    assert m == pytest.approx((auc(big["s"], big["y"]) + auc(small["s"], small["y"])) / 2)


def test_threshold_for_fpr_respects_target_and_ties():
    neg = np.array([0.1] * 50 + [0.5] * 40 + [0.9] * 10)
    tau = threshold_for_fpr(neg, 0.10)
    assert fpr_at_threshold(neg, tau) <= 0.10
    assert tau == pytest.approx(0.9)
    tau5 = threshold_for_fpr(neg, 0.05)  # no attainable point below the max -> just above the max, FPR 0
    assert fpr_at_threshold(neg, tau5) == 0.0
    assert threshold_for_fpr(neg, 1.0) == pytest.approx(0.1)


def test_threshold_for_fpr_sits_at_a_pool_score():
    """Documented convention (as sklearn's roc_curve thresholds): τ is a negative-pool score, so positives lying
    strictly between two consecutive pool scores are not caught although a lower τ would keep the same pool FPR."""
    neg = np.array([0.1] * 90 + [0.5] * 10)
    pos = np.array([0.3] * 50 + [0.6] * 50)
    tpr, tau = tpr_at_fpr(pos, neg, 0.10)
    assert tau == pytest.approx(0.5) and tpr == pytest.approx(0.5)
    lower = np.nextafter(0.1, np.inf)
    assert fpr_at_threshold(neg, lower) == pytest.approx(0.10) and tpr_at_threshold(pos, lower) == 1.0


def test_tpr_at_fpr_monotone_in_fpr():
    rng = np.random.default_rng(2)
    neg = rng.normal(size=3000)
    pos = rng.normal(size=300) + 1.2
    prev_tpr, prev_tau = -1.0, np.inf
    for f in (0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        tpr, tau = tpr_at_fpr(pos, neg, f)
        assert fpr_at_threshold(neg, tau) <= f + 1e-12
        assert tpr >= prev_tpr and tau <= prev_tau
        assert tpr == pytest.approx(tpr_at_threshold(pos, tau))
        prev_tpr, prev_tau = tpr, tau


def test_tau_for_tpr_monotone_and_exact():
    rng = np.random.default_rng(3)
    pos = np.round(rng.normal(size=500), 2)
    prev = np.inf
    for t in (0.5, 0.8, 0.9, 0.95, 1.0):
        tau = tau_for_tpr(pos, t)
        assert tpr_at_threshold(pos, tau) >= t
        assert tau <= prev
        prev = tau
    assert tau_for_tpr(pos, 1.0) == pytest.approx(pos.min())
    with pytest.raises(ValueError):
        tau_for_tpr([], 0.9)


def test_doc_scores_max_over_non_excluded_windows():
    windows = pd.DataFrame({
        "window_id": ["a#w0", "a#w1", "a#w2", "b#w0", "c#w0"],
        "doc_id": ["a", "a", "a", "b", "c"],
        "source": ["deep"] * 3 + ["bipia", "bipia"],
        "cluster_id": ["a", "a", "a", "ctx1", "ctx1"],
        "dedup_excluded": [False, True, False, False, True],
    })
    scores = pd.DataFrame({"window_id": ["a#w0", "a#w1", "a#w2", "b#w0", "c#w0"],
                           "score": [0.2, 0.99, 0.4, 0.7, 0.8]})
    out = doc_scores(scores, windows).set_index("doc_id")
    assert out.loc["a", "score"] == pytest.approx(0.4)  # the excluded 0.99 window does not count
    assert out.loc["b", "score"] == pytest.approx(0.7)
    assert "c" not in out.index  # all windows excluded -> document leaves
    assert list(out.columns) == ["score", "source", "cluster_id"]
    # without a dedup column nothing is excluded
    out2 = doc_scores(scores, windows.drop(columns=["dedup_excluded"])).set_index("doc_id")
    assert out2.loc["a", "score"] == pytest.approx(0.99) and out2.loc["c", "score"] == pytest.approx(0.8)


def test_doc_scores_refuses_unscored_windows():
    """A window without a score must not silently lower the maximum or make its document vanish."""
    windows = pd.DataFrame({"window_id": ["a#w0", "a#w1", "b#w0", "c#w0"], "doc_id": ["a", "a", "b", "c"],
                            "dedup_excluded": [False, False, False, True]})
    partial = pd.DataFrame({"window_id": ["a#w0", "z#w0"], "score": [0.2, 0.9]})  # a#w1 and b#w0 unscored
    with pytest.raises(ValueError, match="2 of 3"):
        doc_scores(partial, windows)
    with pytest.warns(RuntimeWarning, match="2 documents"):
        out = doc_scores(partial, windows, strict=False).set_index("doc_id")
    assert list(out.index) == ["a"] and out.loc["a", "score"] == pytest.approx(0.2)
    # an excluded window needs no score; scores of unknown windows are ignored
    full = pd.DataFrame({"window_id": ["a#w0", "a#w1", "b#w0", "z#w0"], "score": [0.2, 0.5, 0.7, 0.9]})
    out = doc_scores(full, windows).set_index("doc_id")
    assert out.loc["a", "score"] == pytest.approx(0.5) and "c" not in out.index
    with pytest.raises(ValueError, match="duplicated"):
        doc_scores(pd.concat([full, full.iloc[:1]]), windows)
