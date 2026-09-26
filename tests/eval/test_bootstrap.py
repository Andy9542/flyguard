"""bootstrap.py: weighted AUC, cluster bootstrap coverage and determinism, paired and macro variants, H3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from flyguard.eval.bootstrap import (CI, WeightedAUC, auc_stat, cluster_bootstrap, cluster_codes, cluster_weights,
                                     macro_auc_bootstrap, paired_cluster_bootstrap, two_stage_bootstrap_h3)
from flyguard.eval.metrics import auc, macro_auc


def test_weighted_auc_equals_sklearn_on_expanded_sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    tables = np.stack([np.round(rng.normal(size=200) + y, 1), rng.normal(size=200), np.full(200, 0.3)])
    w = rng.integers(0, 4, 200)
    idx = np.repeat(np.arange(200), w)
    got = WeightedAUC(tables, y).auc(w)
    for t, g in zip(tables, got):
        assert g == pytest.approx(roc_auc_score(y[idx], t[idx]), abs=1e-12)
    assert WeightedAUC(tables[0], y).auc(None)[0] == pytest.approx(auc(tables[0], y))
    assert np.isnan(WeightedAUC(tables, y).auc(w * (y == 1))).all()  # negatives gone -> undefined


def test_weighted_auc_multi_chunk_matches_single_chunk():
    """The H3 run has ~2000 tables -> many row chunks; T=5 with chunk_rows=2 exercises a partial last chunk."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 150)
    tables = np.round(rng.normal(size=(5, 150)) + 0.7 * y, 1)  # ties within and across tables
    tables[3] = tables[0]  # identical tables in different chunks
    w = rng.integers(0, 3, 150)
    idx = np.repeat(np.arange(150), w)
    chunked = WeightedAUC(tables, y, chunk_rows=2)
    assert len(chunked._chunks) == 3
    got = chunked.auc(w)
    whole = WeightedAUC(tables, y).auc(w)
    for t, g, s in zip(tables, got, whole):
        expected = roc_auc_score(y[idx], t[idx])
        assert g == pytest.approx(expected, abs=1e-12) and s == pytest.approx(expected, abs=1e-12)
    assert got[3] == got[0]
    assert chunked.auc(None) == pytest.approx([auc(t, y) for t in tables], abs=1e-12)
    # repeated calls reuse the buffers without leaking state between draws
    assert chunked.auc(w) == pytest.approx(got, abs=1e-15)


def test_cluster_weights_resample_whole_clusters():
    codes, k = cluster_codes(["b", "a", "b", "c", "a"])
    assert k == 3 and list(codes) == [1, 0, 1, 2, 0]
    w = cluster_weights(codes, k, np.random.default_rng(1))
    assert w[0] == w[2] and w[1] == w[4] and w.sum() >= 0


def test_cluster_bootstrap_deterministic_and_covers_truth(source_factory):
    """Coverage of the 95 % interval for the AUC of a binormal source with clusters; 40 replicates."""
    make_source = source_factory
    true_auc = norm.cdf(1.0 / np.sqrt(2))
    hits = 0
    for rep in range(40):
        df = make_source(240, 60, 100 + rep, shift=1.0)
        ci = cluster_bootstrap(df, auc_stat("score"), n=200, seed=rep, alpha=0.05)
        hits += ci.contains(true_auc)
        assert ci.low <= ci.point <= ci.high
    assert hits >= 32  # nominal 38 of 40; allow bootstrap/small-sample slack
    df = make_source(240, 60, 5)
    a = cluster_bootstrap(df, auc_stat("score"), n=100, seed=11)
    b = cluster_bootstrap(df, auc_stat("score"), n=100, seed=11)
    c = cluster_bootstrap(df, auc_stat("score"), n=100, seed=12)
    assert a.to_dict() == b.to_dict() and a.to_dict() != c.to_dict()
    assert a.n == 240 and a.n_clusters == 60 and a.n_boot == 100 and a.level == 0.95
    assert set(a.to_dict()) == {"point", "low", "high", "level", "n_boot", "n", "n_clusters"}


def test_paired_bootstrap_identical_detectors_contains_zero(source_factory):
    df = source_factory(200, 40, 7)
    df["same"] = df["score"]
    ci = paired_cluster_bootstrap(df, auc_stat("score"), auc_stat("same"), n=150, seed=3)
    assert ci.point == 0.0 and ci.low == 0.0 and ci.high == 0.0 and ci.contains(0.0)
    fast = macro_auc_bootstrap({"s": df}, n=150, seed=3, score="score", reference="same")
    assert fast.point == 0.0 and fast.width == 0.0
    # paired interval is narrower than the naive difference of independent intervals
    df["other"] = df["ref"]
    paired = macro_auc_bootstrap({"s": df}, n=300, seed=4, score="score", reference="other")
    a = macro_auc_bootstrap({"s": df}, n=300, seed=4, score="score")
    b = macro_auc_bootstrap({"s": df}, n=300, seed=5, score="other")
    assert paired.point == pytest.approx(a.point - b.point)
    assert paired.width < a.width + b.width


def test_macro_auc_bootstrap_point_is_mean_of_sources(three_sources):
    ci = macro_auc_bootstrap(three_sources, n=200, seed=1)
    per = {s: auc(d["score"], d["label"]) for s, d in three_sources.items()}
    assert ci.point == pytest.approx(macro_auc(per))
    assert ci.low < ci.point < ci.high and ci.n == 620 and ci.n_clusters == 190
    same = macro_auc_bootstrap(dict(reversed(list(three_sources.items()))), n=200, seed=1)
    assert same.to_dict() == ci.to_dict()  # independent of dict order
    # a single-class source is dropped, as macroAUC averages present sources only
    one_class = three_sources["deep"].copy()
    one_class["label"] = 0
    dropped = macro_auc_bootstrap({**three_sources, "notinject": one_class}, n=200, seed=1)
    assert dropped.to_dict() == ci.to_dict()
    ci90 = macro_auc_bootstrap(three_sources, n=200, seed=1, alpha=0.10)
    assert ci90.level == pytest.approx(0.90) and ci90.width <= ci.width


def test_two_stage_h3_shape_and_zero_difference(three_sources):
    docs = {s: d[["label", "cluster_id"]] for s, d in three_sources.items()}
    rng = np.random.default_rng(0)
    perms = ["p0", "p1"]
    measured = {p: {s: d["score"].to_numpy() for s, d in three_sources.items()} for p in perms}
    # nulls identical to the measured matrix -> difference exactly 0 in every draw
    nulls = {p: {s: np.stack([measured[p][s]] * 5) for s in three_sources} for p in perms}
    r = two_stage_bootstrap_h3(docs, measured, nulls, n=50, seed=2)
    assert r["n_perms"] == 2 and r["n_null"] == 5
    zero = pytest.approx(0.0, abs=1e-12)
    assert r["diff"].point == zero and r["diff"].low == zero and r["diff"].high == zero and r["diff"].level == 0.90
    assert r["measured"].point == pytest.approx(r["null_mean"].point) == pytest.approx(r["delta_reference"])
    # noisy nulls: the interval is non-degenerate, deterministic per seed and has the measured mean inside
    nulls = {p: {s: measured[p][s][None, :] + rng.normal(0, 0.3, (5, len(measured[p][s]))) for s in three_sources}
             for p in perms}
    r1 = two_stage_bootstrap_h3(docs, measured, nulls, n=60, seed=9)
    r2 = two_stage_bootstrap_h3(docs, measured, nulls, n=60, seed=9)
    assert r1["diff"].to_dict() == r2["diff"].to_dict() and r1["diff"].width > 0
    assert r1["diff"].point > 0  # measured is the noiseless version, so it should score higher
    assert set(r1["per_perm"]) == {"p0", "p1"}


def test_ci_roundtrip_and_predicates():
    ci = CI(point=0.1, low=-0.05, high=0.2, level=0.9, n_boot=10, n=5)
    d = ci.to_dict()
    assert CI.from_dict(d).to_dict() == d
    assert ci.contains(0.0) and ci.inside(-0.1, 0.25) and not ci.inside(0.0, 0.25)
    assert ci.width == pytest.approx(0.25)
