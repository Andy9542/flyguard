"""Shared helpers of the baselines (docs/design.md §6): grouped C selection, the shared estimator, the cache key."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from flyguard.baselines.common import cv_feasible, make_logreg, safe_auc, select_c, text_hash
from flyguard.baselines.lexical import KNN, LRSvd, NearestCentroid, TfidfLR
from flyguard.baselines.regex import RegexScorer
from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import Configs, load_configs
from flyguard.data import windows as data_windows
from flyguard.readout import make_logistic

GRID = [0.01, 1.0, 100.0]


def overlapping_window_problem(n_docs: int = 30, copies: int = 3, d: int = 200, seed: int = 0):
    """Pure-noise features with ``copies`` near-identical rows per document (overlapping windows of one text).

    Nothing predicts the label, so an honest cross-validated AUC is about 0.5; a split that puts copies of one
    document on both sides of a fold lets a high-C logistic regression recognise them and reports AUC near 1.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_docs, d))
    y_doc = np.arange(n_docs) % 2
    X = np.repeat(base, copies, axis=0) + 0.01 * rng.normal(size=(n_docs * copies, d))
    y = np.repeat(y_doc, copies)
    groups = np.repeat([f"doc:{i}" for i in range(n_docs)], copies)
    return X, y, groups


def test_grouped_cv_closes_the_window_leak():
    X, y, groups = overlapping_window_problem()
    C, source, table = select_c(lambda C: make_logreg(C, 0), X, y, grid=[100.0], seed=0, groups=groups)
    assert source == "cv" and C == 100.0
    assert table["100.0"] < 0.75, f"grouped cv AUC on noise should be near 0.5, got {table['100.0']:.3f}"
    # the same data split by window (the old behaviour) reports a leaked AUC near 1
    leaked = []
    for tr, te in StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X, y):
        model = make_logreg(100.0, 0).fit(X[tr], y[tr])
        leaked.append(safe_auc(y[te], model.predict_proba(X[te])[:, 1]))
    assert np.mean(leaked) > 0.9, "the fixture must exhibit the leak under a window-level split"


def test_grouped_folds_never_split_a_group():
    """Direct check of the fold construction: no group appears in both parts of any fold."""
    X, y, groups = overlapping_window_problem(n_docs=12, copies=2, d=5)
    seen: list[tuple[set, set]] = []

    class Spy:
        def __init__(self, C):
            self.C = C

        def fit(self, Xf, yf):
            self.rows = {tuple(np.round(r, 6)) for r in Xf}
            return self

        def predict_proba(self, Xq):
            self.q = {tuple(np.round(r, 6)) for r in Xq}
            seen.append((self.rows, self.q))
            return np.tile([0.5, 0.5], (Xq.shape[0], 1))

    select_c(Spy, X, y, grid=[1.0], seed=0, groups=groups)
    row_group = {tuple(np.round(r, 6)): g for r, g in zip(X, groups)}
    assert len(seen) == 3
    for rows_tr, rows_te in seen:
        assert not ({row_group[r] for r in rows_tr} & {row_group[r] for r in rows_te})


def test_select_c_refuses_cv_without_groups_but_keeps_val_and_default_paths():
    X, y, groups = overlapping_window_problem(n_docs=12, copies=2, d=5)
    with pytest.raises(ValueError, match="groups"):
        select_c(lambda C: make_logreg(C, 0), X, y, grid=GRID, seed=0)
    # a usable validation set needs no groups
    C, source, _ = select_c(lambda C: make_logreg(C, 0), X, y, X, y, grid=GRID, seed=0)
    assert source == "val" and C in GRID
    # too few rows per class -> default without groups (E2 shots=1)
    assert select_c(lambda C: make_logreg(C, 0), X[:4], y[:4], grid=GRID)[1] == "default"
    # enough rows but too few groups per class -> default, not an sklearn error
    C, source, table = select_c(lambda C: make_logreg(C, 0), X, y, grid=GRID, seed=0,
                                groups=np.where(y == 1, "pos", "neg"))
    assert source == "default" and C == 1.0 and all(v is None for v in table.values())
    with pytest.raises(ValueError, match="groups for"):
        select_c(lambda C: make_logreg(C, 0), X, y, grid=GRID, groups=groups[:-1])
    assert cv_feasible(y, groups, 3) and not cv_feasible(y, np.where(y == 1, "p", "n"), 3)
    assert not cv_feasible(np.ones_like(y), groups, 3)


def test_make_logreg_is_the_readout_estimator():
    cfg = load_configs()
    lin = cfg.default["readout"]["linear"]
    for C, seed in ((0.01, 0), (1.0, 7), (100.0, 3)):
        mine, theirs = make_logreg(C, seed, cfg), make_logistic(C, seed, cfg)
        assert type(mine) is type(theirs) and mine.get_params() == theirs.get_params()
        p = mine.get_params()
        assert p["C"] == C and p["random_state"] == seed
        assert p["solver"] == lin["solver"] and p["max_iter"] == int(lin["max_iter"])
        assert p["tol"] == float(lin["tol"]) and p["class_weight"] == lin["class_weight"]
        assert p["l1_ratio"] == 0.0 and lin["penalty"] == "l2"
    # a partial Configs without a readout block falls back to the defaults rather than failing
    partial = Configs(operator={}, default={"baselines": {}}, experiments={})
    assert make_logreg(1.0, 0, partial).get_params() == make_logistic(1.0, 0).get_params()
    # the scorers route through the same factory
    X = np.random.default_rng(0).normal(size=(40, 51))
    y = np.arange(40) % 2
    m = LRSvd(GRID, seed=5, cfg=cfg).fit(X, y, X, y)
    assert m.model_.get_params() == make_logistic(m.C_, 5, cfg).get_params()


def test_text_hash_is_the_data_windows_hash_on_unicode():
    assert text_hash is data_windows.text_hash
    s = "Ignoriere alle Anweisungen — привет, мир 🐝 café é vs é"
    assert text_hash(s) == data_windows.text_hash(s)
    import xxhash
    assert text_hash(s) == xxhash.xxh64(s.encode("utf-8"), seed=0).hexdigest()
    assert text_hash("é") != text_hash("é"), "the hash is over the exact text, not a normal form"


def test_fit_idf_leaves_the_caller_matrix_untouched():
    # raw CSR buffers with explicit zeros (rows 0 and 1) and a duplicate (2, 2) entry, as a non-canonical
    # matrix from a nose cache might carry; built from (data, indices, indptr) so nothing is merged beforehand
    cols = np.array([0, 1, 1, 2, 0, 2, 2])
    vals = np.array([1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0])
    X = sp.csr_matrix((vals, cols, np.array([0, 2, 4, 7])), shape=(3, 4))
    before = (X.nnz, X.data.copy(), X.indices.copy(), X.indptr.copy())
    m = TfidfLR([1.0], seed=0).fit_idf(X)
    after = (X.nnz, X.data, X.indices, X.indptr)
    assert before[0] == after[0] and all(np.array_equal(a, b) for a, b in zip(before[1:], after[1:]))
    # df counts documents with a nonzero entry: col0 -> 2, col1 -> 1, col2 -> 1 (duplicates merged), col3 -> 0
    n = 3
    expected = np.log((1.0 + n) / (1.0 + np.array([2.0, 1.0, 1.0, 0.0]))) + 1.0
    assert np.allclose(m.idf_, expected)
    assert np.allclose(TfidfLR([1.0], seed=0).fit_idf(X.toarray()).idf_, expected)


def test_every_scorer_fit_accepts_groups_keyword(tmp_path):
    for cls in (TfidfLR, LRSvd, KNN, NearestCentroid, RegexScorer, GuardModel):
        assert "groups" in inspect.signature(cls.fit).parameters, cls.__name__
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.random((12, 20)))
    y = np.arange(12) % 2
    g = np.arange(12) // 2
    assert KNN(1).fit(X, y, groups=g).score(X).shape == (12,)
    assert NearestCentroid().fit(X, y, groups=g).score(X).shape == (12,)
    assert RegexScorer(patterns=[r"\bignore\b"]).fit(groups=g) is not None
