"""Lexical baselines (ТЗ 3.1) on a synthetic separable problem: AUC > 0.9, scores in [0, 1], protocol shape."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from flyguard.baselines.common import WindowScorer, select_c, make_logreg, text_hash
from flyguard.baselines.lexical import KNN, LRSvd, NearestCentroid, TfidfLR, make_lexical_scorers
from flyguard.config import Configs

GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def counts_problem(n: int, seed: int, bins: int = 400) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sparse 'hashed n-gram' counts: class 1 favours bins 0-39, class 0 bins 40-79, shared noise elsewhere."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    rows, cols, vals = [], [], []
    for i in range(n):
        signal = rng.integers(0, 40, size=12) + (0 if y[i] == 1 else 40)
        noise = rng.integers(80, bins, size=25)
        for b in np.concatenate([signal, noise]):
            rows.append(i)
            cols.append(int(b))
            vals.append(1.0)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, bins))
    X.sum_duplicates()
    return X, y


def n16k_like(X_counts: sp.csr_matrix) -> sp.csr_matrix:
    """The nose's N16k transform (design §5): log1p, rows summing to one."""
    X = X_counts.copy().astype(np.float64)
    X.data = np.log1p(X.data)
    s = np.asarray(X.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / s) @ X)


@pytest.fixture(scope="module")
def data():
    Xc_tr, y_tr = counts_problem(240, 1)
    Xc_va, y_va = counts_problem(80, 2)
    Xc_te, y_te = counts_problem(160, 3)
    Xc_unl, _ = counts_problem(300, 4)
    return {"counts": (Xc_tr, Xc_va, Xc_te, Xc_unl), "n16k": tuple(n16k_like(X) for X in (Xc_tr, Xc_va, Xc_te, Xc_unl)),
            "y": (y_tr, y_va, y_te)}


def _check(scorer, X_tr, y_tr, X_va, y_va, X_te, y_te, min_auc=0.9):
    assert isinstance(scorer, WindowScorer)
    assert scorer.fit(X_tr, y_tr, X_va, y_va) is scorer
    s = scorer.score(X_te)
    assert s.shape == (X_te.shape[0],) and s.dtype == np.float64
    assert s.min() >= 0.0 and s.max() <= 1.0
    auc = roc_auc_score(y_te, s)
    assert auc > min_auc, f"{scorer.name}: AUC {auc:.3f}"
    return s


def test_tfidf_lr_with_c_unl_idf(data):
    X_tr, X_va, X_te, X_unl = data["n16k"]
    y_tr, y_va, y_te = data["y"]
    m = TfidfLR(GRID, seed=0, X_unl=X_unl)
    assert m.name == "tfidf_lr" and m.idf_source_ == "c_unl" and m.idf_.shape == (X_tr.shape[1],)
    _check(m, X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert m.c_source_ == "val" and m.C_ in GRID and set(m.c_table_) == {str(c) for c in GRID}
    # unit rows after transform
    norms = np.sqrt(np.asarray(m.transform(X_te).multiply(m.transform(X_te)).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)


def test_tfidf_lr_counts_input_and_train_idf_fallback(data):
    Xc_tr, Xc_va, Xc_te, _ = data["counts"]
    y_tr, y_va, y_te = data["y"]
    m = TfidfLR(GRID, seed=0, tf="counts")
    with pytest.warns(RuntimeWarning, match="C_unl"):
        _check(m, Xc_tr, y_tr, Xc_va, y_va, Xc_te, y_te)
    assert m.idf_source_ == "train"
    with pytest.raises(ValueError):
        TfidfLR(GRID, tf="bogus")


def test_tfidf_lr_dense_input_and_determinism(data):
    X_tr, X_va, X_te, X_unl = data["n16k"]
    y_tr, y_va, y_te = data["y"]
    a = TfidfLR(GRID, seed=0, X_unl=X_unl).fit(X_tr.toarray(), y_tr, X_va.toarray(), y_va).score(X_te.toarray())
    b = TfidfLR(GRID, seed=0, X_unl=X_unl).fit(X_tr, y_tr, X_va, y_va).score(X_te)
    assert np.allclose(a, b, atol=1e-6)


def test_knn5_mean_label_and_knn1_similarity_weighted(data):
    X_tr, X_va, X_te, _ = data["n16k"]
    y_tr, y_va, y_te = data["y"]
    s5 = _check(KNN(5), X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert KNN(5).name == "knn5"
    assert set(np.round(np.unique(s5), 6)) <= {round(i / 5, 6) for i in range(6)}
    k1 = KNN(1)
    s1 = _check(k1, X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert k1.name == "knn1"
    assert len(np.unique(s1)) > 2, "k=1 score must not be a 0/1 label (degenerate ROC)"
    # the hard 1-NN decision is recovered at 0.5
    hard = (s1 >= 0.5).astype(int)
    assert (hard == y_te).mean() > 0.9
    with pytest.raises(ValueError):
        KNN(3, metric="euclidean")


def test_knn_handles_dense_query_against_sparse_train_and_small_k(data):
    X_tr, _, X_te, _ = data["n16k"]
    y_tr, _, y_te = data["y"]
    m = KNN(5, block_rows=7).fit(X_tr, y_tr)
    s_dense = m.score(X_te.toarray())
    s_sparse = m.score(X_te)
    assert np.allclose(s_dense, s_sparse)
    tiny = KNN(5).fit(X_tr[:3], y_tr[:3])          # k > n_train -> uses all three rows
    assert tiny.score(X_te[:4]).shape == (4,)


def test_nearest_centroid(data):
    X_tr, X_va, X_te, _ = data["n16k"]
    y_tr, y_va, y_te = data["y"]
    m = NearestCentroid()
    assert m.name == "centroid"
    s = _check(m, X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert len(np.unique(s)) > 10
    with pytest.raises(ValueError):
        NearestCentroid().fit(X_tr[:5], np.zeros(5))


def svd_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 51))
    X[:, 3] += 2.0 * (y - 0.5)
    X[:, 17] -= 1.5 * (y - 0.5)
    return X, y


def test_lr_svd_val_cv_and_default_c_sources():
    X_tr, y_tr = svd_problem(200, 10)
    X_va, y_va = svd_problem(80, 11)
    X_te, y_te = svd_problem(150, 12)
    m = LRSvd(GRID, seed=0)
    assert m.name == "lr_svd"
    _check(m, X_tr, y_tr, X_va, y_va, X_te, y_te)
    assert m.c_source_ == "val"
    m_cv = LRSvd(GRID, seed=0).fit(X_tr, y_tr)
    assert m_cv.c_source_ == "cv" and roc_auc_score(y_te, m_cv.score(X_te)) > 0.9
    m_sparse = LRSvd(GRID, seed=0).fit(sp.csr_matrix(X_tr), y_tr, sp.csr_matrix(X_va), y_va)
    assert np.allclose(m_sparse.score(sp.csr_matrix(X_te)), m.score(X_te), atol=1e-6)


def test_two_row_training_does_not_crash_e2_shots_one(data):
    X_tr, _, X_te, X_unl = data["n16k"]
    y_tr, _, _ = data["y"]
    i1, i0 = int(np.where(y_tr == 1)[0][0]), int(np.where(y_tr == 0)[0][0])
    X2, y2 = X_tr[[i1, i0]], y_tr[[i1, i0]]
    t = TfidfLR(GRID, seed=0, X_unl=X_unl).fit(X2, y2)
    assert t.c_source_ == "default" and t.C_ == 1.0
    assert t.score(X_te).shape == (X_te.shape[0],)
    assert KNN(1).fit(X2, y2).score(X_te).shape == (X_te.shape[0],)
    assert NearestCentroid().fit(X2, y2).score(X_te).shape == (X_te.shape[0],)
    Xs, ys = svd_problem(2, 0)
    ys[:] = [0, 1]
    assert LRSvd(GRID).fit(Xs, ys).c_source_ == "default"
    # single-class validation set is ignored, not fatal
    t2 = TfidfLR(GRID, seed=0, X_unl=X_unl).fit(X_tr, y_tr, X_te[:5], np.ones(5))
    assert t2.c_source_ == "cv"


def test_select_c_prefers_smallest_c_among_ties():
    X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1, 0, 1])
    C, source, table = select_c(lambda C: make_logreg(C, 0), X, y, X, y, [0.1, 1.0, 10.0], seed=0)
    assert source == "val" and C == 0.1 and all(v == 1.0 for v in table.values())


def test_make_lexical_scorers_reads_config(data):
    cfg = Configs(operator={}, default={"baselines": {"tfidf": {"C_grid": GRID}, "knn": {"ks": [1, 5], "metric": "cosine"},
                                                      "centroid": True, "lr_svd": {"C_grid": GRID}}}, experiments={})
    scorers = make_lexical_scorers(cfg, seed=3, X_unl=data["n16k"][3])
    assert set(scorers) == {"tfidf_lr", "knn1", "knn5", "centroid", "lr_svd"}
    assert scorers["tfidf_lr"].idf_source_ == "c_unl" and scorers["lr_svd"].seed == 3


def test_text_hash_is_xxhash64_hex_seed_zero():
    import xxhash
    assert text_hash("abc") == xxhash.xxh64(b"abc").hexdigest() == "44bc2cf5ad770999"
    assert text_hash("abc") != text_hash("abd")
