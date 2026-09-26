"""ТЗ 2.4 / 2.6: Bloom readout (active cells only, gamma = 0 saturation, balancing), linear readout, selection."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from flyguard.readout import BloomReadout, LinearReadout, balance_indices, select_C, select_gamma
from tests.model.conftest import random_codes

M_CELLS, K = 400, 20  # k = 5 % of m


def test_bloom_updates_only_active_cells():
    rng = np.random.default_rng(0)
    z = random_codes(1, M_CELLS, K, rng)
    b = BloomReadout(M_CELLS, K, gamma=0.5, seed_subsample=0)
    b.partial_fit(z, [1])
    active = z.indices
    assert np.allclose(b.F_[1, active], 0.5) and b.F_[1].sum() == pytest.approx(M_CELLS - K + K * 0.5)
    assert (b.F_[0] == 1.0).all()
    inactive = np.setdiff1d(np.arange(M_CELLS), active)
    assert (b.F_[1, inactive] == 1.0).all()
    b.partial_fit(z, [1])
    assert np.allclose(b.F_[1, active], 0.25) and (b.F_[1, inactive] == 1.0).all()


def test_bloom_gamma0_saturates_after_200_examples_per_class():
    rng = np.random.default_rng(1)
    Z = sp.vstack([random_codes(200, M_CELLS, K, rng), random_codes(200, M_CELLS, K, rng)]).tocsr()
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    b = BloomReadout(M_CELLS, K, gamma=0.0, seed_subsample=0).fit(Z, y)
    assert b.F_.mean() <= 0.02 and b.F_[0].mean() <= 0.02 and b.F_[1].mean() <= 0.02
    assert set(np.unique(b.F_)) <= {0.0, 1.0}


def test_bloom_closed_form_equals_sequential_rule():
    rng = np.random.default_rng(2)
    Z = random_codes(40, M_CELLS, K, rng)
    y = rng.integers(0, 2, 40)
    y[:20], y[20:] = 0, 1
    fit = BloomReadout(M_CELLS, K, gamma=0.9).fit(Z, y)
    seq = BloomReadout(M_CELLS, K, gamma=0.9)
    for i in rng.permutation(40):
        seq.partial_fit(Z[i], [y[i]])
    assert np.allclose(fit.F_, seq.F_, atol=1e-6)
    assert np.allclose(fit.F_, 0.9 ** np.vstack([np.asarray(Z[y == c].sum(axis=0)).ravel() for c in (0, 1)]), atol=1e-6)


def test_balancing_equalises_class_counts():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(120, int), np.ones(30, int)]
    idx = balance_indices(y, seed=7)
    assert idx.size == 60 and np.bincount(y[idx]).tolist() == [30, 30]
    assert np.array_equal(idx, np.sort(idx)) and np.unique(idx).size == 60
    assert np.array_equal(balance_indices(y, seed=7), idx) and not np.array_equal(balance_indices(y, seed=8), idx)
    Z = random_codes(150, M_CELLS, K, rng)
    b = BloomReadout(M_CELLS, K, gamma=0.5, seed_subsample=7).fit(Z, y)
    assert b.balanced_counts_.tolist() == [30, 30] and b.class_counts_.tolist() == [120, 30]
    assert np.array_equal(b.balanced_index_, idx) and b.n_seen_.tolist() == [30, 30]
    n0 = np.asarray(Z[idx][y[idx] == 0].sum(axis=0)).ravel()
    assert np.allclose(b.F_[0], 0.5 ** n0, atol=1e-6)         # trained on the subsample only


def test_bloom_normalized_variant_uses_all_data_with_scaled_exponent():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(90, int), np.ones(30, int)]
    Z = random_codes(120, M_CELLS, K, rng)
    b = BloomReadout(M_CELLS, K, gamma=0.5, seed_subsample=0, normalized=True).fit(Z, y)
    n0 = np.asarray(Z[y == 0].sum(axis=0)).ravel()
    n1 = np.asarray(Z[y == 1].sum(axis=0)).ravel()
    assert np.allclose(b.F_[0], 0.5 ** (n0 * 30 / 90), atol=1e-6)
    assert np.allclose(b.F_[1], 0.5 ** n1, atol=1e-6)
    assert b.balanced_index_.size == 120 and b.n_seen_.tolist() == [90, 30]


def test_bloom_scores_in_unit_interval_and_separates_prototypes():
    rng = np.random.default_rng(5)
    proto = [np.sort(rng.choice(M_CELLS, K, replace=False)) for _ in (0, 1)]

    def noisy(c: int, n: int) -> sp.csr_matrix:
        rows = []
        for _ in range(n):
            keep = proto[c][rng.random(K) < 0.7]
            fill = rng.choice(np.setdiff1d(np.arange(M_CELLS), keep), K - keep.size, replace=False)
            rows.append(np.sort(np.r_[keep, fill]))
        cols = np.concatenate(rows)
        return sp.csr_matrix((np.ones(n * K, np.float32), cols, np.arange(0, n * K + 1, K)), shape=(n, M_CELLS))

    Ztr = sp.vstack([noisy(0, 60), noisy(1, 60)]).tocsr()
    ytr = np.r_[np.zeros(60, int), np.ones(60, int)]
    Zte = sp.vstack([noisy(0, 40), noisy(1, 40)]).tocsr()
    yte = np.r_[np.zeros(40, int), np.ones(40, int)]
    for gamma in (0.0, 0.5, 0.9, 0.99):
        b = BloomReadout(M_CELLS, K, gamma, seed_subsample=1).fit(Ztr, ytr)
        s = b.score(Zte)
        assert s.shape == (80,) and s.dtype == np.float32 and s.min() >= 0.0 and s.max() <= 1.0
        assert roc_auc_score(yte, s) > 0.9, gamma
    phi = b.familiarity(Zte)
    assert phi.shape == (80, 2) and np.allclose(b.score(Zte), (phi[:, 1] - phi[:, 0] + 1) / 2, atol=1e-6)
    empty = sp.csr_matrix((1, M_CELLS), dtype=np.float32)
    assert b.score(empty)[0] == pytest.approx(0.5) and b.state_size_bytes() == 2 * M_CELLS * 4
    same = BloomReadout(M_CELLS, K, 0.9, seed_subsample=1).fit(Ztr, ytr)
    assert np.array_equal(same.F_, BloomReadout(M_CELLS, K, 0.9, seed_subsample=1).fit(Ztr, ytr).F_)
    with pytest.raises(ValueError):
        BloomReadout(M_CELLS, K, 1.5)
    with pytest.raises(ValueError):
        b.fit(Ztr, np.full(120, 2))


def make_linear_problem(seed: int = 6):
    rng = np.random.default_rng(seed)
    m, k = 300, 15
    good = np.arange(0, 40)
    bad = np.arange(40, 80)

    def sample(c: int, n: int) -> sp.csr_matrix:
        rows = []
        for _ in range(n):
            pref = rng.choice(good if c else bad, 8, replace=False)
            rest = rng.choice(np.arange(80, m), k - 8, replace=False)
            rows.append(np.sort(np.r_[pref, rest]))
        cols = np.concatenate(rows)
        return sp.csr_matrix((np.ones(n * k, np.float32), cols, np.arange(0, n * k + 1, k)), shape=(n, m))

    Ztr = sp.vstack([sample(0, 70), sample(1, 30)]).tocsr()
    ytr = np.r_[np.zeros(70, int), np.ones(30, int)]
    Zva = sp.vstack([sample(0, 30), sample(1, 30)]).tocsr()
    yva = np.r_[np.zeros(30, int), np.ones(30, int)]
    return m, k, Ztr, ytr, Zva, yva


def test_linear_readout_fits_scores_and_is_deterministic():
    m, k, Ztr, ytr, Zva, yva = make_linear_problem()
    lr = LinearReadout(C=1.0, seed=0).fit(Ztr, ytr)
    s = lr.score(Zva)
    assert s.shape == (60,) and s.dtype == np.float32 and s.min() >= 0 and s.max() <= 1
    assert roc_auc_score(yva, s) > 0.95
    assert np.array_equal(s, LinearReadout(C=1.0, seed=0).fit(Ztr, ytr).score(Zva))
    assert lr.model_.class_weight == "balanced" and lr.state_size_bytes() > 0
    with pytest.raises(ValueError):
        LinearReadout().fit(Ztr, np.zeros(100, int))
    with pytest.raises(RuntimeError):
        LinearReadout().score(Zva)


def test_select_gamma_and_select_c_use_validation_auc():
    m, k, Ztr, ytr, Zva, yva = make_linear_problem()
    grid = [0.0, 0.5, 0.9]
    gamma, table = select_gamma(Ztr, ytr, Zva, yva, m, k, seed_subsample=0, gammas=grid)
    assert gamma in grid and list(table) == grid and all(0 <= v <= 1 for v in table.values())
    assert table[gamma] == max(table.values())
    C_grid = [0.01, 1.0, 100.0]
    C, ctable = select_C(Ztr, ytr, Zva, yva, C_grid=C_grid, seed=0)
    assert C in C_grid and list(ctable) == C_grid and ctable[C] == max(ctable.values())
    g_default, t_default = select_gamma(Ztr, ytr, Zva, yva, m, k)
    assert list(t_default) == [0.0, 0.5, 0.9, 0.99]              # grid from configs/default.yaml
    c_default, ct_default = select_C(Ztr, ytr, Zva, yva)
    assert list(ct_default) == [0.01, 0.1, 1.0, 10.0, 100.0]
