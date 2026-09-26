"""ТЗ 2.2 / 2.6 / 3.3: connectome loader, in-degree check, null models, FlyHash and dense sign matrices."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from flyguard.connectome import (curveball, curveball_n_trades, curveball_nulls, dense_gaussian_sign_matrix,
                                 flyhash_matrix, indegree_stats, load_malecns, random_same_density)


def degrees(M: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(M.sum(axis=1)).ravel(), np.asarray(M.sum(axis=0)).ravel()


def write_malecns_fixture(tmp_path, n_glom=12, n_kc=60, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((n_glom, n_kc), dtype=np.float64)
    for j in range(n_kc):
        inputs = rng.choice(n_glom, 2 + j % 5, replace=False)
        m[inputs, j] = rng.integers(1, 10, size=inputs.size)
    ragged = np.empty(n_glom, dtype=object)
    for g in range(n_glom):
        ragged[g] = list(range(g))
    path = tmp_path / "malecns_R.npz"
    np.savez_compressed(path, m=m, pn_counts=np.arange(1, n_glom + 1), pn_partners=ragged)
    path.with_suffix(".json").write_text(json.dumps({"side": "R", "n_kc": n_kc}), encoding="utf-8")
    return path, m


def test_load_malecns_binary_transposed_weighted_meta(tmp_path):
    path, m = write_malecns_fixture(tmp_path)
    M, meta = load_malecns(path)
    assert isinstance(M, sp.csr_matrix) and M.shape == (60, 12) and M.dtype == np.float32
    assert np.array_equal(M.toarray(), (m > 0).T.astype(np.float32))
    assert np.array_equal(meta["weighted"].toarray(), m.T.astype(np.float32))
    assert meta["n_kc"] == 60 and meta["n_glomeruli"] == 12 and meta["nnz"] == int((m > 0).sum())
    assert meta["unavailable_keys"] == ["pn_partners"]           # object array refused without pickle
    assert np.array_equal(meta["optional"]["pn_counts"], np.arange(1, 13))
    assert meta["json"]["side"] == "R" and meta["n_kc_zero_indegree"] == 0
    assert meta["indegree"]["indegree_mean"] == pytest.approx(4.0) and meta["indegree"]["in_range"]


def test_load_malecns_strict_indegree_check(tmp_path):
    rng = np.random.default_rng(1)
    m = (rng.random((6, 30)) < 0.2).astype(np.float64)    # mean in-degree ~1.2, outside 4-8
    path = tmp_path / "malecns_R.npz"
    np.savez(path, m=m)
    M, meta = load_malecns(path)
    assert not meta["indegree"]["in_range"] and meta["json"] is None
    with pytest.raises(ValueError):
        load_malecns(path, strict=True)


def test_indegree_stats_values(small_M):
    s = indegree_stats(small_M, check=(4, 8))
    row, col = degrees(small_M)
    assert s["n_cells"] == 60 and s["n_inputs"] == 12 and s["nnz"] == small_M.nnz
    assert s["indegree_mean"] == pytest.approx(row.mean()) and s["indegree_min"] == row.min()
    assert s["indegree_max"] == row.max() and s["outdegree_mean"] == pytest.approx(col.mean())
    assert s["in_range"] is True and s["check_range"] == [4.0, 8.0]
    assert indegree_stats(small_M, check=(5, 8))["in_range"] is False
    assert "in_range" in indegree_stats(small_M)                 # default range from the config


def test_random_same_density_preserves_indegrees(small_M):
    R = random_same_density(small_M, seed=5)
    assert R.shape == small_M.shape and R.dtype == np.float32 and R.has_sorted_indices
    assert np.array_equal(np.diff(R.indptr), np.diff(small_M.indptr))
    assert R.max() == 1.0 and R.nnz == small_M.nnz               # distinct inputs per cell
    assert (R != small_M).nnz > 0
    assert (random_same_density(small_M, seed=5) != R).nnz == 0
    assert (random_same_density(small_M, seed=6) != R).nnz > 0


def test_curveball_preserves_both_degree_sequences_and_changes(small_M):
    n_trades = curveball_n_trades(small_M, swaps_per_edge=5)
    assert n_trades == 5 * small_M.nnz
    C = curveball(small_M, n_trades, seed=11)
    row0, col0 = degrees(small_M)
    row1, col1 = degrees(C)
    assert np.array_equal(row0, row1) and np.array_equal(col0, col1)
    assert C.max() == 1.0 and C.has_sorted_indices and C.dtype == np.float32
    assert (C != small_M).nnz > 0
    assert (curveball(small_M, n_trades, seed=11) != C).nnz == 0
    assert (curveball(small_M, n_trades, seed=12) != C).nnz > 0
    assert curveball_n_trades(small_M) >= small_M.nnz             # config swaps_per_edge >= 1


def test_curveball_nulls_are_distinct_and_degree_preserving(small_M):
    nulls = curveball_nulls(small_M, n_null=5, seed=3)
    assert len(nulls) == 5
    row0, col0 = degrees(small_M)
    for N in nulls:
        r, c = degrees(N)
        assert np.array_equal(r, row0) and np.array_equal(c, col0)
    for i in range(5):
        for j in range(i + 1, 5):
            assert (nulls[i] != nulls[j]).nnz > 0
    again = curveball_nulls(small_M, n_null=5, seed=3)
    assert all((a != b).nnz == 0 for a, b in zip(nulls, again))


def test_flyhash_matrix_fan_in_per_row():
    F = flyhash_matrix(64, expansion=3, fan_in=6, seed=0)
    assert F.shape == (192, 64) and F.dtype == np.float32 and F.has_sorted_indices
    assert (np.diff(F.indptr) == 6).all() and F.max() == 1.0 and F.nnz == 192 * 6
    for i in range(F.shape[0]):
        assert np.unique(F.indices[F.indptr[i]:F.indptr[i + 1]]).size == 6
    assert (flyhash_matrix(64, 3, 6, 0) != F).nnz == 0
    assert (flyhash_matrix(64, 3, 6, 1) != F).nnz > 0
    with pytest.raises(ValueError):
        flyhash_matrix(4, 2, 6, 0)
    D = flyhash_matrix(16384, expansion=1, seed=0)               # config fan_in, tiny expansion
    assert (np.diff(D.indptr) == 6).all()


def test_dense_gaussian_sign_matrix_dimension(small_M):
    d, nnz = 51, 11_316
    G = dense_gaussian_sign_matrix(d, nnz, seed=0)
    assert G.shape == (round(nnz / d), d) and G.dtype == np.float32
    assert np.array_equal(G, dense_gaussian_sign_matrix(d, nnz, seed=0))
    assert not np.array_equal(G, dense_gaussian_sign_matrix(d, nnz, seed=1))
    assert abs(G.mean()) < 0.05 and abs(G.std() - 1.0) < 0.05
    assert dense_gaussian_sign_matrix(12, small_M.nnz, 0).shape == (round(small_M.nnz / 12), 12)
