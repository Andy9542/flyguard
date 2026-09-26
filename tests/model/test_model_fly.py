"""ТЗ 2.3 / 2.6 / 3.3: k-WTA exactness and tie rule, expansion with centring, batching, seed sensitivity."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flyguard.connectome import dense_gaussian_sign_matrix
from flyguard.fly import expand, fly_code, k_for, kwta, sign_code
from flyguard.nose import N16k, N51Svd, char_ngram_hash_counts


def brute_force_kwta(A: np.ndarray, k: int) -> np.ndarray:
    Z = np.zeros(A.shape, dtype=np.float32)
    for r in range(A.shape[0]):
        winners = sorted(range(A.shape[1]), key=lambda j: (-A[r, j], j))[:k]
        Z[r, winners] = 1.0
    return Z


def test_kwta_exactly_k_with_lowest_index_ties():
    rng = np.random.default_rng(0)
    for trial in range(40):
        n, m = 6, int(rng.integers(5, 30))
        k = int(rng.integers(1, m))
        A = rng.integers(0, 3, size=(n, m)).astype(np.float32)   # quantised -> many ties at the threshold
        A[0] = 0.0                                               # all-zero row -> first k cells
        A[1] = 7.0                                               # constant row
        Z = kwta(A, k)
        assert isinstance(Z, sp.csr_matrix) and Z.dtype == np.float32 and Z.has_sorted_indices
        assert (np.diff(Z.indptr) == k).all() and (Z.data == 1.0).all()
        assert np.array_equal(Z.toarray(), brute_force_kwta(A, k)), (trial, k)
        assert np.array_equal(Z[0].indices, np.arange(k))


def test_kwta_edge_cases_and_float_input():
    A = np.array([[0.1, 0.5, 0.5, 0.2], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    assert np.array_equal(kwta(A, 1).toarray(), [[0, 1, 0, 0], [1, 0, 0, 0]])
    assert np.array_equal(kwta(A, 2).toarray(), [[0, 1, 1, 0], [1, 1, 0, 0]])
    assert np.array_equal(kwta(A, 3).toarray(), [[0, 1, 1, 1], [1, 1, 1, 0]])
    assert kwta(A, 4).nnz == 8 and kwta(A, 9).nnz == 8 and kwta(A, 0).nnz == 0
    rng = np.random.default_rng(1)
    B = rng.standard_normal((50, 300)).astype(np.float64)
    assert np.array_equal(kwta(B, 15).toarray(), brute_force_kwta(B, 15))
    with pytest.raises(RuntimeError):
        kwta(np.array([[np.nan, 1.0, 2.0]]), 1)


def test_expand_matches_dense_reference_with_centring(small_M):
    rng = np.random.default_rng(2)
    U = rng.standard_normal((5, 12)).astype(np.float32)
    mean = rng.standard_normal(12).astype(np.float32)
    ref = (U - mean) @ small_M.toarray().T
    A = expand(U, small_M, mean)
    assert A.shape == (5, 60) and A.dtype == np.float32 and np.allclose(A, ref, atol=1e-5)
    assert np.allclose(expand(sp.csr_matrix(U), small_M, mean), ref, atol=1e-5)
    assert np.allclose(expand(U, small_M), U @ small_M.toarray().T, atol=1e-5)


def test_fly_code_batch_invariance_and_determinism(small_M):
    rng = np.random.default_rng(3)
    U = sp.random(23, 12, density=0.5, format="csr", dtype=np.float32, random_state=4)
    mean = rng.standard_normal(12).astype(np.float32) * 0.1
    k = k_for(60)
    Z = fly_code(U, small_M, k, mean=mean)
    assert Z.shape == (23, 60) and (np.diff(Z.indptr) == k).all() and Z.dtype == np.float32
    for bs in (1, 3, 100):
        assert (fly_code(U, small_M, k, mean=mean, batch_size=bs) != Z).nnz == 0
    assert (fly_code(U, small_M, k, mean=mean) != Z).nnz == 0
    assert np.array_equal(Z.toarray(), kwta(expand(U, small_M, mean), k).toarray())
    assert fly_code(U[:0], small_M, k).shape == (0, 60)


def test_different_nose_and_perm_seeds_give_different_codes(texts, small_M):
    k = k_for(60)

    def codes(seed_nose: int, seed_perm: int) -> sp.csr_matrix:
        X = char_ngram_hash_counts(texts, bins=1024, seed=seed_nose)
        T = N16k().fit(X).transform(X)
        U = N51Svd(rank=12, seed_svd=0, seed_perm=seed_perm).fit(T).transform(T)
        return fly_code(U, small_M, k)

    base = codes(1, 1)
    assert (codes(1, 1) != base).nnz == 0                      # determinism under fixed seeds
    assert (codes(2, 1) != base).nnz > 0                       # nose seed changes the code
    assert (codes(1, 2) != base).nnz > 0                       # perm seed changes the code
    assert (np.diff(base.indptr) == k).all()


def test_sign_code_dimension_and_values(small_M):
    rng = np.random.default_rng(5)
    U = rng.standard_normal((9, 12)).astype(np.float32)
    mean = rng.standard_normal(12).astype(np.float32)
    G = dense_gaussian_sign_matrix(12, small_M.nnz, seed=0)
    S = sign_code(U, G, mean=mean)
    assert S.shape == (9, round(small_M.nnz / 12)) and S.dtype == np.float32
    ref = ((U - mean) @ G.T > 0).astype(np.float32)
    assert np.array_equal(S.toarray(), ref)
    assert np.array_equal(sign_code(sp.csr_matrix(U), G, mean=mean, batch_size=2).toarray(), ref)
    assert np.array_equal(sign_code(U, G).toarray(), (U @ G.T > 0).astype(np.float32))


def test_k_for_rounding_from_config():
    assert k_for(1886) == 94 and k_for(20 * 16384) == 16384 and k_for(40 * 16384) == 32768
    assert k_for(1886, 0.025) == 47 and k_for(1886, 0.10) == 189
    assert k_for(10, 0.0) == 1 and k_for(10, 5.0) == 10
