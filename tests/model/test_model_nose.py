"""ТЗ 2.1 / 2.6: hashing nose, N16k, N51-svd, N51-hash — shapes, normalisation, determinism, seed sensitivity."""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from flyguard.nose import (N16k, N51Hash, N51Svd, cached_char_ngram_hash_counts, center,
                           char_ngram_hash_counts)

BINS = 1024


def n_grams(text: str, sizes=(3, 4, 5)) -> int:
    return sum(max(len(text) - n + 1, 0) for n in sizes)


def test_counts_shape_dtype_and_total(texts):
    X = char_ngram_hash_counts(texts, bins=BINS, seed=3)
    assert isinstance(X, sp.csr_matrix) and X.shape == (len(texts), BINS) and X.dtype == np.float32
    assert X.has_sorted_indices
    assert X.sum() == sum(n_grams(t) for t in texts)
    assert (np.asarray(X.sum(axis=1)).ravel() == np.array([n_grams(t) for t in texts])).all()


def test_counts_small_cases_lowercase_empty():
    X = char_ngram_hash_counts(["aaaa", "AaAa", "", "ab"], sizes=(3,), bins=BINS, seed=5)
    assert X[0].nnz == 1 and X[0].data[0] == 2.0            # "aaa" twice, one bin
    assert (X[0] != X[1]).nnz == 0                            # lower-cased before hashing
    assert X[2].nnz == 0 and X[3].nnz == 0                    # empty / shorter than n -> empty rows


def test_counts_deterministic_and_seed_sensitive(texts):
    a = char_ngram_hash_counts(texts, bins=BINS, seed=11)
    b = char_ngram_hash_counts(texts, bins=BINS, seed=11)
    c = char_ngram_hash_counts(texts, bins=BINS, seed=12)
    assert (a != b).nnz == 0
    assert (a != c).nnz > 0 and a.sum() == c.sum()


def test_counts_throughput_is_far_above_target():
    rng = np.random.default_rng(0)
    alphabet = list("abcdefghijklmnopqrstuvwxyz   .,")
    windows = ["".join(rng.choice(alphabet, 256)) for _ in range(500)]
    t0 = time.perf_counter()
    char_ngram_hash_counts(windows, seed=1)
    per_min = 60 * len(windows) / (time.perf_counter() - t0)
    assert per_min >= 20_000, f"{per_min:.0f} windows/min"


def test_cached_counts_roundtrip(tmp_path, texts):
    path = tmp_path / "feat" / "counts.npz"
    a = cached_char_ngram_hash_counts(texts, path, bins=BINS, seed=2)
    assert path.exists()
    b = cached_char_ngram_hash_counts(texts, path, bins=BINS, seed=2)
    assert (a != b).nnz == 0 and b.dtype == np.float32


def test_n16k_log1p_unit_sum_and_mean(texts):
    X = char_ngram_hash_counts(texts, bins=BINS, seed=3)
    nose = N16k().fit(X)
    T = nose.transform(X)
    assert isinstance(T, sp.csr_matrix) and T.dtype == np.float32 and T.shape == X.shape
    sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0, atol=1e-5)
    r, j = 0, X[0].indices[0]
    expected = np.log1p(X[0, j]) / np.log1p(X[0].data).sum()
    assert abs(T[r, j] - expected) < 1e-6
    assert nose.mean_.shape == (BINS,) and nose.mean_.dtype == np.float32
    assert np.allclose(nose.mean_, np.asarray(T.mean(axis=0)).ravel(), atol=1e-6)
    D = nose.transform(X, centered=True)
    assert isinstance(D, np.ndarray) and np.allclose(D.mean(axis=0), 0.0, atol=1e-5)
    E = N16k().transform(char_ngram_hash_counts(["", "abc"], sizes=(3,), bins=BINS))
    assert E[0].nnz == 0 and abs(E[1].sum() - 1.0) < 1e-6


@pytest.fixture(scope="module")
def n16k_features(texts):
    X = char_ngram_hash_counts(texts, bins=BINS, seed=3)
    return N16k().fit(X).transform(X)


def test_n51_svd_standardised_permuted_deterministic(n16k_features):
    T = n16k_features
    nose = N51Svd(rank=20, seed_svd=1, seed_perm=2).fit(T)
    Z = nose.transform(T, centered=False)
    assert Z.shape == (T.shape[0], 20) and Z.dtype == np.float32
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-3) and np.allclose(Z.std(axis=0), 1.0, atol=1e-3)
    assert sorted(nose.pi_.tolist()) == list(range(20))
    S = (nose.svd_.transform(T) - nose.comp_mean_) / nose.comp_std_
    assert np.allclose(Z, S[:, nose.pi_], atol=1e-4)
    assert np.allclose(nose.transform(T).mean(axis=0), 0.0, atol=1e-4)
    again = N51Svd(rank=20, seed_svd=1, seed_perm=2).fit(T).transform(T)
    assert np.array_equal(again, nose.transform(T))
    other = N51Svd(rank=20, seed_svd=1, seed_perm=3).fit(T)
    assert not np.array_equal(other.pi_, nose.pi_)
    Zo = other.transform(T, centered=False)
    assert not np.allclose(Zo, Z) and np.allclose(np.sort(Zo, axis=1), np.sort(Z, axis=1), atol=1e-4)
    assert nose.explained_variance_ratio_.shape == (20,)


def test_n51_svd_needs_enough_rows(n16k_features):
    with pytest.raises(ValueError):
        N51Svd(rank=20).fit(n16k_features[:10])


def test_n51_hash_fractions_and_centering(texts):
    X = char_ngram_hash_counts(texts, bins=51, seed=3)
    nose = N51Hash().fit(X)
    F = nose.transform(X, centered=False)
    assert F.shape == (len(texts), 51) and F.dtype == np.float32
    assert np.allclose(F.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(nose.transform(X).mean(axis=0), 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        nose.transform(char_ngram_hash_counts(texts[:2], bins=50))
    with pytest.raises(RuntimeError):
        N51Hash().transform(X)


def test_center_sparse_and_dense():
    X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    mean = np.array([0.5, 0.5], dtype=np.float32)
    C = center(X, mean)
    assert isinstance(C, np.ndarray) and C.dtype == np.float32
    assert np.array_equal(C, np.array([[0.5, -0.5], [-0.5, 1.5]], dtype=np.float32))
    assert np.array_equal(center(X.toarray(), mean), C)
