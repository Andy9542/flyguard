"""The nose: character n-gram hashing and the three input spaces of the fly (ТЗ 2.1, preregistration §3.1).

Every detector in the project shares one hash: H(g) = xxhash64(utf8(g), seed=nose) mod B over lower-cased
character n-grams with n in {3, 4, 5}. Three feature spaces are derived from the hashed counts:

* N16k  — B = 16 384, x_b = log(1 + c_b), rows normalised to unit sum; input of TF-IDF, kNN and FlyHash.
* N51-svd — TruncatedSVD of rank 51 on the N16k rows of C_unl, components standardised by C_unl statistics and
  mapped to glomeruli by a seeded permutation pi; input of the real fly.
* N51-hash — B = 51, x_b = c_b / sum c; the control nose of E5.

All three are centred on the C_unl mean (ТЗ 2.1 "Центрирование на среднем C_unl"). Fitting uses only the
unlabelled corpus C_unl (ТЗ 1.9), never labels or test clusters.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import xxhash
from sklearn.decomposition import TruncatedSVD

from flyguard.config import load_configs
from flyguard.io import atomic_write_bytes


@lru_cache(maxsize=1)
def _nose_cfg() -> dict[str, Any]:
    return load_configs().default["nose"]


def char_ngram_hash_counts(texts: Sequence[str], sizes: Sequence[int] | None = None, bins: int | None = None,
                           seed: int = 0) -> sp.csr_matrix:
    """Hashed character n-gram counts, one csr row per text (ТЗ 2.1 "Нос").

    H(g) = xxhash64(g, seed) mod bins on the lower-cased text, n in ``sizes`` (config ``nose.ngram_sizes``,
    default bins = ``nose.n16k.bins``). The gram is hashed as its UTF-8 bytes (python-xxhash accepts only
    bytes; the ТЗ fixes the hash function and the seed, not the byte encoding). One pass per text: raw 64-bit
    hashes are collected into a uint64 array, reduced mod ``bins`` vectorised, and ``np.unique`` gives the
    sorted (column, count) pairs of the row, so the csr matrix is built directly with sorted indices. Measured
    on this machine at ~7 000 256-char windows/s single-threaded (target of the task: >= 20 000/min).
    Empty or too-short texts give empty rows. Counts are float32 (exact up to 2**24).
    """
    cfg = _nose_cfg()
    sizes = tuple(int(n) for n in (cfg["ngram_sizes"] if sizes is None else sizes))
    bins = int(cfg["n16k"]["bins"] if bins is None else bins)
    if bins <= 0 or any(n <= 0 for n in sizes):
        raise ValueError("bins and n-gram sizes must be positive")
    seed = int(seed)
    h = xxhash.xxh64_intdigest
    modulus = np.uint64(bins)
    n_texts = len(texts)
    indptr = np.zeros(n_texts + 1, dtype=np.int64)
    idx_parts: list[np.ndarray] = []
    cnt_parts: list[np.ndarray] = []
    for r, text in enumerate(texts):
        t = str(text).lower()
        length = len(t)
        parts = []
        for n in sizes:
            cnt = length - n + 1
            if cnt <= 0:
                continue
            parts.append(np.fromiter((h(t[i:i + n].encode("utf-8"), seed) for i in range(cnt)),
                                     dtype=np.uint64, count=cnt))
        if parts:
            cols, counts = np.unique(np.concatenate(parts) % modulus, return_counts=True)
            idx_parts.append(cols.astype(np.int32))
            cnt_parts.append(counts.astype(np.float32))
            indptr[r + 1] = indptr[r] + cols.size
        else:
            indptr[r + 1] = indptr[r]
    indices = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int32)
    data = np.concatenate(cnt_parts) if cnt_parts else np.zeros(0, dtype=np.float32)
    X = sp.csr_matrix((data, indices, indptr), shape=(n_texts, bins), dtype=np.float32)
    X.has_sorted_indices = True
    return X


def cached_char_ngram_hash_counts(texts: Sequence[str], path: str | Path, sizes: Sequence[int] | None = None,
                                  bins: int | None = None, seed: int = 0) -> sp.csr_matrix:
    """Counts of :func:`char_ngram_hash_counts` cached as an ``.npz`` file (design §5: counts are computed once
    per (seed, window set) and cached under ``data/processed/features/<seed>/``).

    The caller chooses ``path`` so that it identifies the window set (e.g. a hash of the window ids); a cached
    matrix is reused only when its row count matches ``len(texts)``. Written atomically through
    :mod:`flyguard.io`.
    """
    path = Path(path)
    want_bins = int(_nose_cfg()["n16k"]["bins"] if bins is None else bins)
    if path.exists():
        X = sp.load_npz(path).tocsr().astype(np.float32)
        if X.shape == (len(texts), want_bins):  # a 51-bin and a 16k-bin cache of one window set must not alias
            X.sort_indices()
            return X
    X = char_ngram_hash_counts(texts, sizes=sizes, bins=bins, seed=seed)
    buf = io.BytesIO()
    sp.save_npz(buf, X, compressed=True)
    atomic_write_bytes(path, buf.getvalue())
    return X


def center(X: np.ndarray | sp.spmatrix, mean: np.ndarray) -> np.ndarray:
    """Return the dense, float32 ``X - mean`` (ТЗ 2.1 "Центрирование на среднем C_unl").

    Sparse input is densified, so for 16 384-bin features call it on batches or let :func:`flyguard.fly.expand`
    subtract ``M @ mean`` instead (mathematically the same activations, no dense 16k matrix).
    """
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    return (Xd.astype(np.float32, copy=False) - np.asarray(mean, dtype=np.float32)).astype(np.float32, copy=False)


def _log1p_unit_sum(X_counts: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    X = sp.csr_matrix(X_counts, dtype=np.float32, copy=True)
    X.sum_duplicates()
    X.data = np.log1p(X.data).astype(np.float32, copy=False)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=np.float64), where=row_sums > 0)
    X.data *= np.repeat(scale, np.diff(X.indptr)).astype(np.float32)
    X.sort_indices()
    return X


class N16k:
    """N16k nose (ТЗ 2.1): x_b = log(1 + c_b), rows normalised to unit sum, centring mean kept from C_unl.

    ``transform`` returns the sparse, *uncentred* features and ``mean_`` holds the C_unl mean, because centring
    a 16 384-bin matrix densifies it; :func:`flyguard.fly.expand` takes ``mean`` and subtracts ``M @ mean`` from
    the activations, which equals expanding the centred input. Use ``transform(..., centered=True)`` for a dense
    centred array when memory allows.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.bins_: int | None = None

    def fit(self, X_counts_unl: sp.spmatrix | np.ndarray) -> "N16k":
        """Store the mean of the transformed C_unl rows (ТЗ 1.9: statistics come from C_unl only)."""
        Xt = _log1p_unit_sum(X_counts_unl)
        self.mean_ = np.asarray(Xt.mean(axis=0)).ravel().astype(np.float32)
        self.bins_ = Xt.shape[1]
        return self

    def transform(self, X_counts: sp.spmatrix | np.ndarray, centered: bool = False) -> sp.csr_matrix | np.ndarray:
        """log1p and unit-sum rows; ``centered=True`` returns the dense ``X - mean_`` instead (needs ``fit``)."""
        Xt = _log1p_unit_sum(X_counts)
        if centered:
            if self.mean_ is None:
                raise RuntimeError("N16k.fit must run before a centred transform")
            return center(Xt, self.mean_)
        return Xt


class N51Svd:
    """N51-svd nose (ТЗ 2.1, preregistration §3.1): TruncatedSVD of rank 51 over N16k, fitted on C_unl.

    Components are standardised by their C_unl mean and standard deviation, then permuted by pi drawn from the
    ``perm`` seed: output column g (glomerulus g of the measured matrix) carries SVD component ``pi_[g]``.
    The permutation is part of the seed variance on purpose: component variance falls with the component index
    and glomerulus degrees in the measured M differ, so a fixed mapping would make H3 depend on one implicit
    choice. Centring on the C_unl mean is applied last (after standardisation it is numerically ~0; kept for
    uniformity with the other noses).
    """

    def __init__(self, rank: int | None = None, seed_svd: int = 0, seed_perm: int = 0, n_iter: int = 5) -> None:
        cfg = _nose_cfg()["n51_svd"]
        self.rank = int(cfg["rank"] if rank is None else rank)
        self.seed_svd = int(seed_svd)
        self.seed_perm = int(seed_perm)
        self.n_iter = int(n_iter)
        self.standardize = bool(cfg.get("standardize", True))
        self.permute = bool(cfg.get("permute_to_glomeruli", True))
        self.svd_: TruncatedSVD | None = None
        self.comp_mean_: np.ndarray | None = None
        self.comp_std_: np.ndarray | None = None
        self.pi_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X16k_unl: sp.spmatrix | np.ndarray) -> "N51Svd":
        """Fit the SVD (``random_state = seed_svd``), the per-component statistics and pi on C_unl."""
        if X16k_unl.shape[0] <= self.rank:
            raise ValueError(f"N51Svd needs more than rank={self.rank} unlabelled rows, got {X16k_unl.shape[0]}")
        self.svd_ = TruncatedSVD(n_components=self.rank, algorithm="randomized", n_iter=self.n_iter,
                                 random_state=self.seed_svd)
        S = np.asarray(self.svd_.fit_transform(X16k_unl), dtype=np.float64)
        self.comp_mean_ = S.mean(axis=0)
        std = S.std(axis=0)
        std[std == 0] = 1.0
        self.comp_std_ = std
        self.pi_ = (np.random.default_rng(self.seed_perm).permutation(self.rank) if self.permute
                    else np.arange(self.rank))
        self.explained_variance_ratio_ = np.asarray(self.svd_.explained_variance_ratio_, dtype=np.float64)
        self.mean_ = self._project(X16k_unl).mean(axis=0).astype(np.float32)
        return self

    def _project(self, X16k: sp.spmatrix | np.ndarray) -> np.ndarray:
        if self.svd_ is None:
            raise RuntimeError("N51Svd.fit must run before transform")
        S = np.asarray(self.svd_.transform(X16k), dtype=np.float64)
        if self.standardize:
            S = (S - self.comp_mean_) / self.comp_std_
        return np.ascontiguousarray(S[:, self.pi_], dtype=np.float32)

    def transform(self, X16k: sp.spmatrix | np.ndarray, centered: bool = True) -> np.ndarray:
        """Project N16k rows to the 51 standardised, permuted components; centred on C_unl by default."""
        Z = self._project(X16k)
        return (Z - self.mean_).astype(np.float32, copy=False) if centered else Z


class N51Hash:
    """N51-hash nose (ТЗ 2.1): B = 51 bins, x_b = c_b / sum c, centred on the C_unl mean; the E5 control.

    ``fit`` is not in the design sketch but is needed to hold the C_unl mean for centring. Input counts come from
    ``char_ngram_hash_counts(texts, bins=51, seed=nose)`` (the same hash reduced mod 51, not the 16k bins folded).
    """

    def __init__(self, bins: int | None = None) -> None:
        self.bins = int(_nose_cfg()["n51_hash"]["bins"] if bins is None else bins)
        self.mean_: np.ndarray | None = None

    def _fractions(self, X_counts: sp.spmatrix | np.ndarray) -> np.ndarray:
        X = X_counts.toarray() if sp.issparse(X_counts) else np.asarray(X_counts)
        X = X.astype(np.float64, copy=False)
        if X.shape[1] != self.bins:
            raise ValueError(f"N51Hash expects {self.bins} bins, got {X.shape[1]}")
        s = X.sum(axis=1, keepdims=True)
        return np.divide(X, s, out=np.zeros_like(X), where=s > 0).astype(np.float32)

    def fit(self, X_counts_unl: sp.spmatrix | np.ndarray) -> "N51Hash":
        """Store the mean fraction vector of C_unl."""
        self.mean_ = self._fractions(X_counts_unl).mean(axis=0).astype(np.float32)
        return self

    def transform(self, X_counts_51: sp.spmatrix | np.ndarray, centered: bool = True) -> np.ndarray:
        """Fractions c_b / sum c per row; centred on the C_unl mean by default (needs ``fit``)."""
        F = self._fractions(X_counts_51)
        if not centered:
            return F
        if self.mean_ is None:
            raise RuntimeError("N51Hash.fit must run before a centred transform")
        return (F - self.mean_).astype(np.float32, copy=False)
