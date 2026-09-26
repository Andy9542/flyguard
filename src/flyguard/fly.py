"""The fly body: expansion a = M u, k-WTA inhibition and the binary codes (ТЗ 2.2, 2.3, 3.3; preregistration
§3.2-3.3).

Sizes range from the measured 1 886 cells to FlyHash's 20·16 384 and 40·16 384 cells with k = 5 %, so every
function that produces an ``[n, m]`` activation works on row batches (``batch_size``; the default is chosen from
a memory budget) and keeps float32 with sparse products.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterator

import numpy as np
import scipy.sparse as sp

from flyguard.config import load_configs

_BATCH_BUDGET_BYTES = 128 << 20  # float32 activations per batch; kwta needs ~3x this transiently


@lru_cache(maxsize=1)
def _inhibition_cfg() -> dict[str, Any]:
    return load_configs().default["inhibition"]


def k_for(m: int, k_frac: float | None = None) -> int:
    """k = round(k_frac · m) winners (ТЗ 2.3; ``inhibition.k_frac`` = 0.05, E6 uses ``e6_k_frac``), at least 1."""
    k_frac = float(_inhibition_cfg()["k_frac"] if k_frac is None else k_frac)
    return int(min(max(1, round(k_frac * int(m))), int(m)))


def _auto_batch(n: int, m: int, batch_size: int | None) -> int:
    if batch_size is not None:
        return max(1, int(batch_size))
    return max(1, min(int(n), _BATCH_BUDGET_BYTES // max(4 * int(m), 1)))


def _row_batches(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def kwta(A: np.ndarray, k: int) -> sp.csr_matrix:
    """k-winners-take-all (ТЗ 2.3, the APL neuron): exactly ``k`` ones per row at the ``k`` largest activations,
    ties broken towards the lowest index.

    Vectorised without a full argsort: ``np.partition`` gives the k-th largest value ``t`` of each row; every
    entry ``> t`` wins, and the remaining ``k - #(> t)`` slots are filled by the lowest-index entries equal to
    ``t`` (a cumulative count over the tie mask, computed only for rows that have more ties than slots). Ties are
    not exotic here: KCs with identical glomerular input sets have identical activations, and an empty window
    gives an all-zero row, which must yield the first ``k`` cells deterministically. Returns a binary float32
    csr with sorted indices. Rows must be finite.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("kwta expects a 2-D activation array")
    n, m = A.shape
    k = int(k)
    if k <= 0:
        return sp.csr_matrix((n, m), dtype=np.float32)
    if k >= m:
        return sp.csr_matrix(np.ones((n, m), dtype=np.float32))
    t = np.partition(A, m - k, axis=1)[:, m - k]  # k-th largest value per row
    strict = A > t[:, None]
    ties = A == t[:, None]
    need = k - strict.sum(axis=1)
    n_ties = ties.sum(axis=1)
    Z = strict
    exact = n_ties == need
    Z[exact] |= ties[exact]
    fill = np.flatnonzero(n_ties > need)
    if fill.size:
        T = ties[fill]
        cum = np.cumsum(T, axis=1, dtype=np.int32)
        Z[fill] = Z[fill] | (T & (cum <= need[fill, None]))
    rows, cols = np.nonzero(Z)
    if cols.size != n * k:  # NaN or a logic error; never silently return a wrong code
        raise RuntimeError("kwta did not produce exactly k winners per row (non-finite activations?)")
    indptr = np.arange(0, n * k + 1, k, dtype=np.int64)
    out = sp.csr_matrix((np.ones(n * k, dtype=np.float32), cols.astype(np.int32), indptr), shape=(n, m))
    out.has_sorted_indices = True
    return out


def expand(U: np.ndarray | sp.spmatrix, M: sp.spmatrix | np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    """Activations a = M (u - mean) for every row of U, as a dense float32 ``[n, m]`` array (ТЗ 2.2).

    With ``mean`` (the C_unl mean of the nose, ТЗ 2.1) the centring is applied as ``U @ M.T - M @ mean``: the
    same numbers as expanding the centred input, but the 16 384-bin N16k rows stay sparse. Sparse U uses a
    sparse-sparse product (FlyHash sizes); dense U (the 51-d noses) uses ``M @ U.T``. Works for the weighted
    (synapse-count) M of E6 too. Call through :func:`fly_code` for batching.
    """
    Ms = sp.csr_matrix(M, dtype=np.float32)
    if sp.issparse(U):
        Us = sp.csr_matrix(U, dtype=np.float32)
        A = (Us @ Ms.T).toarray().astype(np.float32, copy=False)
    else:
        Ud = np.asarray(U, dtype=np.float32)
        A = np.ascontiguousarray((Ms @ Ud.T).T, dtype=np.float32)
    if mean is not None:
        A -= (Ms @ np.asarray(mean, dtype=np.float32)).astype(np.float32)[None, :]
    return A


def fly_code(U: np.ndarray | sp.spmatrix, M: sp.spmatrix | np.ndarray, k: int, mean: np.ndarray | None = None,
             batch_size: int | None = None) -> sp.csr_matrix:
    """z = kWTA(M (u - mean)) for every row (ТЗ 2.2-2.3): the sparse binary KC code, ``[n, m]`` csr.

    Rows are processed in batches (``batch_size`` rows; default sized so one float32 activation batch stays
    within ~128 MiB, i.e. ~100 rows at m = 327 680) because the dense activation matrix is the only large object
    in the pipeline. The result does not depend on the batch size.
    """
    n = U.shape[0]
    m = M.shape[0]
    size = _auto_batch(n, m, batch_size)
    blocks = [kwta(expand(U[b], M, mean), k) for b in _row_batches(n, size)]
    if not blocks:
        return sp.csr_matrix((0, m), dtype=np.float32)
    out = sp.vstack(blocks, format="csr", dtype=np.float32)
    out.sort_indices()
    return out


def sign_code(U: np.ndarray | sp.spmatrix, G: np.ndarray, mean: np.ndarray | None = None,
              batch_size: int | None = None) -> sp.csr_matrix:
    """Dense Gaussian sign code (ТЗ 3.3 null model): bit j = [g_j · (u - mean) > 0], ``[n, round(nnz/d)]`` csr.

    ``G`` comes from :func:`flyguard.connectome.dense_gaussian_sign_matrix`. About half the bits are active, which
    is why this code is evaluated with the linear readout only (preregistration §5.4).
    """
    G = np.asarray(G, dtype=np.float32)
    n = U.shape[0]
    size = _auto_batch(n, G.shape[0], batch_size)
    offset = (G @ np.asarray(mean, dtype=np.float32)) if mean is not None else None
    blocks = []
    for b in _row_batches(n, size):
        Ub = U[b]
        P = (Ub @ G.T) if sp.issparse(Ub) else np.asarray(Ub, dtype=np.float32) @ G.T
        P = np.asarray(P, dtype=np.float32)
        if offset is not None:
            P = P - offset[None, :]
        blocks.append(sp.csr_matrix((P > 0).astype(np.float32)))
    if not blocks:
        return sp.csr_matrix((0, G.shape[0]), dtype=np.float32)
    out = sp.vstack(blocks, format="csr", dtype=np.float32)
    out.sort_indices()
    return out
