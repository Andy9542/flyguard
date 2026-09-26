"""Expansion matrices M (ТЗ 2.2, 3.3): the measured MaleCNS PN->KC wiring, its null models and the controls.

Convention (design §5): every M is a binary float32 csr matrix of shape ``[m_cells, d_inputs]``; the activation of
a cell is ``a = M u``. The measured matrix has 51 glomeruli (inputs) x 1 886 Kenyon cells (right hemisphere,
FlyHash-Connectome's ``flypath build`` selection), binary in the main analysis and weighted by synapse counts in
E6. Null models: a random matrix of the same density (each cell keeps its in-degree), Curveball shuffles that
preserve both degree sequences, and, at FlyHash scale, random matrices with a fixed fan-in per cell. The dense
Gaussian sign matrix of ТЗ 3.3 costs the same multiply-adds as M and is used only with the linear readout.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from flyguard.config import ROOT, load_configs

DEFAULT_MALECNS_PATH = ROOT / "data" / "processed" / "connectome" / "malecns_R.npz"
_OPTIONAL_NPZ_KEYS = ("pn_counts", "pn_partners", "kc_body_ids")


@lru_cache(maxsize=1)
def _expansion_cfg() -> dict[str, Any]:
    return load_configs().default["expansion"]


def as_binary_csr(M: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    """Canonical form used everywhere: binary (nonzero -> 1) float32 csr with sorted, duplicate-free indices."""
    X = sp.csr_matrix(M, dtype=np.float32, copy=True)
    X.sum_duplicates()
    X.eliminate_zeros()
    X.data = np.ones_like(X.data, dtype=np.float32)
    X.sort_indices()
    return X


def indegree_stats(M: sp.spmatrix | np.ndarray, check: tuple[float, float] | None = None) -> dict[str, Any]:
    """Degree statistics of an expansion matrix with the ТЗ 2.2 check that the mean KC in-degree lies in 4-8.

    In-degree of a cell = number of distinct inputs (row sum of the binary M); out-degree of an input (glomerulus)
    = column sum. ``check`` defaults to ``expansion.malecns.indegree_check`` from the config; ``in_range`` is
    recorded rather than enforced so that the audit can report it (acceptance criterion "Средняя степень KC 4-8").
    """
    B = as_binary_csr(M)
    lo, hi = (_expansion_cfg()["malecns"]["indegree_check"] if check is None else check)
    indeg = np.diff(B.indptr)
    outdeg = np.asarray(B.sum(axis=0)).ravel()
    n_cells, n_inputs = B.shape
    mean = float(indeg.mean()) if n_cells else 0.0
    return {
        "n_cells": int(n_cells), "n_inputs": int(n_inputs), "nnz": int(B.nnz),
        "density": float(B.nnz / (n_cells * n_inputs)) if n_cells and n_inputs else 0.0,
        "indegree_mean": mean, "indegree_min": int(indeg.min()) if n_cells else 0,
        "indegree_max": int(indeg.max()) if n_cells else 0,
        "indegree_median": float(np.median(indeg)) if n_cells else 0.0,
        "n_zero_indegree": int((indeg == 0).sum()),
        "outdegree_mean": float(outdeg.mean()) if n_inputs else 0.0,
        "outdegree_min": int(outdeg.min()) if n_inputs else 0, "outdegree_max": int(outdeg.max()) if n_inputs else 0,
        "check_range": [float(lo), float(hi)], "in_range": bool(lo <= mean <= hi),
    }


def load_malecns(path: str | Path | None = None, strict: bool = False) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Load the measured glomerulus x KC matrix written by ``scripts/malecns_matrix.py`` (ТЗ 2.2 "Измеренная M").

    The file holds ``m`` = synapse counts ``[n_glomeruli, n_kc]`` (float64) plus optional ``pn_counts``,
    ``pn_partners`` (possibly ragged object arrays, which ``np.load`` refuses without pickle; they are then
    recorded as unavailable rather than loading pickled data). Returns the binary ``(m > 0).T`` as float32 csr
    ``[n_kc, n_glomeruli]`` and ``meta`` with the weighted (synapse-count) variant for E6 under ``"weighted"``,
    the in-degree statistics, the number of KCs (recorded per ТЗ 2.2; zero-in-degree cells are kept, counted in
    ``n_kc_zero_indegree``), and the companion ``malecns_R.json`` if present. ``strict=True`` raises when the
    mean in-degree is outside the configured 4-8 range.
    """
    path = Path(path) if path is not None else DEFAULT_MALECNS_PATH
    unavailable: list[str] = []
    optional: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as npz:
        if "m" not in npz.files:
            raise KeyError(f"{path} has no key 'm' (keys: {npz.files})")
        m = np.asarray(npz["m"], dtype=np.float64)
        for key in _OPTIONAL_NPZ_KEYS:
            if key in npz.files:
                try:
                    optional[key] = np.asarray(npz[key])
                except ValueError:  # object array -> needs pickle -> refused on a data file
                    unavailable.append(key)
    if m.ndim != 2:
        raise ValueError(f"expected m as [n_glomeruli, n_kc], got shape {m.shape}")
    if np.any(m < 0):
        raise ValueError("synapse counts must be non-negative")
    weighted = sp.csr_matrix(m.T.astype(np.float32))
    weighted.sort_indices()
    M = as_binary_csr(weighted)
    stats = indegree_stats(M)
    if strict and not stats["in_range"]:
        raise ValueError(f"mean KC in-degree {stats['indegree_mean']:.2f} outside {stats['check_range']}")
    json_path = path.with_suffix(".json")
    meta: dict[str, Any] = {
        "path": str(path), "n_glomeruli": int(m.shape[0]), "n_kc": int(m.shape[1]), "nnz": int(M.nnz),
        "n_synapses": float(m.sum()), "n_kc_zero_indegree": stats["n_zero_indegree"], "indegree": stats,
        "weighted": weighted, "optional": optional, "unavailable_keys": unavailable,
        "json": json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None,
    }
    return M, meta


def _rows_to_csr(rows: list[list[int]] | list[set[int]] | list[np.ndarray], shape: tuple[int, int]) -> sp.csr_matrix:
    lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    indptr = np.concatenate([[0], np.cumsum(lens)])
    indices = np.concatenate([np.sort(np.fromiter(r, dtype=np.int32, count=len(r))) for r in rows]) if len(rows) \
        else np.zeros(0, dtype=np.int32)
    X = sp.csr_matrix((np.ones(indices.size, dtype=np.float32), indices.astype(np.int32), indptr), shape=shape)
    X.sort_indices()
    return X


def random_same_density(M: sp.spmatrix | np.ndarray, seed: int) -> sp.csr_matrix:
    """Random matrix of the same density (ТЗ 2.2, seed ``projection``): each cell keeps its in-degree, its inputs
    are drawn uniformly without replacement.

    Uniformity: per row, iid uniform keys sorted by ``argsort`` give a uniformly random permutation of the
    inputs, and its first ``deg`` entries are a uniformly random ``deg``-subset. Rows are processed in chunks of
    at most ~8M keys so the measured size (1 886 x 51) is one chunk and FlyHash sizes stay bounded in memory.
    Column degrees are *not* preserved (that is what distinguishes it from Curveball).
    """
    B = as_binary_csr(M)
    n, d = B.shape
    deg = np.diff(B.indptr)
    rng = np.random.default_rng(int(seed))
    indices = np.empty(B.nnz, dtype=np.int32)
    chunk = max(1, min(n, (1 << 23) // max(d, 1)))
    pos = 0
    col = np.arange(d)[None, :]
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        order = np.argsort(rng.random((stop - start, d)), axis=1)
        sel = order[col < deg[start:stop, None]]  # row-major -> matches indptr order
        indices[pos:pos + sel.size] = sel
        pos += sel.size
    R = sp.csr_matrix((np.ones(B.nnz, dtype=np.float32), indices, B.indptr.copy()), shape=(n, d))
    R.sort_indices()
    return R


def curveball_n_trades(M: sp.spmatrix | np.ndarray, swaps_per_edge: float | None = None) -> int:
    """n_trades = ceil(swaps_per_edge x nnz(M)) (ТЗ 2.2 ">= 5·E обменов"; ``expansion.curveball.swaps_per_edge``).

    This is the number of *effective* exchanges :func:`curveball` performs (trades that change the matrix), so
    the literal ТЗ count holds whichever way "обмен" is read; Strona's own convention counts attempted trades,
    of which ~4 % are no-ops on the measured matrix.
    """
    spe = float(_expansion_cfg()["curveball"]["swaps_per_edge"] if swaps_per_edge is None else swaps_per_edge)
    return int(np.ceil(spe * as_binary_csr(M).nnz))


_CURVEBALL_MAX_ATTEMPT_FACTOR = 20  # attempts are capped at this multiple of n_trades (degenerate matrices)


def _distinct_pairs(rng: np.random.Generator, n: int, size: int) -> np.ndarray:
    pairs = rng.integers(0, n, size=(size, 2))
    same = pairs[:, 0] == pairs[:, 1]
    while same.any():
        pairs[same] = rng.integers(0, n, size=(int(same.sum()), 2))
        same = pairs[:, 0] == pairs[:, 1]
    return pairs


def curveball(M: sp.spmatrix | np.ndarray, n_trades: int, seed: int,
              return_stats: bool = False) -> sp.csr_matrix | tuple[sp.csr_matrix, dict[str, Any]]:
    """Curveball shuffle (Strona et al. 2014) preserving both degree sequences (ТЗ 2.2, 3.3; seed ``curveball``).

    Each trade picks a uniformly random pair of distinct cells (rows) i != j, keeps the inputs they share, pools
    the inputs held by exactly one of them and redistributes the pool uniformly at random so that i and j keep
    their in-degrees; every pooled input goes to exactly one of the two rows, so the input (glomerulus)
    out-degrees are preserved too. Pools are sorted before the RNG touches them so nothing depends on Python set
    iteration order; pairs are drawn in blocks from one generator, so the result is a deterministic function of
    ``(M, n_trades, seed)``.

    Counting. ``n_trades`` (from :func:`curveball_n_trades`, swaps_per_edge x nnz) is the number of *effective*
    trades — trades that change the matrix. A trade is a no-op when one row's private inputs are empty (one set
    contains the other) or when the random split re-draws the original assignment; on the measured matrix these
    are ~0.8 % and ~3 % of attempts. Strona's convention counts attempted trades, under which a literal
    ">= 5·E обменов" would be missed by ~4 %; counting effective trades satisfies the ТЗ under either reading.
    Uniformity caveat: Carstens 2015 proves the uniform stationary law for the lazy chain with a *fixed* number
    of attempted trades (no-ops are self-loops of a symmetric transition matrix); stopping at the N-th effective
    trade samples the embedded jump chain instead, whose stationary law is proportional to 1 - p_hold(X). With
    p_hold ~ 0.04 on the measured matrix and margins fixed, the relative variation of that weight across matrices
    is well below the residual mixing error at 5·E trades, so the bias is negligible for the E4 null; the
    attempted/effective counts are returned with ``return_stats=True`` so the report can state them. Attempts are
    capped at ``_CURVEBALL_MAX_ATTEMPT_FACTOR x n_trades``: a matrix on which no trade can change anything (all
    rows identical or nested) would otherwise never terminate; the cap is recorded in ``stats["capped"]`` and
    warned about, never silent.
    """
    B = as_binary_csr(M)
    n, d = B.shape
    n_trades = int(n_trades)
    stats: dict[str, Any] = {"n_target": n_trades, "n_attempted": 0, "n_effective": 0, "n_subset_noop": 0,
                             "n_redraw_noop": 0, "capped": False}
    if n < 2 or n_trades <= 0:
        return (B, stats) if return_stats else B
    rows: list[set[int]] = [set(B.indices[B.indptr[i]:B.indptr[i + 1]].tolist()) for i in range(n)]
    rng = np.random.default_rng(int(seed))
    max_attempts = _CURVEBALL_MAX_ATTEMPT_FACTOR * n_trades
    block = n_trades
    while stats["n_effective"] < n_trades and stats["n_attempted"] < max_attempts:
        block = min(block, max_attempts - stats["n_attempted"])
        for a, b in _distinct_pairs(rng, n, block).tolist():
            stats["n_attempted"] += 1
            A, Bb = rows[a], rows[b]
            shared = A & Bb
            a_only, b_only = A - shared, Bb - shared
            if not a_only or not b_only:
                stats["n_subset_noop"] += 1
                continue
            pool = sorted(a_only | b_only)
            perm = rng.permutation(len(pool)).tolist()
            na = len(a_only)
            new_a = shared | {pool[j] for j in perm[:na]}
            if new_a == A:
                stats["n_redraw_noop"] += 1
                continue
            rows[a] = new_a
            rows[b] = shared | {pool[j] for j in perm[na:]}
            stats["n_effective"] += 1
            if stats["n_effective"] >= n_trades:
                break
        block = max(256, n_trades // 16)  # top-up blocks for the ~4 % of no-ops
    if stats["n_effective"] < n_trades:
        stats["capped"] = True
        warnings.warn(f"curveball: only {stats['n_effective']} of {n_trades} effective trades after "
                      f"{stats['n_attempted']} attempts (degenerate matrix?)", RuntimeWarning, stacklevel=2)
    out = _rows_to_csr(rows, (n, d))
    return (out, stats) if return_stats else out


def curveball_nulls(M: sp.spmatrix | np.ndarray, n_null: int | None = None, seed: int = 0,
                    swaps_per_edge: float | None = None) -> list[sp.csr_matrix]:
    """The set of Curveball null matrices for E4 (ТЗ 2.2 "200 перемешиваний", ``expansion.curveball.n_null``).

    Each null is an independent chain started from M with ``curveball_n_trades`` trades and its own child seed
    (``SeedSequence(seed).spawn(n_null)``), so the 200 matrices are exchangeable draws rather than autocorrelated
    samples of one chain, which is what the randomisation test of ТЗ Этап 4 assumes.
    """
    n_null = int(_expansion_cfg()["curveball"]["n_null"] if n_null is None else n_null)
    n_trades = curveball_n_trades(M, swaps_per_edge)
    children = np.random.SeedSequence(int(seed)).spawn(n_null)
    return [curveball(M, n_trades, int(c.generate_state(1, dtype=np.uint32)[0])) for c in children]


def flyhash_matrix(d: int, expansion: int | None = None, fan_in: int | None = None, seed: int = 0) -> sp.csr_matrix:
    """FlyHash projection (ТЗ 2.2 "FlyHash"; seed ``projection``): ``[expansion*d, d]`` with exactly ``fan_in``
    distinct inputs per cell (6, the number of KC claws; ``expansion.flyhash``).

    Sampling: draw ``fan_in`` iid uniform inputs per row and redraw the rows that contain a duplicate until none
    does; conditioning iid draws on distinctness gives the uniform distribution over ``fan_in``-subsets, and the
    rejection rate is ~fan_in^2 / (2d) (0.1 % at d = 16 384), so this is fast at m = 655 360 without an
    ``m x d`` intermediate.
    """
    cfg = _expansion_cfg()["flyhash"]
    expansion = int(cfg["primary_expansion"] if expansion is None else expansion)
    fan_in = int(cfg["fan_in"] if fan_in is None else fan_in)
    d = int(d)
    if fan_in > d or fan_in <= 0 or expansion <= 0:
        raise ValueError(f"need 0 < fan_in <= d and expansion > 0 (fan_in={fan_in}, d={d}, expansion={expansion})")
    m = expansion * d
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, d, size=(m, fan_in))
    draws.sort(axis=1)
    bad = np.flatnonzero(np.any(draws[:, 1:] == draws[:, :-1], axis=1)) if fan_in > 1 else np.zeros(0, dtype=int)
    while bad.size:
        redraw = rng.integers(0, d, size=(bad.size, fan_in))
        redraw.sort(axis=1)
        draws[bad] = redraw
        bad = bad[np.any(redraw[:, 1:] == redraw[:, :-1], axis=1)]
    indptr = np.arange(0, m * fan_in + 1, fan_in, dtype=np.int64)
    X = sp.csr_matrix((np.ones(m * fan_in, dtype=np.float32), draws.ravel().astype(np.int32), indptr), shape=(m, d))
    X.has_sorted_indices = True
    return X


def dense_gaussian_sign_matrix(d: int, nnz: int, seed: int) -> np.ndarray:
    """Dense N(0,1) matrix ``[round(nnz/d), d]`` (ТЗ 3.3 "плотный гауссов знаковый код при равном числе операций").

    A sparse M costs ``nnz`` multiply-adds per input vector; a dense matrix with ``r`` rows costs ``r*d``, so
    ``r = round(nnz/d)`` equalises the operation count. Used with :func:`flyguard.fly.sign_code` and the linear
    readout only (with half the bits active a Bloom filter saturates after a few examples, preregistration §5.4).
    """
    r = max(1, int(round(int(nnz) / int(d))))
    return np.random.default_rng(int(seed)).standard_normal((r, int(d))).astype(np.float32)
