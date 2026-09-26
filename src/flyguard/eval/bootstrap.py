"""Cluster bootstrap of ТЗ Этап 4 "Статистика": percentile intervals, paired differences, macroAUC, two-stage H3.

Why clusters: documents of one BIPIA context, one AgentDojo task or one paraphrase base are not independent (ТЗ
1.10), so every interval resamples ``cluster_id`` with replacement *within a source*; 1000 draws, percentile 95 %,
seed = the ``bootstrap`` child of the global seed (``configs/default.yaml: stats.bootstrap``).

Why weights: a cluster resample is a vector of integer multiplicities per document. AUC of the resampled sample is
the *weighted* Mann-Whitney statistic, which :class:`WeightedAUC` evaluates from a one-off sort of each score table.
This makes one draw cost O(n) per table instead of O(n log n), and lets the H3 bootstrap (clusters × null matrices
× π permutations, up to 10 × 201 tables) share one set of cluster draws across every table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from flyguard.config import Configs, load_configs

Stat = Callable[[pd.DataFrame], float]


# ----------------------------------------------------------------------------------------------------------------
# CI container (docs/design.md §8)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class CI:
    """A point estimate with a percentile bootstrap interval; ``to_dict`` gives the design §8 shape
    ``{point, low, high, level, n_boot, n_valid, n}`` (plus ``n_clusters``). ``n_boot`` is the number of draws
    made, ``n_valid`` the number whose statistic was defined (a cluster resample can lose a class, see
    :func:`percentile_ci`): an interval built from far fewer valid draws than requested must be visible in the
    results files, so the report can flag it. ``samples`` keeps the bootstrap draws so that p-values
    (``tost.bootstrap_p``) and Holm steps can be derived without redoing the resampling."""

    point: float
    low: float
    high: float
    level: float = 0.95
    n_boot: int = 0
    n: int = 0
    n_clusters: int | None = None
    n_valid: int | None = None
    samples: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"point": _f(self.point), "low": _f(self.low), "high": _f(self.high),
                             "level": float(self.level), "n_boot": int(self.n_boot),
                             "n_valid": int(self.n_boot if self.n_valid is None else self.n_valid), "n": int(self.n)}
        if self.n_clusters is not None:
            d["n_clusters"] = int(self.n_clusters)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CI":
        n_boot = int(d.get("n_boot", 0))
        return cls(point=float(d["point"]), low=float(d["low"]), high=float(d["high"]),
                   level=float(d.get("level", 0.95)), n_boot=n_boot, n=int(d.get("n", 0)),
                   n_clusters=d.get("n_clusters"), n_valid=int(d.get("n_valid", n_boot)))

    @property
    def width(self) -> float:
        return float(self.high - self.low)

    @property
    def valid_share(self) -> float:
        """Share of draws with a defined statistic (1.0 when every draw was valid or nothing was drawn)."""
        if not self.n_boot:
            return 1.0
        return float((self.n_boot if self.n_valid is None else self.n_valid) / self.n_boot)

    def contains(self, x: float) -> bool:
        return bool(self.low <= x <= self.high)

    def inside(self, low: float, high: float) -> bool:
        """True when the whole interval lies within [low, high] (TOST reading of ТЗ Этап 4)."""
        return bool(low <= self.low and self.high <= high)


def _f(x: Any) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def as_ci(x: CI | Mapping[str, Any]) -> CI:
    """Accept either a :class:`CI` or its dict form (results files store the dict)."""
    return x if isinstance(x, CI) else CI.from_dict(x)


def bootstrap_defaults(cfg: Configs | None = None) -> tuple[int, float]:
    """(n draws, alpha) from ``stats.bootstrap`` of configs/default.yaml (1000, 0.05 per ТЗ Этап 4)."""
    cfg = cfg or load_configs()
    b = cfg.default["stats"]["bootstrap"]
    return int(b["n"]), float(b["alpha"])


# ----------------------------------------------------------------------------------------------------------------
# Cluster resampling
# ----------------------------------------------------------------------------------------------------------------
def cluster_codes(cluster_ids: Sequence[Any] | pd.Series | np.ndarray) -> tuple[np.ndarray, int]:
    """Map cluster ids to dense integer codes (sorted order, so the same data always gets the same codes)."""
    codes, uniques = pd.factorize(pd.Series(list(cluster_ids)).astype(str), sort=True)
    return codes.astype(np.int64), int(len(uniques))


def cluster_weights(codes: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """One cluster-bootstrap draw as per-row multiplicities: draw ``n_clusters`` clusters with replacement and
    give every row of a drawn cluster the number of times its cluster was drawn."""
    counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
    return counts[codes]


def iter_cluster_weights(codes: np.ndarray, n_clusters: int, n: int, seed: int) -> Iterator[np.ndarray]:
    """``n`` deterministic cluster draws from ``numpy.random.default_rng(seed)``."""
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n)):
        yield cluster_weights(codes, n_clusters, rng)


def resample_indices(weights: np.ndarray) -> np.ndarray:
    """Row indices of the resampled data set (rows repeated by their multiplicity)."""
    return np.repeat(np.arange(weights.size), weights)


def percentile_ci(samples: np.ndarray, point: float, alpha: float, n: int, n_clusters: int | None = None) -> CI:
    """Percentile interval (``stats.bootstrap.method: percentile``): quantiles alpha/2 and 1 − alpha/2 of the finite
    draws. Draws where the statistic is undefined (a resample lost a class) are dropped, not imputed; their number
    shows as ``n_boot − n_valid`` in the result."""
    s = np.asarray(samples, dtype=float)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        low = high = float("nan")
    else:
        low, high = (float(v) for v in np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return CI(point=float(point), low=low, high=high, level=1 - alpha, n_boot=int(s.size), n=int(n),
              n_clusters=n_clusters, n_valid=int(finite.size), samples=s)


# ----------------------------------------------------------------------------------------------------------------
# Generic statistics on a DataFrame
# ----------------------------------------------------------------------------------------------------------------
def cluster_bootstrap(df: pd.DataFrame, stat: Stat, n: int, seed: int, alpha: float = 0.05,
                      cluster_key: str = "cluster_id") -> CI:
    """Cluster bootstrap of an arbitrary statistic ``stat(resampled_df) -> float`` (ТЗ Этап 4 "Статистика").

    Clusters (``cluster_key``) are drawn with replacement, ``n`` times, from ``default_rng(seed)``; the interval
    is percentile at level 1 − alpha. Use this for TPR/FPR at a threshold, contract metrics and other non-AUC
    statistics; AUC and macroAUC have the faster :func:`macro_auc_bootstrap`.
    """
    codes, k = cluster_codes(df[cluster_key].values)
    point = float(stat(df))
    samples = np.empty(int(n), dtype=float)
    for b, w in enumerate(iter_cluster_weights(codes, k, n, seed)):
        samples[b] = float(stat(df.iloc[resample_indices(w)]))
    return percentile_ci(samples, point, alpha, n=len(df), n_clusters=k)


def paired_cluster_bootstrap(df: pd.DataFrame, stat_a: Stat, stat_b: Stat, n: int, seed: int,
                             alpha: float = 0.05, cluster_key: str = "cluster_id") -> CI:
    """Interval of ``stat_a − stat_b`` on the *same* documents and the *same* cluster resamples (paired bootstrap of
    ТЗ Этап 4: "разности детекторов на одних документах"). Pairing removes the between-document variance shared by
    the two detectors, which is what makes the equivalence corridors of H1a/H1b/H3 reachable at these sample sizes."""
    codes, k = cluster_codes(df[cluster_key].values)
    point = float(stat_a(df)) - float(stat_b(df))
    samples = np.empty(int(n), dtype=float)
    for b, w in enumerate(iter_cluster_weights(codes, k, n, seed)):
        sub = df.iloc[resample_indices(w)]
        samples[b] = float(stat_a(sub)) - float(stat_b(sub))
    return percentile_ci(samples, point, alpha, n=len(df), n_clusters=k)


def auc_stat(score_col: str, label_col: str = "label") -> Stat:
    """A ``stat`` for :func:`cluster_bootstrap`/:func:`paired_cluster_bootstrap`: AUC of ``score_col``."""
    from flyguard.eval.metrics import auc

    return lambda d: auc(d[score_col].values, d[label_col].values)


# ----------------------------------------------------------------------------------------------------------------
# Weighted AUC for many score tables at once
# ----------------------------------------------------------------------------------------------------------------
class WeightedAUC:
    """Mann-Whitney AUC (ties 1/2) of ``T`` score tables over the same ``n`` documents under integer row weights.

    ``scores`` is ``[T, n]`` (or ``[n]``), ``labels`` is ``[n]``. Each table is sorted once; ``auc(weights)`` then
    needs only gathers and one cumulative sum per table: with C_lt(i) / C_le(i) the negative weight strictly below /
    at or below the score of positive i, AUC = Σ_i w_i (C_lt(i) + C_le(i)) / 2 / (W_pos W_neg). With unit weights
    this is exactly ``metrics.auc``; with cluster multiplicities it is the AUC of the resampled sample.

    Tables are processed in chunks of ``chunk_rows`` with preallocated work buffers and ``out=`` arguments: a
    fresh [T, n] temporary per draw would be mmapped and page-faulted on every call (measured 6–8× slower than
    the arithmetic itself). Memory: 9 bytes × T × n (sort order, negative mask) + 24 bytes × T × n_pos (positive
    positions) + about 48 bytes × chunk_rows × n of buffers.
    """

    def __init__(self, scores: np.ndarray, labels: np.ndarray, chunk_rows: int = 256):
        s = np.asarray(scores, dtype=float)
        if s.ndim == 1:
            s = s[None, :]
        y = np.asarray(labels).astype(int) == 1
        if s.shape[1] != y.size:
            raise ValueError("scores and labels disagree on the number of documents")
        t, n = s.shape
        self.n_tables, self.n = t, n
        self.pos = y
        self.n_pos = int(y.sum())
        order = np.argsort(s, axis=1, kind="stable")
        s_sorted = np.take_along_axis(s, order, axis=1)
        idx = np.arange(n)[None, :]
        if n > 1:
            change = s_sorted[:, 1:] != s_sorted[:, :-1]
            start_flag = np.concatenate([np.ones((t, 1), bool), change], axis=1)
            end_flag = np.concatenate([change, np.ones((t, 1), bool)], axis=1)
        else:
            start_flag = end_flag = np.ones((t, 1), bool)
        self.order = order.astype(np.intp)  # intp: np.take converts (copies) any other index dtype on every call
        gstart = np.maximum.accumulate(np.where(start_flag, idx, 0), axis=1)
        gend = np.minimum.accumulate(np.where(end_flag, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]
        pos_sorted = y[order]  # [T, n] label of the document at each sorted position
        self.neg_mask = ~pos_sorted
        # flat indices (row * n + position) of the positive positions and of their tie-group ends/starts, so that
        # one draw needs only 1-D ``np.take`` gathers over [rows, n_pos] instead of 2-D take_along_axis over [T, n];
        # indices are relative to the first row of their chunk
        rows, cols = np.nonzero(pos_sorted)  # rows ascending, n_pos entries per row
        self._chunks: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for r0 in range(0, t, int(chunk_rows)):
            r1 = min(t, r0 + int(chunk_rows))
            sel = slice(r0 * self.n_pos, r1 * self.n_pos)
            rr, cc = rows[sel], cols[sel]
            base = (rr - r0) * n
            shape = (r1 - r0, self.n_pos)
            self._chunks.append((r0, r1, (base + cc).reshape(shape), (base + gend[rr, cc]).reshape(shape),
                                 (base + gstart[rr, cc]).reshape(shape)))
        m = min(int(chunk_rows), t)
        self._buf_n = [np.empty((m, n)) for _ in range(3)]
        self._buf_p = [np.empty((m, self.n_pos)) for _ in range(3)]

    def auc(self, weights: np.ndarray | None = None) -> np.ndarray:
        """AUC of every table under per-document multiplicities ``weights`` (None = all ones). ``nan`` where the
        weighted sample has no positive or no negative."""
        w = np.ones(self.n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
        w_pos_total = float(w[self.pos].sum())
        w_neg_total = float(w.sum() - w_pos_total)
        if w_pos_total == 0 or w_neg_total == 0:
            return np.full(self.n_tables, np.nan)
        num = np.empty(self.n_tables)
        for r0, r1, flat_pos, flat_gend, flat_gstart in self._chunks:
            m = r1 - r0
            ws, wneg, cs = (b[:m] for b in self._buf_n)
            c_le, c_lt, wpos = (b[:m] for b in self._buf_p)
            np.take(w, self.order[r0:r1], out=ws)  # weights in sorted order
            np.multiply(ws, self.neg_mask[r0:r1], out=wneg)  # negative weights only
            np.cumsum(wneg, axis=1, out=cs)  # negative weight at or below each sorted position (inclusive)
            np.subtract(cs, wneg, out=wneg)  # exclusive cumulative sum: negative weight strictly below
            np.take(cs.ravel(), flat_gend, out=c_le)  # negative weight <= score of each positive
            np.take(wneg.ravel(), flat_gstart, out=c_lt)  # negative weight < score of each positive
            np.take(ws.ravel(), flat_pos, out=wpos)
            np.add(c_le, c_lt, out=c_le)
            np.multiply(c_le, wpos, out=c_le)
            num[r0:r1] = c_le.sum(axis=1)
        return num / (2.0 * w_pos_total * w_neg_total)


# ----------------------------------------------------------------------------------------------------------------
# macroAUC bootstrap (resampling within each source)
# ----------------------------------------------------------------------------------------------------------------
def _source_seeds(seed: int, sources: Sequence[str]) -> dict[str, int]:
    """One child seed per source (sorted names), so the draws do not depend on dict order."""
    children = np.random.SeedSequence(int(seed)).spawn(len(sources))
    return {s: int(c.generate_state(1, dtype=np.uint32)[0]) for s, c in zip(sorted(sources), children)}


def macro_auc_draws(by_source: Mapping[str, pd.DataFrame], tables: Mapping[str, np.ndarray] | None, n: int,
                    seed: int, score_cols: Sequence[str] = ("score",), label: str = "label",
                    cluster_key: str = "cluster_id") -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Per-draw macroAUC of many score tables with cluster resampling inside each source.

    ``by_source[s]`` holds ``label``/``cluster_key`` (and the ``score_cols`` when ``tables`` is None); ``tables[s]``
    is an optional ``[T, n_s]`` array of scores aligned with the rows of ``by_source[s]``. Returns ``(point [T],
    draws [n, T], n_clusters by source)``. A draw in which some source loses a class is ``nan`` (macroAUC would
    silently change its set of sources otherwise).
    """
    seeds = _source_seeds(seed, sorted(by_source))
    evals: dict[str, WeightedAUC] = {}
    for s in sorted(by_source):
        d = by_source[s]
        sc = np.asarray(tables[s], dtype=float) if tables is not None else d[list(score_cols)].to_numpy(float).T
        ev = WeightedAUC(sc, d[label].to_numpy())
        if np.all(np.isnan(ev.auc(None))):
            continue  # single-class source: not "present" for macroAUC (ТЗ 0), so it does not enter the draws
        evals[s] = ev
    sources = sorted(evals)
    if not sources:
        raise ValueError("no source with both classes")
    t = evals[sources[0]].n_tables
    if any(e.n_tables != t for e in evals.values()):
        raise ValueError("every source must provide the same number of score tables")
    point = np.mean(np.stack([evals[s].auc(None) for s in sources]), axis=0)
    draws = np.zeros((int(n), t), dtype=float)
    n_clusters: dict[str, int] = {}
    for s in sources:
        codes, k = cluster_codes(by_source[s][cluster_key].values)
        n_clusters[s] = k
        acc = np.zeros((int(n), t), dtype=float)
        for b, w in enumerate(iter_cluster_weights(codes, k, n, seeds[s])):
            acc[b] = evals[s].auc(w)
        draws += acc / len(sources)
    return point, draws, n_clusters


def macro_auc_bootstrap(by_source: Mapping[str, pd.DataFrame], n: int, seed: int, alpha: float = 0.05,
                        score: str = "score", reference: str | None = None, label: str = "label",
                        cluster_key: str = "cluster_id") -> CI:
    """Cluster-bootstrap interval of macroAUC (ТЗ Этап 4: "macroAUC пересэмплируется внутри каждого источника").

    ``by_source`` maps a source name to its document frame (``label``, ``cluster_id``, score columns). With
    ``reference`` given, the interval is of macroAUC(``score``) − macroAUC(``reference``) on the same documents and
    the same resamples (paired). A single-source dict gives an ordinary per-source AUC interval, so one code path
    serves AUC_S and macroAUC.
    """
    cols = [score] + ([reference] if reference else [])
    point, draws, n_clusters = macro_auc_draws(by_source, None, n, seed, score_cols=cols, label=label,
                                               cluster_key=cluster_key)
    if reference:
        pt, samples = float(point[0] - point[1]), draws[:, 0] - draws[:, 1]
    else:
        pt, samples = float(point[0]), draws[:, 0]
    n_docs = int(sum(len(by_source[s]) for s in n_clusters))  # present sources only
    return percentile_ci(samples, pt, alpha, n=n_docs, n_clusters=int(sum(n_clusters.values())))


# ----------------------------------------------------------------------------------------------------------------
# Two-stage bootstrap for H3
# ----------------------------------------------------------------------------------------------------------------
def two_stage_bootstrap_h3(docs_by_source: Mapping[str, pd.DataFrame], measured: Mapping[Any, np.ndarray],
                           nulls: Mapping[Any, np.ndarray], n: int, seed: int, alpha: float | None = None,
                           label: str = "label", cluster_key: str = "cluster_id",
                           cfg: Configs | None = None) -> dict[str, Any]:
    """H3 interval of macroAUC(measured M) − mean macroAUC(null matrices) over clusters × null matrices × π (ТЗ
    Этап 4 "Двухступенчатый бутстреп H3"; acceptance: "перестановка π ... входит в двухступенчатый бутстреп H3").

    Inputs are precomputed document-score tables, one per (matrix, π):
    * ``docs_by_source[s]``: frame with ``label`` and ``cluster_id`` for source ``s`` (row order fixes the columns);
    * ``measured[p]``: dict of source -> ``[n_s]`` scores of the measured matrix under permutation ``p``;
    * ``nulls[p]``: dict of source -> ``[J, n_s]`` scores of the J null matrices under ``p`` (row j = matrix j for
      every p, so that level 2 resamples *matrices*).
    Every draw resamples (1) clusters within each source -- shared by all tables, (2) the J null matrices with
    replacement, (3) the P permutations with replacement; the statistic is the π-averaged measured macroAUC minus
    the (π, matrix)-averaged null macroAUC. ``alpha=None`` gives the TOST level of ``stats.tost.ci`` (90 %),
    because the result feeds the H3 equivalence test.
    Returns ``{"diff": CI, "measured": CI, "null_mean": CI, "n_perms", "n_null", "delta_reference"}`` where
    ``delta_reference`` is the point null mean (the reference value of the ±δ corridor).
    """
    if alpha is None:
        cfg = cfg or load_configs()
        alpha = 1.0 - float(cfg.default["stats"]["tost"]["ci"])
    perms = sorted(measured, key=str)
    if not perms or any(p not in nulls for p in perms):
        raise ValueError("measured and nulls must cover the same permutation ids")
    sources = sorted(docs_by_source)
    j_null = None
    tables: dict[str, np.ndarray] = {}
    for s in sources:
        rows = []
        for p in perms:
            rows.append(np.asarray(measured[p][s], dtype=float)[None, :])
        for p in perms:
            block = np.asarray(nulls[p][s], dtype=float)
            block = block[None, :] if block.ndim == 1 else block
            if j_null is None:
                j_null = block.shape[0]
            elif block.shape[0] != j_null:
                raise ValueError("every permutation must carry the same number of null matrices")
            rows.append(block)
        tables[s] = np.concatenate(rows, axis=0)
    n_p, j_null = len(perms), int(j_null or 0)
    point, draws, n_clusters = macro_auc_draws(docs_by_source, tables, n, seed, label=label, cluster_key=cluster_key)
    meas_pt, null_pt = point[:n_p], point[n_p:].reshape(n_p, j_null)
    # a stream distinct from the per-source cluster streams (which are children of SeedSequence(seed))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4833]))
    meas_s = np.empty(int(n))
    null_s = np.empty(int(n))
    for b in range(int(n)):
        m_p = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p) / n_p
        m_j = np.bincount(rng.integers(0, j_null, j_null), minlength=j_null) / j_null
        meas_s[b] = float(m_p @ draws[b, :n_p])
        null_s[b] = float(m_p @ draws[b, n_p:].reshape(n_p, j_null) @ m_j)
    n_docs = int(sum(len(docs_by_source[s]) for s in n_clusters))
    k = int(sum(n_clusters.values()))
    meas_point, null_point = float(np.mean(meas_pt)), float(np.mean(null_pt))
    return {
        "diff": percentile_ci(meas_s - null_s, meas_point - null_point, alpha, n_docs, k),
        "measured": percentile_ci(meas_s, meas_point, alpha, n_docs, k),
        "null_mean": percentile_ci(null_s, null_point, alpha, n_docs, k),
        "n_perms": n_p,
        "n_null": j_null,
        "delta_reference": null_point,
        "per_perm": {str(p): {"measured": float(meas_pt[i]), "null_mean": float(np.mean(null_pt[i]))}
                     for i, p in enumerate(perms)},
    }
