"""Equivalence and multiplicity of ТЗ Этап 4 "Статистика": TOST by the 90 % interval, Holm, randomisation p.

TOST (Lakens 2017, cited by the preregistration) declares two detectors equivalent when the 90 % CI of their
difference lies entirely inside the corridor ±δ, δ = ``stats.tost.delta_rel`` (5 %) × the reference value of the
metric. The 90 % interval is the two one-sided 5 % tests combined, hence ``stats.tost.ci: 0.90``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from flyguard.config import Configs, load_configs
from flyguard.eval.bootstrap import CI, as_ci


def equivalence_margin(reference: float, delta_rel: float | None = None, cfg: Configs | None = None) -> float:
    """δ = delta_rel × reference (ТЗ 0: "коридор эквивалентности δ = 5% от референсного значения метрики"). The
    reference is the metric of the detector the hypothesis compares against (ProtectAI v2 for H1a, LR-on-N51 / TF-IDF
    for H1b, the mean null macroAUC for H3), so the corridor scales with how good the reference is."""
    if delta_rel is None:
        cfg = cfg or load_configs()
        delta_rel = float(cfg.default["stats"]["tost"]["delta_rel"])
    return float(delta_rel) * abs(float(reference))


def tost_equivalent(diff_ci90: CI | Mapping[str, Any], delta: float) -> bool:
    """Equivalence by TOST: the whole 90 % interval of the difference lies within [−δ, +δ] (closed corridor)."""
    ci = as_ci(diff_ci90)
    if not (math.isfinite(ci.low) and math.isfinite(ci.high)):
        return False
    return ci.inside(-abs(delta), abs(delta))


def tost(diff_ci90: CI | Mapping[str, Any], delta: float) -> dict[str, Any]:
    """TOST report: decision plus the distinction the ТЗ asks for in H3 ("эквивалентна в коридоре" vs "не
    отличается"): ``differs_from_zero`` is True when the same interval excludes 0, i.e. a small but real difference
    inside the corridor, as in the FlyHash-Connectome reference (−2.06 %, p = 0.02)."""
    ci = as_ci(diff_ci90)
    eq = tost_equivalent(ci, delta)
    finite = math.isfinite(ci.low) and math.isfinite(ci.high)
    return {
        "equivalent": eq,
        "delta": float(abs(delta)),
        "point": ci.point, "low": ci.low, "high": ci.high, "level": ci.level,
        "differs_from_zero": bool(finite and not ci.contains(0.0)),
        "outside_corridor": bool(finite and (ci.low > abs(delta) or ci.high < -abs(delta))),
    }


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (``stats.holm: true``): sort ascending, p_(k) × (m − k + 1), enforce
    monotonicity, cap at 1. Controls the family-wise error over the sources of the H1a semantic half and over the
    secondary sources/metrics of the H3 randomisation test."""
    items = [(k, float(v)) for k, v in pvalues.items() if v is not None and not math.isnan(float(v))]
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1])):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[k] = running
    return out


def holm_reject(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Rejections at family-wise level ``alpha`` (adjusted p <= alpha)."""
    return {k: bool(p <= alpha) for k, p in holm(pvalues).items()}


def randomization_p(observed: float, nulls: Sequence[float] | np.ndarray, two_sided: bool = True,
                    center: float | str = "mean", alternative: str = "greater") -> float:
    """Exact randomisation p-value of one observed statistic against N null statistics (ТЗ 3.3 / Этап 4: measured M
    against 200 curveball matrices): p = (1 + #{|null − c| >= |obs − c|}) / (N + 1), c = mean of the nulls (or a
    number, 0 for statistics that are already centred). The +1 counts the observed value as one of the N + 1
    exchangeable outcomes, so p is never 0 and the smallest attainable value is 1/(N + 1) (1/201 for 200 nulls).
    One-sided (``two_sided=False``): ``alternative`` "greater" counts nulls >= obs, "less" nulls <= obs."""
    x = np.asarray(nulls, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    obs = float(observed)
    if two_sided:
        c = float(np.mean(x)) if isinstance(center, str) else float(center)
        k = int(np.sum(np.abs(x - c) >= abs(obs - c) - 1e-12))
    elif alternative == "greater":
        k = int(np.sum(x >= obs - 1e-12))
    elif alternative == "less":
        k = int(np.sum(x <= obs + 1e-12))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return (1 + k) / (n + 1)


def bootstrap_p(samples: Sequence[float] | np.ndarray, null: float = 0.0, two_sided: bool = True) -> float:
    """Percentile-bootstrap p-value consistent with the percentile CI: two-sided p = 2 × min(P(θ* <= null),
    P(θ* >= null)) (capped at 1), so the 95 % CI excludes ``null`` iff p < 0.05. Used to feed Holm from the same
    draws that produced the interval (H1a semantic half: "нижняя граница ... выше нуля, Хольм по источникам")."""
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    lo = float(np.mean(s <= null))
    hi = float(np.mean(s >= null))
    if two_sided:
        return float(min(1.0, 2.0 * min(lo, hi)))
    return hi  # P(θ* >= null): small when the effect is clearly below null


def p_from_ci(ci: CI | Mapping[str, Any], null: float = 0.0) -> float:
    """Normal-approximation two-sided p-value from an interval (fallback when bootstrap draws were not kept):
    z = (point − null) / se with se = (high − low) / (2 z_{1−α/2}). Flagged as approximate wherever it is used."""
    c = as_ci(ci)
    if not (math.isfinite(c.low) and math.isfinite(c.high)) or c.high <= c.low:
        return float("nan")
    z_level = norm.ppf(0.5 + c.level / 2)
    se = (c.high - c.low) / (2 * z_level)
    z = (c.point - null) / se
    return float(2 * norm.sf(abs(z)))
