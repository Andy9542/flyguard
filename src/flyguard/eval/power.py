"""E0, power (ТЗ Этап 0): what each source can carry, before any test label is seen.

Inputs are sizes only (positives, negatives, cluster sizes per source, pool sizes) plus validation runs; test labels
and scores never enter. The binormal model puts negatives at N(0, 1) and positives at N(μ, 1) with μ = √2 Φ⁻¹(AUC);
documents are grouped into clusters drawn from the observed cluster-size distribution and share a cluster effect, so
the cluster bootstrap on the synthetic data has the same dependence structure as the real one. Two detectors are
simulated on the same documents with correlated noise (paired design, as in the verdict rules).

Outputs: ``power_table`` -> the dict written to ``results/power.json`` with the table "источник × метрика -> несёт /
только 5% / только AUC / не хватило данных", the MDD of an AUC difference and the TOST power per source and for
macroAUC at AUC ∈ {0.75, 0.85, 0.95}, the NotInject 339-pair interval width for H2 and the empirical spread hooks
(20 curveball nulls, 10 π seeds on validation).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from flyguard.config import Configs, load_configs
from flyguard.eval.bootstrap import macro_auc_draws, percentile_ci
from flyguard.eval.thresholds import fpr_target_for_pool
from flyguard.eval.tost import equivalence_margin, tost_equivalent

CARRIES = "несёт"
ONLY_5 = "только 5%"
ONLY_AUC = "только AUC"
INSUFFICIENT = "не хватило данных"
ONLY_FPR = "только FPR"

DEFAULT_DELTA_GRID = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15)


# ----------------------------------------------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------------------------------------------
def binormal_shift(auc: float) -> float:
    """μ of the positive class in the binormal model with unit variances: AUC = Φ(μ/√2)."""
    return float(math.sqrt(2.0) * norm.ppf(float(auc)))


def _cluster_assignment(n_docs: int, cluster_sizes: Sequence[int] | None, rng: np.random.Generator,
                        prefix: str) -> np.ndarray:
    """Cluster ids for ``n_docs`` documents by resampling the observed cluster sizes with replacement (the last
    cluster is truncated to fit); without an observed distribution every document is its own cluster."""
    if n_docs <= 0:
        return np.array([], dtype=object)
    sizes = np.asarray([int(s) for s in (cluster_sizes or []) if int(s) > 0])
    if sizes.size == 0:
        return np.array([f"{prefix}{i}" for i in range(n_docs)], dtype=object)
    out: list[str] = []
    c = 0
    while len(out) < n_docs:
        size = int(sizes[rng.integers(0, sizes.size)])
        out.extend([f"{prefix}{c}"] * size)
        c += 1
    return np.array(out[:n_docs], dtype=object)


def simulate_source(auc_ref: float, delta: float, n_pos: int, n_neg: int, cluster_sizes: Sequence[int] | None,
                    rng: np.random.Generator, icc: float = 0.2, corr: float = 0.5) -> pd.DataFrame:
    """One synthetic source: reference detector at AUC ``auc_ref`` and candidate at ``auc_ref + delta`` on the same
    documents. Score = μ·y + b_cluster + ε with Var(b) = icc, Var(ε) = 1 − icc, corr(ε_ref, ε_cand) = ``corr``;
    positives and negatives live in separate clusters (labels are cluster-constant in every source, ТЗ 1.10).
    Returns ``label, cluster_id, ref, cand``."""
    mu_ref, mu_cand = binormal_shift(auc_ref), binormal_shift(min(0.999999, max(1e-6, auc_ref + delta)))
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    clusters = np.concatenate([_cluster_assignment(n_pos, cluster_sizes, rng, "p"),
                               _cluster_assignment(n_neg, cluster_sizes, rng, "n")])
    codes, uniques = pd.factorize(clusters)
    b = rng.normal(0.0, math.sqrt(max(icc, 0.0)), len(uniques))[codes] if len(uniques) else np.zeros(0)
    sd_eps = math.sqrt(max(1.0 - icc, 1e-9))
    z1 = rng.normal(size=y.size)
    z2 = corr * z1 + math.sqrt(max(1 - corr ** 2, 0.0)) * rng.normal(size=y.size)
    ref = mu_ref * y + b + sd_eps * z1
    cand = mu_cand * y + b + sd_eps * z2
    return pd.DataFrame({"label": y, "cluster_id": clusters, "ref": ref, "cand": cand})


# ----------------------------------------------------------------------------------------------------------------
# MDD and TOST power of a paired AUC / macroAUC difference
# ----------------------------------------------------------------------------------------------------------------
def _paired_diff_ci(by_source: Mapping[str, pd.DataFrame], n_boot: int, seed: int, alpha: float):
    point, draws, _ = macro_auc_draws(by_source, None, n_boot, seed, score_cols=("cand", "ref"))
    return percentile_ci(draws[:, 0] - draws[:, 1], float(point[0] - point[1]), alpha, 0)


def power_cell(sizes: Mapping[str, Mapping[str, Any]], auc_ref: float, delta_rel: float, n_rep: int = 30,
               n_boot: int = 1000, alpha: float = 0.05, power_target: float = 0.8,
               delta_grid: Sequence[float] = DEFAULT_DELTA_GRID, seed: int = 0, icc: float = 0.2,
               corr: float = 0.5, tost_level: float = 0.90) -> dict[str, Any]:
    """MDD and TOST power for one cell (a source, or macroAUC over several sources) at reference AUC ``auc_ref``.

    ``sizes[s] = {"n_pos", "n_neg", "cluster_sizes"}``. For each true difference in ``delta_grid`` the candidate is
    simulated ``n_rep`` times and the paired cluster-bootstrap (1 − alpha) interval of AUC(cand) − AUC(ref) (macroAUC
    when several sources) is computed; power(Δ) = share of replicates whose interval excludes 0; MDD = smallest grid
    Δ with power >= ``power_target`` (grid values above the first with power >= 0.99 are not simulated). TOST power
    at ±δ, δ = delta_rel × auc_ref, is the share of replicates at Δ = 0 whose ``tost_level`` interval lies inside
    the corridor. ``se_diff`` (mean bootstrap sd at Δ = 0) also gives the normal-approximation MDD
    (z_{1−α/2} + z_power)·se and TOST power 2Φ(δ/se − z_{0.95}) − 1 as a cross-check of the simulation.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE0]))
    delta = equivalence_margin(auc_ref, delta_rel)
    total_pos = sum(int(v["n_pos"]) for v in sizes.values())
    total_neg = sum(int(v["n_neg"]) for v in sizes.values())
    if total_pos == 0 or total_neg == 0 or not sizes:
        return {"auc": auc_ref, "delta": delta, "mdd": None, "tost_power": None, "se_diff": None,
                "power_curve": {}, "status": INSUFFICIENT, "n_rep": 0, "n_boot": int(n_boot)}

    def _replicates(d: float) -> list:
        cis = []
        for r in range(int(n_rep)):
            by = {s: simulate_source(auc_ref, d, int(v["n_pos"]), int(v["n_neg"]), v.get("cluster_sizes"), rng,
                                     icc=icc, corr=corr) for s, v in sizes.items()}
            by = {s: df for s, df in by.items() if len(df) and df["label"].nunique() == 2}
            if not by:
                continue
            cis.append(_paired_diff_ci(by, n_boot, int(rng.integers(0, 2 ** 31 - 1)), alpha))
        return cis

    zero = _replicates(0.0)
    if not zero:
        return {"auc": auc_ref, "delta": delta, "mdd": None, "tost_power": None, "se_diff": None,
                "power_curve": {}, "status": INSUFFICIENT, "n_rep": 0, "n_boot": int(n_boot)}
    se = float(np.mean([np.nanstd(c.samples) for c in zero]))
    tost_hits = []
    for c in zero:
        lo, hi = np.nanpercentile(c.samples, [100 * (1 - tost_level) / 2, 100 * (1 + tost_level) / 2])
        tost_hits.append(-delta <= lo and hi <= delta)
    tost_power = float(np.mean(tost_hits))
    curve: dict[str, float] = {}
    mdd = None
    for d in sorted(float(x) for x in delta_grid):
        cis = _replicates(d)
        p = float(np.mean([not c.contains(0.0) for c in cis])) if cis else float("nan")
        curve[f"{d:g}"] = p
        if mdd is None and p >= power_target:
            mdd = d
        if p >= 0.99:
            break
    z_a, z_p = norm.ppf(1 - alpha / 2), norm.ppf(power_target)
    z_t = norm.ppf(0.5 + tost_level / 2)
    return {
        "auc": float(auc_ref), "delta": float(delta), "mdd": mdd, "tost_power": tost_power, "se_diff": se,
        "mdd_normal_approx": float((z_a + z_p) * se) if se > 0 else 0.0,
        "tost_power_normal_approx": float(max(0.0, 2 * norm.cdf(delta / se - z_t) - 1)) if se > 0 else 1.0,
        "power_curve": curve, "n_rep": int(len(zero)), "n_boot": int(n_boot),
        "status": CARRIES if tost_power >= power_target else INSUFFICIENT,
    }


# ----------------------------------------------------------------------------------------------------------------
# Rules that need no simulation
# ----------------------------------------------------------------------------------------------------------------
def carrier_rule(n_pool: int, cfg: Configs | None = None) -> tuple[str, float | None]:
    """ТЗ Этап 0: |P_test| >= 2000 -> TPR@1 % ("несёт"), 500–1999 -> "только 5%", else "только AUC". Returns the
    status and the FPR target (None when only AUC is reported). The numbers come from ``thresholds.pool_min_docs``."""
    fpr = fpr_target_for_pool(int(n_pool), cfg)
    cfg = cfg or load_configs()
    primary = float(cfg.default["thresholds"]["fpr_targets"]["primary"])
    if fpr is None:
        return ONLY_AUC, None
    return (CARRIES if fpr == primary else ONLY_5), fpr


def proportion_diff_ci_width(n: int, p1: float, p2: float, rho: float = 0.0, level: float = 0.95) -> float:
    """Width of the (1 − level) Wald interval of a difference of two proportions measured on the same ``n`` units
    (H2: FPR of two detectors on the 339 NotInject prompts). Var = [p1(1−p1) + p2(1−p2) − 2ρ√(p1(1−p1)p2(1−p2))]/n;
    ρ = 0 (independent alarms) is the conservative case for the positively correlated detectors expected here."""
    v1, v2 = p1 * (1 - p1), p2 * (1 - p2)
    var = (v1 + v2 - 2 * rho * math.sqrt(v1 * v2)) / n
    return float(2 * norm.ppf(0.5 + level / 2) * math.sqrt(max(var, 0.0)))


def notinject_table(n: int, margin_pp: float, fprs: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
                    level: float = 0.95) -> dict[str, Any]:
    """H2 planning table (ТЗ Этап 0 "ширина ДИ разности долей на 339 парных наблюдениях"): for equal FPR scenarios
    the interval width and the largest observed |ΔFPR| still compatible with the ±margin corridor
    (margin − width/2); negative means the corridor cannot be reached at that FPR even with a zero point estimate."""
    rows = {}
    m = margin_pp / 100.0
    for p in fprs:
        w = proportion_diff_ci_width(n, p, p, level=level)
        rows[f"{p:g}"] = {"ci_width": w, "max_abs_diff_for_corridor": m - w / 2}
    worst = proportion_diff_ci_width(n, 0.5, 0.5, level=level)
    return {"n": int(n), "margin_pp": float(margin_pp), "level": level, "by_fpr": rows,
            "worst_case_width": worst, "corridor_reachable_worst_case": bool(m - worst / 2 >= 0)}


def spread(values: Sequence[float] | np.ndarray | None, delta: float | None = None) -> dict[str, Any] | None:
    """Empirical spread hook (ТЗ Этап 0: "разброс по нулевым матрицам по 20 curveball ... по перестановкам perm по 10
    сидам"): mean, sd, range and half-width 1.96·sd of validation macroAUCs; with ``delta`` also sd/δ, the share
    of the corridor eaten by that variance source."""
    if values is None:
        return None
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if x.size == 0:
        return None
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    out = {"n": int(x.size), "mean": float(np.mean(x)), "sd": sd, "min": float(x.min()), "max": float(x.max()),
           "half_width_95": 1.96 * sd}
    if delta:
        out["sd_over_delta"] = sd / float(delta)
    return out


# ----------------------------------------------------------------------------------------------------------------
# The E0 table
# ----------------------------------------------------------------------------------------------------------------
def _nearest_level(levels: Sequence[float], val_auc: float | None) -> float:
    if val_auc is None or not math.isfinite(float(val_auc)):
        return float(sorted(levels)[len(levels) // 2])
    return float(min(levels, key=lambda a: abs(a - float(val_auc))))


def power_table(cfg: Configs | None, sizes_by_source: Mapping[str, Mapping[str, Any]],
                val_runs: Mapping[str, Any] | None = None, pools: Mapping[str, int] | None = None,
                seed: int = 0, n_rep: int = 30, n_boot: int | None = None, icc: float = 0.2, corr: float = 0.5,
                delta_grid: Sequence[float] = DEFAULT_DELTA_GRID, power_target: float = 0.8,
                min_class_docs: int = 10) -> dict[str, Any]:
    """The E0 result (ТЗ Этап 0) for ``results/power.json``.

    ``sizes_by_source[s] = {"n_pos", "n_neg", "cluster_sizes": [...]}`` for the test sources (deep, bipia, dojo,
    dyn, para; notinject with ``n_pos = 0``); ``pools = {"P_test": n, "P_val": n}``; ``val_runs`` may hold
    ``"val_auc": {source: AUC of the reference detector on validation}`` (picks the planning AUC level per source),
    ``"curveball_val_macro_auc": [20 values]`` and ``"perm_val_macro_auc": [10 values]`` (empirical spread hooks).
    Levels, δ, α, the 339 NotInject pairs and the bootstrap size come from ``configs/experiments/E0.yaml`` and
    ``configs/default.yaml``; ``icc``/``corr`` are the simulation's dependence assumptions (recorded in the output).
    Returns::

        {"levels", "delta_rel", "alpha", "sizes", "pools", "fpr_target",
         "cells": {source|"macro": {level: power_cell}}, "planning_level": {source: level},
         "carriers": {source|"macro": {"auc": status, "auc_diff": status, "tpr_at_fpr": status}},
         "notinject": notinject_table, "spread": {"curveball", "perm"}, "hypotheses": {H1a, H1b, H2, H3},
         "assumptions": {...}}

    ``carriers`` is the table "источник × метрика -> несёт / только 5% / только AUC / не хватило данных" the
    verdicts read: ``auc`` needs both classes (>= ``min_class_docs`` each), ``auc_diff`` needs TOST power >=
    ``power_target`` at the planning level, ``tpr_at_fpr`` follows the |P_test| rule.
    """
    cfg = cfg or load_configs()
    e0 = cfg.exp("E0")
    levels = [float(a) for a in e0["aucs"]]
    delta_rel = float(cfg.default["stats"]["tost"]["delta_rel"])
    alpha = float(cfg.default["stats"]["bootstrap"]["alpha"])
    tost_level = float(cfg.default["stats"]["tost"]["ci"])
    margin_pp = float(cfg.default["stats"]["h2_margin_pp"])
    n_boot = int(e0.get("synthetic_bootstrap", cfg.default["stats"]["bootstrap"]["n"])) if n_boot is None else n_boot
    val_runs = val_runs or {}
    pools = dict(pools or {})
    val_auc = dict(val_runs.get("val_auc") or {})

    n_test_pool = int(pools.get("P_test", 0))
    tpr_status, fpr_target = carrier_rule(n_test_pool, cfg)
    sizes = {s: {"n_pos": int(v["n_pos"]), "n_neg": int(v["n_neg"]),
                 "n_clusters": int(len(v.get("cluster_sizes") or [])) or None} for s, v in sizes_by_source.items()}
    cells: dict[str, dict[str, Any]] = {}
    carriers: dict[str, dict[str, str]] = {}
    planning: dict[str, float] = {}
    auc_sources = {s: v for s, v in sizes_by_source.items() if int(v["n_pos"]) > 0 and int(v["n_neg"]) > 0}
    cell_seed = 0
    for s, v in sizes_by_source.items():
        n_pos, n_neg = int(v["n_pos"]), int(v["n_neg"])
        if n_pos == 0:
            carriers[s] = {"auc": ONLY_FPR, "auc_diff": ONLY_FPR, "tpr_at_fpr": ONLY_FPR}
            continue
        level = _nearest_level(levels, val_auc.get(s))
        planning[s] = level
        cells[s] = {}
        for a in levels:
            cell_seed += 1
            cells[s][f"{a:g}"] = power_cell({s: v}, a, delta_rel, n_rep=n_rep, n_boot=n_boot, alpha=alpha,
                                            power_target=power_target, delta_grid=delta_grid,
                                            seed=int(seed) * 1000 + cell_seed, icc=icc, corr=corr,
                                            tost_level=tost_level)
        enough = n_pos >= min_class_docs and n_neg >= min_class_docs
        carriers[s] = {
            "auc": CARRIES if enough else INSUFFICIENT,
            "auc_diff": cells[s][f"{level:g}"]["status"] if enough else INSUFFICIENT,
            "tpr_at_fpr": tpr_status if enough else INSUFFICIENT,
        }
    if auc_sources:
        macro_level = _nearest_level(levels, np.mean([val_auc[s] for s in auc_sources if s in val_auc])
                                     if any(s in val_auc for s in auc_sources) else None)
        planning["macro"] = macro_level
        cells["macro"] = {}
        for a in levels:
            cell_seed += 1
            cells["macro"][f"{a:g}"] = power_cell(auc_sources, a, delta_rel, n_rep=n_rep, n_boot=n_boot,
                                                  alpha=alpha, power_target=power_target, delta_grid=delta_grid,
                                                  seed=int(seed) * 1000 + cell_seed, icc=icc, corr=corr,
                                                  tost_level=tost_level)
        carriers["macro"] = {"auc": CARRIES, "auc_diff": cells["macro"][f"{macro_level:g}"]["status"],
                             "tpr_at_fpr": tpr_status}

    def _cell(name: str) -> dict[str, Any] | None:
        lvl = planning.get(name)
        return cells.get(name, {}).get(f"{lvl:g}") if lvl is not None else None

    delta_macro = (_cell("macro") or {}).get("delta")
    notinj_n = int(e0.get("notinject_pairs", sizes.get("notinject", {}).get("n_neg", 339)))
    out: dict[str, Any] = {
        "levels": levels, "delta_rel": delta_rel, "alpha": alpha, "tost_level": tost_level,
        "power_target": power_target, "sizes": sizes, "pools": pools, "fpr_target": fpr_target,
        "planning_level": planning, "cells": cells, "carriers": carriers,
        "notinject": notinject_table(notinj_n, margin_pp),
        "spread": {"curveball": spread(val_runs.get("curveball_val_macro_auc"), delta_macro),
                   "perm": spread(val_runs.get("perm_val_macro_auc"), delta_macro)},
        "assumptions": {"icc": icc, "detector_corr": corr, "n_rep": n_rep, "n_boot": n_boot,
                        "delta_grid": [float(d) for d in delta_grid], "min_class_docs": min_class_docs,
                        "model": "binormal, unit variances, cluster effect shared by both detectors"},
    }
    out["hypotheses"] = {
        "H1a": {"template": {s: carriers.get(s, {}).get("auc_diff", INSUFFICIENT) for s in ("deep", "dojo")},
                "semantic_mdd": {s: (_cell(s) or {}).get("mdd") for s in ("para", "bipia", "dyn") if s in cells}},
        "H1b": {"macro": carriers.get("macro", {}).get("auc_diff", INSUFFICIENT),
                "macro_cell": _cell("macro")},
        "H2": {"notinject": out["notinject"]},
        "H3": {"macro": carriers.get("macro", {}).get("auc_diff", INSUFFICIENT), "macro_cell": _cell("macro"),
               "spread": out["spread"]},
    }
    return out
