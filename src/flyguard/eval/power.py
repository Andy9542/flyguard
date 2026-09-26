"""E0, power (ТЗ Этап 0): what each source can carry, before any test label is seen.

Inputs are sizes only (positives, negatives, cluster composition per source, pool sizes) plus validation runs; test
labels and scores never enter. The binormal model puts negatives at N(0, 1) and positives at N(μ, 1) with
μ = √2 Φ⁻¹(AUC); documents are grouped into clusters drawn from the *observed* clusters and share a cluster effect,
so the cluster bootstrap on the synthetic data has the same dependence structure as the real one. That structure
differs by source (docs/design.md §2, ТЗ 1.4, 1.10): a deepset document is its own cluster and a paraphrase base
carries one label, but a BIPIA pair (clean 0 + attacked 1) shares its ``cluster_id`` and an AgentDojo / AgentDyn
task cluster holds clean and injected steps. Hence two cluster inputs: ``cluster_label_sizes`` = the observed
``(n_pos, n_neg)`` of every cluster (mixed clusters are resampled as they are), or ``cluster_sizes`` = label-constant
clusters of the given sizes (deep, para; also the fallback when only sizes are known). Two detectors are simulated on
the same documents with correlated noise (paired design, as in the verdict rules).

Every simulation constant (ICC, detector correlation, replicates, Δ grid, power target, minimum documents per class)
is read from ``stats.power`` of configs/default.yaml, the bootstrap size from ``configs/experiments/E0.yaml``
(``smoke.power`` in smoke mode); keyword arguments override them and the resolved values are written to
``assumptions`` of the result, so the frozen config hash describes what ran.

Outputs: ``power_table`` -> the dict written to ``results/power.json`` with the table "источник × метрика -> несёт /
только 5% / только AUC / не хватило данных", the MDD of an AUC difference and the TOST power per source and for
macroAUC at AUC ∈ {0.75, 0.85, 0.95}, the NotInject 339-pair interval width for H2 and the empirical spread hooks
(20 curveball nulls, 10 π seeds on validation).
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from flyguard.config import Configs, load_configs
from flyguard.eval.bootstrap import macro_auc_draws, percentile_ci
from flyguard.eval.thresholds import fpr_target_for_pool
from flyguard.eval.tost import equivalence_margin

CARRIES = "несёт"
ONLY_5 = "только 5%"
ONLY_AUC = "только AUC"
INSUFFICIENT = "не хватило данных"
ONLY_FPR = "только FPR"

CLUSTER_MODEL_OBSERVED = "observed_label_composition"
CLUSTER_MODEL_CONSTANT = "label_constant_clusters"
CLUSTER_MODEL_INDEPENDENT = "independent_documents"


# ----------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------
def power_params(cfg: Configs | None = None, smoke: bool = False, **overrides: Any) -> dict[str, Any]:
    """The E0 simulation constants: ``stats.power`` (icc, detector_corr -> ``corr``, n_rep, delta_grid,
    power_target, min_class_docs), ``stats.bootstrap.alpha`` (``alpha``), ``stats.tost.ci`` (``tost_level``) and the
    synthetic bootstrap size ``n_boot`` (``E0.synthetic_bootstrap``, else ``stats.bootstrap.n``); ``smoke=True``
    takes ``n_rep``/``n_boot`` from ``smoke.power``. Keyword ``overrides`` that are not ``None`` win, so callers can
    still shrink a cell for tests while the defaults stay in the frozen config."""
    cfg = cfg or load_configs()
    st = cfg.default["stats"]
    p = st["power"]
    out: dict[str, Any] = {
        "icc": float(p["icc"]), "corr": float(p["detector_corr"]), "n_rep": int(p["n_rep"]),
        "delta_grid": tuple(float(d) for d in p["delta_grid"]), "power_target": float(p["power_target"]),
        "min_class_docs": int(p["min_class_docs"]), "alpha": float(st["bootstrap"]["alpha"]),
        "tost_level": float(st["tost"]["ci"]),
        "n_boot": int(cfg.exp("E0").get("synthetic_bootstrap", st["bootstrap"]["n"])),
    }
    if smoke:
        sp = cfg.default["smoke"]["power"]
        out["n_rep"], out["n_boot"] = int(sp["n_rep"]), int(sp["n_boot"])
    for k, v in overrides.items():
        if k not in out:
            raise TypeError(f"unknown power parameter {k!r}")
        if v is not None:
            out[k] = tuple(float(d) for d in v) if k == "delta_grid" else type(out[k])(v)
    return out


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


def _mixed_cluster_assignment(n_pos: int, n_neg: int, cluster_label_sizes: Sequence[Sequence[int]],
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Labels and cluster ids by resampling the observed clusters *with their label composition*: each drawn
    cluster contributes its ``n_pos_c`` positives and ``n_neg_c`` negatives (a BIPIA pair stays a pair, a task
    cluster keeps its clean/injected mix) until both class totals are reached; the surplus documents of the last
    clusters are dropped so the totals are exact. Returns ``(labels, cluster_ids)``."""
    comp = np.asarray([[int(p), int(q)] for p, q in cluster_label_sizes if int(p) + int(q) > 0], dtype=int)
    if comp.size == 0:
        raise ValueError("cluster_label_sizes holds no non-empty cluster")
    if (n_pos > 0 and comp[:, 0].sum() == 0) or (n_neg > 0 and comp[:, 1].sum() == 0):
        raise ValueError("cluster_label_sizes is inconsistent with n_pos/n_neg: a class is requested that no "
                         "observed cluster contains")
    labels: list[int] = []
    clusters: list[str] = []
    got_pos = got_neg = 0
    c = 0
    while got_pos < n_pos or got_neg < n_neg:
        p, q = (int(v) for v in comp[rng.integers(0, comp.shape[0])])
        p, q = min(p, n_pos - got_pos), min(q, n_neg - got_neg)
        if p + q == 0:
            continue
        labels.extend([1] * p + [0] * q)
        clusters.extend([f"m{c}"] * (p + q))
        got_pos, got_neg, c = got_pos + p, got_neg + q, c + 1
    return np.asarray(labels, dtype=int), np.asarray(clusters, dtype=object)


def cluster_model(spec: Mapping[str, Any]) -> str:
    """Which dependence model a ``sizes_by_source`` entry selects (recorded in ``assumptions``)."""
    if spec.get("cluster_label_sizes"):
        return CLUSTER_MODEL_OBSERVED
    if spec.get("cluster_sizes"):
        return CLUSTER_MODEL_CONSTANT
    return CLUSTER_MODEL_INDEPENDENT


def simulate_source(auc_ref: float, delta: float, n_pos: int, n_neg: int, cluster_sizes: Sequence[int] | None,
                    rng: np.random.Generator, icc: float = 0.2, corr: float = 0.5,
                    cluster_label_sizes: Sequence[Sequence[int]] | None = None) -> pd.DataFrame:
    """One synthetic source: reference detector at AUC ``auc_ref`` and candidate at ``auc_ref + delta`` on the same
    documents. Score = μ·y + b_cluster + ε with Var(b) = icc, Var(ε) = 1 − icc, corr(ε_ref, ε_cand) = ``corr``.

    Clusters: with ``cluster_label_sizes`` (``[(n_pos_c, n_neg_c), ...]`` of the observed clusters) they are resampled
    with their label mix, which is the real structure of bipia (pairs), dojo and dyn (task clusters holding clean
    and injected steps); otherwise positives and negatives live in separate clusters of the ``cluster_sizes``
    distribution (deep: singletons; para: one label per base). Measured effect of the choice (this model, 40
    replicates × 200 draws, n = 150 + 150): for the *paired* difference AUC(cand) − AUC(ref) at Δ = 0 the two cluster
    models give the same bootstrap sd (ratio 0.99–1.02 at ICC 0, 0.2, 0.5), because the cluster effect is shared
    by both detectors and cancels in the difference, so MDD and TOST power of E0 barely depend on it; for a
    *single-detector* AUC interval label-constant clusters overstate the sd of a mixed-cluster source by 16–53 %
    at ICC 0.2–0.5 (a mixed cluster's effect cannot reorder its own positives and negatives). The observed
    composition is therefore the faithful model and the label-constant fallback a conservative one for anything
    built on single-detector intervals. Returns ``label, cluster_id, ref, cand``."""
    mu_ref, mu_cand = binormal_shift(auc_ref), binormal_shift(min(0.999999, max(1e-6, auc_ref + delta)))
    if cluster_label_sizes:
        y, clusters = _mixed_cluster_assignment(int(n_pos), int(n_neg), cluster_label_sizes, rng)
    else:
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


def _insufficient_cell(auc_ref: float, delta: float, n_boot: int) -> dict[str, Any]:
    return {"auc": float(auc_ref), "delta": float(delta), "mdd": None, "tost_power": None, "se_diff": None,
            "power_curve": {}, "status": INSUFFICIENT, "n_rep": 0, "n_boot": int(n_boot)}


def power_cell(sizes: Mapping[str, Mapping[str, Any]], auc_ref: float, delta_rel: float, n_rep: int | None = None,
               n_boot: int | None = None, alpha: float | None = None, power_target: float | None = None,
               delta_grid: Sequence[float] | None = None, seed: int = 0, icc: float | None = None,
               corr: float | None = None, tost_level: float | None = None,
               cfg: Configs | None = None) -> dict[str, Any]:
    """MDD and TOST power for one cell (a source, or macroAUC over several sources) at reference AUC ``auc_ref``.

    ``sizes[s] = {"n_pos", "n_neg", "cluster_label_sizes" | "cluster_sizes"}``. For each true difference in
    ``delta_grid`` the candidate is simulated ``n_rep`` times and the paired cluster-bootstrap (1 − alpha) interval of
    AUC(cand) − AUC(ref) (macroAUC when several sources) is computed; power(Δ) = share of replicates whose interval
    excludes 0; MDD = smallest grid Δ with power >= ``power_target`` (grid values above the first with power >= 0.99
    are not simulated). TOST power at ±δ, δ = delta_rel × auc_ref, is the share of replicates at Δ = 0 whose
    ``tost_level`` interval lies inside the corridor. ``se_diff`` (mean bootstrap sd at Δ = 0) also gives the
    normal-approximation MDD (z_{1−α/2} + z_power)·se and TOST power 2Φ(δ/se − z_{0.95}) − 1 as a cross-check of the
    simulation. ``None`` arguments come from :func:`power_params` (configs/default.yaml). Sources are simulated in
    sorted order from one stream, so the result does not depend on the order of ``sizes``.
    """
    p = power_params(cfg, n_rep=n_rep, n_boot=n_boot, alpha=alpha, power_target=power_target, delta_grid=delta_grid,
                     icc=icc, corr=corr, tost_level=tost_level)
    n_rep, n_boot, alpha, power_target = p["n_rep"], p["n_boot"], p["alpha"], p["power_target"]
    delta_grid, icc, corr, tost_level = p["delta_grid"], p["icc"], p["corr"], p["tost_level"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE0]))
    delta = equivalence_margin(auc_ref, delta_rel)
    total_pos = sum(int(v["n_pos"]) for v in sizes.values())
    total_neg = sum(int(v["n_neg"]) for v in sizes.values())
    if total_pos == 0 or total_neg == 0 or not sizes:
        return _insufficient_cell(auc_ref, delta, n_boot)
    names = sorted(sizes)

    def _replicates(d: float) -> list:
        cis = []
        for _ in range(int(n_rep)):
            by = {}
            for s in names:
                v = sizes[s]
                df = simulate_source(auc_ref, d, int(v["n_pos"]), int(v["n_neg"]), v.get("cluster_sizes"), rng,
                                     icc=icc, corr=corr, cluster_label_sizes=v.get("cluster_label_sizes"))
                if len(df) and df["label"].nunique() == 2:
                    by[s] = df
            if not by:
                continue
            cis.append(_paired_diff_ci(by, n_boot, int(rng.integers(0, 2 ** 31 - 1)), alpha))
        return cis

    zero = _replicates(0.0)
    if not zero:
        return _insufficient_cell(auc_ref, delta, n_boot)
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
        pw = float(np.mean([not c.contains(0.0) for c in cis])) if cis else float("nan")
        curve[f"{d:g}"] = pw
        if mdd is None and pw >= power_target:
            mdd = d
        if pw >= 0.99:
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


def spread(values: Sequence[float] | np.ndarray | None, delta: float | None = None,
           level: float = 0.95) -> dict[str, Any] | None:
    """Empirical spread hook (ТЗ Этап 0: "разброс по нулевым матрицам по 20 curveball ... по перестановкам perm по 10
    сидам"): mean, sd, range and the normal half-width z_{(1+level)/2}·sd of validation macroAUCs; with ``delta``
    also sd/δ, the share of the corridor eaten by that variance source."""
    if values is None:
        return None
    x = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if x.size == 0:
        return None
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    out = {"n": int(x.size), "mean": float(np.mean(x)), "sd": sd, "min": float(x.min()), "max": float(x.max()),
           "level": float(level), "half_width": float(norm.ppf(0.5 + level / 2) * sd)}
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


def cell_seed(seed: int, name: str, level: float) -> int:
    """Seed of one E0 cell derived from the global seed, the cell name and the AUC level (not from insertion order),
    so the numbers of a source do not change when another source is added or the caller reorders its dict."""
    ss = np.random.SeedSequence([int(seed), zlib.crc32(str(name).encode("utf-8")), int(round(float(level) * 1000))])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _n_clusters(spec: Mapping[str, Any]) -> int | None:
    comp = spec.get("cluster_label_sizes")
    if comp:
        return int(sum(1 for p, q in comp if int(p) + int(q) > 0))
    sizes = spec.get("cluster_sizes")
    return int(len(sizes)) if sizes else None


def power_table(cfg: Configs | None, sizes_by_source: Mapping[str, Mapping[str, Any]],
                val_runs: Mapping[str, Any] | None = None, pools: Mapping[str, int] | None = None,
                seed: int = 0, n_rep: int | None = None, n_boot: int | None = None, icc: float | None = None,
                corr: float | None = None, delta_grid: Sequence[float] | None = None,
                power_target: float | None = None, min_class_docs: int | None = None,
                smoke: bool = False) -> dict[str, Any]:
    """The E0 result (ТЗ Этап 0) for ``results/power.json``.

    ``sizes_by_source[s] = {"n_pos", "n_neg", "cluster_label_sizes": [[n_pos_c, n_neg_c], ...]}`` for the test
    sources (deep, bipia, dojo, dyn, para; notinject with ``n_pos = 0``), the composition of every observed cluster
    (``documents.groupby("cluster_id")["label"]`` -> sum and count − sum); ``"cluster_sizes": [...]`` instead selects
    label-constant clusters (see :func:`simulate_source`). ``pools = {"P_test": n, "P_val": n}``; ``val_runs`` may
    hold ``"val_auc": {source: AUC of the reference detector on validation}`` (picks the planning AUC level per
    source), ``"curveball_val_macro_auc": [20 values]`` and ``"perm_val_macro_auc": [10 values]`` (empirical spread
    hooks). Levels, δ, α, the 339 NotInject pairs and the bootstrap size come from ``configs/experiments/E0.yaml``
    and ``configs/default.yaml``; ``icc``/``corr``/``n_rep``/``delta_grid``/``power_target``/``min_class_docs``
    default to ``stats.power`` (``smoke=True``: ``smoke.power`` for ``n_rep``/``n_boot``) and are recorded in
    ``assumptions``. Returns::

        {"levels", "delta_rel", "alpha", "sizes", "pools", "fpr_target",
         "cells": {source|"macro": {level: power_cell}}, "planning_level": {source: level},
         "carriers": {source|"macro": {"auc": status, "auc_diff": status, "tpr_at_fpr": status}},
         "notinject": notinject_table, "spread": {"curveball", "perm"}, "hypotheses": {H1a, H1b, H2, H3},
         "assumptions": {...}}

    ``carriers`` is the table "источник × метрика -> несёт / только 5% / только AUC / не хватило данных" the
    verdicts read: ``auc`` needs both classes (>= ``min_class_docs`` each), ``auc_diff`` needs TOST power >=
    ``power_target`` at the planning level, ``tpr_at_fpr`` follows the |P_test| rule; the ``macro`` row sums the
    sources with both classes and carries AUC only when at least one of them does.
    """
    cfg = cfg or load_configs()
    p = power_params(cfg, smoke=smoke, n_rep=n_rep, n_boot=n_boot, icc=icc, corr=corr, delta_grid=delta_grid,
                     power_target=power_target, min_class_docs=min_class_docs)
    n_rep, n_boot, icc, corr = p["n_rep"], p["n_boot"], p["icc"], p["corr"]
    delta_grid, power_target, min_class_docs = p["delta_grid"], p["power_target"], p["min_class_docs"]
    alpha, tost_level = p["alpha"], p["tost_level"]
    e0 = cfg.exp("E0")
    levels = [float(a) for a in e0["aucs"]]
    delta_rel = float(cfg.default["stats"]["tost"]["delta_rel"])
    margin_pp = float(cfg.default["stats"]["h2_margin_pp"])
    ci_level = 1.0 - alpha
    val_runs = val_runs or {}
    pools = dict(pools or {})
    val_auc = dict(val_runs.get("val_auc") or {})

    n_test_pool = int(pools.get("P_test", 0))
    tpr_status, fpr_target = carrier_rule(n_test_pool, cfg)
    names = sorted(sizes_by_source)
    sizes = {s: {"n_pos": int(sizes_by_source[s]["n_pos"]), "n_neg": int(sizes_by_source[s]["n_neg"]),
                 "n_clusters": _n_clusters(sizes_by_source[s])} for s in names}
    models = {s: cluster_model(sizes_by_source[s]) for s in names}
    cells: dict[str, dict[str, Any]] = {}
    carriers: dict[str, dict[str, str]] = {}
    planning: dict[str, float] = {}
    auc_sources = {s: sizes_by_source[s] for s in names if sizes[s]["n_pos"] > 0 and sizes[s]["n_neg"] > 0}
    enough_by_source: dict[str, bool] = {}
    for s in names:
        v = sizes_by_source[s]
        n_pos, n_neg = sizes[s]["n_pos"], sizes[s]["n_neg"]
        if n_pos == 0:
            carriers[s] = {"auc": ONLY_FPR, "auc_diff": ONLY_FPR, "tpr_at_fpr": ONLY_FPR}
            continue
        level = _nearest_level(levels, val_auc.get(s))
        planning[s] = level
        cells[s] = {}
        for a in levels:
            cells[s][f"{a:g}"] = power_cell({s: v}, a, delta_rel, n_rep=n_rep, n_boot=n_boot, alpha=alpha,
                                            power_target=power_target, delta_grid=delta_grid,
                                            seed=cell_seed(seed, s, a), icc=icc, corr=corr,
                                            tost_level=tost_level, cfg=cfg)
        enough = n_pos >= min_class_docs and n_neg >= min_class_docs
        enough_by_source[s] = enough
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
            cells["macro"][f"{a:g}"] = power_cell(auc_sources, a, delta_rel, n_rep=n_rep, n_boot=n_boot,
                                                  alpha=alpha, power_target=power_target, delta_grid=delta_grid,
                                                  seed=cell_seed(seed, "macro", a), icc=icc, corr=corr,
                                                  tost_level=tost_level, cfg=cfg)
        any_enough = any(enough_by_source.get(s, False) for s in auc_sources)
        carriers["macro"] = {"auc": CARRIES if any_enough else INSUFFICIENT,
                             "auc_diff": cells["macro"][f"{macro_level:g}"]["status"] if any_enough else INSUFFICIENT,
                             "tpr_at_fpr": tpr_status if any_enough else INSUFFICIENT}
        macro_clusters = [sizes[s]["n_clusters"] for s in auc_sources]
        sizes["macro"] = {"n_pos": int(sum(sizes[s]["n_pos"] for s in auc_sources)),
                          "n_neg": int(sum(sizes[s]["n_neg"] for s in auc_sources)),
                          "n_clusters": int(sum(c for c in macro_clusters if c is not None))
                          if all(c is not None for c in macro_clusters) else None,
                          "sources": sorted(auc_sources)}

    def _cell(name: str) -> dict[str, Any] | None:
        lvl = planning.get(name)
        return cells.get(name, {}).get(f"{lvl:g}") if lvl is not None else None

    delta_macro = (_cell("macro") or {}).get("delta")
    notinj_n = int(e0.get("notinject_pairs", sizes.get("notinject", {}).get("n_neg", 339)))
    out: dict[str, Any] = {
        "levels": levels, "delta_rel": delta_rel, "alpha": alpha, "tost_level": tost_level,
        "power_target": power_target, "sizes": sizes, "pools": pools, "fpr_target": fpr_target,
        "planning_level": planning, "cells": cells, "carriers": carriers,
        "notinject": notinject_table(notinj_n, margin_pp, level=ci_level),
        "spread": {"curveball": spread(val_runs.get("curveball_val_macro_auc"), delta_macro, level=ci_level),
                   "perm": spread(val_runs.get("perm_val_macro_auc"), delta_macro, level=ci_level)},
        "assumptions": {"icc": icc, "detector_corr": corr, "n_rep": n_rep, "n_boot": n_boot,
                        "delta_grid": [float(d) for d in delta_grid], "power_target": power_target,
                        "min_class_docs": min_class_docs, "smoke": bool(smoke),
                        "config_source": "stats.power / stats.bootstrap / stats.tost / experiments/E0",
                        "cluster_model": models,
                        "model": "binormal, unit variances, cluster effect shared by both detectors; clusters "
                                 "resampled from the observed clusters with their label composition "
                                 f"({CLUSTER_MODEL_OBSERVED}) or label-constant clusters of the observed sizes "
                                 f"({CLUSTER_MODEL_CONSTANT}; same sd of the paired AUC difference, larger sd of "
                                 "a single-detector AUC for mixed-cluster sources, see simulate_source)"},
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
