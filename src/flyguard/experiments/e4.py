"""E4 -- wiring: the measured MaleCNS matrix against its null models (ТЗ Этап 4 "E4", 3.3 "Нулевые модели
проводки", "Вклады", "Статистика"; docs/design_experiments.md §3). Produces the inputs of H3.

Setting: E1 (train = deepset train windows, every hyperparameter on deepset validation, test = every positive test
source), the real fly's nose N51-svd with its permutation π, k-WTA at 5 % and both readouts. Matrices
(``configs/experiments/E4.yaml: nulls``): the measured M, the ``n_null`` Curveball shuffles (seed ``curveball``; both
degree sequences preserved; ``Context.n_null()`` is smoke-aware, and a real run refuses to start when
``E4.yaml: n_curveball`` differs from it -- :func:`check_n_curveball`), the random matrix of the same density (seed
``projection``) and the dense Gaussian sign code with the same number of multiply-adds (linear readout only, ТЗ 3.3).
Readouts (``E4.yaml: readouts``): primary ``bloom_full``, secondary ``linear`` and ``bloom_10shot`` (10 training
documents per class, ``FeatureContext.fewshot_set(10, rep=0)``, γ chosen on validation separately). The measured
detectors are named ``real_fly_bloom``, ``real_fly_linear``, ``real_fly_bloom_10shot``; the random ones
``random_fly_*``; the sign code ``dense_sign_linear``; the Curveball family is the pseudo-detector ``curveball_mean``.

Two views of the same protocol
------------------------------
1. *Own π* (the seed's ``perm`` child): measured, random and sign-code detectors go through the engine
   (:func:`flyguard.experiments.engine.standard_evaluation`): ``auc/<src>/<det>``, ``macro_auc/<det>``, paired
   ``diff/<metric>/real_fly_*-random_fly_*`` and ``diff/<metric>/real_fly_linear-dense_sign_linear`` (95 %, and
   ``diff90/...``), ``val_auc/<src>/<det>`` and ``val_macro_auc/real_fly_bloom`` (the H3 precondition).
2. *π grid*: for every permutation π_p of the ten global seeds (their ``perm`` children -- the seed's own π is one of
   them; ТЗ 2.1: "без разыгрывания π результат H3 зависел бы от одного неявного выбора") and every matrix (measured,
   the J nulls, random; the sign code for the linear readout) the readouts are refitted with the same protocol and
   scored on the test documents -> one score table ``[n_docs]`` per (π, matrix, readout, source). π acts on the nose
   output only, so U_π is a column permutation of the seed's N51-svd features (:func:`permute_columns`; the SVD is
   fitted once per seed on C_unl). The own-π tables must coincide with the engine's detectors; the run raises if not.

Statistics (ТЗ Этап 4 "Статистика"), ``<det>`` = the measured detector of a readout, ``<metric>`` in ``macro_auc``,
``auc/<src>``
-------------------------------------------------------------------------------------------------------------------
* Two-stage bootstrap (:func:`flyguard.eval.bootstrap.two_stage_bootstrap_h3`): clusters within source × the J null
  matrices × the P permutations. ``diff90/macro_auc/<det>-curveball_mean`` = macroAUC(measured, π-averaged) −
  macroAUC(nulls, π- and matrix-averaged) with its 90 % interval (TOST level) and the extras ``reference`` (the null
  mean, the reference of the ±δ corridor), ``p_randomization``, ``delta``, ``tost_equivalent``, ``n_perms``,
  ``n_null``; ``diff/macro_auc/<det>-curveball_mean`` the same draws at 95 %; ``macro_auc/curveball_mean/<det>`` and
  ``macro_auc/pi_mean/<det>`` the null mean and the π-averaged measured macroAUC with their intervals;
  ``tost_delta/macro_auc/<det>-curveball_mean`` (δ = ``stats.tost.delta_rel`` × null mean) and
  ``tost_equivalent/macro_auc/<det>-curveball_mean`` (1 = the 90 % interval lies inside ±δ).
* Randomisation test: two-sided p of the measured statistic against the J Curveball statistics around their mean
  (:func:`flyguard.eval.tost.randomization_p`). Canonical statistic = the π-averaged AUC (the quantity whose interval
  the TOST uses): ``p_randomization/<metric>/<det>``; the single-π version at the own π is
  ``p_randomization_ownperm/<metric>/<det>``. Holm (``p_randomization_holm/...``) runs over the secondary family =
  every (readout, metric) except the primary (``real_fly_bloom``, ``macro_auc``). With J nulls the smallest
  attainable p is 1/(J+1), so an adjusted p in a family of m tests cannot fall below min(1, m/(J+1)): a property of
  the design, stated in ``notes``.
* Contribution "проводка" (ТЗ "Вклады": AUC(измеренная) − среднее AUC(curveball)) at the own π with the matrices
  fixed and a cluster-bootstrap interval: ``contrib/wiring/<det>/<metric>`` (``p`` = the own-π randomisation p).
* Spread (E0 style): ``curveball_mean_ownperm/<metric>/<det>`` and ``curveball_sd/<metric>/<det>`` over the J nulls
  at the own π; ``pi_sd/macro_auc/<det>`` and ``pi_sd/macro_auc/curveball_mean/<det>`` over the P permutations.

Tables: ``curveball`` (histogram data of the primary readout at the own π, one row per null: ``j``, ``matrix``,
``macro_auc``, ``auc_<src>``), ``nulls_secondary`` (the same for the secondary readouts), ``perm_grid`` (per π,
readout and metric: measured, null mean/sd/min/max, random / sign code), ``h3`` (the TOST record per readout),
``grid_hyper`` (how often each γ / C was chosen over the grid) plus ``detectors`` / ``sources`` of the engine.

H3 inputs (:mod:`flyguard.experiments.verdicts_run` -> :func:`flyguard.eval.verdicts.verdict_h3`): the primary row
is the ``diff90/macro_auc/real_fly_bloom-curveball_mean`` record (``reference`` and ``p_randomization`` extras), the
secondary rows the same records of ``real_fly_linear`` and ``real_fly_bloom_10shot``; the precondition is
``val_macro_auc/real_fly_bloom``.

Memory and time: the grid keeps P·(J+2)·Σ_s n_s float64 per readout (10 · 202 · n_docs · 8 B ≈ 16 MB per 1 000 test
documents) and never keeps codes (one ``[n_windows, m]`` code at a time); the two-stage bootstrap builds one
``WeightedAUC`` per source over P·(J+1) tables (≈ 33 B per table and document, ≈ 330 MB at 5 000 documents). A seed
costs P·(J+3) codings + refits of every readout (the linear readout's C grid dominates: 5 + 1 logistic fits on
``[n_train, 1886]`` per combination). Measured on the smoke tables (141 train windows, 754 windows in play, J = 20,
machine load 50–70 on 16 cores): 469 s for 230 combinations, ≈ 2 s each; at the E1 sizes (526 train windows, J = 200,
2 030 combinations) expect roughly 35–60 min per seed for the grid, plus three two-stage bootstraps (n = 1 000 draws
over ≈ 2 010 tables per source: about a minute each once the trace sources are in the tables). The combinations
are independent, so a process pool over π is the obvious speed-up if the pipeline budget needs it.

Entry points: ``python -m flyguard.experiments.run E4 [--seeds ...] [--smoke]`` (``run_cli``: one ``Context`` shared
by the seeds, so the test files are journaled once per experiment run), ``python -m flyguard.experiments.e4`` and
:func:`run` for one seed with an existing context (tests).
"""
from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyguard import fly
from flyguard.config import ROOT, Configs, load_configs, seeds_for
from flyguard.eval.bootstrap import CI, macro_auc_draws, percentile_ci, two_stage_bootstrap_h3
from flyguard.eval.metrics import auc as auc_point
from flyguard.eval.tost import equivalence_margin, holm, randomization_p, tost
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import (POSITIVE_SOURCES, Evaluator, FeatureContext, FittedDetector, ResultBuilder,
                                         Runner, fly_spec, standard_evaluation)
from flyguard.readout import BloomReadout, LinearReadout, select_C, select_gamma

EXPERIMENT = "E4"
NOSE = "n51_svd"
PRIMARY_METRIC = "macro_auc"
NULL_FAMILY = "curveball_mean"
NULL_KIND = {"curveball": "curveball", "random_same_density": "random", "dense_gaussian_sign": "dense_sign"}
MATRIX_PREFIX = {"measured": "real_fly", "random": "random_fly", "dense_sign": "dense_sign"}
SCORE_TOLERANCE = 1e-6


# ----------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Readout:
    """A readout of the E4/E5 configs: ``id`` (``bloom_full``, ``linear``, ``bloom_<n>shot``, E5's ``bloom``),
    ``kind`` (``bloom`` / ``linear``) and ``shots`` (documents per class for a few-shot Bloom, else None)."""

    id: str
    kind: str
    shots: int | None = None

    @property
    def suffix(self) -> str:
        """Detector-name suffix: ``bloom`` for the full Bloom, otherwise the id itself."""
        return "bloom" if self.kind == "bloom" and self.shots is None else self.id


def parse_readout(rid: str) -> Readout:
    rid = str(rid)
    if rid == "linear":
        return Readout(rid, "linear")
    if rid in ("bloom_full", "bloom"):
        return Readout(rid, "bloom")
    m = re.fullmatch(r"bloom_(\d+)shot", rid)
    if m:
        return Readout(rid, "bloom", int(m.group(1)))
    raise ValueError(f"unknown readout id {rid!r} (expected linear, bloom_full or bloom_<n>shot)")


def e4_config(cfg: Configs) -> tuple[list[Readout], list[str]]:
    """``(readouts, null kinds)`` from ``configs/experiments/E4.yaml``: the primary readout first; null kinds are the
    engine's matrix kinds (``curveball``, ``random``, ``dense_sign``)."""
    e4 = cfg.exp(EXPERIMENT)
    readouts = [parse_readout(e4["readouts"]["primary"])] + [parse_readout(r) for r in e4["readouts"]["secondary"]]
    nulls = []
    for n in e4["nulls"]:
        if n not in NULL_KIND:
            raise KeyError(f"E4.yaml null {n!r} is not one of {sorted(NULL_KIND)}")
        nulls.append(NULL_KIND[n])
    return readouts, nulls


def check_n_curveball(cfg: Configs, n_null: int, smoke: bool) -> None:
    """``E4.yaml: n_curveball`` must equal the J the engine actually draws (``expansion.curveball.n_null`` of
    default.yaml through ``Context.n_null()``): the engine's Curveball set is shared with E5, so E4 cannot honour a
    different count, and a silent mismatch would put a J in the report that the run never used. Smoke mode uses
    ``smoke.n_null`` by design and is exempt. Raises before any heavy work."""
    if smoke:
        return
    declared = cfg.exp(EXPERIMENT).get("n_curveball")
    if declared is not None and int(declared) != int(n_null):
        raise ValueError(f"configs/experiments/E4.yaml n_curveball={int(declared)} but expansion.curveball.n_null="
                         f"{int(n_null)} (default.yaml): make the two agree before running E4")


def detector_name(kind: str, readout: Readout) -> str:
    return f"{MATRIX_PREFIX[kind]}_{readout.suffix}"


# ----------------------------------------------------------------------------------------------------------------
# Permutations π
# ----------------------------------------------------------------------------------------------------------------
def perm_for_seed(seed_perm: int, rank: int) -> np.ndarray:
    """π of :class:`flyguard.nose.N51Svd` for a ``perm`` seed (ASSUMPTIONS A14: ``default_rng(perm).permutation(rank)``);
    the run checks it against the fitted nose's ``pi_`` so this rule can never drift silently."""
    return np.random.default_rng(int(seed_perm)).permutation(int(rank))


def permute_columns(U_own: np.ndarray, pi_own: np.ndarray, pi_p: np.ndarray) -> np.ndarray:
    """Features under another permutation: with S the standardised SVD components, the nose gives
    ``U_own[:, g] = S[:, pi_own[g]]``; the same rows under π_p are ``U_p[:, g] = S[:, pi_p[g]] = U_own[:, inv_own[pi_p[g]]]``
    (centring on C_unl is per column, so it commutes with the permutation)."""
    pi_own = np.asarray(pi_own)
    inv = np.empty_like(pi_own)
    inv[pi_own] = np.arange(pi_own.size)
    return np.ascontiguousarray(np.asarray(U_own)[:, inv[np.asarray(pi_p)]])


def grid_perms(cfg: Configs, own_perm: int) -> list[dict[str, Any]]:
    """The P permutations of the grid: the ``perm`` children of the global seeds (``seeds.global``), the seed's own
    π appended when it is not among them (a seed outside the configured list)."""
    perms = [{"global_seed": int(g), "perm": seeds_for(cfg, int(g))["perm"]} for g in cfg.default["seeds"]["global"]]
    if own_perm not in {p["perm"] for p in perms}:
        perms.append({"global_seed": None, "perm": int(own_perm)})
    for i, p in enumerate(perms):
        p["index"] = i
        p["own"] = p["perm"] == own_perm
    return perms


# ----------------------------------------------------------------------------------------------------------------
# Score tables of refitted readouts
# ----------------------------------------------------------------------------------------------------------------
class ScoreTables:
    """Document-score tables of readouts refitted on arbitrary (features, matrix) pairs with the engine's protocol.

    Built once per nose over the windows in play (train, deepset validation, every test source of ``doc_tables``);
    :meth:`tables` codes the stacked features with one matrix, fits every readout (γ / C on validation, few-shot
    Bloom on ``fewshot_set(shots, 0)``) and returns per readout and source the document scores (max over windows)
    aligned with the rows of ``doc_tables[source]``. Used by E4 for the π grid and by E5 for the Curveball wiring row.
    """

    def __init__(self, fc: FeatureContext, nose: str, doc_tables: Mapping[str, pd.DataFrame],
                 readouts: Sequence[Readout]) -> None:
        self.fc, self.cfg, self.nose = fc, fc.cfg, nose
        self.readouts = list(readouts)
        self.sources = sorted(doc_tables)
        self.train, self.val = fc.window_set("train"), fc.window_set("val")
        self.tests = {s: fc.window_set(f"test:{s}") for s in self.sources}
        parts = [fc.features(nose, self.train), fc.features(nose, self.val)] + \
                [fc.features(nose, self.tests[s]) for s in self.sources]
        parts = [np.asarray(p.toarray() if sp.issparse(p) else p, dtype=np.float32) for p in parts]
        self.U = np.vstack(parts)
        bounds = np.cumsum([0] + [p.shape[0] for p in parts])
        names = ["train", "val"] + self.sources
        self.slices = {n: slice(int(bounds[i]), int(bounds[i + 1])) for i, n in enumerate(names)}
        self.y_train, self.y_val = self.train.labels, self.val.labels
        self.few_rows = {r.shots: fc.fewshot_set(r.shots, 0).rows for r in self.readouts if r.shots}
        self.doc_index: dict[str, tuple[np.ndarray, int]] = {}
        for s in self.sources:
            docs = doc_tables[s]["doc_id"].to_numpy().astype(str)
            order = np.argsort(docs, kind="stable")
            wanted = self.tests[s].doc_ids.astype(str)
            pos = order[np.minimum(np.searchsorted(docs[order], wanted), docs.size - 1)]
            if not np.array_equal(docs[pos], wanted):
                raise ValueError(f"test windows of {s!r} name documents absent from its document table")
            self.doc_index[s] = (pos, int(docs.size))
        self.seed = fc.seeds["subsample"]
        self.gammas = [float(g) for g in self.cfg.default["readout"]["bloom"]["gammas"]]
        self.C_grid = [float(c) for c in self.cfg.default["readout"]["linear"]["C_grid"]]
        self.n_windows = int(self.U.shape[0])

    def code(self, M: Any, kind: str, U: np.ndarray | None = None) -> sp.csr_matrix:
        U = self.U if U is None else U
        if kind == "dense_sign":
            return fly.sign_code(U, M)
        return fly.fly_code(U, M, self.fc.k_for(M.shape[0]))

    def _doc_max(self, source: str, window_scores: np.ndarray) -> np.ndarray:
        pos, n = self.doc_index[source]
        out = np.full(n, -np.inf)
        np.maximum.at(out, pos, np.asarray(window_scores, dtype=np.float64))
        return out

    def tables(self, M: Any, kind: str, U: np.ndarray | None = None
               ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
        """``({readout id: {source: doc scores}}, {readout id: chosen hyperparameter})`` for one matrix; Bloom
        readouts are skipped on the sign code (ТЗ 3.3: linear only)."""
        Z = self.code(M, kind, U)
        m = int(Z.shape[1])
        k = self.fc.k_for(m) if kind != "dense_sign" else m // 2
        Ztr, Zva = Z[self.slices["train"]], Z[self.slices["val"]]
        out: dict[str, dict[str, np.ndarray]] = {}
        choices: dict[str, dict[str, float]] = {}
        for r in self.readouts:
            if r.kind == "bloom":
                if kind == "dense_sign":
                    continue
                rows = self.few_rows[r.shots] if r.shots else slice(None)
                Zt, yt = Ztr[rows], self.y_train[rows]
                gamma, _ = select_gamma(Zt, yt, Zva, self.y_val, m, k, seed_subsample=self.seed, gammas=self.gammas,
                                        cfg=self.cfg)
                model: Any = BloomReadout(m, k, gamma, seed_subsample=self.seed).fit(Zt, yt)
                choices[r.id] = {"gamma": float(gamma)}
            else:
                C, _ = select_C(Ztr, self.y_train, Zva, self.y_val, C_grid=self.C_grid, seed=self.seed, cfg=self.cfg)
                model = LinearReadout(C=C, seed=self.seed, cfg=self.cfg).fit(Ztr, self.y_train)
                choices[r.id] = {"C": float(C)}
            out[r.id] = {s: self._doc_max(s, model.score(Z[self.slices[s]])) for s in self.sources}
        return out, choices


# ----------------------------------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------------------------------
def metric_names(sources: Sequence[str]) -> list[str]:
    return [PRIMARY_METRIC] + [f"auc/{s}" for s in sources]


def point_metrics(tables: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> dict[str, float]:
    """``{"macro_auc": ..., "auc/<src>": ...}`` of one score table set (macroAUC = mean over the present sources)."""
    per = {s: auc_point(tables[s], labels[s]) for s in sorted(tables)}
    finite = [v for v in per.values() if math.isfinite(v)]
    out = {PRIMARY_METRIC: float(np.mean(finite)) if finite else float("nan")}
    out.update({f"auc/{s}": v for s, v in per.items()})
    return out


def fixed_matrix_diff_ci(by_source: Mapping[str, pd.DataFrame], measured: Mapping[str, np.ndarray],
                         nulls: Mapping[str, np.ndarray], n: int, seed: int, alpha: float) -> dict[str, CI]:
    """Cluster-bootstrap intervals of AUC(measured) − mean_j AUC(null_j) with the matrices fixed (the ТЗ "Вклады"
    wiring contribution): ``{"macro_auc": CI, "auc/<src>": CI}``; every draw resamples clusters within each source
    and evaluates all J + 1 tables on the same draw (:func:`flyguard.eval.bootstrap.macro_auc_draws`)."""
    out: dict[str, CI] = {}

    def one(frames: Mapping[str, pd.DataFrame]) -> CI:
        tables = {s: np.vstack([np.asarray(measured[s], dtype=float)[None, :], np.asarray(nulls[s], dtype=float)])
                  for s in frames}
        point, draws, n_clusters = macro_auc_draws(frames, tables, n, seed)
        samples = draws[:, 0] - draws[:, 1:].mean(axis=1)
        pt = float(point[0] - point[1:].mean())
        n_docs = int(sum(len(frames[s]) for s in n_clusters))
        return percentile_ci(samples, pt, alpha, n_docs, int(sum(n_clusters.values())))

    out[PRIMARY_METRIC] = one(by_source)
    for s in sorted(by_source):
        out[f"auc/{s}"] = one({s: by_source[s]})
    return out


def _spread(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0, "min": float(x.min()),
            "max": float(x.max()), "n": int(x.size)}


def _metric_col(metric: str) -> str:
    """Table column of a metric: ``macro_auc`` stays, ``auc/<src>`` becomes ``auc_<src>``."""
    return metric.replace("/", "_")


# ----------------------------------------------------------------------------------------------------------------
# The experiment body
# ----------------------------------------------------------------------------------------------------------------
def fit_own_detectors(fc: FeatureContext, readouts: Sequence[Readout], nulls: Sequence[str]) -> dict[str, FittedDetector]:
    """Measured, random and sign-code detectors at the seed's own π through the engine (every readout; the sign code
    with the linear readout only)."""
    fitted: dict[str, FittedDetector] = {}
    kinds = ["measured"] + [k for k in nulls if k != "curveball"]
    for r in readouts:
        for kind in kinds:
            if kind == "dense_sign" and r.kind != "linear":
                continue
            name = detector_name(kind, r)
            spec = fly_spec(name, NOSE, kind, r.kind)
            train = fc.fewshot_set(r.shots, 0) if r.shots else "train"
            fitted[name] = fc.fit(spec, train=train)
    return fitted


def own_pairs(readouts: Sequence[Readout], nulls: Sequence[str], sources: Sequence[str]) -> list[tuple[str, str, str]]:
    pairs = []
    for r in readouts:
        for kind in nulls:
            if kind == "curveball" or (kind == "dense_sign" and r.kind != "linear"):
                continue
            for metric in metric_names(sources):
                pairs.append((metric, detector_name("measured", r), detector_name(kind, r)))
    return pairs


def body(fc: FeatureContext, rb: ResultBuilder) -> None:
    """E4 for one global seed (see the module docstring for every key)."""
    cfg, ctx = fc.cfg, fc.ctx
    readouts, nulls = e4_config(cfg)
    check_n_curveball(cfg, ctx.n_null(), ctx.smoke)
    primary = readouts[0]
    sources = [s for s in ctx.test_sources if s in POSITIVE_SOURCES]
    t0 = time.perf_counter()
    fitted = fit_own_detectors(fc, readouts, nulls)
    out = standard_evaluation(fc, fitted, sources=sources, pairs=own_pairs(readouts, nulls, sources), validation=True,
                              latency=False, notinject=False)
    rb.merge(out)
    rb.note(f"E4 own-perm detectors: {len(fitted)} fitted and evaluated in {time.perf_counter() - t0:.1f} s; "
            f"readouts={[r.id for r in readouts]}; nulls={list(nulls)}")
    doc_tables = {s: df for s, df in out["doc_tables"].items() if s in POSITIVE_SOURCES and df["label"].nunique() == 2}
    if not doc_tables:
        rb.note("no positive test source with both classes: the π grid, randomisation test and two-stage bootstrap "
                "were skipped (не хватило данных)")
        return
    src = sorted(doc_tables)
    labels = {s: doc_tables[s]["label"].to_numpy().astype(int) for s in src}
    by_source = {s: doc_tables[s][["doc_id", "label", "cluster_id"]] for s in src}
    metrics = metric_names(src)
    det_of = {r.id: detector_name("measured", r) for r in readouts}

    # -- matrices and permutations ----------------------------------------------------------------------------------
    M = fc.matrix("measured")
    curve = fc.curveball_set() if "curveball" in nulls else []
    matrices: list[tuple[str, Any]] = [("measured", M)] + [(f"curveball:{j}", N) for j, N in enumerate(curve)]
    for kind in nulls:
        if kind != "curveball":
            matrices.append((kind, fc.matrix(kind)))
    pi_own = fc.n51_svd.pi_
    if not np.array_equal(perm_for_seed(fc.seeds["perm"], pi_own.size), pi_own):
        raise RuntimeError("perm_for_seed disagrees with N51Svd.pi_ (ASSUMPTIONS A14 rule changed?)")
    perms = grid_perms(cfg, fc.seeds["perm"])
    own_idx = next(p["index"] for p in perms if p["own"])
    P, J = len(perms), len(curve)

    # -- the π grid -------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    st = ScoreTables(fc, NOSE, doc_tables, readouts)
    meas: list[dict[str, dict[str, np.ndarray]]] = []
    null: list[dict[str, list[dict[str, np.ndarray]]]] = []
    other: list[dict[str, dict[str, dict[str, np.ndarray]]]] = []   # per perm: kind -> readout -> source
    hyper: dict[str, Counter] = {r.id: Counter() for r in readouts}
    for p in perms:
        U_p = permute_columns(st.U, pi_own, perm_for_seed(p["perm"], pi_own.size))
        meas_p: dict[str, dict[str, np.ndarray]] = {}
        null_p: dict[str, list[dict[str, np.ndarray]]] = {r.id: [] for r in readouts}
        other_p: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for kind, mat in matrices:
            base_kind = kind.split(":")[0]
            tabs, choices = st.tables(mat, base_kind, U_p)
            for rid, ch in choices.items():
                (param, value), = ch.items()
                hyper[rid][(base_kind, param, value)] += 1
            if kind == "measured":
                meas_p = tabs
            elif base_kind == "curveball":
                for rid, t in tabs.items():
                    null_p[rid].append(t)
            else:
                other_p[kind] = tabs
        meas.append(meas_p)
        null.append(null_p)
        other.append(other_p)
    grid_seconds = time.perf_counter() - t0

    # -- own-π consistency with the engine (a wrong permutation composition or protocol drift raises here) -----------
    for r in readouts:
        checks = [("measured", meas[own_idx][r.id])] + [(k, t[r.id]) for k, t in other[own_idx].items() if r.id in t]
        for kind, tabs in checks:
            name = detector_name(kind, r)
            for s in src:
                got, ref = tabs[s], doc_tables[s][name].to_numpy(dtype=float)
                dev = float("inf") if got.shape != ref.shape else float(np.max(np.abs(got - ref)))
                if dev > SCORE_TOLERANCE:
                    raise RuntimeError(f"π-grid scores of {name} at the own π differ from the engine's on {s} (max abs "
                                       f"difference {dev:.3g}, tolerance {SCORE_TOLERANCE:g}): a difference of order 1e-5 "
                                       f"points at solver noise, a larger one at a permutation or protocol mismatch")

    # -- point statistics of every table (rows in the order of ``metrics``) --------------------------------------------
    def row(tables: Mapping[str, np.ndarray]) -> list[float]:
        pm = point_metrics(tables, labels)
        return [pm[m] for m in metrics]

    A_meas = {r.id: np.array([row(meas[p][r.id]) for p in range(P)]) for r in readouts}
    A_null = {r.id: np.array([[row(null[p][r.id][j]) for j in range(J)] for p in range(P)]).reshape(P, J, len(metrics))
              for r in readouts}
    A_other = {k: {r.id: np.array([row(other[p][k][r.id]) for p in range(P)]) for r in readouts if r.id in other[0][k]}
               for k in other[0]}

    # -- randomisation test (π-averaged canonical, own-π alongside) and Holm over the secondary family --------------
    p_can: dict[tuple[str, str], float] = {}
    p_own: dict[tuple[str, str], float] = {}
    for r in readouts:
        for mi, m in enumerate(metrics):
            if J:
                p_can[(r.id, m)] = randomization_p(float(A_meas[r.id][:, mi].mean()),
                                                   A_null[r.id][:, :, mi].mean(axis=0), two_sided=True, cfg=cfg)
                p_own[(r.id, m)] = randomization_p(float(A_meas[r.id][own_idx, mi]), A_null[r.id][own_idx, :, mi],
                                                   two_sided=True, cfg=cfg)
    family = {k: v for k, v in p_can.items() if k != (primary.id, PRIMARY_METRIC)}
    adj = holm({f"{rid}/{m}": v for (rid, m), v in family.items()})
    for (rid, m), p in p_can.items():
        det = det_of[rid]
        rb.add_number(f"p_randomization/{m}/{det}", p, n=J,
                      note="two-sided randomisation p of the pi-averaged statistic against J curveball nulls")
        rb.add_number(f"p_randomization_ownperm/{m}/{det}", p_own[(rid, m)], n=J,
                      note="two-sided randomisation p at the seed's own pi")
        if f"{rid}/{m}" in adj:
            rb.add_number(f"p_randomization_holm/{m}/{det}", adj[f"{rid}/{m}"], n=len(family),
                          note="Holm over the secondary family (every readout x metric except the primary macroAUC)")

    # -- two-stage bootstrap and TOST per readout -------------------------------------------------------------------
    ev = Evaluator(fc)
    ids = [f"p{p['index']:02d}" for p in perms]
    t0 = time.perf_counter()
    h3_rows = []
    for r in readouts:
        if not J:
            continue
        det = det_of[r.id]
        measured = {ids[p]: meas[p][r.id] for p in range(P)}
        nulls_r = {ids[p]: {s: np.stack([null[p][r.id][j][s] for j in range(J)]) for s in src} for p in range(P)}
        res = two_stage_bootstrap_h3(by_source, measured, nulls_r, n=ev.n_boot, seed=ev.seed, cfg=cfg)
        d90: CI = res["diff"]
        d95 = percentile_ci(d90.samples, d90.point, 0.05, d90.n, d90.n_clusters)
        delta = equivalence_margin(res["delta_reference"], cfg=cfg)
        t = tost(d90, delta)
        p_r = p_can.get((r.id, PRIMARY_METRIC))
        pair = f"{PRIMARY_METRIC}/{det}-{NULL_FAMILY}"
        rb.add_ci(f"diff90/{pair}", d90, reference=float(res["delta_reference"]), p_randomization=p_r, delta=float(delta),
                  tost_equivalent=bool(t["equivalent"]), n_perms=res["n_perms"], n_null=res["n_null"],
                  note=f"H3: pi-averaged over {P} permutations, two_stage bootstrap (clusters x {J} null matrices x pi), "
                       f"90 % = TOST level; p = randomisation p of the pi-averaged macroAUC")
        rb.add_ci(f"diff/{pair}", d95, p=p_r, reference=float(res["delta_reference"]),
                  note=f"pi-averaged over {P} permutations, two_stage bootstrap, same draws at 95 %")
        rb.add_ci(f"{PRIMARY_METRIC}/{NULL_FAMILY}/{det}", res["null_mean"],
                  note="mean macroAUC of the curveball nulls (two_stage interval); reference of the +/-delta corridor")
        rb.add_ci(f"{PRIMARY_METRIC}/pi_mean/{det}", res["measured"],
                  note="macroAUC of the measured matrix averaged over the pi permutations (two_stage interval)")
        rb.add_number(f"tost_delta/{pair}", delta, note="delta_rel x null mean")
        rb.add_number(f"tost_equivalent/{pair}", float(t["equivalent"]), note="1 = 90% CI inside +/-delta")
        h3_rows.append({"readout": r.id, "detector": det, "primary": r.id == primary.id, **t,
                        "reference": res["delta_reference"], "n_perms": res["n_perms"], "n_null": res["n_null"],
                        "p_randomization": p_r, "p_randomization_ownperm": p_own.get((r.id, PRIMARY_METRIC)),
                        "diff95_low": d95.low, "diff95_high": d95.high, "n_valid": d90.n_valid,
                        "per_perm": res["per_perm"]})
    rb.add_table("h3", h3_rows)
    boot_seconds = time.perf_counter() - t0

    # -- wiring contribution at the own π (fixed matrices, cluster CI), null spreads, tables ---------------------------
    for r in readouts:
        det = det_of[r.id]
        if J:
            null_own = {s: np.stack([null[own_idx][r.id][j][s] for j in range(J)]) for s in src}
            cis = fixed_matrix_diff_ci(by_source, meas[own_idx][r.id], null_own, ev.n_boot, ev.seed, ev.alpha)
            for m, ci in cis.items():
                rb.add_ci(f"contrib/wiring/{det}/{m}", ci, p=p_own.get((r.id, m)),
                          note=f"contribution of the ТЗ table at the seed's own pi only, {J} null matrices fixed "
                               f"(cluster bootstrap); p = own-pi randomisation p; the H3 interval is diff90/macro_auc/"
                               f"{det}-{NULL_FAMILY}")
            for mi, m in enumerate(metrics):
                spr = _spread(A_null[r.id][own_idx, :, mi])
                rb.add_number(f"curveball_mean_ownperm/{m}/{det}", spr["mean"], n=J)
                rb.add_number(f"curveball_sd/{m}/{det}", spr["sd"], n=J)
            rb.add_number(f"pi_sd/{PRIMARY_METRIC}/{NULL_FAMILY}/{det}", _spread(A_null[r.id][:, :, 0].mean(axis=1))["sd"],
                          n=P, note="sd over the P permutations of the null mean macroAUC")
        rb.add_number(f"pi_sd/{PRIMARY_METRIC}/{det}", _spread(A_meas[r.id][:, 0])["sd"], n=P,
                      note="sd over the P permutations of macroAUC(measured)")
    hist_rows: dict[str, list[dict[str, Any]]] = {"curveball": [], "nulls_secondary": []}
    for r in readouts:
        for j in range(J):
            rec = {"readout": r.id, "detector": det_of[r.id], "j": j}
            if r.id == primary.id:
                rec["matrix"] = f"curveball:{j}"
            rec.update({_metric_col(m): float(A_null[r.id][own_idx, j, mi]) for mi, m in enumerate(metrics)})
            hist_rows["curveball" if r.id == primary.id else "nulls_secondary"].append(rec)
    for name, rows in hist_rows.items():
        rb.add_table(name, rows)
    grid_rows = []
    for p in perms:
        for r in readouts:
            for mi, m in enumerate(metrics):
                rec = {"perm_index": p["index"], "global_seed": p["global_seed"], "perm_seed": p["perm"], "own": p["own"],
                       "readout": r.id, "detector": det_of[r.id], "metric": m, "measured": float(A_meas[r.id][p["index"], mi])}
                if J:
                    rec.update({f"null_{k}": v for k, v in _spread(A_null[r.id][p["index"], :, mi]).items() if k != "n"})
                for kind, arr in A_other.items():
                    if r.id in arr:
                        rec[kind] = float(arr[r.id][p["index"], mi])
                grid_rows.append(rec)
    rb.add_table("perm_grid", grid_rows)
    rb.add_table("grid_hyper", [{"readout": rid, "matrix": key[0], "param": key[1], "value": key[2], "count": c}
                                for rid, cnt in hyper.items() for key, c in sorted(cnt.items(), key=str)])
    n_docs = int(sum(len(df) for df in doc_tables.values()))
    rb.note(f"E4 pi grid: P={P} permutations x {len(matrices)} matrices (J={J} curveball) x {len(readouts)} readouts "
            f"on {st.n_windows} windows / {n_docs} test documents of {src} in {grid_seconds:.1f} s "
            f"({grid_seconds / max(P * len(matrices), 1):.2f} s per (pi, matrix) incl. every readout); score tables "
            f"{P * len(matrices) * n_docs * 8 * len(readouts)} bytes; two_stage bootstrap n={ev.n_boot} in {boot_seconds:.1f} s")
    rb.note(f"randomisation floor: min p = 1/(J+1) = {1 / (J + 1) if J else float('nan'):.4f}; Holm family of "
            f"{len(family)} secondary tests cannot fall below {min(1.0, len(family) / (J + 1)) if J else float('nan'):.4f}")
    rb.note(f"bloom few-shot readouts use fewshot_set(shots, rep=0) of the subsample seed; own perm index {own_idx}")


# ----------------------------------------------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------------------------------------------
def seeds_to_run(cfg: Configs, smoke: bool) -> list[int]:
    seeds = [int(s) for s in cfg.default["seeds"]["global"]]
    return seeds[: int(cfg.default["smoke"]["seeds"])] if smoke else seeds


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path | None = None,
        cfg: Configs | None = None, keep_cache: bool = False, force: bool = False, access_log: Any = None,
        guard_factory: Any = None, cache: bool = True) -> Path:
    """Run E4 for one global seed and return the results path (``Runner`` semantics: skipped when current)."""
    root = Path(root) if root is not None else (ctx.root if ctx is not None else ROOT)
    return Runner(EXPERIMENT, seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                  access_log=access_log, guard_factory=guard_factory, cache=cache).run(body)


def run_seeds(seeds: Sequence[int], smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None,
              force: bool = False, keep_cache: bool = False, summarize: bool = True,
              experiment: str = EXPERIMENT, body_fn: Any = None) -> list[Path]:
    """Run the seeds with one shared :class:`Context` (the test files are journaled once per experiment run) and
    rewrite ``summary.json``."""
    root = Path(root)
    cfg = cfg or (load_configs(root) if (root / "configs").is_dir() else load_configs())
    ctx = Context(cfg, smoke=smoke, root=root)
    paths = [Runner(experiment, s, smoke=smoke, root=Path(root), cfg=cfg, ctx=ctx, keep_cache=keep_cache,
                    force=force).run(body_fn or body) for s in seeds]
    if summarize:
        results_mod.summarize(experiment, smoke=smoke, root=Path(root))
    return paths


def run_cli(args: argparse.Namespace) -> list[Path]:
    """Plug-in entry of ``python -m flyguard.experiments.run E4`` (``args.seeds, smoke, root, force, keep_cache``)."""
    return run_seeds(args.seeds, smoke=args.smoke, root=args.root, force=args.force, keep_cache=args.keep_cache)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E4: measured wiring against its null models (H3 inputs)")
    ap.add_argument("--seed", type=int, action="append", help="global seed (repeatable); default: all configured")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--force", action="store_true", help="rerun even when the result file is current")
    ap.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_configs(args.root) if (args.root / "configs").is_dir() else load_configs()
    for p in run_seeds(args.seed or seeds_to_run(cfg, args.smoke), args.smoke, args.root, cfg, args.force, args.keep_cache):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
