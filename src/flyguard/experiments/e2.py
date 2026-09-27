"""E2 -- learning curves (ТЗ Этап 4 "E2 кривые обучения", design_experiments §3; feeds H1b(ii)).

What is measured
----------------
For every training size ``shots`` in ``configs/experiments/E2.yaml`` (``1, 10, 100, full`` documents *per class*
of deepset train, the only labels of the cross-dataset split, ТЗ 1.10) and every subsample ``rep`` (``subsamples_per_
point``, drawn without replacement from ``SeedSequence([subsample, shots, rep])`` by ``FeatureContext.fewshot_set``;
``full`` is the whole train set, one rep), every detector of the config (real Bloom fly, FlyHash Bloom fly, real
linear fly, TF-IDF + LR, kNN(1), nearest centroid) is fitted on the subsample and scored on every positive test
source. γ and C are chosen on the deepset validation windows *per fit*, i.e. separately for every shot level and
subsample (ТЗ 2.4: "γ по валидации, отдельно для few-shot и полного обучения"); the per-rep choices are tabulated and
the modal choice per level is a number.

Statistics: one set of cluster draws (``eval.bootstrap.macro_auc_draws``; clusters resampled inside each source,
``n`` draws from the ``bootstrap`` child) serves every fit at once, so the curve of a detector, its band and the paired
differences of H1b(ii) come from the same resamples. Per (detector, level) the number is the *mean over subsamples*
of macroAUC with the percentile interval of that mean over the draws; ``band_low`` / ``band_high`` are the lowest and
highest per-subsample point macroAUC (the band of the learning-curve figure: variability over training subsamples,
which the bootstrap interval does not contain), ``rep_sd`` their standard deviation. The paired difference
macroAUC(a) − macroAUC(b) is the mean over subsamples of the per-subsample paired differences (the same rep of both
detectors), with the 95 % interval, the 90 % interval of the same draws and the two-sided bootstrap p.

Keys (results naming rules of :mod:`flyguard.experiments.results`; the level segment ``<L>`` is ``shots1``,
``shots10``, ``shots100`` or ``full``; tables use the bare level ``1`` / ``10`` / ``100`` / ``full``)
-------------------------------------------------------------------------------------------------------------
* ``fewshot/macro_auc/<L>/<detector>``        mean over subsamples, CI, ``band_low``/``band_high``/``rep_sd``/``n_reps``;
* ``fewshot/auc/<source>/<L>/<detector>``     the same per positive source (``para_deep`` = deep paraphrase stratum);
* ``diff/macro_auc/<L>/<a>-<b>``              paired difference (95 %, ``p``) for the H1b(ii) pairs ``real_fly_bloom-knn1``
  and ``flyhash_bloom-knn1`` -- ``verdicts_run.h1b_inputs`` looks these up as ``diff/macro_auc/shots<k>/<fly>_bloom-knn1``
  for k in {1, 10} -- and the full-training cross-check ``real_fly_bloom-tfidf_lr``, ``flyhash_bloom-tfidf_lr`` (the
  verdict's ``full`` interval comes from E1); ``diff90/macro_auc/<L>/<a>-<b>`` the 90 % interval of the same draws;
* ``fewshot/hyper/<detector>/<param>/<L>``    modal validated γ / C over the subsamples (``note`` = the histogram);
* ``fewshot/n_train_docs/<L>``                training documents per fit (mean over reps).

Tables: ``fewshot_levels`` (per level: reps, documents per class, status -- a level a class cannot supply gets
``не хватило данных`` with the counts instead of failing, as in smoke or a small train set), ``fewshot_fits`` (per fit:
the validated hyperparameters, train sizes, seconds), ``fewshot_curve`` (per fit: macroAUC with its own interval and
the per-source point AUCs). Notes carry counts only, never window text.

Cost: FlyHash-20 codes of the large test sets are streamed, so all fits are collected first and each test source is
scored once with the whole set (``score_many`` groups detectors by code key); the per-seed cost is one coding pass per
test source plus the bootstrap.

Entry points: :func:`run_e2` (the experiment body, ``body(fc, rb)`` of :class:`flyguard.experiments.engine.Runner`)
and :func:`run` (``run(ctx, seed, smoke)``, the signature ``experiments.run`` imports by module name).
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from flyguard.config import ROOT, Configs
from flyguard.eval.bootstrap import macro_auc_draws, percentile_ci
from flyguard.eval.tost import bootstrap_p
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import (POSITIVE_SOURCES, Evaluator, FeatureContext, FittedDetector, ResultBuilder,
                                         Runner, detector_spec)

EXPERIMENT = "E2"
FULL = "full"
PREFIX = "fewshot"
H1B_PAIRS: tuple[tuple[str, str], ...] = (("real_fly_bloom", "knn1"), ("flyhash_bloom", "knn1"),
                                           ("real_fly_bloom", "tfidf_lr"), ("flyhash_bloom", "tfidf_lr"))
STATUS_OK = "выполнено"
STATUS_NO_DATA = "не хватило данных"


@dataclasses.dataclass(frozen=True)
class Fit:
    """One point of the curve: detector ``det`` at ``level`` (int shots or ``full``), subsample ``rep``, stored in the
    fitted dict under the unique ``name`` (``<det>@<level>:<rep>``; the detector's code key is unchanged, so every
    fit of one fly shares one coding pass)."""

    det: str
    level: str
    rep: int
    name: str
    n_train_docs: int = 0


def shot_levels(shots: Sequence[Any]) -> list[int | str]:
    """Parse the config list: integers plus the literal ``full``; duplicates and unknown entries are rejected."""
    out: list[int | str] = []
    for s in shots:
        if isinstance(s, str) and s.strip().lower() == FULL:
            v: int | str = FULL
        else:
            v = int(s)
            if v <= 0:
                raise ValueError(f"shots must be positive, got {s!r}")
        if v in out:
            raise ValueError(f"duplicate shot level {s!r}")
        out.append(v)
    return out


def fit_name(det: str, level: int | str, rep: int) -> str:
    return f"{det}@{level}:{int(rep)}"


def level_key(level: int | str) -> str:
    """Key segment of a level: ``shots<k>`` (the spelling ``verdicts_run`` searches for) or ``full``."""
    return FULL if str(level) == FULL else f"shots{int(level)}"


def _class_doc_counts(ws) -> dict[int, int]:
    docs = ws.frame.groupby("doc_id")["label"].max()
    return {int(c): int((docs == c).sum()) for c in (0, 1)}


# ----------------------------------------------------------------------------------------------------------------
# Fitting the curve
# ----------------------------------------------------------------------------------------------------------------
def fit_curve(fc: FeatureContext, detectors: Sequence[str], levels: Sequence[int | str], n_reps: int,
              val: str = "val", parent: str = "train") -> tuple[dict[str, FittedDetector], list[Fit], list[dict[str, Any]]]:
    """Fit every detector at every level x rep; returns ``(fitted by name, index, level rows)``.

    A level that ``fewshot_set`` cannot draw (a class has fewer documents than ``shots``) is skipped for *all*
    detectors with status ``не хватило данных`` and the class counts in its row; ``full`` has a single rep on the
    whole parent set.
    """
    fitted: dict[str, FittedDetector] = {}
    index: list[Fit] = []
    rows: list[dict[str, Any]] = []
    counts = _class_doc_counts(fc.window_set(parent))
    for level in levels:
        key = str(level)
        reps = [0] if level == FULL else list(range(int(n_reps)))
        sets = []
        reason = None
        for rep in reps:
            try:
                sets.append((rep, fc.window_set(parent) if level == FULL else fc.fewshot_set(int(level), rep, parent)))
            except ValueError as exc:  # not enough documents of a class (smoke / small train sets)
                reason = str(exc)
                break
        if reason is not None:
            rows.append({"level": key, "n_reps": 0, "docs_per_class": None if level == FULL else int(level),
                         "n_train_docs": None, "status": STATUS_NO_DATA,
                         "reason": f"train has {counts[1]} positive / {counts[0]} negative documents: {reason}"})
            continue
        n_docs = int(np.mean([ws.frame["doc_id"].nunique() for _, ws in sets]))
        rows.append({"level": key, "n_reps": len(sets), "docs_per_class": None if level == FULL else int(level),
                     "n_train_docs": n_docs, "status": STATUS_OK, "reason": None})
        for rep, ws in sets:
            for det in detectors:
                spec = dataclasses.replace(detector_spec(det), name=fit_name(det, level, rep))
                f = fc.fit(spec, train=ws, val=val)
                fitted[f.name] = f
                index.append(Fit(det, key, int(rep), f.name, int(ws.frame["doc_id"].nunique())))
    return fitted, index, rows


def fit_rows(fitted: Mapping[str, FittedDetector], index: Sequence[Fit]) -> list[dict[str, Any]]:
    """The ``fewshot_fits`` table: validated hyperparameters and sizes per fit (no data text)."""
    rows = []
    for it in index:
        f = fitted[it.name]
        row: dict[str, Any] = {"level": it.level, "rep": it.rep, "detector": it.det, "n_train_windows": f.n_train,
                               "n_train_docs": it.n_train_docs, "fit_seconds": f.fit_seconds}
        for k in ("gamma", "gamma_source", "C", "C_source", "val_auc_window"):
            if k in f.choices:
                row[k] = f.choices[k]
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------------------------
def score_curve(fc: FeatureContext, fitted: Mapping[str, FittedDetector], sources: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Document tables per test source (one scoring pass per source over every fit) plus ``para_deep``."""
    tables: dict[str, pd.DataFrame] = {}
    for s in sources:
        ws = fc.window_set(f"test:{s}")
        if ws.n == 0:
            continue
        tables[s] = fc.doc_frame(fc.score_many(fitted, ws), ws)
    if "para" in tables:
        deep = tables["para"][tables["para"]["stratum"] == "deep"]
        if len(deep):
            tables["para_deep"] = deep.reset_index(drop=True)
    return tables


# ----------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------
def _mode(values: Sequence[Any]) -> tuple[Any, str]:
    """Modal value (ties -> first seen) and a ``value:count`` histogram string."""
    c = Counter(values)
    best = max(c.items(), key=lambda kv: (kv[1], -list(c).index(kv[0])))[0]
    return best, " ".join(f"{k}:{v}" for k, v in c.items())


def _n_docs(by_source: Mapping[str, pd.DataFrame], n_clusters: Mapping[str, int]) -> int:
    return int(sum(len(by_source[s]) for s in n_clusters))


def curve_numbers(ev: Evaluator, by_source: Mapping[str, pd.DataFrame], index: Sequence[Fit], metric: str,
                  pairs: Sequence[tuple[str, str]] = H1B_PAIRS) -> tuple[dict[str, Any], dict[str, tuple[float, float, float]]]:
    """Numbers of one metric (``macro_auc`` over ``by_source`` or ``auc/<source>`` for a single source) from one set
    of shared cluster draws: per (detector, level) the mean over reps with CI and band, and the paired differences
    of ``pairs`` per level. Returns ``(numbers, per-fit (point, low, high))``."""
    cols = [it.name for it in index]
    tables = {s: df[cols].to_numpy(dtype=float).T for s, df in by_source.items()}
    point, draws, n_clusters = macro_auc_draws(by_source, tables, ev.n_boot, ev.seed)
    n_docs, n_cl = _n_docs(by_source, n_clusters), int(sum(n_clusters.values()))
    pos = {it.name: j for j, it in enumerate(index)}
    numbers: dict[str, Any] = {}
    per_fit: dict[str, tuple[float, float, float]] = {}
    groups: dict[tuple[str, str], list[Fit]] = {}
    for it in index:
        groups.setdefault((it.det, it.level), []).append(it)
        ci = percentile_ci(draws[:, pos[it.name]], float(point[pos[it.name]]), ev.alpha, n_docs, n_cl)
        per_fit[it.name] = (ci.point, ci.low, ci.high)
    for (det, level), fits in groups.items():
        idx = [pos[f.name] for f in fits]
        pts = point[idx]
        ci = percentile_ci(draws[:, idx].mean(axis=1), float(np.mean(pts)), ev.alpha, n_docs, n_cl)
        numbers[f"{PREFIX}/{metric}/{level_key(level)}/{det}"] = results_mod.number(
            None, ci, note="sources=" + ",".join(sorted(n_clusters)), band_low=float(np.min(pts)),
            band_high=float(np.max(pts)), rep_sd=(float(np.std(pts, ddof=1)) if len(pts) > 1 else None), n_reps=len(pts))
    if metric == "macro_auc":
        for a, b in pairs:
            for level in sorted({it.level for it in index}, key=_level_order):
                fa = {f.rep: pos[f.name] for f in groups.get((a, level), [])}
                fb = {f.rep: pos[f.name] for f in groups.get((b, level), [])}
                reps = sorted(set(fa) & set(fb))
                if not reps:
                    continue
                ia, ib = [fa[r] for r in reps], [fb[r] for r in reps]
                d_point = float(np.mean(point[ia] - point[ib]))
                d_samples = (draws[:, ia] - draws[:, ib]).mean(axis=1)
                ci95 = percentile_ci(d_samples, d_point, ev.alpha, n_docs, n_cl)
                ci90 = percentile_ci(d_samples, d_point, 1.0 - ev.tost_level, n_docs, n_cl)
                numbers[f"diff/{metric}/{level_key(level)}/{a}-{b}"] = results_mod.number(
                    None, ci95, p=bootstrap_p(d_samples), n_reps=len(reps))
                numbers[f"diff90/{metric}/{level_key(level)}/{a}-{b}"] = results_mod.number(None, ci90, n_reps=len(reps))
    return numbers, per_fit


def _level_order(level: str) -> tuple[int, int]:
    return (1, 0) if level == FULL else (0, int(level))


def hyper_numbers(fitted: Mapping[str, FittedDetector], index: Sequence[Fit]) -> dict[str, Any]:
    """``fewshot/hyper/<det>/<param>/<level>``: the modal validated γ / C over the subsamples of a level."""
    out: dict[str, Any] = {}
    by: dict[tuple[str, str, str], list[float]] = {}
    for it in index:
        for param in ("gamma", "C"):
            v = fitted[it.name].choices.get(param)
            if v is not None:
                by.setdefault((it.det, param, it.level), []).append(float(v))
    for (det, param, level), vals in by.items():
        mode, hist = _mode(vals)
        out[f"{PREFIX}/hyper/{det}/{param}/{level_key(level)}"] = results_mod.number(mode, n=len(vals), note=f"reps {hist}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Body and entry point
# ----------------------------------------------------------------------------------------------------------------
def run_e2(fc: FeatureContext, rb: ResultBuilder, detectors: Sequence[str] | None = None,
           shots: Sequence[Any] | None = None, n_reps: int | None = None,
           sources: Sequence[str] | None = None) -> dict[str, Any]:
    """The E2 body: fit the curve, score every positive test source once, write numbers and tables into ``rb``.
    Returns the document tables and the fitted index (for tests and the report's figures)."""
    e2 = fc.cfg.exp(EXPERIMENT)
    detectors = list(detectors if detectors is not None else e2["detectors"])
    levels = shot_levels(shots if shots is not None else e2["shots"])
    n_reps = int(n_reps if n_reps is not None else e2["subsamples_per_point"])
    ctx = fc.ctx
    sources = list(sources if sources is not None else [s for s in ctx.test_sources if s != "notinject"])
    ev = Evaluator(fc)

    fitted, index, level_rows = fit_curve(fc, detectors, levels, n_reps)
    rb.add_table("fewshot_levels", level_rows)
    rb.add_table("fewshot_fits", fit_rows(fitted, index))
    for row in level_rows:
        if row["status"] != STATUS_OK:
            rb.note(f"E2 level {row['level']}: {row['status']} ({row['reason']})")
        elif row["n_train_docs"] is not None:
            rb.add_number(f"{PREFIX}/n_train_docs/{level_key(row['level'])}", row["n_train_docs"], n=row["n_reps"])
    for k, v in hyper_numbers(fitted, index).items():
        rb.numbers[k] = v
    if not index:
        rb.note("E2: no shot level could be fitted; no curve")
        return {"doc_tables": {}, "index": index, "fitted": fitted}

    doc_tables = score_curve(fc, fitted, sources)
    pos_sources = {s: df for s, df in doc_tables.items()
                   if s in POSITIVE_SOURCES and df["label"].nunique() == 2}
    per_fit: dict[str, dict[str, tuple[float, float, float]]] = {}
    if pos_sources:
        nums, pf = curve_numbers(ev, pos_sources, index, "macro_auc")
        rb.numbers.update(nums)
        per_fit["macro"] = pf
    else:
        rb.note("E2: no positive test source with both classes; macroAUC not computed")
    for s, df in doc_tables.items():
        if s == "notinject" or df["label"].nunique() < 2:
            continue
        nums, pf = curve_numbers(ev, {s: df}, index, f"auc/{s}", pairs=())
        rb.numbers.update(nums)
        per_fit[s] = pf
    curve_rows = []
    for it in index:
        row: dict[str, Any] = {"level": it.level, "rep": it.rep, "detector": it.det}
        m = per_fit.get("macro", {}).get(it.name)
        row.update({"macro_auc": m[0] if m else None, "macro_ci_low": m[1] if m else None,
                    "macro_ci_high": m[2] if m else None})
        for s in sorted(per_fit):
            if s != "macro" and it.name in per_fit[s]:
                row[f"auc_{s}"] = per_fit[s][it.name][0]
        curve_rows.append(row)
    rb.add_table("fewshot_curve", curve_rows)
    rb.add_table("fewshot_sources", [{"source": s, "n_docs": int(len(df)), "n_pos": int((df["label"] == 1).sum()),
                                      "n_neg": int((df["label"] == 0).sum()), "n_clusters": int(df["cluster_id"].nunique()),
                                      "in_macro": s in pos_sources} for s, df in doc_tables.items()])
    rb.note(f"E2: {len(index)} fits ({len(detectors)} detectors x levels {[r['level'] for r in level_rows if r['status'] == STATUS_OK]}"
            f" x {n_reps} subsamples, full = 1); macroAUC over {sorted(pos_sources)}; bootstrap n={ev.n_boot}, "
            f"alpha={ev.alpha}; gamma/C validated per fit on deepset val")
    return {"doc_tables": doc_tables, "index": index, "fitted": fitted}


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path | None = None,
        cfg: Configs | None = None, keep_cache: bool = False, force: bool = False, access_log: Any = None,
        guard_factory: Any = None, cache: bool = True) -> Path:
    """Run E2 for one global seed and write ``results/E2/<seed>.json`` (``results/smoke/E2/`` in smoke mode); the
    ``run(ctx, seed, smoke)`` signature shared by ``e1`` … ``e6`` (``ctx`` may be ``None``: the Runner builds one;
    ``root`` defaults to the context's root). Skipped when the result file is current unless ``force``."""
    root = Path(root) if root is not None else (ctx.root if ctx is not None else ROOT)
    runner = Runner(EXPERIMENT, seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                    access_log=access_log, guard_factory=guard_factory, cache=cache)
    return runner.run(lambda fc, rb: run_e2(fc, rb))


def run_seed(seed: int, smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None, force: bool = False,
             keep_cache: bool = False, **kwargs: Any) -> Path:
    """The plug-in entry of ``flyguard.experiments.run`` (``python -m flyguard.experiments.run E2 --seeds ...``)."""
    return run(None, seed, smoke, root=root, cfg=cfg, force=force, keep_cache=keep_cache, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E2 learning curves (one global seed per call)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-cache", action="store_true")
    a = p.parse_args(argv)
    path = run(None, a.seed, a.smoke, force=a.force, keep_cache=a.keep_cache)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
