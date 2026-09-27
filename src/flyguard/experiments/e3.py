"""E3 -- transfer across AgentDojo templates and suites (ТЗ 1.10 "Кросс-шаблонный / Кросс-сьютовый / Двойное
удержание", Этап 4 "E3 перенос по 1.10 с интерпретируемыми метриками фолдов"; design_experiments §3).

Folds come from ``splits.json["e3"]`` (``flyguard.data.splits.build_e3_folds``): ``cross_template`` (4, one held-out
attack template), ``cross_suite`` (4, one held-out suite), ``double_holdout`` (16, suite x template), each with
``train`` / ``test`` document lists, the four class counts and a ``usable`` flag (positives and negatives on both
sides). E3 trains on AgentDojo labels -- the fold structure *is* the experiment -- so fold documents carry every E1
role (``val``, ``unused``, ``test``); the ``test``-role rows enter through the single journaled door
(``Context.load_test_windows("dojo", ..., doc_ids=..., name="e3")``, one read per run) and never feed a choice of E1.

Per usable fold (one global seed):

1. *Fold validation.* ``splits.val_fraction`` of the fold's *train-side* clusters (``cluster_id`` = suite/user_task)
   is held out, drawn from ``SeedSequence([subsample, crc32("<family>/<fold>")])``; a draw that leaves one class
   in the hold-out or in the remainder is redrawn with the attempt index appended to the seed (at most
   :data:`HOLDOUT_ATTEMPTS` draws, the count is in the fold row), because a one-class hold-out cannot validate γ / C
   by AUC and small folds (smoke, few episodes) hit this often. The remainder is the fit set. γ, C and the
   baselines' C are validated on the hold-out windows (``FeatureContext.fit``), the fold threshold τ_fold on its
   *documents* with label 0. Nothing of the fold's test side is used for any choice.
2. *τ_fold* (ТЗ 1.10: "порог по негативам фолда"): ``threshold_for_fpr`` on the hold-out negatives at the FPR level
   of the ТЗ Этап 0 carrier rule applied to their count (``fpr_target_for_pool``: >= 2000 -> 1 %, >= 500 -> 5 %);
   below 500 negatives the config ``fallback`` level (5 %) is used and the record is flagged ``below_pool_min`` --
   E3 folds never reach the pool sizes of E1, and without a level the family would have no metric at all. Every
   record keeps ``{value, source, target, n, achieved_fpr}`` (ТЗ 2.5).
3. *Metrics on the test side* (dedup-excluded windows dropped, document score = max over windows), each with the
   cluster-bootstrap interval and the class counts: ``cross_template`` -> TPR of the new template's positives at
   τ_fold and AUC; ``cross_suite`` -> FPR of the unfamiliar suite's negatives at τ_fold only (ТЗ: "TPR не
   интерпретируется как перенос"); ``double_holdout`` -> TPR, FPR and AUC with ``n_pos``.

Detectors: :data:`E3_DETECTORS` (the trainable fly and lexical detectors; ``flyhash_linear`` is left out because
every FlyHash fit on a fold train set of thousands of windows is a full 327 680-cell coding pass and the Bloom
variant already carries the FlyHash column) plus :data:`E3_REFERENCES` (``regex``, inference only, as the fixed
point of every fold; guard models can be passed as references and are scored from their hash cache). Fold sets are
registered as root sets, not subsets of one AgentDojo pool, so the FlyHash codes are computed for the fold's rows
only.

C_unl of E3 (ТЗ 1.9: SVD, idf, centring and standardisation are fit on C_unl, and test clusters never enter it): the
E1 C_unl minus every AgentDojo window (:func:`e3_c_unl`), registered as ``"c_unl"`` before any nose is fitted
(:func:`register_c_unl`). The E1 C_unl holds the clean E1-validation AgentDojo documents, and those sit on the test
side of E3 folds -- every document of the held-out suite (``cross_suite``), the benign episodes of E1-validation tasks
(``cross_template``), the held-out suite's benign episodes (``double_holdout``) -- where they are negatives of
``fpr_at_tau``; E3 folds contain AgentDojo documents only, so dropping the ``dojo`` source removes every fold document
from C_unl. The choice is fold-agnostic (one nose per
seed, not per fold), so the E3 noses see no AgentDojo text at all (ASSUMPTIONS A32); the composition goes into the
``transfer_c_unl`` table and a note.

Status instead of failure: a family whose folds cannot be built or are all unusable (DEVIATIONS D6: only
``important_instructions`` was generated, so every ``cross_template`` and ``double_holdout`` fold lacks positives on
one side) is recorded with status ``не хватило данных`` and the reason (templates present / missing, the usable
rule); a usable fold whose hold-out lacks a class gets the same status with its own reason.

Keys (``<family>`` in cross_template / cross_suite / double_holdout, ``<fold>`` the builder's name, e.g. ``travel``
or ``travelximportant_instructions``):
* ``transfer/<family>/<fold>/<metric>/<detector>``  ``tpr_at_tau`` / ``fpr_at_tau`` / ``auc`` with CI, ``n_pos``,
  ``n_neg`` and ``tau`` as extra fields;
* ``transfer/<family>/<metric>/<detector>``         mean over the family's folds that were run (``n`` = folds,
  ``fold_min`` / ``fold_max``);
thresholds ``tau_fold/<family>/<fold>/<detector>``; tables ``transfer_families`` (status per family),
``transfer_folds`` (per fold: usability, counts, hold-out sizes, FPR level), ``transfer_metrics`` (per fold x
detector x metric), ``transfer_fits`` (validated hyperparameters per fold x detector), ``transfer_c_unl`` (windows and
documents of the E1 C_unl and of the E3 C_unl per source).

Caveat carried into the notes: AgentDojo environments are static (contract §10), the same injection strings recur
across tasks and suites, so fold sides can share near-identical windows; the benign side of each fold is the
builder's stated choice, not a leakage-free guarantee.

Entry points: :func:`run_e3` (the Runner body) and :func:`run` (``run(ctx, seed, smoke)``).
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from flyguard.config import ROOT, Configs
from flyguard.eval.thresholds import fpr_target_for_pool, tau_fpr_record
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import Evaluator, FeatureContext, ResultBuilder, Runner

EXPERIMENT = "E3"
PREFIX = "transfer"
FAMILIES = ("cross_template", "cross_suite", "double_holdout")
FAMILY_METRICS: dict[str, tuple[str, ...]] = {
    "cross_template": ("tpr_at_tau", "auc"),
    "cross_suite": ("fpr_at_tau",),
    "double_holdout": ("tpr_at_tau", "fpr_at_tau", "auc"),
}
E3_DETECTORS: tuple[str, ...] = ("tfidf_lr", "knn1", "centroid", "lr_svd", "real_fly_bloom", "real_fly_linear",
                                 "flyhash_bloom")
E3_REFERENCES: tuple[str, ...] = ("regex",)
HOLDOUT_ATTEMPTS = 20
STATUS_OK = "выполнено"
STATUS_PARTIAL = "частично"
STATUS_NO_DATA = "не хватило данных"
STATIC_ENV_NOTE = ("E3: AgentDojo environments are static (contract §10): the same injection strings recur across "
                   "tasks and suites, so fold train and test sides can share near-identical windows; the benign side "
                   "of each fold is the splits builder's stated choice, not a leakage-free guarantee")


# ----------------------------------------------------------------------------------------------------------------
# Manifest-level status
# ----------------------------------------------------------------------------------------------------------------
def family_reason(e3: Mapping[str, Any], family: str, folds: Sequence[Mapping[str, Any]]) -> str:
    """The Russian reason string of a family that cannot (fully) run, from the manifest alone."""
    configured = list(e3.get("templates_configured") or [])
    present = list(e3.get("templates_present") or [])
    missing = list(e3.get("templates_missing") or [])
    n_usable = sum(1 for f in folds if f.get("usable"))
    parts = []
    if not folds:
        parts.append("в splits.json нет фолдов этого семейства (нет документов AgentDojo или трассы не извлечены)")
    parts.append(f"шаблонов атак в трассах {len(present)} из {len(configured)} ({', '.join(present) or '—'}); "
                 f"отсутствуют: {', '.join(missing) or '—'}")
    if missing:
        parts.append("см. DEVIATIONS D6")
    parts.append(f"пригодных фолдов {n_usable} из {len(folds)} (правило: {e3.get('usable_rule', '')})")
    return "; ".join(parts)


def fold_reason(fold: Mapping[str, Any]) -> str:
    counts = {k: fold.get(k) for k in ("n_train_pos", "n_train_neg", "n_test_pos", "n_test_neg")}
    zero = [k for k, v in counts.items() if not v]
    return "нет " + ", ".join(zero) if zero else "usable"


# ----------------------------------------------------------------------------------------------------------------
# C_unl of E3
# ----------------------------------------------------------------------------------------------------------------
C_UNL_EXCLUDED_SOURCES: tuple[str, ...] = ("dojo",)


def e3_c_unl(ctx: Context) -> pd.DataFrame:
    """The E3 C_unl (ТЗ 1.9): the E1 C_unl without the sources E3 folds are built from (:data:`C_UNL_EXCLUDED_SOURCES`).

    ``splits.build_e3_folds`` draws every fold from AgentDojo documents, and the clean E1-validation AgentDojo
    documents of the E1 C_unl lie on the test side of the folds of all three families; fitting the
    N16k centring, the N51-svd SVD / standardisation and the TF-IDF idf on them would let the fold's test negatives
    shape the noses that score them."""
    base = ctx.c_unl_windows
    return base[~base["source"].isin(C_UNL_EXCLUDED_SOURCES)].reset_index(drop=True)


def register_c_unl(fc: FeatureContext) -> pd.DataFrame:
    """Register :func:`e3_c_unl` as the feature context's ``"c_unl"`` before any nose exists (the pattern of
    ``contract_run``). Idempotent for the same windows. ``FeatureContext.register_set`` does not invalidate noses,
    counts or features already derived from another ``"c_unl"``, so a context that holds any of them raises instead
    of silently scoring E3 with noses fit on the E1 C_unl (the Runner hands every experiment a fresh context). The
    per-seed count cache keys on the texts' hash, so sharing the ``c_unl`` file name with E1 is safe."""
    c_unl = e3_c_unl(fc.ctx)
    prior = fc._sets.get("c_unl")
    if prior is not None and prior.frame["window_id"].tolist() == c_unl["window_id"].tolist():
        return prior.frame
    stale = (bool(fc._noses) or any(k[0] == "c_unl" for k in fc._counts)
             or any(k[1] == "c_unl" for k in fc._features))
    if stale:
        raise RuntimeError("E3 needs a feature context without noses: C_unl-derived state already exists, fit on a "
                           "C_unl that contains fold test documents (ТЗ 1.9); run E3 on a fresh FeatureContext")
    fc.register_set("c_unl", c_unl)
    return c_unl


def c_unl_composition(ctx: Context, c_unl: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows of ``transfer_c_unl``: windows and documents per source in the E1 and the E3 C_unl."""
    base = ctx.c_unl_windows
    rows = []
    for src in sorted(set(base["source"]) | set(c_unl["source"])):
        b, e = base[base["source"] == src], c_unl[c_unl["source"] == src]
        rows.append({"source": src, "n_windows_e1_c_unl": int(len(b)), "n_docs_e1_c_unl": int(b["doc_id"].nunique()),
                     "n_windows_e3_c_unl": int(len(e)), "n_docs_e3_c_unl": int(e["doc_id"].nunique()),
                     "excluded": src in C_UNL_EXCLUDED_SOURCES})
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Windows of the folds
# ----------------------------------------------------------------------------------------------------------------
def load_fold_pool(fc: FeatureContext, doc_ids: set[str]) -> pd.DataFrame:
    """Windows of the given AgentDojo documents: non-test roles from the context frame, test-role rows through the
    journaled door under the memo key ``dojo#e3`` (one read per run, whatever the number of folds)."""
    ctx = fc.ctx
    frames = [ctx.windows[ctx.windows["doc_id"].isin(doc_ids)]]
    test_ids = sorted(doc_ids - set(ctx.documents["doc_id"]))
    if test_ids:
        if "dojo" not in ctx.splits["e1"]["test"]:
            raise KeyError(f"{len(test_ids)} fold documents are neither non-test nor in an AgentDojo test list")
        frames.append(ctx.load_test_windows("dojo", fc.purpose, doc_ids=test_ids, name="e3"))
    pool = pd.concat(frames, ignore_index=True)
    if "dedup_excluded" not in pool.columns:
        pool["dedup_excluded"] = False
    pool["dedup_excluded"] = pool["dedup_excluded"].fillna(False).astype(bool)
    return pool.reset_index(drop=True)


def draw_holdout(frame: pd.DataFrame, val_fraction: float, seed: int, tag: str, labels: bool = False,
                 max_attempts: int = HOLDOUT_ATTEMPTS) -> tuple[set[str], int]:
    """``val_fraction`` of the frame's clusters (at least one when there are two or more, never all), drawn from
    ``SeedSequence([seed, crc32(tag)])`` over the sorted cluster ids (content-determined, not order-determined).
    With ``labels`` the draw is repeated -- attempt ``a >= 1`` uses ``SeedSequence([seed, crc32(tag), a])`` -- until
    both the hold-out and the remainder contain both classes of ``frame["label"]``, at most ``max_attempts`` times;
    the last draw is returned when none qualifies (the caller then reports the one-class hold-out). Returns
    ``(cluster ids, attempts made)``."""
    clusters = sorted(frame["cluster_id"].astype(str).unique())
    if len(clusters) < 2:
        return set(), 0
    n_val = min(max(1, int(round(float(val_fraction) * len(clusters)))), len(clusters) - 1)
    base = [int(seed), zlib.crc32(tag.encode("utf-8"))]
    ids = frame["cluster_id"].astype(str).to_numpy()
    y = frame["label"].to_numpy() if labels else None
    chosen: set[str] = set()
    attempts = 0
    for attempt in range(max(1, int(max_attempts)) if labels else 1):
        rng = np.random.default_rng(np.random.SeedSequence(base if attempt == 0 else base + [attempt]))
        chosen = set(rng.choice(clusters, n_val, replace=False).tolist())
        attempts = attempt + 1
        if y is None:
            break
        mask = np.isin(ids, list(chosen))
        if len(np.unique(y[mask])) == 2 and len(np.unique(y[~mask])) == 2:
            break
    return chosen, attempts


def holdout_clusters(frame: pd.DataFrame, val_fraction: float, seed: int, tag: str, labels: bool = False) -> set[str]:
    """The cluster ids of :func:`draw_holdout` (the first draw when ``labels`` is False)."""
    return draw_holdout(frame, val_fraction, seed, tag, labels=labels)[0]


def fold_fpr_level(n_neg: int, cfg: Configs) -> tuple[float, bool]:
    """FPR level of τ_fold: the carrier rule on the hold-out negatives, else the config fallback (flagged)."""
    level = fpr_target_for_pool(int(n_neg), cfg)
    if level is None:
        return float(cfg.default["thresholds"]["fpr_targets"]["fallback"]), True
    return float(level), False


def _doc_label_counts(frame: pd.DataFrame) -> tuple[int, int]:
    docs = frame.groupby("doc_id")["label"].max()
    return int((docs == 1).sum()), int((docs == 0).sum())


# ----------------------------------------------------------------------------------------------------------------
# One fold
# ----------------------------------------------------------------------------------------------------------------
def run_fold(fc: FeatureContext, ev: Evaluator, family: str, fold: Mapping[str, Any], pool: pd.DataFrame | None,
             detectors: Sequence[str], references: Sequence[str], val_fraction: float) -> dict[str, Any]:
    """Fit, threshold and evaluate one fold; returns ``{status, reason, fold_row, numbers, thresholds, metric_rows,
    fit_rows}`` (numbers etc. empty when the status is ``не хватило данных``)."""
    name = str(fold["fold"])
    tag = f"{family}/{name}"
    row: dict[str, Any] = {"family": family, "fold": name, "usable": bool(fold.get("usable")),
                           **{k: fold.get(k) for k in ("n_train_pos", "n_train_neg", "n_test_pos", "n_test_neg")}}
    out: dict[str, Any] = {"status": STATUS_NO_DATA, "reason": None, "fold_row": row, "numbers": {},
                           "thresholds": {}, "metric_rows": [], "fit_rows": []}

    def fail(reason: str) -> dict[str, Any]:
        out["reason"] = reason
        row.update({"status": STATUS_NO_DATA, "reason": reason})
        return out

    if not fold.get("usable"):
        return fail(fold_reason(fold))
    if pool is None:
        return fail("окна фолда не загружены")
    train = pool[pool["doc_id"].isin(set(fold["train"]))]
    test = pool[pool["doc_id"].isin(set(fold["test"])) & ~pool["dedup_excluded"]]
    val_clusters, attempts = draw_holdout(train, val_fraction, fc.seeds["subsample"], tag, labels=True)
    row["holdout_attempts"] = attempts
    if not val_clusters:
        return fail(f"на обучающей стороне {train['cluster_id'].nunique()} кластер(а); для отложенной валидации нужно >= 2")
    val_mask = train["cluster_id"].astype(str).isin(val_clusters).to_numpy()
    fit_frame, val_frame = train[~val_mask], train[val_mask]
    if fit_frame["label"].nunique() < 2:
        return fail("после отложения валидационных кластеров в обучающей части остался один класс")
    if val_frame["label"].nunique() < 2:
        return fail("в отложенных валидационных кластерах один класс: γ/C не подобрать по AUC")
    val_pos, val_neg = _doc_label_counts(val_frame)
    if val_neg == 0:
        return fail("в отложенных валидационных кластерах нет документов без инъекции для τ_fold")
    if test.empty:
        return fail("тестовая сторона фолда пуста после исключения дубликатов")
    register_c_unl(fc)  # no-op after run_e3; guards direct calls against noses fit on the E1 C_unl
    fit_ws = fc.register_set(f"e3:{family}:{name}:fit", fit_frame)
    val_ws = fc.register_set(f"e3:{family}:{name}:val", val_frame)
    test_ws = fc.register_set(f"e3:{family}:{name}:test", test)
    fitted = fc.fit_many(list(detectors) + [r for r in references if r not in detectors], train=fit_ws, val=val_ws)
    dets = {n: f for n, f in fitted.items() if f.available}
    val_df = fc.doc_frame(fc.score_many(dets, val_ws), val_ws)
    test_df = fc.doc_frame(fc.score_many(dets, test_ws), test_ws)
    neg_val = val_df[val_df["label"] == 0]
    level, below = fold_fpr_level(len(neg_val), fc.cfg)
    test_pos, test_neg = _doc_label_counts(test)
    fit_pos, fit_neg = _doc_label_counts(fit_frame)
    row.update({"status": STATUS_OK, "reason": None, "n_fit_docs": fit_pos + fit_neg, "n_fit_pos_docs": fit_pos,
                "n_fit_windows": fit_ws.n, "n_val_docs": val_pos + val_neg, "n_val_neg_docs": val_neg,
                "n_val_clusters": len(val_clusters), "n_train_clusters": int(train["cluster_id"].nunique()),
                "n_test_docs": int(len(test_df)), "n_test_pos_docs": test_pos, "n_test_neg_docs": test_neg,
                "n_test_clusters": int(test_df["cluster_id"].nunique()), "n_test_windows": test_ws.n,
                "fpr_level": level, "below_pool_min": below})
    for det, f in dets.items():
        fit_row = {"family": family, "fold": name, "detector": det, "trained": det in detectors,
                   "n_train_windows": f.n_train, "fit_seconds": f.fit_seconds}
        fit_row.update({k: f.choices[k] for k in ("gamma", "gamma_source", "C", "C_source", "val_auc_window") if k in f.choices})
        out["fit_rows"].append(fit_row)
        if det not in val_df.columns or det not in test_df.columns:
            continue
        rec = tau_fpr_record(neg_val[det].to_numpy(dtype=float), fpr=level, source=f"fold_val_negatives:{tag}", cfg=fc.cfg)
        rec.update({"below_pool_min": below, "n_val_clusters": len(val_clusters)})
        out["thresholds"][f"tau_fold/{family}/{name}/{det}"] = rec
        tau = float(rec["value"])
        for metric in FAMILY_METRICS[family]:
            if metric == "tpr_at_tau":
                sub = test_df[test_df["label"] == 1]
                ci = ev.rate_ci(sub, tau, det) if len(sub) else None
            elif metric == "fpr_at_tau":
                sub = test_df[test_df["label"] == 0]
                ci = ev.rate_ci(sub, tau, det) if len(sub) else None
            else:
                sub = test_df
                ci = ev.auc_ci(test_df, det) if test_df["label"].nunique() == 2 else None
            if ci is None:
                out["metric_rows"].append({"family": family, "fold": name, "detector": det, "metric": metric,
                                           "value": None, "note": "no documents of the needed class on the test side"})
                continue
            num = results_mod.number(None, ci, n_pos=test_pos, n_neg=test_neg, tau=tau,
                                     note=f"tau_fold fpr<={level:g} on {len(neg_val)} hold-out negatives")
            out["numbers"][f"{PREFIX}/{family}/{name}/{metric}/{det}"] = num
            out["metric_rows"].append({"family": family, "fold": name, "detector": det, "metric": metric,
                                       "value": num["value"], "ci_low": num["ci_low"], "ci_high": num["ci_high"],
                                       "n": num["n"], "n_pos": test_pos, "n_neg": test_neg, "tau": tau,
                                       "gamma": f.choices.get("gamma"), "C": f.choices.get("C")})
    out.update({"status": STATUS_OK, "reason": None})
    return out


# ----------------------------------------------------------------------------------------------------------------
# Body and entry point
# ----------------------------------------------------------------------------------------------------------------
def run_e3(fc: FeatureContext, rb: ResultBuilder, detectors: Sequence[str] | None = None,
           references: Sequence[str] | None = None, val_fraction: float | None = None,
           families: Sequence[str] | None = None) -> dict[str, Any]:
    """The E3 body: family status from the manifest, one journaled read of the fold documents, per-fold fits,
    thresholds and metrics, family means. Returns ``{family: [fold results]}`` for tests."""
    detectors = tuple(detectors if detectors is not None else E3_DETECTORS)
    references = tuple(references if references is not None else E3_REFERENCES)
    val_fraction = float(fc.cfg.default["splits"]["val_fraction"] if val_fraction is None else val_fraction)
    families = tuple(families if families is not None else FAMILIES)
    c_unl = register_c_unl(fc)  # before any nose: SVD / idf / centring never see a fold document (ТЗ 1.9)
    e3 = fc.ctx.splits.get("e3") or {}
    ev = Evaluator(fc)
    usable = {fam: [f for f in (e3.get(fam) or []) if f.get("usable")] for fam in families}
    needed: set[str] = set()
    for folds in usable.values():
        for f in folds:
            needed.update(f["train"])
            needed.update(f["test"])
    pool = load_fold_pool(fc, needed) if needed else None
    results: dict[str, list[dict[str, Any]]] = {}
    family_rows, fold_rows, metric_rows, fit_rows = [], [], [], []
    for fam in families:
        folds = list(e3.get(fam) or [])
        runs: list[dict[str, Any]] = []
        for fold in folds:
            r = run_fold(fc, ev, fam, fold, pool, detectors, references, val_fraction)
            runs.append(r)
            fold_rows.append(r["fold_row"])
            metric_rows.extend(r["metric_rows"])
            fit_rows.extend(r["fit_rows"])
            rb.numbers.update(r["numbers"])
            rb.thresholds.update(r["thresholds"])
            if r["status"] != STATUS_OK:
                rb.note(f"E3 {fam}/{fold.get('fold')}: {STATUS_NO_DATA} ({r['reason']})")
        results[fam] = runs
        n_run = sum(1 for r in runs if r["status"] == STATUS_OK)
        n_usable = len(usable[fam])
        status = STATUS_OK if folds and n_run == len(folds) else (STATUS_PARTIAL if n_run else STATUS_NO_DATA)
        reason = None if status == STATUS_OK else family_reason(e3, fam, folds)
        family_rows.append({"family": fam, "n_folds": len(folds), "n_usable": n_usable, "n_run": n_run,
                            "status": status, "reason": reason, "metrics": ",".join(FAMILY_METRICS[fam]),
                            "templates_present": ",".join(e3.get("templates_present") or []),
                            "templates_missing": ",".join(e3.get("templates_missing") or [])})
        rb.note(f"E3 {fam}: {status}; folds run {n_run}/{len(folds)}" + (f" ({reason})" if reason else ""))
        # family means over the folds that ran
        per: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for r in runs:
            for key, num in r["numbers"].items():
                _, _, fold_name, metric, det = key.split("/")
                if num["value"] is not None:
                    per.setdefault((metric, det), []).append((fold_name, float(num["value"])))
        for (metric, det), vals in per.items():
            v = np.array([x for _, x in vals], dtype=float)
            rb.add_number(f"{PREFIX}/{fam}/{metric}/{det}", float(v.mean()), n=len(vals),
                          note="folds=" + ",".join(f for f, _ in vals), fold_min=float(v.min()), fold_max=float(v.max()),
                          fold_sd=(float(v.std(ddof=1)) if len(v) > 1 else None))
    rb.add_table("transfer_families", family_rows)
    rb.add_table("transfer_folds", fold_rows)
    rb.add_table("transfer_metrics", metric_rows)
    rb.add_table("transfer_fits", fit_rows)
    composition = c_unl_composition(fc.ctx, c_unl)
    rb.add_table("transfer_c_unl", composition)
    rb.note(STATIC_ENV_NOTE)
    rb.note(f"E3: C_unl (SVD, idf, centring; ТЗ 1.9) = E1 C_unl without {', '.join(C_UNL_EXCLUDED_SOURCES)} windows, "
            f"because the E1-validation AgentDojo documents lie on the test side of E3 folds: {len(c_unl)} windows ("
            + ", ".join(f"{r['source']} {r['n_windows_e3_c_unl']}" for r in composition if not r["excluded"])
            + "); one C_unl for all folds, so the E3 noses see no AgentDojo text (ASSUMPTIONS A32)")
    rb.note(f"E3: detectors {list(detectors)}, references {list(references)}; fold validation = {val_fraction:g} of the "
            f"train-side clusters (seed subsample, redrawn up to {HOLDOUT_ATTEMPTS} times until both sides have both "
            f"classes); tau_fold on hold-out negative documents; bootstrap n={ev.n_boot}")
    return results


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path | None = None,
        cfg: Configs | None = None, keep_cache: bool = False, force: bool = False, access_log: Any = None,
        guard_factory: Any = None, cache: bool = True) -> Path:
    """Run E3 for one global seed and write ``results/E3/<seed>.json`` (``results/smoke/E3/`` in smoke mode); the
    ``run(ctx, seed, smoke)`` signature shared by ``e1`` … ``e6`` (``ctx`` may be ``None``: the Runner builds one;
    ``root`` defaults to the context's root). Skipped when the result file is current unless ``force``."""
    root = Path(root) if root is not None else (ctx.root if ctx is not None else ROOT)
    runner = Runner(EXPERIMENT, seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                    access_log=access_log, guard_factory=guard_factory, cache=cache)
    return runner.run(lambda fc, rb: run_e3(fc, rb))


def run_seed(seed: int, smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None, force: bool = False,
             keep_cache: bool = False, **kwargs: Any) -> Path:
    """The plug-in entry of ``flyguard.experiments.run`` (``python -m flyguard.experiments.run E3 --seeds ...``)."""
    return run(None, seed, smoke, root=root, cfg=cfg, force=force, keep_cache=keep_cache, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E3 transfer folds (one global seed per call)")
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
