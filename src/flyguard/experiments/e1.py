"""E1 -- the main cross-dataset experiment (ТЗ Этап 4 "E1", docs/design_experiments.md §2–§3).

Per global seed: every detector of ``configs/experiments/E1.yaml`` is fitted on the deepset-train windows with its
hyperparameters chosen on deepset validation (``FeatureContext.fit``), scored on the test sources of the config
that exist in the tables (``para``/``dojo``/``dyn`` appear only after traces and paraphrases; the missing ones are
listed in ``notes``), and :func:`flyguard.experiments.engine.standard_evaluation` produces the full table of
design_experiments §3: ``auc/<source>/<detector>`` and ``macro_auc/<detector>`` with cluster-bootstrap 95 % CIs,
the paired differences of the hypothesis pairs (``diff/...`` at 95 % with the bootstrap p, ``diff90/...`` from the
same draws for TOST), τ_FPR on P_val at the E0 target with ``tpr_at_fpr/<source>/<detector>`` and
``fpr_ptest/<detector>``, τ_90(deep) / τ_90(dojo), FPR on NotInject at τ_FPR and τ_90(deep) overall, by subset and
by language stratum with the H2 paired differences, the validation AUCs behind the H2/H3 preconditions, latency per
document and state size, and every threshold as a ТЗ 2.5 record.

On top of the engine's numbers this module writes the per-seed *verdict inputs* so that ``verdicts_run`` and the
report read one file: ``tost/<metric>/<a>-<b>`` (1 = the 90 % interval lies inside ±δ, δ = ``delta_rel`` × the
reference detector's point estimate on that metric; extras ``delta``, ``reference``, ``differs_from_zero``,
``outside_corridor``) for every pair with a ``diff90``, and ``p_holm/auc/<source>/protectai_v2-tfidf_lr`` -- the
Holm-adjusted bootstrap p-values over the present semantic sources of H1a (``para_deep``, ``bipia``, ``dyn``).

Guards are timed (``latency_ms/protectai_v2`` ...) only for the first seed of a run by default (``latency_guards=
"auto"``): a forward pass over 200 documents does not depend on the seed and is the most expensive step of E1.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from flyguard.config import ROOT, Configs, load_configs
from flyguard.eval.tost import equivalence_margin, holm, tost
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import FeatureContext, ResultBuilder, Runner, standard_evaluation

SEMANTIC_SOURCES = ("para_deep", "bipia", "dyn")


def hypothesis_inputs(numbers: Mapping[str, Mapping[str, Any]], cfg: Configs,
                      comparator: str = "protectai_v2", lexical: str = "tfidf_lr") -> dict[str, dict[str, Any]]:
    """TOST decisions for every ``diff90/<metric>/<a>-<b>`` present and Holm over the H1a semantic sources.

    The reference of the corridor is detector ``b``'s own point estimate on ``<metric>`` (``auc/<source>/<b>`` or
    ``macro_auc/<b>``), as :func:`flyguard.eval.tost.equivalence_margin` prescribes; pairs whose reference or interval
    is missing are skipped. Returns ``{key: number dict}`` ready for :class:`ResultBuilder`.
    """
    st = cfg.default["stats"]
    delta_rel = float(st["tost"]["delta_rel"])
    alpha = float(st["bootstrap"]["alpha"])
    out: dict[str, dict[str, Any]] = {}
    for key, rec in numbers.items():
        if not key.startswith("diff90/"):
            continue
        metric, pair = key[len("diff90/"):].rsplit("/", 1)
        if "-" not in pair:
            continue
        _, b = pair.split("-", 1)
        ref_key = f"{metric}/{b}"
        ref = (numbers.get(ref_key) or {}).get("value")
        if ref is None or rec.get("ci_low") is None or rec.get("ci_high") is None or rec.get("value") is None:
            continue
        delta = equivalence_margin(float(ref), delta_rel)
        t = tost({"point": rec["value"], "low": rec["ci_low"], "high": rec["ci_high"],
                  "level": rec.get("level", 1.0 - alpha)}, delta)
        out[f"tost/{metric}/{pair}"] = results_mod.number(
            1.0 if t["equivalent"] else 0.0, note=f"reference={ref_key}", delta=delta, reference=float(ref),
            differs_from_zero=bool(t["differs_from_zero"]), outside_corridor=bool(t["outside_corridor"]),
            level=rec.get("level"))
    pvals: dict[str, float] = {}
    for s in SEMANTIC_SOURCES:
        rec = numbers.get(f"diff/auc/{s}/{comparator}-{lexical}")
        p = None if rec is None else rec.get("p")
        if p is not None and math.isfinite(float(p)):
            pvals[s] = float(p)
    if pvals:
        adj = holm(pvals)
        for s, p_adj in adj.items():
            rec = numbers[f"diff/auc/{s}/{comparator}-{lexical}"]
            low = rec.get("ci_low")
            positive = bool(low is not None and low > 0)
            out[f"p_holm/auc/{s}/{comparator}-{lexical}"] = results_mod.number(
                p_adj, note="family=" + ",".join(sorted(pvals)), p_raw=pvals[s], lower_bound_positive=positive,
                passes=bool(p_adj <= alpha and positive), alpha=alpha)
    return out


def e1_body(fc: FeatureContext, rb: ResultBuilder, latency_guards: bool = False) -> dict[str, Any]:
    """The E1 run of one seed (see the module docstring); returns the evaluation for callers that want the
    document tables (never written to disk)."""
    cfg, ctx = fc.cfg, fc.ctx
    e1 = cfg.exp("E1")
    names = [str(n) for n in e1["detectors"]]
    wanted = [str(s) for s in e1["test_sources"]]
    sources = [s for s in wanted if s in ctx.test_sources]
    missing = [s for s in wanted if s not in ctx.test_sources]
    fitted = fc.fit_many(names)
    out = standard_evaluation(fc, fitted, sources=sources, latency=True, latency_guards=latency_guards)
    rb.merge(out)
    for key, rec in hypothesis_inputs(out["numbers"], cfg,
                                      comparator=str(cfg.default["baselines"]["transformers"]["comparator"])).items():
        rb.numbers[results_mod.check_key(key)] = rec
    available = [n for n, f in fitted.items() if f.available]
    rb.note(f"E1: train_labels={e1.get('train_labels')}, sources={sources}, missing_sources={missing}, "
            f"detectors={names}, available={available}, latency_guards={latency_guards}")
    if missing:
        rb.note(f"E1: test sources absent from the tables at this stage: {missing} (their keys are not written)")
    rb.extra["e1"] = {"sources": sources, "missing_sources": missing, "detectors": names, "available": available,
                      "latency_guards": bool(latency_guards)}
    return out


def run_e1(seeds: Sequence[int], smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None,
           force: bool = False, keep_cache: bool = False, latency_guards: str | bool = "auto",
           ctx: Context | None = None, access_log: Callable | None = None, guard_factory: Callable | None = None,
           cache: bool = True) -> list[Path]:
    """Run E1 for ``seeds`` (one shared :class:`Context`, so test files are read and journaled once per process),
    then rewrite ``results/E1/summary.json``. ``latency_guards``: ``"auto"`` = the first seed of ``seeds`` only,
    ``"always"`` / ``True``, ``"never"`` / ``False``. Seeds whose result file is current are skipped unless ``force``."""
    root = Path(root)
    cfg = cfg or load_configs(root)
    ctx = ctx or Context(cfg, smoke=smoke, root=root, access_log=access_log)
    seeds = [int(s) for s in seeds]
    paths: list[Path] = []
    for s in seeds:
        if latency_guards in ("always", True):
            lg = True
        elif latency_guards in ("never", False):
            lg = False
        else:
            lg = s == seeds[0]
        runner = Runner("E1", s, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                        access_log=access_log, guard_factory=guard_factory, cache=cache)
        paths.append(runner.run(lambda fc, rb, lg=lg: e1_body(fc, rb, latency_guards=lg)))
    results_mod.summarize("E1", smoke, root)
    return paths
