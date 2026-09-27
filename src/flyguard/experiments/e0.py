"""E0 -- power (ТЗ Этап 0, docs/design_experiments.md §3): ``results/power.json`` and ``results/E0/<seed>.json``.

What enters (ТЗ: "Расчёт использует размеры наборов и валидационные прогоны; тестовые метки и оценки не участвуют"):

* **Sizes.** Per test source the number of positives, negatives and clusters and the label composition of every
  cluster (``eval.power`` resamples the observed composition: BIPIA pairs stay pairs, AgentDojo task clusters keep
  their clean/injected mix). ``splits.json`` / ``pools.json`` carry only totals, so the counts are aggregated from
  the ``label`` / ``cluster_id`` columns of the test documents, read through the single journaled door
  (``Context.test_documents`` -> ``log_data_access(..., "test", "E0 ... counts only")``). No document label enters
  the simulation individually and no text or score is touched -- these are the "размеры наборов" the ТЗ lists as
  E0's first output ("число позитивов, негативов, кластеров"). Pool sizes |P_test| / |P_val| come from
  ``pools.json``.
* **Validation runs** (validation documents only: deepset val, BIPIA val, AgentDojo val tasks): the validation AUC
  of the planning detectors :data:`PLANNING_DETECTORS` per source picks the binormal level of each cell (the *lower*
  of the two, because in-distribution deepset validation overstates cross-dataset AUC and a lower level is the
  conservative power assumption); the spread of the validation macroAUC of the Bloom fly over
  ``E0.curveball_null_on_val`` Curveball null matrices (same nose and π, seed ``curveball``) and over the ``perm``
  children of the first ``E0.perm_seeds`` global seeds (same SVD, π redrawn; the fly of the E0 seed itself is one of
  them), γ chosen on deepset validation exactly as the engine does.
* **Rules that need no data**: the |P_test| carrier rule for TPR@FPR and the 339-pair NotInject interval width.

Stages. ``--stage 1`` runs before traces and paraphrases are in the tables, ``--stage 2`` after them and freezes the
result (``frozen: true``); the acceptance check compares its timestamp with the first E1 test read. Both stages
write ``results/power.json`` (``results/smoke/power.json`` in smoke mode, see :mod:`results`) and keep a copy under
``results/E0/power_stage<k>.json``. Idempotency is decided here, not by :class:`Runner` (whose skip rule keys on the
seed file and the config hash only, which would let stage 2 be skipped after stage 1): a stage is skipped when
``power.json`` exists with the current ``config_hash``, the same stage and seed, the matching ``frozen`` flag and a
current seed file.

The freeze is final (ТЗ «Задача выполнена» item 4: the power table is written before the final run). A frozen
``power.json`` that is not skipped -- the configs changed after the freeze, another seed, a missing seed file -- is
never recomputed silently: both stages raise unless ``force``, because a re-freeze after test results were read
would move ``created_at`` past them and the acceptance check (``created_at`` earlier than every E1 seed file) would
still pass. With ``force`` (a deliberate, journaled re-freeze: DEVIATIONS) the old table is first archived as
``results/E0/power_frozen_<hash>_<created_at>.json`` and the new table records it (path relative to the root) under ``refrozen_over``. Smoke
mode (``results/smoke/``) re-freezes over a stale table without ``force`` -- smoke results never feed the real
verdicts and ``run_all.sh --smoke`` has no ``--force`` -- but archives and records it the same way.

The per-seed result file holds the same numbers under stable keys (``size/<source>/n_pos``,
``power/<cell>/<level>/mdd`` ...), the tables ``carriers`` (источник × метрика -> статус), ``cells``, ``sizes``,
``spread_values`` and ``val_auc``, and ``stage`` / ``frozen`` as top-level fields. ``power.json`` carries the code
provenance of every results writer (``results.provenance``, ASSUMPTIONS A41/A52): ``git_commit``, ``git_dirty`` and
``timing.threads``; the π-permutation fits run under ``threadpool_limits(1)`` like the engine's fits.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from flyguard import fly, nose
from flyguard.config import ROOT, Configs, config_hash, load_configs, seeds_for
from flyguard.eval.metrics import auc, macro_auc
from flyguard.eval.power import power_table
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import FeatureContext, ResultBuilder, Runner, by_source, fly_spec
from flyguard.io import atomic_write_json, read_json
from flyguard.readout import BloomReadout, select_gamma

PLANNING_DETECTORS = ("real_fly_bloom", "tfidf_lr")
SPREAD_DETECTOR = "real_fly_bloom"
POWER_OVERRIDE_KEYS = ("n_rep", "n_boot", "icc", "corr", "delta_grid", "power_target", "min_class_docs")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fin(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ----------------------------------------------------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------------------------------------------------
def sizes_by_source(ctx: Context, purpose: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``{source: {n_pos, n_neg, cluster_label_sizes}}`` of the E1 test sources (the input of
    :func:`flyguard.eval.power.power_table`) plus ``extra`` with the deep paraphrase stratum (H1a's ``para_deep``
    row of E1, reported beside the ``para`` cell because macroAUC must not count it twice). Counts only; the read
    goes through the journaled door with ``purpose``."""
    sizes: dict[str, dict[str, Any]] = {}
    extra: dict[str, Any] = {}
    for s in ctx.test_sources:
        docs = ctx.test_documents(s, purpose)
        if "dedup_dropped" in docs.columns:
            docs = docs[~docs["dedup_dropped"].fillna(False).astype(bool)]
        sizes[s] = _composition(docs)
        if s == "para" and len(docs):
            strata = np.asarray([m.get("stratum") for m in docs["meta"]], dtype=object)
            deep = docs[strata == "deep"]
            if len(deep):
                comp = _composition(deep)
                extra["para_deep"] = {"n_pos": comp["n_pos"], "n_neg": comp["n_neg"],
                                      "n_clusters": len(comp["cluster_label_sizes"]),
                                      "note": "deep paraphrase stratum (H1a semantic half); subset of the para cell"}
    return sizes, extra


def _composition(docs: pd.DataFrame) -> dict[str, Any]:
    y = docs["label"].astype(int).to_numpy()
    comp = pd.DataFrame({"c": docs["cluster_id"].astype(str).to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
    return {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
            "cluster_label_sizes": [[int(p), int(c - p)] for p, c in zip(comp["sum"], comp["count"])]}


# ----------------------------------------------------------------------------------------------------------------
# Validation runs
# ----------------------------------------------------------------------------------------------------------------
def _per_source_auc(frames: Mapping[str, pd.DataFrame], col: str) -> dict[str, float]:
    return {s: float(auc(df[col].to_numpy(dtype=float), df["label"].to_numpy())) for s, df in frames.items()}


def _val_macro(fc: FeatureContext, scores: pd.Series, ws) -> tuple[float, dict[str, float]]:
    per = _per_source_auc(by_source(fc.doc_frame(scores, ws)), "score")
    return float(macro_auc(per)), per


def validation_runs(fc: FeatureContext, n_null: int, perm_seeds: Sequence[int],
                    notes: list[str] | None = None) -> dict[str, Any]:
    """The ``val_runs`` argument of ``power_table`` plus everything the report shows about it: per-source
    validation AUC of the planning detectors, the planning level input (their minimum per source), the Bloom fly's
    validation macroAUC over ``n_null`` Curveball nulls and over the π permutations of ``perm_seeds``."""
    notes = notes if notes is not None else []
    cfg = fc.cfg
    val_ws = fc.window_set("val_all")
    out: dict[str, Any] = {"val_auc": {}, "val_auc_by_detector": {}, "val_macro_auc": {}, "planning_detectors":
                           list(PLANNING_DETECTORS), "planning_rule": "min over planning detectors per source",
                           "curveball_val_macro_auc": [], "perm_val_macro_auc": [], "curveball_gamma": [],
                           "perm_gamma": [], "perm_seeds": [int(s) for s in perm_seeds], "n_null": int(n_null),
                           "val_sources": [], "n_val_docs": {}}
    if val_ws.n == 0:
        notes.append("E0: no validation windows -> planning level defaults to the middle level, no spread")
        return out
    fitted = fc.fit_many(PLANNING_DETECTORS)
    frames = by_source(fc.doc_frame(fc.score_many(fitted, val_ws), val_ws))
    if not frames:
        notes.append("E0: no validation source with both classes -> planning level defaults, no spread")
        return out
    out["val_sources"] = sorted(frames)
    out["n_val_docs"] = {s: int(len(df)) for s, df in frames.items()}
    for det in fitted:
        per = _per_source_auc(frames, det)
        out["val_auc_by_detector"][det] = per
        out["val_macro_auc"][det] = float(macro_auc(per))
    out["val_auc"] = {s: min(out["val_auc_by_detector"][d][s] for d in fitted) for s in frames}
    out["hyper"] = {det: {k: v for k, v in f.choices.items() if k in ("gamma", "C")} for det, f in fitted.items()}
    tr, vd = fc.window_set("train"), fc.window_set("val")
    if tr.n == 0 or vd.n == 0 or len(set(vd.labels)) < 2:
        notes.append("E0: no two-class deepset validation windows -> no curveball / perm spread")
        return out
    # Curveball nulls: the same nose, π, inhibition and readout; only the wiring changes (ТЗ 3.3)
    for j in range(int(n_null)):
        f = fc.fit(fly_spec(f"e0_curveball{j}", "n51_svd", f"curveball:{j}", "bloom"))
        macro, _ = _val_macro(fc, fc.score_windows(f, val_ws), val_ws)
        out["curveball_val_macro_auc"].append(macro)
        out["curveball_gamma"].append(float(f.choices["gamma"]))
    # π permutations: the SVD of this seed with the perm child of every global seed (ТЗ 2.1: π is seed variance).
    # The randomized TruncatedSVD and the fits run with the BLAS/OpenMP pools pinned to one thread, as the engine's
    # N51-svd and ``FeatureContext.fit`` do (ASSUMPTIONS A41): at 8 vs 16 threads the SVD features differed by up to
    # 2e-3, which would make the π spread depend on ``run_all.sh --jobs``.
    X_unl, X_tr, X_vd, X_va = (fc.features("n16k", w) for w in ("c_unl", tr, vd, val_ws))
    M = fc.matrix("measured")
    m = int(M.shape[0])
    k = fc.k_for(m)
    gammas = [float(g) for g in cfg.default["readout"]["bloom"]["gammas"]]
    sub = fc.seeds["subsample"]
    for sp in perm_seeds:
        with threadpool_limits(limits=1):
            n51 = nose.N51Svd(rank=fc.d_glom, seed_svd=fc.seeds["svd"], seed_perm=int(sp)).fit(X_unl)
            Z_tr, Z_vd, Z_va = (fly.fly_code(n51.transform(X), M, k) for X in (X_tr, X_vd, X_va))
            gamma, _ = select_gamma(Z_tr, tr.labels, Z_vd, vd.labels, m, k, seed_subsample=sub, gammas=gammas,
                                    cfg=cfg)
            model = BloomReadout(m, k, gamma, seed_subsample=sub).fit(Z_tr, tr.labels)
        scores = pd.Series(np.asarray(model.score(Z_va), dtype=float), index=val_ws.window_ids)
        macro, _ = _val_macro(fc, scores, val_ws)
        out["perm_val_macro_auc"].append(macro)
        out["perm_gamma"].append(float(gamma))
    notes.append(f"E0 validation runs: sources={out['val_sources']}, n_null={n_null}, n_perm={len(perm_seeds)}, "
                 f"planning AUC (min of {list(PLANNING_DETECTORS)}) = "
                 + ", ".join(f"{s}={v:.3f}" for s, v in sorted(out["val_auc"].items())))
    return out


# ----------------------------------------------------------------------------------------------------------------
# The power table of one run
# ----------------------------------------------------------------------------------------------------------------
def build_power(fc: FeatureContext, rb: ResultBuilder, stage: int,
                power_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Body of the E0 run: sizes -> validation runs -> ``power_table`` -> numbers/tables of the seed file; returns
    the ``power.json`` payload."""
    ctx, cfg = fc.ctx, fc.cfg
    purpose = (f"E0 stage {stage} seed={fc.seed}: label/cluster counts of the test documents (sizes only; "
               f"no text, no scores)")
    sizes, sizes_extra = sizes_by_source(ctx, purpose)
    pools = {"P_test": int(ctx.p_test_size), "P_val": int(len(ctx.p_val_doc_ids))}
    e0 = cfg.exp("E0")
    n_null = min(int(e0["curveball_null_on_val"]), int(ctx.n_null()))
    n_perm = int(e0["perm_seeds"])
    perm_seeds = [seeds_for(cfg, s)["perm"] for s in list(cfg.default["seeds"]["global"])[:n_perm]]
    val = validation_runs(fc, n_null, perm_seeds, rb.notes)
    overrides = {k: v for k, v in (power_overrides or {}).items()}
    bad = set(overrides) - set(POWER_OVERRIDE_KEYS)
    if bad:
        raise TypeError(f"unknown power overrides {sorted(bad)}")
    table = power_table(cfg, sizes, val_runs=val, pools=pools, seed=fc.seed, smoke=ctx.smoke, **overrides)
    for levels in table["cells"].values():   # ``power`` = the TOST power under the report's column name (additive alias)
        for cell in levels.values():
            cell.setdefault("power", cell.get("tost_power"))
    prov = results_mod.provenance(ctx.root)
    table.update({
        "experiment": "E0", "stage": int(stage), "frozen": stage == 2, "created_at": _stamp(),
        "config_hash": config_hash(ctx.root), "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"],
        "timing": {**dict(table.get("timing") or {}), "threads": prov["threads"]}, "seed": int(fc.seed),
        "seeds": dict(fc.seeds), "smoke": bool(ctx.smoke), "sources": list(ctx.test_sources),
        "sizes_extra": sizes_extra, "val_runs": val,
        "provenance": {"sizes": "aggregate label/cluster counts of the E1 test documents (journaled read, counts "
                                "only)", "pools": "pools.json", "validation": "validation documents only",
                       "test_reads": [k for k, _ in ctx.test_reads]},
    })
    _fill_result(rb, table, sizes_extra, val)
    rb.extra.update({"stage": int(stage), "frozen": stage == 2})
    return table


def _fill_result(rb: ResultBuilder, table: Mapping[str, Any], sizes_extra: Mapping[str, Any],
                 val: Mapping[str, Any]) -> None:
    for name, sz in table["sizes"].items():
        for k in ("n_pos", "n_neg", "n_clusters"):
            rb.add_number(f"size/{name}/{k}", sz.get(k))
        rb.add_table("sizes", [{"source": name, **{k: v for k, v in sz.items() if k != "sources"},
                                "sources": ",".join(sz.get("sources", [])) or None}])
    for name, sz in sizes_extra.items():
        for k in ("n_pos", "n_neg", "n_clusters"):
            rb.add_number(f"size/{name}/{k}", sz.get(k), note=sz.get("note"))
    for k, v in table["pools"].items():
        rb.add_number(f"pool/{k.lower()}", v)
    rb.add_number("fpr_target", table["fpr_target"], note="None = TPR@FPR withdrawn (AUC only)")
    for name, levels in table["cells"].items():
        for lvl, cell in levels.items():
            for k in ("mdd", "tost_power", "se_diff", "delta"):
                rb.add_number(f"power/{name}/{lvl}/{k}", cell.get(k), n=cell.get("n_rep"), note=cell.get("status"))
            rb.add_table("cells", [{"cell": name, "level": float(lvl), "planning": table["planning_level"].get(name) == float(lvl),
                                    **{k: cell.get(k) for k in ("mdd", "tost_power", "se_diff", "delta", "status",
                                                                "n_rep", "n_boot", "mdd_normal_approx",
                                                                "tost_power_normal_approx")},
                                    "power_curve": ";".join(f"{d}:{p:.3f}" for d, p in cell.get("power_curve", {}).items())}])
    for name, lvl in table["planning_level"].items():
        rb.add_number(f"planning_level/{name}", lvl)
    for name, row in table["carriers"].items():
        rb.add_table("carriers", [{"source": name, "metric": m, "status": st} for m, st in row.items()])
    for kind in ("curveball", "perm"):
        spr = table["spread"].get(kind)
        if spr:
            for k in ("mean", "sd", "min", "max", "half_width", "sd_over_delta"):
                if k in spr:
                    rb.add_number(f"spread/{kind}/{k}", spr[k], n=spr.get("n"))
        values = val.get(f"{kind}_val_macro_auc") or []
        gam = val.get(f"{kind}_gamma") or []
        ids = val.get("perm_seeds") if kind == "perm" else list(range(len(values)))
        rb.add_table("spread_values", [{"kind": kind, "index": i, "seed_perm": (ids[i] if kind == "perm" else None),
                                        "val_macro_auc": v, "gamma": (gam[i] if i < len(gam) else None)}
                                       for i, v in enumerate(values)])
    for det, per in (val.get("val_auc_by_detector") or {}).items():
        for s, a in per.items():
            rb.add_number(f"val_auc/{s}/{det}", a, n=(val.get("n_val_docs") or {}).get(s))
            rb.add_table("val_auc", [{"detector": det, "source": s, "auc": a, "n_docs": (val.get("n_val_docs") or {}).get(s)}])
        rb.add_number(f"val_macro_auc/{det}", (val.get("val_macro_auc") or {}).get(det),
                      note="sources=" + ",".join(val.get("val_sources") or []))
    ni = table["notinject"]
    rb.add_number("notinject/n", ni["n"])
    rb.add_number("notinject/worst_case_width", ni["worst_case_width"], n=ni["n"])
    for fpr, row in ni["by_fpr"].items():
        rb.add_number(f"notinject/ci_width/{fpr}", row["ci_width"], n=ni["n"])
        rb.add_number(f"notinject/max_abs_diff_for_corridor/{fpr}", row["max_abs_diff_for_corridor"], n=ni["n"])
    hyp = table["hypotheses"]
    rows = [{"hypothesis": "H1a", "item": f"template/{s}", "status": st} for s, st in hyp["H1a"]["template"].items()]
    rows += [{"hypothesis": "H1a", "item": f"semantic_mdd/{s}", "status": None if v is None else f"mdd={v:g}"}
             for s, v in hyp["H1a"]["semantic_mdd"].items()]
    rows += [{"hypothesis": h, "item": "macro", "status": hyp[h]["macro"]} for h in ("H1b", "H3")]
    rows += [{"hypothesis": "H2", "item": "notinject_corridor_reachable_worst_case",
              "status": str(ni["corridor_reachable_worst_case"])}]
    rb.add_table("hypotheses", rows)
    rb.note(f"E0: sources={table['sources']}, pools={table['pools']}, fpr_target={table['fpr_target']}, "
            f"stage={table['stage']}, frozen={table['frozen']}")
    rb.note("E0 sizes are aggregate counts of test labels/clusters read through the journaled door; "
            "no test score or text was computed or read")


# ----------------------------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------------------------
def power_copy_path(stage: int, root: Path = ROOT, smoke: bool = False) -> Path:
    return results_mod.results_dir(root, smoke) / "E0" / f"power_stage{int(stage)}.json"


def current_power(root: Path = ROOT, smoke: bool = False) -> dict[str, Any] | None:
    path = results_mod.power_path(root, smoke)
    return read_json(path) if path.exists() else None


def frozen_archive_path(table: Mapping[str, Any], root: Path = ROOT, smoke: bool = False) -> Path:
    """Where a frozen table is kept before it is overwritten (non-numeric stem: ``list_results`` ignores it)."""
    h = str(table.get("config_hash") or "nohash")[:12]
    stamp = "".join(ch for ch in str(table.get("created_at") or "undated") if ch.isalnum())
    return results_mod.results_dir(root, smoke) / "E0" / f"power_frozen_{h}_{stamp}.json"


def run_e0(stage: int, seed: int = 0, smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None,
           force: bool = False, keep_cache: bool = False, ctx: Context | None = None,
           access_log: Callable | None = None, guard_factory: Callable | None = None,
           power_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run E0 at ``stage`` (1 or 2) for one global seed; writes ``power.json``, its stage copy, the seed file and
    ``results/E0/summary.json``. Returns ``{"power_path", "result_path", "frozen", "skipped", "power"}``.
    ``power_overrides`` (``n_rep``, ``n_boot``, ``delta_grid`` ...) exist for tests; the CLI passes none. A frozen
    ``power.json`` is overwritten only with ``force`` (or in smoke mode), after archiving it (module docstring)."""
    if int(stage) not in (1, 2):
        raise ValueError("E0 stage must be 1 or 2")
    stage = int(stage)
    root = Path(root)
    cfg = cfg or load_configs(root)
    ppath = results_mod.power_path(root, smoke)
    existing = current_power(root, smoke)
    if existing is not None and not force:
        if stage == 1 and existing.get("frozen"):
            raise RuntimeError(f"{ppath} is frozen (stage 2); stage 1 would unfreeze it -- pass force to overwrite")
        same = (existing.get("config_hash") == config_hash(root) and int(existing.get("stage", 0)) == stage
                and bool(existing.get("frozen")) == (stage == 2) and int(existing.get("seed", -1)) == int(seed))
        if same and results_mod.is_current("E0", seed, smoke, root):
            return {"power_path": ppath, "result_path": results_mod.result_path("E0", seed, smoke, root),
                    "frozen": bool(existing.get("frozen")), "skipped": True, "power": existing}
        if existing.get("frozen") and not smoke:
            why = ("config_hash differs from the current configs" if existing.get("config_hash") != config_hash(root)
                   else f"frozen for seed {existing.get('seed')}" if int(existing.get("seed", -1)) != int(seed)
                   else "the E0 seed file is missing or stale")
            raise RuntimeError(
                f"{ppath} is frozen (created_at {existing.get('created_at')}) and would be re-frozen ({why}). "
                f"A re-freeze after the freeze is a deviation: journal it in DEVIATIONS.md and rerun "
                f"`python -m flyguard.experiments.run E0 --stage {stage} --seed {seed} --force` (the old table is "
                f"archived under results/E0/power_frozen_*.json)")
    refrozen: dict[str, Any] | None = None
    if existing is not None and existing.get("frozen"):
        refrozen = {"path": frozen_archive_path(existing, root, smoke).relative_to(root).as_posix(),
                    "config_hash": existing.get("config_hash"),
                    "created_at": existing.get("created_at"), "stage": existing.get("stage"),
                    "seed": existing.get("seed"), "reason": "force" if force else "smoke"}
    runner = Runner("E0", seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=True,
                    access_log=access_log, guard_factory=guard_factory)
    holder: dict[str, Any] = {}

    def body(fc: FeatureContext, rb: ResultBuilder) -> None:
        holder["power"] = build_power(fc, rb, stage, power_overrides)
        if refrozen is not None:
            holder["power"]["refrozen_over"] = refrozen
            rb.extra["refrozen_over"] = refrozen
            rb.note(f"E0: re-froze over the frozen power.json of {refrozen['created_at']} (config_hash "
                    f"{str(refrozen['config_hash'])[:12]}, reason={refrozen['reason']}); "
                    f"archived at {refrozen['path']}")

    result_path = runner.run(body)
    table = holder["power"]
    if refrozen is not None:
        atomic_write_json(root / refrozen["path"], existing)
        print(f"E0: frozen {ppath} re-frozen (reason={refrozen['reason']}); the old table is kept at "
              f"{refrozen['path']}", file=sys.stderr)
    atomic_write_json(ppath, table)
    atomic_write_json(power_copy_path(stage, root, smoke), table)
    results_mod.summarize("E0", smoke, root)
    return {"power_path": ppath, "result_path": result_path, "frozen": bool(table["frozen"]), "skipped": False,
            "power": table}
