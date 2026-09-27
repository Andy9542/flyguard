"""E5 -- the ablation grid in the E1 setting (ТЗ Этап 4 "E5", "Вклады"; docs/design_experiments.md §3).

Grid (``configs/experiments/E5.yaml``): nose {N51-svd, N51-hash, N16k} × expansion {none, measured, random_1886,
random_327680} × readout {linear, bloom}. Rules of the ТЗ: the measured matrix only for noses with as many channels
as the matrix has inputs ("только 51-канальные носы"); Bloom only with a KC layer ("Bloom при KC-слое"); the linear
readout everywhere. Every cell is fitted on deepset train with γ / C chosen on deepset validation and evaluated on
every positive test source through :func:`flyguard.experiments.engine.standard_evaluation`.

Cells are detectors named ``<nose>__<expansion>__<readout>`` (e.g. ``n51_svd__measured__bloom``,
``n16k__random_327680__linear``); the ``cells`` table carries the structured fields (nose, expansion, readout, matrix
kind, m, k, chosen γ / C, per-source and macro AUC), so nothing needs to parse the names. Matrix kinds: ``none`` ->
no expansion (logistic regression on the nose output); ``measured`` -> the MaleCNS matrix; ``random_<m>`` -> when the
nose has the matrix's channel count and ``m`` equals its cell count, the random matrix of the *same density* (each
cell keeps its measured in-degree; identical to E4's ``random_fly_*`` null, seed ``projection``), otherwise the
engine's ``random:<m>`` (m cells with the FlyHash fan-in of ``expansion.flyhash.fan_in``; at m = 20 · 16 384 over N16k
this is the FlyHash matrix of E1).

Contributions (ТЗ "Вклады"; keys ``contrib/<kind>/.../<metric>`` with ``<metric>`` in ``macro_auc``, ``auc/<src>``,
every one with its cluster-bootstrap interval; the ``contributions`` table repeats them as rows):

* ``contrib/nose/<nose>/<metric>``                     = AUC(nose, no layer, linear);
* ``contrib/expansion/<nose>/<expansion>/<metric>``    = AUC(nose, expansion, linear) − AUC(nose, none, linear),
  paired on the same documents and cluster draws (``diff/<metric>/<cell>-<cell>`` of the engine);
* ``contrib/rule/<nose>/<expansion>/<metric>``         = AUC(Bloom) − AUC(linear) in the cell (paired);
* ``contrib/wiring/<nose>/<readout>/<metric>``         = AUC(measured) − mean_j AUC(curveball_j) for the 51-channel
  noses, the J Curveball nulls of the seed refitted with the same protocol inside E5
  (:class:`flyguard.experiments.e4.ScoreTables`), interval with the matrices fixed
  (:func:`flyguard.experiments.e4.fixed_matrix_diff_ci`), ``p`` = two-sided randomisation p. E4 carries the
  two-stage (clusters × matrices × π) version of the same quantity for N51-svd; the N51-hash row exists only here;
* ``contrib/wiring_random/<nose>/<readout>/<metric>``  = AUC(measured) − AUC(random same density) (paired), the
  cheap companion of the wiring row.

The seed's E5 cell ``n51_svd__measured__bloom`` and E4's ``real_fly_bloom`` are the same detector under the same
protocol and must agree number for number; likewise ``n51_svd__random_1886__*`` and ``random_fly_*``.
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from flyguard.config import ROOT, Configs, load_configs
from flyguard.eval.tost import randomization_p
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.e4 import (PRIMARY_METRIC, SCORE_TOLERANCE, Readout, ScoreTables, fixed_matrix_diff_ci,
                                     metric_names, point_metrics, run_seeds, seeds_to_run)
from flyguard.experiments.engine import (POSITIVE_SOURCES, Evaluator, FeatureContext, ResultBuilder, Runner, fly_spec,
                                         standard_evaluation)

EXPERIMENT = "E5"
NO_EXPANSION = "none"
MEASURED = "measured"


@dataclass(frozen=True)
class Cell:
    """One grid cell: nose, expansion label of the config, readout, the engine's matrix kind and the detector name."""

    nose: str
    expansion: str
    readout: str
    matrix: str | None

    @property
    def name(self) -> str:
        return f"{self.nose}__{self.expansion}__{self.readout}"


def matrix_kind(expansion: str, same_channels: bool, n_cells: int) -> str | None:
    """Engine matrix kind of an expansion label (module docstring); ``None`` for no expansion."""
    if expansion == NO_EXPANSION:
        return None
    if expansion == MEASURED:
        return MEASURED
    m = re.fullmatch(r"random_(\d+)", expansion)
    if not m:
        raise ValueError(f"unknown E5 expansion {expansion!r}")
    cells = int(m.group(1))
    return "random" if same_channels and cells == n_cells else f"random:{cells}"


def cells_from_config(cfg: Configs, input_dims: dict[str, int], d_glom: int, n_cells: int) -> list[Cell]:
    """The grid of ``E5.yaml`` with the ТЗ exclusions applied (measured only for ``d_glom``-channel noses, Bloom
    only with an expansion); ``input_dims`` maps each nose to its channel count."""
    e5 = cfg.exp(EXPERIMENT)
    out: list[Cell] = []
    for nose in e5["noses"]:
        same = int(input_dims[nose]) == int(d_glom)
        for exp in e5["expansions"]:
            if exp == MEASURED and not same:
                continue
            kind = matrix_kind(str(exp), same, n_cells)
            for readout in e5["readouts"]:
                if readout == "bloom" and kind is None:
                    continue
                out.append(Cell(str(nose), str(exp), str(readout), kind))
    return out


def random_partner(cells: Sequence[Cell], c: Cell) -> Cell | None:
    """The same-density random cell (matrix kind ``random``) with the nose and readout of ``c``, if any."""
    return next((x for x in cells if x.nose == c.nose and x.readout == c.readout and x.matrix == "random"), None)


def contribution_pairs(cells: Sequence[Cell], sources: Sequence[str]) -> list[tuple[str, str, str]]:
    """Paired differences the contributions need, for macroAUC and every source."""
    by = {(c.nose, c.expansion, c.readout): c for c in cells}
    pairs: list[tuple[str, str]] = []
    for c in cells:
        base = by.get((c.nose, NO_EXPANSION, "linear"))
        if c.readout == "linear" and c.expansion != NO_EXPANSION and base is not None:
            pairs.append((c.name, base.name))                                        # expansion
        if c.readout == "bloom" and (c.nose, c.expansion, "linear") in by:
            pairs.append((c.name, by[(c.nose, c.expansion, "linear")].name))        # learning rule
        rnd = random_partner(cells, c) if c.expansion == MEASURED else None
        if rnd is not None:
            pairs.append((c.name, rnd.name))                                         # wiring vs random
    return [(m, a, b) for a, b in pairs for m in metric_names(sources)]


def body(fc: FeatureContext, rb: ResultBuilder) -> None:
    """E5 for one global seed (see the module docstring for every key)."""
    cfg, ctx = fc.cfg, fc.ctx
    M = fc.matrix("measured")
    noses = [str(n) for n in cfg.exp(EXPERIMENT)["noses"]]
    cells = cells_from_config(cfg, {n: fc.input_dim(n) for n in noses}, fc.d_glom, int(M.shape[0]))
    sources = [s for s in ctx.test_sources if s in POSITIVE_SOURCES]
    t0 = time.perf_counter()
    fitted = {c.name: fc.fit(fly_spec(c.name, c.nose, c.matrix, c.readout)) for c in cells}
    out = standard_evaluation(fc, fitted, sources=sources, pairs=contribution_pairs(cells, sources), validation=True,
                              latency=False, notinject=False)
    rb.merge(out)
    nums = out["numbers"]
    rb.note(f"E5 grid: {len(cells)} cells fitted and evaluated in {time.perf_counter() - t0:.1f} s")
    doc_tables = {s: df for s, df in out["doc_tables"].items() if s in POSITIVE_SOURCES and df["label"].nunique() == 2}
    src = sorted(doc_tables)
    metrics = metric_names(src)

    # -- cells table ------------------------------------------------------------------------------------------------
    rows = []
    for c in cells:
        f = fitted[c.name]
        m, k = (fc.code_width(f.spec.code_key) if f.spec.code_key else (None, None))
        row: dict[str, Any] = {"cell": c.name, "nose": c.nose, "expansion": c.expansion, "readout": c.readout,
                               "matrix": c.matrix, "m": m, "k": k, "gamma": f.choices.get("gamma"), "C": f.choices.get("C"),
                               "val_auc_window": f.choices.get("val_auc_window")}
        for metric in metrics:
            key = f"{metric}/{c.name}" if metric != PRIMARY_METRIC else f"macro_auc/{c.name}"
            row[metric.replace("/", "_")] = (nums.get(key) or {}).get("value")
        rows.append(row)
    rb.add_table("cells", rows)

    # -- contributions from the engine's numbers ---------------------------------------------------------------------
    contrib: list[dict[str, Any]] = []

    def copy(kind: str, path: str, source_key: str, c: Cell, metric: str, **extra: Any) -> None:
        """Republish an engine number under a ``contrib/`` key and add the table row."""
        rec = nums.get(source_key)
        if rec is None:
            return
        rb.numbers[results_mod.check_key(f"contrib/{kind}/{path}/{metric}")] = dict(rec)
        contrib.append({"kind": kind, "nose": c.nose, "expansion": c.expansion, "readout": c.readout, "metric": metric,
                        "value": rec.get("value"), "ci_low": rec.get("ci_low"), "ci_high": rec.get("ci_high"),
                        "level": rec.get("level"), "p": rec.get("p"), "cell": c.name, **extra})

    by = {(c.nose, c.expansion, c.readout): c for c in cells}
    for c in cells:
        rnd = random_partner(cells, c) if c.expansion == MEASURED else None
        for metric in metrics:
            own_key = f"macro_auc/{c.name}" if metric == PRIMARY_METRIC else f"{metric}/{c.name}"
            if c.expansion == NO_EXPANSION:
                copy("nose", c.nose, own_key, c, metric)
            if c.readout == "linear" and c.expansion != NO_EXPANSION and (c.nose, NO_EXPANSION, "linear") in by:
                base = by[(c.nose, NO_EXPANSION, "linear")]
                copy("expansion", f"{c.nose}/{c.expansion}", f"diff/{metric}/{c.name}-{base.name}", c, metric)
            if c.readout == "bloom" and (c.nose, c.expansion, "linear") in by:
                lin = by[(c.nose, c.expansion, "linear")]
                copy("rule", f"{c.nose}/{c.expansion}", f"diff/{metric}/{c.name}-{lin.name}", c, metric)
            if rnd is not None:
                copy("wiring_random", f"{c.nose}/{c.readout}", f"diff/{metric}/{c.name}-{rnd.name}", c, metric,
                     random_expansion=rnd.expansion)

    # -- wiring against the Curveball nulls, refitted inside E5 for every 51-channel nose ---------------------------
    if src:
        ev = Evaluator(fc)
        labels = {s: doc_tables[s]["label"].to_numpy().astype(int) for s in src}
        by_source = {s: doc_tables[s][["doc_id", "label", "cluster_id"]] for s in src}
        nulls = fc.curveball_set()
        measured_cells = [c for c in cells if c.expansion == MEASURED]
        t0 = time.perf_counter()
        for nose in sorted({c.nose for c in measured_cells}):
            readouts = [Readout(c.readout, c.readout) for c in measured_cells if c.nose == nose]
            st = ScoreTables(fc, nose, doc_tables, readouts)
            own, _ = st.tables(M, MEASURED)
            for r in readouts:
                name = by[(nose, MEASURED, r.id)].name
                for s in src:
                    if float(np.max(np.abs(own[r.id][s] - doc_tables[s][name].to_numpy(dtype=float)))) > SCORE_TOLERANCE:
                        raise RuntimeError(f"E5 refit of {name} differs from the engine's scores on {s}")
            null_tabs: dict[str, list[dict[str, np.ndarray]]] = {r.id: [] for r in readouts}
            for N in nulls:
                tabs, _ = st.tables(N, "curveball")
                for rid, t in tabs.items():
                    null_tabs[rid].append(t)
            for r in readouts:
                c = by[(nose, MEASURED, r.id)]
                if not null_tabs[r.id]:
                    continue
                stacked = {s: np.stack([t[s] for t in null_tabs[r.id]]) for s in src}
                cis = fixed_matrix_diff_ci(by_source, own[r.id], stacked, ev.n_boot, ev.seed, ev.alpha)
                obs = point_metrics(own[r.id], labels)
                null_pts = [point_metrics(t, labels) for t in null_tabs[r.id]]
                for metric, ci in cis.items():
                    p = randomization_p(obs[metric], [q[metric] for q in null_pts], two_sided=True, cfg=cfg)
                    key = f"contrib/wiring/{nose}/{r.id}/{metric}"
                    rb.add_ci(key, ci, note=f"AUC(measured) - mean AUC over {len(nulls)} curveball nulls, matrices fixed", p=p)
                    contrib.append({"kind": "wiring", "nose": nose, "expansion": MEASURED, "readout": r.id, "metric": metric,
                                    "value": ci.point, "ci_low": ci.low, "ci_high": ci.high, "level": ci.level, "p": p,
                                    "n_null": len(nulls), "cell": c.name})
        rb.note(f"E5 wiring rows: {len(nulls)} curveball nulls refitted per 51-channel nose in {time.perf_counter() - t0:.1f} s")
    else:
        rb.note("no positive test source with both classes: contributions limited to the engine numbers")
    rb.add_table("contributions", contrib)


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path | None = None,
        cfg: Configs | None = None, keep_cache: bool = False, force: bool = False, access_log: Any = None,
        guard_factory: Any = None, cache: bool = True) -> Path:
    """Run E5 for one global seed and return the results path (``Runner`` semantics: skipped when current)."""
    root = Path(root) if root is not None else (ctx.root if ctx is not None else ROOT)
    return Runner(EXPERIMENT, seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                  access_log=access_log, guard_factory=guard_factory, cache=cache).run(body)


def run_cli(args: argparse.Namespace) -> list[Path]:
    """Plug-in entry of ``python -m flyguard.experiments.run E5`` (``args.seeds, smoke, root, force, keep_cache``):
    one shared :class:`Context` for the seeds, then ``summary.json``."""
    return run_seeds(args.seeds, smoke=args.smoke, root=args.root, force=args.force, keep_cache=args.keep_cache,
                     experiment=EXPERIMENT, body_fn=body)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E5: ablation grid nose x expansion x readout with contributions")
    ap.add_argument("--seed", type=int, action="append", help="global seed (repeatable); default: all configured")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--force", action="store_true", help="rerun even when the result file is current")
    ap.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_configs(args.root) if (args.root / "configs").is_dir() else load_configs()
    for p in run_seeds(args.seed or seeds_to_run(cfg, args.smoke), args.smoke, args.root, cfg, args.force,
                       args.keep_cache, experiment=EXPERIMENT, body_fn=body):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
