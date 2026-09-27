"""Shared command line of the experiments layer (docs/design_experiments.md §3–§5)::

    python -m flyguard.experiments.run E0 --stage 1|2 [--seed s] [--smoke] [--force]
    python -m flyguard.experiments.run E1 [--seeds 0,1,...] [--smoke] [--force] [--keep-cache]
                                          [--latency-guards auto|always|never]
    python -m flyguard.experiments.run E2 ... E6 | contract   (plug-in protocol below)
    python -m flyguard.experiments.run verdicts [--smoke]     (= python -m flyguard.experiments.verdicts_run)

Every command reads the configs through :func:`flyguard.config.load_configs`, writes only through
:mod:`flyguard.experiments.results` (atomic JSON with ``config_hash`` / ``git_commit``) and, after the seed runs,
rewrites ``results/<E>/summary.json`` with :func:`flyguard.experiments.results.summarize`. ``--smoke`` switches to
the smoke tables (``data/processed/smoke``) and to ``results/smoke/`` (see :mod:`results`), ``--root`` points at
another repository tree (tests), ``--force`` reruns a seed whose result file is current.

Plug-in protocol for the other experiment modules (E2–E6, contract) so that ``run_all.sh`` has one entry point:
the module ``flyguard.experiments.<e>`` (lower-cased name, ``contract_run`` for ``contract``) may define

* ``add_cli_arguments(parser)`` -- extra options of its sub-command (optional);
* ``run_cli(args) -> Any``     -- runs the whole experiment from the parsed arguments (``args.seeds``, ``args.smoke``,
  ``args.root``, ``args.force``, ``args.keep_cache`` are always present); or, when absent,
* ``run_seed(seed, smoke=..., root=..., cfg=..., force=..., keep_cache=...) -> Path`` -- one seed; the CLI loops over
  ``--seeds`` and calls ``summarize`` afterwards; or, as the last resort,
* ``main(argv)`` accepting ``--seed s [--smoke] [--force] [--keep-cache]`` -- called once per seed (this checkout only).

Modules are imported lazily, so a not-yet-written experiment does not break the others, and importing this module
loads no torch / transformers (the engine defers the guard imports).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from flyguard.config import ROOT, Configs, load_configs
from flyguard.experiments import results as results_mod

EXPERIMENT_MODULES = {"E0": "e0", "E1": "e1", "E2": "e2", "E3": "e3", "E4": "e4", "E5": "e5", "E6": "e6",
                      "contract": "contract_run", "verdicts": "verdicts_run"}


# ----------------------------------------------------------------------------------------------------------------
# Seeds
# ----------------------------------------------------------------------------------------------------------------
def default_seeds(cfg: Configs, smoke: bool) -> list[int]:
    """The configured global seeds (``seeds.global``); smoke mode keeps the first ``smoke.seeds`` of them."""
    seeds = [int(s) for s in cfg.default["seeds"]["global"]]
    if smoke:
        return seeds[: int(cfg.default["smoke"]["seeds"])]
    return seeds


def parse_seeds(text: str | None, cfg: Configs, smoke: bool) -> list[int]:
    """``"0,1,2"``, ``"0-9"``, a mix of both, ``"all"`` or ``None`` (the configured seeds)."""
    if text is None or text.strip().lower() in ("", "all", "default"):
        return default_seeds(cfg, smoke)
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen: list[int] = []
    for s in out:
        if s not in seen:
            seen.append(s)
    return seen


# ----------------------------------------------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------------------------------------------
def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="smoke tables and results/smoke/")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--force", action="store_true", help="rerun even when the result file is current")
    parser.add_argument("--keep-cache", dest="keep_cache", action="store_true",
                        help="keep the per-seed feature cache after the run")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m flyguard.experiments.run",
                                 description="FlyGuard experiments (docs/design_experiments.md)")
    sub = ap.add_subparsers(dest="experiment", required=True)

    p0 = sub.add_parser("E0", help="power table -> results/power.json (ТЗ Этап 0)")
    p0.add_argument("--stage", type=int, choices=(1, 2), required=True,
                    help="1 = before traces/paraphrases, 2 = final (frozen)")
    p0.add_argument("--seed", type=int, default=None, help="global seed of the validation runs (default: first)")
    _common(p0)

    p1 = sub.add_parser("E1", help="main cross-dataset table (ТЗ Этап 4)")
    p1.add_argument("--seeds", default=None, help="'0,1,2', '0-9' or 'all' (default: configured seeds)")
    p1.add_argument("--latency-guards", dest="latency_guards", choices=("auto", "always", "never"), default="auto",
                    help="time the transformer guards on 200 test documents: auto = first seed only")
    _common(p1)

    for name in ("E2", "E3", "E4", "E5", "E6", "contract"):
        p = sub.add_parser(name, help=f"{name} (module flyguard.experiments.{EXPERIMENT_MODULES[name]})")
        p.add_argument("--seeds", default=None, help="'0,1,2', '0-9' or 'all' (default: configured seeds)")
        _common(p)
        mod = _try_import(name)
        if mod is not None and hasattr(mod, "add_cli_arguments"):
            mod.add_cli_arguments(p)

    pv = sub.add_parser("verdicts", help="assemble results/verdicts.json from the summaries and power.json")
    pv.add_argument("--smoke", action="store_true")
    pv.add_argument("--root", type=Path, default=ROOT)
    return ap


def _try_import(name: str, strict: bool = False):
    """Import ``flyguard.experiments.<module>``; ``strict=False`` (parser construction) returns ``None`` on *any*
    failure so that one broken or unwritten module never disables the other sub-commands, ``strict=True``
    (dispatch) lets the real error surface."""
    target = f"flyguard.experiments.{EXPERIMENT_MODULES[name]}"
    try:
        return importlib.import_module(target)
    except ImportError as exc:  # the module is not written yet, or one of its imports is missing
        if strict and exc.name != target:
            raise
        return None
    except Exception:  # noqa: BLE001 - a syntax or config error in another engineer's module
        if strict:
            raise
        return None


# ----------------------------------------------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------------------------------------------
def run_generic(name: str, args: argparse.Namespace, cfg: Configs) -> Any:
    """The plug-in protocol of the module docstring for E2–E6 and the contract run."""
    mod = _try_import(name, strict=True)
    if mod is None:
        raise SystemExit(f"{name}: module flyguard.experiments.{EXPERIMENT_MODULES[name]} is not available yet")
    args.seeds = parse_seeds(getattr(args, "seeds", None), cfg, args.smoke)
    if hasattr(mod, "run_cli"):
        return mod.run_cli(args)
    if hasattr(mod, "run_seed"):
        paths = [mod.run_seed(s, smoke=args.smoke, root=args.root, cfg=cfg, force=args.force,
                              keep_cache=args.keep_cache) for s in args.seeds]
        results_mod.summarize(name, args.smoke, args.root)
        return paths
    if hasattr(mod, "main"):  # a module with only its own ``main(argv)`` (``--seed s --smoke --force --keep-cache``)
        if Path(args.root).resolve() != ROOT.resolve():
            raise SystemExit(f"{name}: its main() runs on this checkout only; --root is not supported")
        flags = (["--smoke"] if args.smoke else []) + (["--force"] if args.force else []) \
            + (["--keep-cache"] if args.keep_cache else [])
        return [mod.main(["--seed", str(s), *flags]) for s in args.seeds]
    raise SystemExit(f"{name}: the module defines none of run_cli(args), run_seed(seed, ...), main(argv)")


def main(argv: Sequence[str] | None = None, log: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    name = args.experiment
    if name == "verdicts":
        from flyguard.experiments.verdicts_run import main as verdicts_main

        return verdicts_main((["--smoke"] if args.smoke else []) + ["--root", str(args.root)])
    cfg = load_configs(args.root)
    if name == "E0":
        from flyguard.experiments.e0 import run_e0

        seed = args.seed if args.seed is not None else default_seeds(cfg, args.smoke)[0]
        out = run_e0(stage=args.stage, seed=seed, smoke=args.smoke, root=args.root, cfg=cfg, force=args.force,
                     keep_cache=args.keep_cache)
        log(f"E0 stage {args.stage}: {out['power_path']} (frozen={out['frozen']}, skipped={out['skipped']})")
        return 0
    if name == "E1":
        from flyguard.experiments.e1 import run_e1

        seeds = parse_seeds(args.seeds, cfg, args.smoke)
        paths = run_e1(seeds, smoke=args.smoke, root=args.root, cfg=cfg, force=args.force,
                       keep_cache=args.keep_cache, latency_guards=args.latency_guards)
        for p in paths:
            log(f"E1: {p}")
        return 0
    run_generic(name, args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
