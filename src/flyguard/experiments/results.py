"""Results files (docs/design.md §9) and their per-experiment summaries; the key naming rules of
docs/design_experiments.md §3 live here so that every experiment script and ``make_report.py`` read one text.

File layout
-----------
``results/<E>/<seed>.json`` for a real run and ``results/smoke/<E>/<seed>.json`` for a smoke run (``smoke=True``).
The design wording puts both under ``results/<E>/``; they are separated because ``run_all.sh`` skips an experiment
whose ``<seed>.json`` exists with the current ``config_hash`` (design §9), and a smoke file with the same hash
would silently make the real seed 0 be skipped. ``results/power.json`` / ``results/smoke/power.json`` follow the
same rule (:func:`power_path`). Every file is written through :func:`flyguard.io.atomic_write_json` (sorted keys,
NaN -> null) and carries ``config_hash`` (:func:`flyguard.config.config_hash`) and ``git_commit``.

Result file (one per experiment and global seed)::

    {"experiment": "E1", "seed": 0, "config_hash": "...", "git_commit": "...", "seeds": {children}, "smoke": false,
     "created_at": "...Z", "timing": {..., "threads": {env, cpu_count, pools}}, "numbers": {key: number},
     "tables": {name: [row dicts]},
     "thresholds": {name: {value, source, target, n, ...}}, "notes": [str]}

A *number* is ``{"value", "ci_low", "ci_high", "n", "note"}`` (design §9) plus optional ``level`` (CI level, 0.95
unless stated), ``n_boot``, ``n_valid``, ``n_clusters`` and ``p`` (see :func:`number`). Notes carry counts and chosen
hyperparameters only, never text from the data (CLAUDE.md data-safety rule).

Key naming rules (design_experiments §3)
---------------------------------------
Keys are ``/``-separated ASCII segments; detector names are the ``configs/experiments/E1.yaml`` names (``tfidf_lr``,
``real_fly_bloom``, ``protectai_v2`` ...; a detector name never contains ``-``), sources are ``deep, bipia, dojo,
dyn, para, notinject``.

* ``auc/<source>/<detector>``            per-source test ROC-AUC with its cluster-bootstrap 95 % CI;
* ``macro_auc/<detector>``               macroAUC over the present test sources (equal weights), CI resampled inside
  each source;
* ``val_auc/<source>/<detector>``        document-level AUC on validation documents of that source (H2/H3
  preconditions); ``val_macro_auc/<detector>`` over the validation sources;
* ``tpr_at_fpr/<source>/<detector>``     TPR of that source's test positives at the frozen τ_FPR (set on P_val, target
  from E0); ``fpr_ptest/<detector>`` the FPR actually reached on P_test at the same τ;
* ``fpr_notinject/<tau>/<detector>[/<subset>|/<lang>]`` with ``<tau>`` in ``tau_fpr``, ``tau90_deep``,
  ``tau90_dojo``, ``tau80_deep`` (E6); the subsets are ``one, two, three``, the language strata ``en, non-en``;
* ``diff/<metric>/<a>-<b>``              paired difference of detector ``a`` minus ``b`` on the same documents and the
  same cluster draws at 95 % (``p`` = two-sided percentile-bootstrap p-value of the same draws), and
  ``diff90/<metric>/<a>-<b>`` the 90 % interval of the *same* draws for TOST; ``<metric>`` is ``macro_auc``,
  ``auc/<source>`` or ``fpr_notinject/<tau>``;
* ``hyper/<detector>/<param>``           the validated hyperparameter (``gamma``, ``C``) as a number;
* ``latency_ms/<detector>``              milliseconds per document over 200 test documents, full pipeline;
* ``state_bytes/<detector>``             size of the trained readout state in bytes.

Thresholds are named ``tau_fpr/<detector>``, ``tau90_deep/<detector>`` etc. and hold the ТЗ 2.5 record
``{value, source, target, n}`` (from :mod:`flyguard.eval.thresholds`).

Summary (``results/<E>/summary.json`` via :func:`summarize`)
-----------------------------------------------------------
Per number key: ``mean``, ``sd`` (ddof 1, ``None`` for one seed), ``min``, ``max``, ``n_seeds``, ``per_seed``
(``{seed: {value, ci_low, ci_high, n, ...}}`` -- the per-seed bootstrap intervals are kept as they are) and
``seed_ci_low`` / ``seed_ci_high``: a t-interval of the seed mean recomputed from the per-seed points
(``mean ± t_{1-α/2, n-1} · sd / √n``, ``None`` for one seed). Averaging the per-seed CI bounds is *not* offered:
a mean of percentile bounds is not an interval of anything. Tables are concatenated with a ``seed`` column,
thresholds and notes are kept per seed. ``config_hash`` is the common hash of the seed files; mismatching hashes
are recorded under ``config_hashes`` and flagged in ``warnings`` so that the report cannot mix runs of two configs.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from flyguard.config import ROOT, config_hash, git_commit
from flyguard.io import atomic_write_json, read_json

NUMBER_KEYS = ("value", "ci_low", "ci_high", "n", "note")
SMOKE_DIRNAME = "smoke"


# ----------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------
def results_dir(root: Path = ROOT, smoke: bool = False) -> Path:
    """``results/`` or ``results/smoke/`` (module docstring: smoke files never shadow real ones)."""
    base = Path(root) / "results"
    return base / SMOKE_DIRNAME if smoke else base


def result_path(experiment: str, seed: int, smoke: bool = False, root: Path = ROOT) -> Path:
    return results_dir(root, smoke) / str(experiment) / f"{int(seed)}.json"


def summary_path(experiment: str, smoke: bool = False, root: Path = ROOT) -> Path:
    return results_dir(root, smoke) / str(experiment) / "summary.json"


def power_path(root: Path = ROOT, smoke: bool = False) -> Path:
    """``results/power.json`` (design §9), or its smoke twin."""
    return results_dir(root, smoke) / "power.json"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def thread_info() -> dict[str, Any]:
    """The thread counts in force when a result was computed (``timing.threads`` of every result file).

    Why: floating-point reductions in BLAS/OpenMP depend on the thread count, and ``run_all.sh --jobs N`` sets it to
    ``nproc / N`` for E1–E6 only (ASSUMPTIONS A39), so a result cannot be re-derived bit for bit without knowing it.
    The engine pins the thread-sensitive steps (the randomized SVD of N51-svd and every detector fit) to one
    thread; this record makes any remaining dependence (guard inference, code outside the engine) auditable. No file paths: only the variables, the CPU count and, per loaded
    pool, its API, implementation, version and thread count."""
    try:
        from threadpoolctl import threadpool_info

        pools = [{k: p.get(k) for k in ("user_api", "internal_api", "version", "num_threads")} for p in threadpool_info()]
    except Exception:  # noqa: BLE001 - provenance only; never fail a result write over it
        pools = []
    return {"env": {k: os.environ.get(k) for k in THREAD_ENV}, "cpu_count": os.cpu_count(), "pools": pools}


def _finite(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ----------------------------------------------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------------------------------------------
def number(value: Any, ci: Mapping[str, Any] | Any | None = None, n: int | None = None, note: str | None = None,
           **extra: Any) -> dict[str, Any]:
    """One entry of ``numbers`` (design §9): ``{value, ci_low, ci_high, n, note}``.

    ``ci`` may be a :class:`flyguard.eval.bootstrap.CI` or its dict form; its ``low``/``high``/``level``/``n_boot``/
    ``n_valid``/``n_clusters`` are copied and its ``n`` is used when ``n`` is not given. Non-finite values become
    ``None`` (the JSON writer would do it anyway; doing it here keeps ``summarize`` simple). ``extra`` keys (``p``,
    ``level`` ...) are added when not ``None``.
    """
    rec: dict[str, Any] = {"value": _finite(value), "ci_low": None, "ci_high": None, "n": None, "note": note}
    if ci is not None:
        d = ci.to_dict() if hasattr(ci, "to_dict") else dict(ci)
        rec["ci_low"], rec["ci_high"] = _finite(d.get("low")), _finite(d.get("high"))
        for k in ("level", "n_boot", "n_valid", "n_clusters"):
            if d.get(k) is not None:
                rec[k] = d[k]
        if n is None and d.get("n") is not None:
            n = int(d["n"])
        if rec["value"] is None and d.get("point") is not None:
            rec["value"] = _finite(d["point"])
    if n is not None:
        rec["n"] = int(n)
    for k, v in extra.items():
        if v is not None:
            rec[k] = _finite(v) if isinstance(v, (float, np.floating)) else v
    return rec


def check_key(key: str) -> str:
    """Validate a results key against the naming rules (ASCII, ``/``-separated, no empty segment)."""
    if not isinstance(key, str) or not key or not key.isascii():
        raise ValueError(f"results key must be a non-empty ASCII string, got {key!r}")
    if any(seg == "" or seg != seg.strip() for seg in key.split("/")):
        raise ValueError(f"results key has an empty or padded segment: {key!r}")
    return key


# ----------------------------------------------------------------------------------------------------------------
# Writing and reading one seed file
# ----------------------------------------------------------------------------------------------------------------
def write_result(experiment: str, seed: int, numbers: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping]],
                 thresholds: Mapping[str, Any], notes: Sequence[str], smoke: bool = False, *, root: Path = ROOT,
                 seeds: Mapping[str, int] | None = None, timing: Mapping[str, Any] | None = None,
                 extra: Mapping[str, Any] | None = None) -> Path:
    """Write ``results/<E>/<seed>.json`` (design §9 format) atomically; returns the path.

    ``numbers`` values may be plain floats (wrapped by :func:`number`) or number dicts; keys are validated.
    ``config_hash`` and ``git_commit`` are taken at write time from the repository at ``root``; ``timing`` gains
    ``threads`` (:func:`thread_info`) unless the caller already recorded it.
    """
    nums: dict[str, Any] = {}
    for key, val in numbers.items():
        check_key(key)
        nums[key] = dict(val) if isinstance(val, Mapping) else number(val)
        for k in NUMBER_KEYS:
            nums[key].setdefault(k, None)
    payload: dict[str, Any] = {
        "experiment": str(experiment), "seed": int(seed), "config_hash": config_hash(Path(root)),
        "git_commit": git_commit(Path(root)), "seeds": dict(seeds or {}), "smoke": bool(smoke),
        "created_at": _stamp(), "timing": {"threads": thread_info(), **dict(timing or {})}, "numbers": nums,
        "tables": {str(k): [dict(r) for r in v] for k, v in tables.items()},
        "thresholds": {str(k): (dict(v) if isinstance(v, Mapping) else v) for k, v in thresholds.items()},
        "notes": [str(n) for n in notes],
    }
    if extra:
        for k, v in extra.items():
            if k in payload:
                raise ValueError(f"extra key {k!r} collides with a standard result field")
            payload[k] = v
    path = result_path(experiment, seed, smoke, root)
    atomic_write_json(path, payload)
    return path


def read_result(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def is_current(experiment: str, seed: int, smoke: bool = False, root: Path = ROOT) -> bool:
    """True when the seed file exists and its ``config_hash`` equals the current one (the ``run_all.sh`` skip rule)."""
    path = result_path(experiment, seed, smoke, root)
    if not path.exists():
        return False
    try:
        return read_json(path).get("config_hash") == config_hash(Path(root))
    except (OSError, ValueError):
        return False


def list_results(experiment: str, smoke: bool = False, root: Path = ROOT) -> list[Path]:
    """Seed files of an experiment, sorted by seed (``summary.json`` excluded)."""
    d = results_dir(root, smoke) / str(experiment)
    if not d.is_dir():
        return []
    files = [p for p in d.glob("*.json") if p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


# ----------------------------------------------------------------------------------------------------------------
# Summary over seeds
# ----------------------------------------------------------------------------------------------------------------
def seed_interval(values: Iterable[float], level: float = 0.95) -> tuple[float | None, float | None]:
    """t-interval of the mean over seeds recomputed from the per-seed points; ``(None, None)`` below two points."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=float)
    if v.size < 2:
        return None, None
    sd = float(v.std(ddof=1))
    half = float(student_t.ppf(0.5 + level / 2, v.size - 1)) * sd / math.sqrt(v.size)
    m = float(v.mean())
    return m - half, m + half


def summarize_numbers(per_seed_numbers: Mapping[int, Mapping[str, Mapping[str, Any]]],
                      level: float = 0.95) -> dict[str, Any]:
    """Aggregate ``{seed: numbers}`` into the summary shape of the module docstring."""
    keys = sorted({k for nums in per_seed_numbers.values() for k in nums})
    out: dict[str, Any] = {}
    for key in keys:
        per_seed = {int(s): dict(nums[key]) for s, nums in per_seed_numbers.items() if key in nums}
        vals = [_finite(rec.get("value")) for rec in per_seed.values()]
        finite = [v for v in vals if v is not None]
        lo, hi = seed_interval(finite, level)
        ns = {rec.get("n") for rec in per_seed.values() if rec.get("n") is not None}
        notes = [rec.get("note") for rec in per_seed.values() if rec.get("note")]
        out[key] = {
            "mean": float(np.mean(finite)) if finite else None,
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "n_seeds": len(per_seed), "n_finite": len(finite),
            "seed_ci_low": lo, "seed_ci_high": hi, "seed_ci_level": level,
            "n": (ns.pop() if len(ns) == 1 else None),
            "note": notes[0] if notes else None,
            "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)},
        }
    return out


def summarize(experiment: str, smoke: bool = False, root: Path = ROOT, write: bool = True) -> dict[str, Any]:
    """Build (and write) ``results/<E>/summary.json`` from every ``<seed>.json`` of the experiment."""
    files = list_results(experiment, smoke, root)
    if not files:
        raise FileNotFoundError(f"no seed results for {experiment} under {results_dir(root, smoke)}")
    runs = {int(p.stem): read_json(p) for p in files}
    hashes = {s: r.get("config_hash") for s, r in runs.items()}
    distinct = sorted({h for h in hashes.values() if h is not None})
    warnings: list[str] = []
    if len(distinct) > 1:
        warnings.append(f"config_hash differs between seed files: {hashes}")
    current = config_hash(Path(root))
    if distinct and distinct != [current]:
        warnings.append("config_hash of the seed files differs from the current configs")
    tables: dict[str, list[dict[str, Any]]] = {}
    for s in sorted(runs):
        for name, rows in (runs[s].get("tables") or {}).items():
            tables.setdefault(name, []).extend({**dict(r), "seed": s} for r in rows)
    summary = {
        "experiment": str(experiment), "smoke": bool(smoke), "created_at": _stamp(),
        "seeds": sorted(runs), "n_seeds": len(runs),
        "config_hash": distinct[0] if len(distinct) == 1 else None,
        "config_hashes": {str(s): h for s, h in hashes.items()},
        "config_hash_current": current,
        "git_commits": {str(s): r.get("git_commit") for s, r in runs.items()},
        "child_seeds": {str(s): r.get("seeds") for s, r in runs.items()},
        "numbers": summarize_numbers({s: r.get("numbers") or {} for s, r in runs.items()}),
        "tables": tables,
        "thresholds": {str(s): r.get("thresholds") or {} for s, r in runs.items()},
        "notes": {str(s): r.get("notes") or [] for s, r in runs.items()},
        "timing": {str(s): r.get("timing") or {} for s, r in runs.items()},
        "warnings": warnings,
    }
    if write:
        atomic_write_json(summary_path(experiment, smoke, root), summary)
    return summary
