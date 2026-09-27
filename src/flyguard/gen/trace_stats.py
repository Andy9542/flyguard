"""Targeted ASR and utility of our frozen traces next to the published AgentDojo / AgentDyn runs (ТЗ Этап 6, report
section 4: "сверка с опубликованными ASR и utility") -> ``results/traces_stats.json``.

    python -m flyguard.gen.trace_stats [--root .] [--out results/traces_stats.json]

Ours: ``results/shared/traces_manifest.json`` (one entry per frozen log with ``benchmark``, ``suite``, ``attack``,
``episode_class``, ``utility``, ``security``). Per benchmark and suite (plus the row ``all`` over the suites):
``n_attacked`` = ``important_instructions`` episodes without error, ``n_hijacked`` = those with ``security`` true
(contract §2: the ``hijacked`` class), ``targeted_asr = n_hijacked / n_attacked``, ``n_clean`` = clean runs without
error, ``utility_clean`` = mean utility over them, ``utility_under_attack`` = mean utility over the attacked runs.

Published: the ``runs/`` trees of the AgentDojo and AgentDyn repositories already on this machine
(``data/ext/published_runs_agentdojo``, ``data/ext/published_runs_AgentDyn``, ASSUMPTIONS A8), laid out as
``runs/<model>/<suite>/<user_task>/<attack|none>/<injection_task|none>.json``. Only undefended pipelines count:
model directories whose name ends with a defense suffix (:data:`DEFENSE_SUFFIXES`) or names Meta-SecAlign are
skipped, as are ``injection_task_*`` user-task directories (injection tasks run as user tasks), other attacks and
logs with a non-empty ``error``. The same statistics per model and suite (``targeted_asr``, ``utility_clean``,
``utility_under_attack``, ``n_attacked``, ``n_hijacked``, ``n_clean``), plus ``all``. The commit of each repository
(``git -C <dir> rev-parse HEAD``) and its origin are recorded (``published_source`` as one string
``"<repo>@<commit>, runs; ..."`` and ``published_sources`` per benchmark), because the published logs change
between commits and between benchmark versions (the ``benchmark_version`` values met are counted per model).

Data safety (CLAUDE.md): a log is parsed as JSON by code and only its scalar fields (:data:`SCALAR_FIELDS`) are
kept; messages, tool outputs and injection strings are never read into the result or printed. Nothing goes to the
network. The manifest read is journaled as split ``test`` (it carries outcomes of test episodes, no text), the
published trees as ``external``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from flyguard.config import ROOT
from flyguard.io import atomic_write_json, read_json

ATTACK = "important_instructions"
OUT = Path("results") / "traces_stats.json"
MANIFEST = Path("results") / "shared" / "traces_manifest.json"
PUBLISHED_DIRS: dict[str, Path] = {"agentdojo": Path("data") / "ext" / "published_runs_agentdojo",
                                   "agentdyn": Path("data") / "ext" / "published_runs_AgentDyn"}
RUNS = "runs"
DEFENSE_SUFFIXES = ("-tool_filter", "-spotlighting_with_delimiting", "-repeat_user_prompt", "-transformers_pi_detector",
                    "-piguard_detector", "-prompt_guard_2_detector", "-progent", "-drift", "-camel")
"""Pipeline suffixes of the defended runs in the published trees (AgentDojo defenses, AgentDyn baselines)."""
DEFENDED_MARKERS = ("meta-secalign",)
"""Model directories of defended models (a model trained against prompt injection), matched case-insensitively."""
SCALAR_FIELDS = ("suite_name", "pipeline_name", "user_task_id", "injection_task_id", "attack_type", "error",
                 "utility", "security", "benchmark_version")
ALL = "all"
AccessLog = Callable[[Path, str, str], None]


def _default_access_log(path: Path, split: str, purpose: str) -> None:
    from flyguard.netlog import log_data_access

    log_data_access(path, split=split, purpose=purpose)


# ----------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------
def _rate(num: int, den: int) -> float | None:
    return float(num) / float(den) if den else None


class _Acc:
    """Counters of one (benchmark|model, suite) cell."""

    def __init__(self) -> None:
        self.n_attacked = self.n_hijacked = self.n_clean = self.n_error = 0
        self.util_clean = self.util_attacked = 0

    def add(self, attacked: bool, utility: Any, hijacked: bool) -> None:
        if attacked:
            self.n_attacked += 1
            self.n_hijacked += int(bool(hijacked))
            self.util_attacked += int(bool(utility))
        else:
            self.n_clean += 1
            self.util_clean += int(bool(utility))

    def merge(self, other: "_Acc") -> None:
        for k in ("n_attacked", "n_hijacked", "n_clean", "n_error", "util_clean", "util_attacked"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict[str, Any]:
        return {"n_attacked": self.n_attacked, "n_hijacked": self.n_hijacked,
                "targeted_asr": _rate(self.n_hijacked, self.n_attacked), "n_clean": self.n_clean,
                "utility_clean": _rate(self.util_clean, self.n_clean),
                "utility_under_attack": _rate(self.util_attacked, self.n_attacked), "n_error": self.n_error}


def _cells_to_dict(cells: Mapping[str, _Acc]) -> dict[str, dict[str, Any]]:
    total = _Acc()
    out: dict[str, dict[str, Any]] = {}
    for suite in sorted(cells):
        total.merge(cells[suite])
        out[suite] = cells[suite].as_dict()
    if cells:
        out[ALL] = total.as_dict()
    return out


def _is_clean(attack: Any) -> bool:
    return attack is None or str(attack) in ("", "none", "None")


def _has_error(error: Any) -> bool:
    return error is not None and str(error).strip() != ""


def ours_stats(manifest: Mapping[str, Any], attack: str = ATTACK) -> dict[str, dict[str, dict[str, Any]]]:
    """``{benchmark: {suite: stats}}`` from the frozen traces manifest (module docstring). ``error`` episodes
    (``episode_class == "error"`` or a non-empty ``error``) are counted in ``n_error`` only; other templates are
    ignored."""
    cells: dict[str, dict[str, _Acc]] = {}
    for f in manifest.get("files") or []:
        clean = _is_clean(f.get("attack"))
        if not clean and str(f.get("attack")) != attack:
            continue
        acc = cells.setdefault(str(f.get("benchmark")), {}).setdefault(str(f.get("suite")), _Acc())
        if f.get("episode_class") == "error" or _has_error(f.get("error")):
            acc.n_error += 1
            continue
        hijacked = f.get("episode_class") == "hijacked" if f.get("episode_class") else bool(f.get("security"))
        acc.add(not clean, f.get("utility"), hijacked and not clean)
    return {b: _cells_to_dict(c) for b, c in sorted(cells.items())}


def is_defended(model_dir: str) -> bool:
    """True for a defended pipeline or model directory (skipped: the comparison is with undefended agents)."""
    low = model_dir.lower()
    return any(low.endswith(s) for s in DEFENSE_SUFFIXES) or any(m in low for m in DEFENDED_MARKERS)


def read_scalars(path: Path) -> dict[str, Any]:
    """The scalar fields of one published log (:data:`SCALAR_FIELDS`); the parsed messages are dropped at once."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: data.get(k) for k in SCALAR_FIELDS if not isinstance(data.get(k), (dict, list))}


def published_stats(runs_dir: Path, attack: str = ATTACK) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """``({model: {suite: stats}}, info)`` over the undefended model directories of one ``runs/`` tree."""
    models: dict[str, dict[str, _Acc]] = {}
    info: dict[str, Any] = {"skipped_defended_models": [], "skipped_injection_task_dirs": 0, "skipped_error_logs": 0,
                            "skipped_unreadable_logs": 0, "logs_read": 0, "benchmark_versions": {}}
    if not runs_dir.is_dir():
        info["missing"] = True
        return {}, info
    for model_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if is_defended(model_dir.name):
            info["skipped_defended_models"].append(model_dir.name)
            continue
        cells: dict[str, _Acc] = {}
        versions: Counter = Counter()
        for suite_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for task_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
                if task_dir.name.startswith("injection_task_"):
                    info["skipped_injection_task_dirs"] += 1
                    continue
                files = [(False, p) for p in sorted((task_dir / attack).glob("*.json"))]
                clean = task_dir / "none" / "none.json"
                if clean.is_file():
                    files.append((True, clean))
                for is_clean, path in files:
                    try:
                        rec = read_scalars(path)
                    except (OSError, ValueError):
                        info["skipped_unreadable_logs"] += 1
                        continue
                    info["logs_read"] += 1
                    acc = cells.setdefault(suite_dir.name, _Acc())
                    if _has_error(rec.get("error")):
                        info["skipped_error_logs"] += 1
                        acc.n_error += 1
                        continue
                    versions[str(rec.get("benchmark_version") or "unrecorded")] += 1
                    acc.add(not is_clean, rec.get("utility"), (not is_clean) and bool(rec.get("security")))
        if cells:
            models[model_dir.name] = cells
            info["benchmark_versions"][model_dir.name] = dict(sorted(versions.items()))
    return {m: _cells_to_dict(c) for m, c in models.items()}, info


def _git(directory: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return out.strip() or None


def repo_info(directory: Path) -> dict[str, Any]:
    """Origin URL and HEAD commit of a published-runs checkout (``None`` when it is not a git checkout)."""
    repo = _git(directory, "remote", "get-url", "origin")
    if repo and repo.endswith(".git"):
        repo = repo[:-4]
    return {"repo": repo, "commit": _git(directory, "rev-parse", "HEAD"), "path": RUNS}


def _same_model(agent: str | None, models: Iterable[str]) -> bool:
    if not agent:
        return False
    a = agent.lower().replace("/", "_")
    return any(m.lower() == a or m.lower().endswith("_" + a) for m in models)


# ----------------------------------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------------------------------
def build_stats(root: Path = ROOT, manifest_path: Path | None = None,
                published_dirs: Mapping[str, Path] | None = None, attack: str = ATTACK,
                access_log: AccessLog | None = None) -> dict[str, Any]:
    """The ``traces_stats.json`` payload (module docstring)."""
    root = Path(root)
    log = access_log or _default_access_log
    mpath = Path(manifest_path) if manifest_path is not None else root / MANIFEST
    dirs = {b: (Path(d) if Path(d).is_absolute() else root / d)
            for b, d in (published_dirs if published_dirs is not None else PUBLISHED_DIRS).items()}
    notes: list[str] = []
    manifest: dict[str, Any] = {}
    if mpath.exists():
        log(mpath, "test", "flyguard.gen.trace_stats: episode classes, utility and security per suite of the frozen "
                           "traces (manifest fields only, no text) for the comparison with the published runs")
        manifest = read_json(mpath)
    else:
        notes.append(f"{MANIFEST.as_posix()} missing: our statistics are empty")
    ours = ours_stats(manifest, attack)
    published: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    info_all: dict[str, Any] = {}
    for bench, d in sorted(dirs.items()):
        runs = d / RUNS
        if runs.is_dir():
            log(runs, "external", f"flyguard.gen.trace_stats: scalar fields (utility, security, error) of the published "
                                  f"{bench} runs; messages never read into results or printed")
        stats, info = published_stats(runs, attack)
        published[bench] = stats
        info_all[bench] = info
        sources[bench] = {**repo_info(d), "local": _rel(runs, root)}
        if info.get("missing"):
            notes.append(f"{bench}: published runs not found at {_rel(runs, root)}")
        else:
            notes.append(f"{bench}: {len(stats)} undefended models, {info['logs_read']} logs read, "
                         f"{info['skipped_error_logs']} error logs and {info['skipped_injection_task_dirs']} "
                         f"injection_task_* user-task directories skipped, defended models skipped: "
                         f"{info['skipped_defended_models']}")
    agent = manifest.get("agent_model")
    same = _same_model(agent, {m for stats in published.values() for m in stats})
    notes.append(f"targeted ASR = hijacked / attacked {attack} episodes without error (security true); utility_clean "
                 f"= share of clean runs with utility true; the row 'all' pools the suites of a benchmark")
    notes.append(f"our agent model {agent!r} is {'among' if same else 'not among'} the published models "
                 f"(same_model_published={str(same).lower()}); the published rows are other models of the same "
                 f"benchmarks, a sanity range rather than a replication")
    return {
        "attack": attack, "agent_model": agent,
        "ours": ours,
        "ours_source": {"manifest": _rel(mpath, root), "generated": manifest.get("generated"),
                        "harnesses": manifest.get("harnesses")},
        "published": published,
        "published_source": "; ".join(f"{s.get('repo') or 'unknown repo'}@{s.get('commit') or 'unknown commit'}, "
                                      f"{s['path']}" for _, s in sorted(sources.items())),
        "published_sources": sources,
        "published_info": info_all,
        "same_model_published": bool(same),
        "rules": {"defense_suffixes": list(DEFENSE_SUFFIXES), "defended_markers": list(DEFENDED_MARKERS),
                  "attack": attack, "scalar_fields": list(SCALAR_FIELDS)},
        "notes": notes,
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def write_stats(root: Path = ROOT, out: Path | None = None, **kwargs: Any) -> tuple[Path, dict[str, Any]]:
    payload = build_stats(root, **kwargs)
    path = Path(out) if out is not None else Path(root) / OUT
    if not path.is_absolute():
        path = Path(root) / path
    atomic_write_json(path, payload)
    return path, payload


def summary_lines(payload: Mapping[str, Any]) -> list[str]:
    """Counts and rates only (data-safety rule)."""
    lines = []
    for bench, suites in (payload.get("ours") or {}).items():
        a = suites.get(ALL, {})
        lines.append(f"ours {bench}: attacked={a.get('n_attacked')} hijacked={a.get('n_hijacked')} "
                     f"asr={a.get('targeted_asr')} clean={a.get('n_clean')} utility_clean={a.get('utility_clean')}")
    for bench, models in (payload.get("published") or {}).items():
        lines.append(f"published {bench}: {len(models)} models")
    return lines


def main(argv: list[str] | None = None, access_log: AccessLog | None = None) -> int:
    ap = argparse.ArgumentParser(description="ASR / utility of our traces vs the published runs -> results/traces_stats.json")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/results/traces_stats.json")
    ap.add_argument("--manifest", type=Path, default=None, help="default: <root>/results/shared/traces_manifest.json")
    ap.add_argument("--agentdojo", type=Path, default=None, help="published AgentDojo checkout (holds runs/)")
    ap.add_argument("--agentdyn", type=Path, default=None, help="published AgentDyn checkout (holds runs/)")
    args = ap.parse_args(argv)
    dirs = dict(PUBLISHED_DIRS)
    if args.agentdojo is not None:
        dirs["agentdojo"] = args.agentdojo
    if args.agentdyn is not None:
        dirs["agentdyn"] = args.agentdyn
    path, payload = write_stats(args.root, args.out, manifest_path=args.manifest, published_dirs=dirs,
                                access_log=access_log)
    for line in summary_lines(payload):
        print(line)
    print(f"trace_stats: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
