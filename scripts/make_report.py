#!/usr/bin/env python
"""scripts/make_report.py — REPORT.md (ТЗ Этап 6, sections 1–10) rendered from results/*.json and the manifests.

Usage::

    .venv/bin/python scripts/make_report.py            # REPORT.md + results/figures/ from results/
    .venv/bin/python scripts/make_report.py --smoke    # results/smoke/REPORT.md + results/smoke/figures/ from results/smoke/

Rules the renderer follows (CLAUDE.md "Numbers in the report come only from results/*.json", ТЗ "Критерии приёмки":
"каждое число REPORT.md прослеживается до results/"):

* Every number is printed as ``value [low, high] (path#key)`` or ``value (path#key)``: the parenthesised reference
  names the file (relative to the repository root) and the JSON/YAML node the number was read from. Keys of the
  results files contain ``/`` themselves, so a reference is resolved greedily: at every dict the longest prefix of
  the remaining segments that is a key is taken (``results/E1/summary.json#numbers/auc/deep/tfidf_lr`` -> the
  summary's ``numbers`` dict -> its ``auc/deep/tfidf_lr`` record). :func:`trace_numbers` is the verifier of that
  rule; ``scripts/check_acceptance.py`` and ``tests/experiments/test_report.py`` run it over the rendered text. The
  match is exact after the rendering format (:func:`token_matches`): no tolerance, an integer never matches a
  fraction. Text files (``requirements.lock``, ``audit.md``) are referenced as ``(path)`` and checked by whole-number
  substring.
* Per-seed summaries (``results/<E>/summary.json``, :mod:`flyguard.experiments.results`), ASSUMPTIONS A55: with one
  seed a number is that seed's ``value [ci_low, ci_high]`` (the 95 % cluster bootstrap of ТЗ Этап 4); with several
  seeds the value is the mean over seeds and the interval is the *envelope* of the per-seed bootstrap intervals,
  ``[min ci_low, max ci_high]`` -- a conservative summary that keeps the ТЗ interval type (a t-interval of the seed
  mean would measure only seed-to-seed variation and collapses to zero width for seed-independent detectors). The
  seed spread follows the reference: ``sd`` of the per-seed points and the number of seeds. Both bounds and the
  spread are leaves of the referenced record (``per_seed/<s>/ci_low``, ``sd``, ``n_seeds``), so they trace exactly.
* Numbers the report derives -- lengths of id lists of the manifests, sums over labels, the median of a published
  range -- are written first to ``results/report_derived.json`` (``entries/<key>/{value, from, rule}``) and referenced
  there, like the environment facts of ``results/setup.json``. Neither snapshot contains data text or document ids.
* An experiment that has not run renders as "не выполнено" cells, a part the smoke profile leaves out (ASSUMPTIONS
  A54: guard latency, the guard-heavy E6 parts, the contamination re-audit) as "не выполнено в смоуке"; the report
  is complete at every stage of ``run_all.sh`` (smoke criterion: every section present).
* Facts of the setup (versions, pins of the harnesses, KC count and in-degree, seeds, comparator) are written first to
  ``results/setup.json`` through ``flyguard.io`` (config_hash + git_commit). Its ``git_commit`` is the commit the
  *report* is rendered on; section 2 lists separately the commits the results were computed on (``git_commits`` of
  every summary, the ``git_commit`` / ``git_dirty`` of power/contract/verdicts, ASSUMPTIONS A52), compares them with
  ``git diff --quiet A B -- src scripts configs`` and prints a warning line when their code differs; the thread
  counts come from ``timing.threads`` of the result files (ASSUMPTIONS A39, A41).
* Prose carries no bare numerals: constants come with a config reference, versions and hashes sit in code spans,
  and the decimal separator is a dot. Journal ids (``D13``, ``A38``) are cited as text.
* No text of any dataset example is read or printed: only counts, statistics, ids of experiments and journals.

Figures (matplotlib, Agg) under ``results/figures/``: ROC curves by source (E1 tables ``roc``/``roc_points``,
ASSUMPTIONS A49: TPR on the common FPR grid averaged vertically over seeds, the regexes as operating points), AUC by
source with intervals (an extra view), learning curves with the min/max band over subsamples (E2, A28), the Curveball
histogram with the measured M (E4), NotInject FPR at τ_90 by detector and subset (E1), AUC by paraphrase stratum
(E1/E6) and the ablation grid as a text table (E5). Every figure is skipped, with a note, when its numbers are
missing.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from flyguard.config import ROOT, config_hash, git_commit, load_configs  # noqa: E402
from flyguard.data.build import output_dirs  # noqa: E402
from flyguard.experiments import results as R  # noqa: E402
from flyguard.io import atomic_write_json, atomic_write_text  # noqa: E402

EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
SOURCE_ORDER = ("deep", "bipia", "dojo", "dyn", "para", "para_deep", "para_shallow", "bipia_all", "notinject")
DETECTOR_ORDER = ("regex", "tfidf_lr", "knn1", "knn5", "centroid", "lr_svd", "real_fly_bloom", "real_fly_linear",
                  "flyhash_bloom", "flyhash_linear", "protectai_v2", "piguard", "prompt_guard_2")
GUARDS = ("protectai_v2", "piguard", "prompt_guard_2")
ROC_DETECTORS = ("real_fly_bloom", "flyhash_bloom", "tfidf_lr", "protectai_v2", "piguard", "prompt_guard_2")
ROC_POINT_DETECTORS = ("regex",)
ROC_TABLES = ("roc", "roc_points")          # drawn as a figure, never dumped as a table
MISSING = "не выполнено"
MISSING_SMOKE = "не выполнено в смоуке"
NA = "—"
MAX_ROWS = 60          # rows of a results table shown inline (the file keeps the rest)
NUMBER_ROWS = 400      # rows of the "remaining numbers" table of an experiment
PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "torch", "transformers", "xxhash", "datasketch",
            "langdetect", "matplotlib", "agentdojo")
HYPOTHESIS_TITLES = {"H1a": "H1a, бенчмарки", "H1b": "H1b, алгоритм мухи", "H2": "H2, слова-триггеры",
                     "H3": "H3, проводка"}
AUDIT_SECTIONS = (("Документы по источникам", "Документы по источникам (audit.md)"),
                  ("Окна", "Окна по источникам (audit.md)"),
                  ("Языки", "Языковые страты (langdetect) и доля немецкого в deepset (audit.md)"),
                  ("Длины", "Длины нормализованного текста (audit.md)"),
                  ("Состав NotInject", "Состав NotInject (audit.md)"),
                  ("BIPIA", "Задачи BIPIA (audit.md)"))


# ================================================================================================= references
def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ref(path: str, key: str | None = None) -> str:
    return f"({path}#{key})" if key else f"({path})"


def code(v: Any) -> str:
    """A string shown verbatim in a code span (versions, hashes, names): never read as a number of the report."""
    return f"`{str(v).replace('`', '')}`"


def finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(v: Any, nd: int = 3) -> str:
    """A number for the report: ints as ints, floats with ``nd`` decimals, missing as a dash."""
    if v is None or isinstance(v, bool):
        return NA if v is None else str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        # A string is a label even when it looks numeric (E2's ``level: "1"`` / ``"full"``): it is rendered as it
        # is stored, so the traceability pass finds it in the referenced row verbatim ("1", never "1.000").
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return NA
    return f"{f:.{nd}f}"


def num(value: Any, path: str, key: str, lo: Any = None, hi: Any = None, nd: int = 3) -> str:
    """``value [lo, hi] (path#key)``; ``value (path#key)`` without an interval; a dash for a missing value."""
    if value is None and lo is None:
        return NA
    s = fmt(value, nd)
    if lo is not None and hi is not None:
        s += f" [{fmt(lo, nd)}, {fmt(hi, nd)}]"
    return f"{s} {ref(path, key)}"


def seed_sort(kv: tuple[Any, Any]) -> tuple[int, str]:
    k = str(kv[0])
    return (int(k), k) if k.lstrip("-").isdigit() else (1 << 30, k)


def seed_records(rec: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    return [(str(s), r) for s, r in sorted(((rec or {}).get("per_seed") or {}).items(), key=seed_sort)
            if isinstance(r, dict)]


def envelope(rec: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """``[min ci_low, max ci_high]`` over the per-seed cluster-bootstrap intervals (ASSUMPTIONS A55)."""
    lows = [x for x in (finite(r.get("ci_low")) for _, r in seed_records(rec)) if x is not None]
    highs = [x for x in (finite(r.get("ci_high")) for _, r in seed_records(rec)) if x is not None]
    return (min(lows) if lows else None), (max(highs) if highs else None)


def n_seeds_of(rec: dict[str, Any] | None) -> int:
    if not rec:
        return 0
    if rec.get("n_seeds") is not None:
        return int(rec["n_seeds"])
    return len(seed_records(rec))


def rec_triple(rec: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    """(value, low, high) of a summary record: one seed -> its value and bootstrap CI; several seeds -> the mean over
    seeds and the envelope of the per-seed bootstrap CIs (module docstring, ASSUMPTIONS A55)."""
    if not rec:
        return None, None, None
    per = seed_records(rec)
    if per and n_seeds_of(rec) <= 1:
        one = per[0][1]
        return one.get("value"), one.get("ci_low"), one.get("ci_high")
    if per:
        lo, hi = envelope(rec)
        return rec.get("mean"), lo, hi
    if "mean" in rec:
        return rec.get("mean"), None, None
    return rec.get("value"), rec.get("ci_low"), rec.get("ci_high")


def spread(rec: dict[str, Any] | None, nd: int = 3) -> str:
    """``; sd x, сидов n`` after a multi-seed number (the seed spread next to the envelope), empty for one seed."""
    n = n_seeds_of(rec)
    if n <= 1:
        return ""
    return f"; sd {fmt(rec.get('sd'), nd)}, сидов {n}"


def rec_num(summary: dict[str, Any] | None, path: str, key: str, nd: int = 3) -> str:
    rec = ((summary or {}).get("numbers") or {}).get(key)
    if rec is None:
        return NA
    v, lo, hi = rec_triple(rec)
    s = num(v, path, f"numbers/{key}", lo, hi, nd)
    return s if s == NA else s + spread(rec, nd)


def rec_p(rec: dict[str, Any] | None, path: str, key: str) -> str:
    """The bootstrap p of a paired difference: one seed -> ``p``, several -> its range over seeds."""
    ps = [(s, r.get("p")) for s, r in seed_records(rec) if r.get("p") is not None]
    if not ps:
        return NA
    if len(ps) == 1:
        return num(ps[0][1], path, f"numbers/{key}/per_seed/{ps[0][0]}/p")
    vals = [p for _, p in ps]
    lo, hi = min(ps, key=lambda x: x[1]), max(ps, key=lambda x: x[1])
    return (f"{fmt(min(vals))}–{fmt(max(vals))} {ref(path, f'numbers/{key}/per_seed/{lo[0]}/p')}"
            f" {ref(path, f'numbers/{key}/per_seed/{hi[0]}/p')}")


def interval_rule(n_seeds: int) -> str:
    """The words for the intervals of a block with ``n_seeds`` seeds (ASSUMPTIONS A55)."""
    if n_seeds <= 1:
        return "значение и кластерный бутстреп-интервал одного сида"
    return ("среднее по сидам; интервал — огибающая [мин. нижняя, макс. верхняя граница] кластерных бутстреп-интервалов "
            "сидов (консервативно); после ссылки — sd точечных оценок по сидам и число сидов")


# ================================================================================================= derived numbers
def safe_key(key: str) -> str:
    return re.sub(r"[\s()#]+", "_", str(key))


class Derived:
    """``results/report_derived.json``: numbers the report computes from the manifests and results (lengths of id
    lists, sums over labels, min/median/max of a published range). Each entry keeps the value, the node it came
    from and the rule, so the reference ``(results/report_derived.json#entries/<key>/value)`` stays traceable."""

    def __init__(self, path_abs: Path, path_rel: str) -> None:
        self.path_abs, self.path = path_abs, path_rel
        self.entries: dict[str, dict[str, Any]] = {}

    def put(self, key: str, value: Any, source: str, rule: str) -> str:
        key = safe_key(key)
        self.entries[key] = {"value": value, "from": source, "rule": rule}
        return key

    def num(self, key: str, value: Any, source: str, rule: str, nd: int = 3) -> str:
        if value is None:
            return NA
        return num(value, self.path, f"entries/{self.put(key, value, source, rule)}/value", nd=nd)

    def stats(self, key: str, values: Iterable[Any], source: str, rule: str) -> tuple[dict[str, Any] | None, str]:
        """min / median / max / n of ``values`` -> (dict, reference to the entry)."""
        vals = sorted(v for v in (finite(x) for x in values) if v is not None)
        if not vals:
            return None, ""
        d = {"min": vals[0], "median": float(statistics.median(vals)), "max": vals[-1], "n": len(vals)}
        k = self.put(key, d, source, rule)
        return d, ref(self.path, f"entries/{k}/value")

    def write(self, provenance: dict[str, Any] | None = None) -> None:
        """``provenance``: ``config_hash`` / ``git_commit`` / ``smoke`` of the render (design §9: every results JSON)."""
        atomic_write_json(self.path_abs, {
            **dict(provenance or {}),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "numbers derived by scripts/make_report.py from the manifests and results (counts only, no text)",
            "entries": dict(sorted(self.entries.items()))})


# ================================================================================================= markdown
def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    rows = [list(r) for r in rows]
    if not rows:
        return [f"_{MISSING}_", ""]
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def truncated(rows: list, note: list[str], limit: int = MAX_ROWS) -> list:
    if len(rows) > limit:
        note.append("_Показаны первые строки таблицы; полностью она лежит в файле результатов._")
        return rows[:limit]
    return rows


def order_key(seq: Sequence[str]):
    idx = {n: i for i, n in enumerate(seq)}
    return lambda x: (idx.get(x, len(seq)), x)


def md_section(text: str | None, heading: str, src: str) -> list[str]:
    """The lines of the ``## <heading>…`` section of a markdown file written by our own code (``audit.md``, counts
    only) with a reference to that file on every line: table rows get a reference column, other lines a trailing
    reference. Numbers are then checked against the file by whole-number substring."""
    if not text:
        return []
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## ") and l[3:].strip().startswith(heading)), None)
    if start is None:
        return []
    out: list[str] = []
    in_table = False
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        s = line.rstrip()
        if not s.strip():
            in_table = False
            out.append("")
            continue
        if s.startswith("|"):
            if not in_table:
                if out and out[-1] != "":
                    out.append("")
                out.append(s + " ссылка |")
                in_table = True
            elif set(s.replace("|", "").strip()) <= set("-: "):
                out.append(s + "---|")
            else:
                out.append(s + f" {ref(src)} |")
        else:
            if in_table:
                out.append("")
            in_table = False
            out.append(f"{s} {ref(src)}")
    while out and out[-1] == "":
        out.pop()
    return out + [""] if out else []


# ================================================================================================= inputs
class Inputs:
    """Everything the report reads, loaded tolerantly (a missing file is ``None``)."""

    def __init__(self, root: Path, smoke: bool) -> None:
        self.root, self.smoke = Path(root), bool(smoke)
        self.cfg = load_configs(self.root)
        self.rdir = R.results_dir(self.root, self.smoke)
        self.processed, self.manifests = output_dirs(self.root, self.smoke)
        self.summaries: dict[str, dict[str, Any] | None] = {e: self.summary(e) for e in EXPERIMENTS}
        self.power = self.json(R.power_path(self.root, self.smoke))
        self.power_stage1_path = self.rdir / "E0" / "power_stage1.json"
        self.power_stage1 = self.json(self.power_stage1_path)
        self.verdicts = self.json(self.rdir / "verdicts.json")
        self.contract = self.json(self.rdir / "contract.json")
        self.spend = self.json(self.root / "results" / "spend.json")
        self.pilot = self.json(self.root / "results" / "pilot.json")
        self.paraphrases = self.json(self.root / "results" / "paraphrases.json")
        self.traces_stats_path = self.root / "results" / "traces_stats.json"
        self.traces_stats = self.json(self.traces_stats_path)
        self.splits = self.json(self.manifests / "splits.json")
        self.pools = self.json(self.manifests / "pools.json")
        self.dedup = self.json(self.manifests / "dedup.json")
        self.contamination_path = self.manifests / "contamination.json"
        self.contamination = self.json(self.contamination_path)
        self.audit_path = self.manifests / "audit.md"
        try:
            self.audit_text: str | None = self.audit_path.read_text(encoding="utf-8")
        except OSError:
            self.audit_text = None
        self.sources = self.json(self.root / "data" / "manifests" / "sources.json")
        self.traces_manifest = self.json(self.root / "results" / "shared" / "traces_manifest.json")
        self.split_manifest = self.json(self.root / "results" / "shared" / "split_manifest.json")
        self.extraction = self.json(self.root / "data" / "manifests" / "traces_extraction.json")
        self.para_manifest = self.json(self.root / "data" / "paraphrases" / "paraphrases_manifest.json")
        self.smoke_cfg = dict(self.cfg.default.get("smoke") or {})
        self.derived = Derived(self.rdir / "report_derived.json", self.p(self.rdir / "report_derived.json"))

    def json(self, path: Path) -> dict[str, Any] | None:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def summary(self, exp: str) -> dict[str, Any] | None:
        """``results/<E>/summary.json``, regenerated when a seed file is newer than it."""
        files = R.list_results(exp, self.smoke, self.root)
        spath = R.summary_path(exp, self.smoke, self.root)
        if files:
            newest = max(p.stat().st_mtime for p in files)
            if not spath.exists() or spath.stat().st_mtime < newest:
                try:
                    R.summarize(exp, self.smoke, self.root, write=True)
                except Exception as exc:  # noqa: BLE001 - a broken seed file must not kill the report
                    print(f"[make_report] summary of {exp} not rebuilt: {exc}", file=sys.stderr)
        return self.json(spath)

    def p(self, path: Path) -> str:
        return rel(path, self.root)

    def spath(self, exp: str) -> str:
        return self.p(R.summary_path(exp, self.smoke, self.root))

    def numbers(self, exp: str) -> dict[str, Any]:
        return ((self.summaries.get(exp) or {}).get("numbers")) or {}

    def smoke_skips(self, flag: str) -> bool:
        """True in smoke mode when the smoke profile leaves the part ``flag`` out (ASSUMPTIONS A54)."""
        return self.smoke and not bool(self.smoke_cfg.get(flag, True))

    def missing(self, smoke_flag: str | None = None) -> str:
        """The marker of an absent cell: "не выполнено в смоуке" when the smoke profile skips the part."""
        return MISSING_SMOKE if smoke_flag and self.smoke_skips(smoke_flag) else MISSING

    def contamination_skipped(self) -> bool:
        c = self.contamination
        return (not c) or bool(c.get("skipped")) or "document_level" not in c

    def first_rows(self, exp: str, name: str) -> list[tuple[int, dict[str, Any]]]:
        """(row index, row) of a summary table restricted to the first seed (data-driven tables repeat per seed)."""
        rows = ((self.summaries.get(exp) or {}).get("tables") or {}).get(name) or []
        seeds = [r.get("seed") for r in rows if r.get("seed") is not None]
        first = min(seeds) if seeds else None
        return [(i, r) for i, r in enumerate(rows) if first is None or r.get("seed") == first]

    def journal_ids(self, name: str, prefix: str) -> list[str]:
        try:
            text = (self.root / name).read_text(encoding="utf-8")
        except OSError:
            return []
        return sorted(set(re.findall(rf"\*\*({prefix}\d+)", text)), key=lambda s: int(s[len(prefix):]))


# ================================================================================================= setup snapshot
def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def write_setup(inp: Inputs) -> tuple[dict[str, Any], str]:
    """``results/setup.json``: the environment facts of ТЗ Этап 6 section 2 (no data text)."""
    d = inp.cfg.default
    conn: dict[str, Any] = {"path": inp.p(inp.root / "data/processed/connectome/malecns_R.npz"), "available": False}
    try:
        from flyguard.connectome import indegree_stats, load_malecns

        M, meta = load_malecns(inp.root / "data/processed/connectome/malecns_R.npz")
        conn.update(indegree_stats(M), available=True,
                    glomeruli=int(M.shape[1]), kenyon_cells=int(M.shape[0]),
                    meta={k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool, list))})
    except Exception as exc:  # noqa: BLE001
        conn["error"] = type(exc).__name__
    guards = {}
    for name, spec in d["baselines"]["transformers"]["models"].items():
        guards[name] = {"path": spec["path"], "present": (inp.root / spec["path"] / "config.json").exists(),
                        "optional": bool(spec.get("optional", False))}
    setup = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": config_hash(inp.root), "git_commit": git_commit(inp.root), "smoke": inp.smoke,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: package_version(p) for p in PACKAGES},
        "seeds": {"global": list(d["seeds"]["global"]), "children": list(d["seeds"]["children"]),
                  "smoke_seeds": int(d["smoke"]["seeds"])},
        "windows": dict(d["windows"]), "connectome": conn,
        "malecns": {**dict(d["expansion"]["malecns"]),
                    **((inp.sources or {}).get("pins", {}).get("malecns", {}))},
        "pins": (inp.sources or {}).get("pins", {}),
        "comparator": d["baselines"]["transformers"]["comparator"], "guards": guards,
        "agent_model": (inp.traces_manifest or {}).get("agent_model"),
        "paraphrase_models": (inp.cfg.operator.get("llm_api") or {}).get("paraphrase"),
        "bootstrap": dict(d["stats"]["bootstrap"]), "tost": dict(d["stats"]["tost"]),
        "h2_margin_pp": d["stats"]["h2_margin_pp"], "preconditions": dict(d["stats"]["preconditions"]),
        "smoke_limits": dict(d["smoke"]),
    }
    path = inp.rdir / "setup.json"
    atomic_write_json(path, setup)
    return setup, inp.p(path)


# ================================================================================================= generic dumps
def scalar_rows(obj: Any, path: str, key: str, rows: list, depth: int = 0, max_depth: int = 4,
                list_limit: int | None = MAX_ROWS) -> None:
    """Flatten scalar leaves of ``obj`` into ``(key, value (ref))`` rows (lists of scalars stay on one row)."""
    sub = (lambda k: f"{key}/{k}") if key else (lambda k: str(k))
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            if depth < max_depth:
                scalar_rows(obj[k], path, sub(k), rows, depth + 1, max_depth, list_limit)
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            rows.append((key, ", ".join(fmt(x) for x in obj[:20]) + (" …" if len(obj) > 20 else "") + " " + ref(path, key)))
        else:
            for i, x in enumerate(obj if list_limit is None else obj[:list_limit]):
                if depth < max_depth:
                    scalar_rows(x, path, sub(i), rows, depth + 1, max_depth, list_limit)
    else:
        rows.append((key, f"{fmt(obj)} {ref(path, key)}"))


def dump(obj: Any, path: str, key: str, title: str | None = None, limit: int | None = MAX_ROWS) -> list[str]:
    """Scalar leaves as a key/value table; ``limit=None`` renders every row (no truncation)."""
    rows: list = []
    scalar_rows(obj, path, key, rows, list_limit=limit)
    note: list[str] = []
    out = [f"**{title}**", ""] if title else []
    return out + table(["ключ", "значение"], rows if limit is None else truncated(rows, note, limit)) + note


def table_rows(summary: dict[str, Any] | None, name: str, path: str) -> list[str]:
    """A results table (rows of dicts) as markdown, one reference per row."""
    rows = ((summary or {}).get("tables") or {}).get(name) or []
    if not rows:
        return [f"_{MISSING}_", ""]
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c != "seed", c))
    note: list[str] = []
    body = [[fmt(r.get(c)) for c in cols] + [ref(path, f"tables/{name}/{i}")]
            for i, r in enumerate(truncated(rows, note))]
    return table(cols + ["ссылка"], body) + note


# ================================================================================================= pivots
def split_key(key: str) -> list[str]:
    return key.split("/")


def pivot_by_source(numbers: dict[str, Any], prefix: str, path: str, extra: str | None = None) -> list[str]:
    """``<prefix>/<source>/<row>`` -> rows x sources; ``extra`` (e.g. ``macro_auc``) adds a ``<extra>/<row>`` column."""
    cells: dict[str, dict[str, str]] = {}
    for key in numbers:
        parts = split_key(key)
        if len(parts) >= 3 and parts[0] == prefix:
            src, row = parts[1], "/".join(parts[2:])
            cells.setdefault(row, {})[src] = key
    if not cells:
        return [f"_{MISSING}_", ""]
    sources = sorted({s for m in cells.values() for s in m}, key=order_key(SOURCE_ORDER))
    headers = ["детектор"] + sources + ([extra] if extra else [])
    rows = []
    for row in sorted(cells, key=order_key(DETECTOR_ORDER)):
        line = [row] + [rec_num({"numbers": numbers}, path, cells[row][s]) if s in cells[row] else NA for s in sources]
        if extra:
            line.append(rec_num({"numbers": numbers}, path, f"{extra}/{row}"))
        rows.append(line)
    return table(headers, rows)


def pivot_detectors(numbers: dict[str, Any], prefixes: Sequence[str], path: str,
                    fill: Callable[[str, str], str] | None = None, extra_rows: Sequence[str] = ()) -> list[str]:
    """``<prefix>/<detector>[/<param>]`` -> detector rows with one column per prefix (and parameter); ``fill(column,
    detector)`` renders an absent cell (a smoke-skipped measurement), a dash by default; ``extra_rows`` adds detectors
    that have no number of these prefixes at all (the guards when their latency was not measured)."""
    cols: dict[str, dict[str, str]] = {}
    for key in numbers:
        parts = split_key(key)
        if len(parts) >= 2 and parts[0] in prefixes:
            col = parts[0] + ("/" + "/".join(parts[2:]) if len(parts) > 2 else "")
            cols.setdefault(col, {})[parts[1]] = key
    if not cols:
        return [f"_{MISSING}_", ""]
    names = sorted({d for m in cols.values() for d in m} | set(extra_rows), key=order_key(DETECTOR_ORDER))
    colnames = sorted(cols, key=order_key(list(prefixes)))
    rows = [[d] + [rec_num({"numbers": numbers}, path, cols[c][d], nd=(1 if c.startswith("latency") else 3))
                   if d in cols[c] else (fill(c, d) if fill else NA) for c in colnames] for d in names]
    return table(["детектор"] + colnames, rows)


def pivot_notinject(numbers: dict[str, Any], path: str) -> list[str]:
    out: list[str] = []
    taus = sorted({split_key(k)[1] for k in numbers if k.startswith("fpr_notinject/") and len(split_key(k)) >= 3})
    if not taus:
        return [f"_{MISSING}_", ""]
    for tau in taus:
        cells: dict[str, dict[str, str]] = {}
        for key in numbers:
            parts = split_key(key)
            if parts[0] == "fpr_notinject" and len(parts) >= 3 and parts[1] == tau:
                cells.setdefault(parts[2], {})[parts[3] if len(parts) > 3 else "все"] = key
        subsets = sorted({s for m in cells.values() for s in m}, key=order_key(("все", "one", "two", "three", "en", "non-en")))
        rows = [[d] + [rec_num({"numbers": numbers}, path, cells[d][s]) if s in cells[d] else NA for s in subsets]
                for d in sorted(cells, key=order_key(DETECTOR_ORDER))]
        out += [f"Порог {code(tau)}:", ""] + table(["детектор"] + subsets, rows)
    return out


def pivot_diffs(numbers: dict[str, Any], path: str) -> list[str]:
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    for key in numbers:
        parts = split_key(key)
        if parts[0] in ("diff", "diff90") and len(parts) >= 3:
            metric, pair = "/".join(parts[1:-1]), parts[-1]
            pairs.setdefault((metric, pair), {})[parts[0]] = key
    if not pairs:
        return [f"_{MISSING}_", ""]
    rows = []
    for (metric, pair), keys in sorted(pairs.items()):
        d95, d90 = keys.get("diff"), keys.get("diff90")
        rows.append([code(metric), code(pair.replace("-", " − ", 1)),
                     rec_num({"numbers": numbers}, path, d95) if d95 else NA,
                     rec_num({"numbers": numbers}, path, d90) if d90 else NA,
                     rec_p(numbers.get(d95), path, d95) if d95 else NA])
    return table(["метрика", "пара (a − b)", "разность, интервал бутстрепа", "интервал уровня TOST", "p"], rows)


KNOWN_PREFIXES = ("auc", "macro_auc", "val_auc", "val_macro_auc", "tpr_at_fpr", "fpr_ptest", "fpr_notinject",
                  "diff", "diff90", "hyper", "latency_ms", "state_bytes")


def generic_numbers(numbers: dict[str, Any], path: str, skip_prefixes: Sequence[str] = KNOWN_PREFIXES) -> list[str]:
    keys = [k for k in sorted(numbers) if split_key(k)[0] not in skip_prefixes]
    if not keys:
        return []
    note: list[str] = []
    rows = [[code(k), rec_num({"numbers": numbers}, path, k),
             fmt(numbers[k].get("n")) + (" " + ref(path, f"numbers/{k}/n") if numbers[k].get("n") is not None else ""),
             str(numbers[k].get("note") or "") + (" " + ref(path, f"numbers/{k}/note") if numbers[k].get("note") else "")]
            for k in truncated(keys, note, NUMBER_ROWS)]
    return table(["ключ", "значение [интервал]", "n", "примечание"], rows) + note


def e6_parts(inp: Inputs) -> list[str]:
    """E6 parts (``configs/experiments/E6.yaml`` flags) with their status; a part the smoke profile leaves out
    (``smoke.e6_parts``, ASSUMPTIONS A54) is "не выполнено в смоуке"."""
    try:
        from flyguard.experiments.e6 import PART_FLAGS
    except Exception:  # noqa: BLE001 - the table is informative, never fatal
        return []
    try:
        e6cfg = inp.cfg.exp("E6") or {}
    except Exception:  # noqa: BLE001
        e6cfg = {}
    s = inp.summaries.get("E6") or {}
    logged = inp.first_rows("E6", "parts")          # the table E6 writes itself (part, status, flag, reason)
    if logged:
        path = inp.spath("E6")
        body = [[code(r.get("part")),
                 f"{code(r.get('flag'))} {ref('configs/experiments/E6.yaml', str(r.get('flag')))}" if r.get("flag") else NA,
                 f"{r.get('status')} {ref(path, f'tables/parts/{i}/status')}",
                 f"{r.get('reason')} {ref(path, f'tables/parts/{i}/reason')}" if r.get("reason") else NA] for i, r in logged]
        return ["**Части E6 и их статус** (таблица `parts` E6, первый сид)", ""] + table(["часть", "флаг конфига", "статус", "причина"], body)
    notes = s.get("notes") or {}
    first = sorted(notes, key=lambda x: int(x))[0] if notes else None
    ran: set[str] | None = None
    for n in (notes.get(first) or []) if first is not None else []:
        m = re.match(r"E6 parts: \[(.*?)\]", str(n))
        if m:
            ran = {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
    smoke_parts = set(inp.smoke_cfg.get("e6_parts") or [])
    rows = []
    for part, flag in PART_FLAGS.items():
        enabled = bool(e6cfg.get(flag))
        if not s:
            status = MISSING
        elif ran is not None and part in ran:
            status = "выполнено"
        elif inp.smoke and part not in smoke_parts:
            status = MISSING_SMOKE
        elif not enabled:
            status = "выключено в конфиге"
        else:
            status = MISSING if ran is not None else "см. примечания"
        rows.append([code(part), f"{code(flag)} {ref('configs/experiments/E6.yaml', flag)}", status])
    return ["**Части E6 и их статус**", ""] + table(["часть", "флаг конфига", "статус"], rows)


def experiment_block(inp: Inputs, exp: str, title: str) -> list[str]:
    """Section-6 block of one experiment: known pivots, remaining numbers, tables, thresholds, notes."""
    s = inp.summaries.get(exp)
    path = inp.spath(exp)
    out = [f"### {exp}. {title}", ""]
    if not s:
        return out + [f"_{MISSING}: файла {path} нет._", ""]
    numbers = s.get("numbers") or {}
    seeds = s.get("seeds") or []
    out += [f"Сиды: {', '.join(str(x) for x in seeds)} {ref(path, 'seeds')}; числа: {interval_rule(len(seeds))}; "
            f"хеш конфига {code(s.get('config_hash'))} {ref(path, 'config_hash')}.", ""]
    if s.get("warnings"):
        out += ["Предупреждения сводки: " + "; ".join(str(w) for w in s["warnings"]) + " " + ref(path, "warnings"), ""]
    if exp == "E6":
        out += e6_parts(inp)
    if any(k.startswith("auc/") for k in numbers):
        out += ["**AUC по источникам и macroAUC**", ""] + pivot_by_source(numbers, "auc", path, extra="macro_auc")
    if any(k.startswith("val_auc/") for k in numbers):
        out += ["**AUC на валидации (предусловия H2/H3)**", ""] + pivot_by_source(numbers, "val_auc", path, extra="val_macro_auc")
    if any(k.startswith("tpr_at_fpr/") for k in numbers):
        out += ["**TPR при τ_FPR по источникам позитивов; FPR на P_test при том же пороге**", ""]
        out += pivot_by_source(numbers, "tpr_at_fpr", path, extra="fpr_ptest")
    if any(k.startswith("fpr_notinject/") for k in numbers):
        out += ["**FPR на NotInject по подмножествам и языковой страте**", ""] + pivot_notinject(numbers, path)
    if any(k.startswith("diff") for k in numbers):
        out += ["**Парные разности детекторов**", ""] + pivot_diffs(numbers, path)
    if any(split_key(k)[0] in ("hyper", "latency_ms", "state_bytes") for k in numbers):
        skip_guards = inp.smoke_skips("guard_latency")

        def fill(col: str, det: str) -> str:
            return MISSING_SMOKE if (skip_guards and col.startswith("latency") and det in GUARDS) else NA
        guards_seen = [g for g in GUARDS if any(k.endswith("/" + g) for k in numbers)] if skip_guards else []
        out += ["**Гиперпараметры, задержка на документ (мс) и размер состояния (байт)**", ""]
        out += pivot_detectors(numbers, ("hyper", "latency_ms", "state_bytes"), path, fill=fill, extra_rows=guards_seen)
    rest = generic_numbers(numbers, path)
    if rest:
        out += ["**Остальные числа эксперимента**", ""] + rest
    for name in sorted(s.get("tables") or {}):
        if name in ROC_TABLES or (exp == "E6" and name == "parts"):      # drawn as a figure / shown above
            continue
        out += [f"**Таблица {code(name)}**", ""] + table_rows(s, name, path)
    roc_present = [n for n in ROC_TABLES if (s.get("tables") or {}).get(n)]
    if roc_present:
        out += ["Таблицы " + ", ".join(code(n) for n in roc_present) + " (ROC-кривые, ASSUMPTIONS A49) показаны на рисунке "
                "«ROC по источникам» " + " ".join(ref(path, f"tables/{n}") for n in roc_present) + ".", ""]
    thresholds = s.get("thresholds") or {}
    if thresholds:
        first = sorted(thresholds, key=lambda x: int(x))[0]
        base = f"thresholds/{first}"
        items = [(name, rec) for name, rec in sorted((thresholds.get(first) or {}).items()) if isinstance(rec, dict)]
        rows = [[code(name), num(rec.get("value"), path, f"{base}/{name}/value"),
                 f"{rec.get('source')} {ref(path, f'{base}/{name}/source')}",
                 f"{rec.get('target')} {ref(path, f'{base}/{name}/target')}",
                 num(rec.get("n"), path, f"{base}/{name}/n")] for name, rec in items]
        out += [f"**Пороги (ТЗ 2.5) сида {first} {ref(path, 'seeds')}; пороги остальных сидов — в файле**", ""]
        out += table(["порог", "значение", "источник", "цель", "n"], rows)
    notes = s.get("notes") or {}
    if notes:
        first = sorted(notes, key=lambda x: int(x))[0]
        out += [f"Примечания сида {first} {ref(path, 'seeds')}:", ""]
        out += [f"- {n} {ref(path, f'notes/{first}/{i}')}" for i, n in enumerate(notes[first])] + [""]
    return out


# ================================================================================================= figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig, figdir: Path, name: str) -> Path:
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


def roc_data(inp: Inputs) -> tuple[dict[str, dict[str, list[tuple[float, float]]]], dict[str, dict[str, list[tuple[float, float]]]]]:
    """E1 ``roc`` rows averaged vertically over seeds per (source, detector, FPR grid point), and the ``roc_points``
    operating points of the point detectors averaged over seeds per (source, detector, threshold) (ASSUMPTIONS A49)."""
    tables = ((inp.summaries.get("E1") or {}).get("tables") or {})
    acc: dict[tuple[str, str], dict[float, list[float]]] = {}
    for r in tables.get("roc") or []:
        det, src, f, t = r.get("detector"), r.get("source"), finite(r.get("fpr")), finite(r.get("tpr"))
        if det in ROC_DETECTORS and src and f is not None and t is not None:
            acc.setdefault((str(src), str(det)), {}).setdefault(round(f, 9), []).append(t)
    curves: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for (src, det), grid in acc.items():
        curves.setdefault(src, {})[det] = [(f, sum(ts) / len(ts)) for f, ts in sorted(grid.items())]
    pacc: dict[tuple[str, str, float], list[tuple[float, float]]] = {}
    for r in tables.get("roc_points") or []:
        det, src, f, t = r.get("detector"), r.get("source"), finite(r.get("fpr")), finite(r.get("tpr"))
        thr = finite(r.get("threshold"))
        if det in ROC_POINT_DETECTORS and src and f is not None and t is not None:
            pacc.setdefault((str(src), str(det), round(thr if thr is not None else 0.0, 9)), []).append((f, t))
    points: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for (src, det, _), pts in sorted(pacc.items()):
        points.setdefault(src, {}).setdefault(det, []).append(
            (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    return curves, points


def fig_roc(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    """ROC by source (ТЗ Этап 6 «ROC по источникам»): one panel per source, the main detectors as curves averaged
    over seeds, the regexes as points, the E0 FPR operating point as a dotted line."""
    curves, points = roc_data(inp)
    sources = sorted(set(curves) | set(points), key=order_key(SOURCE_ORDER))
    if not sources:
        return None, "ROC по источникам: " + MISSING + " (в E1 нет таблиц roc / roc_points)"
    plt = _plt()
    ncols = min(3, len(sources))
    nrows = math.ceil(len(sources) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    colors = {d: plt.cm.tab10(i % 10) for i, d in enumerate(ROC_DETECTORS)}
    fpr_target = finite((inp.power or {}).get("fpr_target"))
    handles: dict[str, Any] = {}
    for ax, src in zip(axes.flat, sources):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
        for det in ROC_DETECTORS:
            pts = curves.get(src, {}).get(det)
            if pts:
                (h,) = ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.4, color=colors[det], label=det)
                handles.setdefault(det, h)
        for det, pts in points.get(src, {}).items():
            h = ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker="X", s=60, color="black", zorder=5, label=det)
            handles.setdefault(det, h)
        if fpr_target is not None:
            ax.axvline(fpr_target, ls=":", lw=1.0, color="#cc3333")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_title(src); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    for ax in list(axes.flat)[len(sources):]:
        ax.axis("off")
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=min(7, len(handles)), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E1: ROC по источникам (среднее по сидам; регулярки — точки)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = save(fig, figdir, "e1_roc_by_source.png"); plt.close(fig)
    return path, ("ROC-кривые E1 по источникам: TPR на общей сетке FPR, усреднение по вертикали по сидам; "
                  "регулярки — рабочие точки; пунктир — рабочая точка FPR из E0")


def fig_auc_by_source(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    """An extra view next to the ROC curves: AUC by source with the intervals of the report (A55 envelope)."""
    numbers = inp.numbers("E1")
    cells: dict[str, dict[str, tuple]] = {}
    for key, rec in numbers.items():
        parts = split_key(key)
        if parts[0] == "auc" and len(parts) == 3:
            cells.setdefault(parts[2], {})[parts[1]] = rec_triple(rec)
    if not cells:
        return None, "AUC по источникам: " + MISSING
    plt = _plt()
    dets = sorted(cells, key=order_key(DETECTOR_ORDER))
    sources = sorted({s for m in cells.values() for s in m}, key=order_key(SOURCE_ORDER))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(dets)), 4))
    width = 0.8 / max(1, len(sources))
    for j, src in enumerate(sources):
        xs, ys, err = [], [], [[], []]
        for i, d in enumerate(dets):
            v, lo, hi = cells[d].get(src, (None, None, None))
            if v is None:
                continue
            xs.append(i + j * width); ys.append(v)
            err[0].append(max(0.0, v - lo) if lo is not None else 0.0); err[1].append(max(0.0, hi - v) if hi is not None else 0.0)
        ax.bar(xs, ys, width=width, yerr=err, capsize=2, label=src)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(dets))]); ax.set_xticklabels(dets, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0); ax.set_ylabel("AUC"); ax.set_title("E1: AUC по источникам с интервалами"); ax.legend(fontsize=7)
    path = save(fig, figdir, "e1_auc_by_source.png"); plt.close(fig)
    return path, "AUC по источникам с интервалами (дополнение к ROC-кривым)"


E2_LEVELS = ("shots1", "shots10", "shots100", "full")


def learning_series(inp: Inputs) -> dict[str, dict[str, tuple[Any, Any, Any]]]:
    """``fewshot/macro_auc/<level>/<detector>`` -> {detector: {level: (mean, band_low, band_high)}}; the band is the
    min / max of the per-subsample macroAUC over all seeds (``band_low`` / ``band_high`` of the E2 records, A28)."""
    series: dict[str, dict[str, tuple[Any, Any, Any]]] = {}
    for key, rec in inp.numbers("E2").items():
        parts = split_key(key)
        if len(parts) == 4 and parts[0] == "fewshot" and parts[1] == "macro_auc":
            per = seed_records(rec)
            lows = [x for x in (finite(r.get("band_low")) for _, r in per) if x is not None]
            highs = [x for x in (finite(r.get("band_high")) for _, r in per) if x is not None]
            series.setdefault(parts[3], {})[parts[2]] = (rec_triple(rec)[0], min(lows) if lows else None,
                                                        max(highs) if highs else None)
    return series


def fig_learning_curves(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    series = learning_series(inp)
    if not series:
        return None, "Кривые обучения: " + MISSING
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    has_band = False
    for det, pts in sorted(series.items(), key=lambda kv: order_key(DETECTOR_ORDER)(kv[0])):
        order = sorted(pts, key=order_key(E2_LEVELS))
        x = [order_key(E2_LEVELS)(lvl)[0] for lvl in order]
        y = [pts[lvl][0] for lvl in order]
        ax.plot(x, y, marker="o", label=det)
        if all(pts[lvl][1] is not None and pts[lvl][2] is not None for lvl in order):
            ax.fill_between(x, [pts[lvl][1] for lvl in order], [pts[lvl][2] for lvl in order], alpha=0.15)
            has_band = True
    ax.set_xticks(range(len(E2_LEVELS))); ax.set_xticklabels([lvl.replace("shots", "") for lvl in E2_LEVELS])
    ax.set_xlabel("примеров на класс"); ax.set_ylabel("macroAUC"); ax.set_title("E2: кривые обучения")
    ax.legend(fontsize=7)
    path = save(fig, figdir, "e2_learning_curves.png"); plt.close(fig)
    return path, ("кривые обучения: среднее macroAUC по подвыборкам и сидам; полоса — минимум и максимум по подвыборкам "
                  "всех сидов (A28)" if has_band else "кривые обучения (полос нет: в записях E2 нет band_low/band_high)")


def curveball_values(inp: Inputs) -> list[float]:
    """Per-null macroAUC of the primary readout (real fly, Bloom, full training) from the E4 table ``curveball``."""
    rows = ((inp.summaries.get("E4") or {}).get("tables") or {}).get("curveball") or []
    vals = []
    for r in rows:
        if "detector" in r and r.get("detector") != "real_fly_bloom":
            continue
        if "readout" in r and ("10shot" in str(r.get("readout")) or not str(r.get("readout")).startswith("bloom")):
            continue
        v = finite(r.get("macro_auc"))
        if v is not None:
            vals.append(v)
    return vals


def fig_curveball(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    vals = curveball_values(inp)
    measured = None
    for exp in ("E4", "E1"):
        rec = inp.numbers(exp).get("macro_auc/real_fly_bloom")
        if rec:
            measured = rec_triple(rec)[0]
            break
    if not vals:
        return None, "Гистограмма curveball: " + MISSING
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=min(30, max(5, len(vals) // 5)), color="#8899aa")
    if measured is not None:
        ax.axvline(measured, color="red", label="измеренная M")
        ax.legend()
    ax.set_xlabel("macroAUC (Bloom)"); ax.set_ylabel("число перемешиваний"); ax.set_title("E4: curveball-нули и измеренная M")
    path = save(fig, figdir, "e4_curveball_hist.png"); plt.close(fig)
    return path, "гистограмма macroAUC по curveball-перемешиваниям всех сидов с отметкой измеренной M (среднее по сидам)"


def fig_notinject(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    numbers = inp.numbers("E1")
    cells: dict[str, dict[str, float]] = {}
    for key, rec in numbers.items():
        parts = split_key(key)
        if parts[0] == "fpr_notinject" and len(parts) in (3, 4) and parts[1] == "tau90_deep":
            sub = parts[3] if len(parts) == 4 else "все"
            if sub in ("все", "one", "two", "three") and rec_triple(rec)[0] is not None:
                cells.setdefault(parts[2], {})[sub] = float(rec_triple(rec)[0])
    if not cells:
        return None, "FPR на NotInject при τ_90: " + MISSING
    plt = _plt()
    dets = sorted(cells, key=order_key(DETECTOR_ORDER)); subs = ["все", "one", "two", "three"]
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(dets)), 4)); width = 0.2
    for j, sub in enumerate(subs):
        ax.bar([i + j * width for i in range(len(dets))], [cells[d].get(sub, 0.0) for d in dets], width=width, label=sub)
    ax.set_xticks([i + 1.5 * width for i in range(len(dets))]); ax.set_xticklabels(dets, rotation=45, ha="right")
    ax.set_ylabel("FPR"); ax.set_title("E1: FPR на NotInject при τ_90(deep)"); ax.legend(fontsize=7)
    path = save(fig, figdir, "e1_notinject_fpr_tau90.png"); plt.close(fig)
    return path, "FPR на NotInject при τ_90(deep) по детекторам и подмножествам"


def fig_para_strata(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    cells: dict[str, dict[str, float]] = {}
    for exp in ("E1", "E6"):
        for key, rec in inp.numbers(exp).items():
            parts = split_key(key)
            if parts[0] == "auc" and len(parts) == 3 and parts[1].startswith("para"):
                v = rec_triple(rec)[0]
                if v is not None:
                    cells.setdefault(parts[2], {}).setdefault(parts[1], float(v))
    if not cells:
        return None, "AUC по стратам парафраз: " + MISSING
    plt = _plt()
    dets = sorted(cells, key=order_key(DETECTOR_ORDER)); strata = sorted({s for m in cells.values() for s in m}, key=order_key(SOURCE_ORDER))
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(dets)), 4)); width = 0.8 / len(strata)
    for j, st in enumerate(strata):
        ax.bar([i + j * width for i in range(len(dets))], [cells[d].get(st, 0.0) for d in dets], width=width, label=st)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(dets))]); ax.set_xticklabels(dets, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("AUC"); ax.set_title("AUC по стратам парафраз"); ax.legend(fontsize=7)
    path = save(fig, figdir, "para_strata_auc.png"); plt.close(fig)
    return path, "AUC по стратам парафраз (все / глубокая / мелкая)"


def ablation_grid(inp: Inputs) -> list[str]:
    """E5 grid as a text table: rows = detector names of E5, columns = sources (+ macroAUC)."""
    numbers = inp.numbers("E5")
    if not numbers:
        return [f"_{MISSING}_", ""]
    return pivot_by_source(numbers, "auc", inp.spath("E5"), extra="macro_auc")


# ================================================================================================= sections
def sec_verdicts(inp: Inputs) -> list[str]:
    out = ["## 1. Итог с вердиктами", ""]
    v = inp.verdicts
    path = inp.p(inp.rdir / "verdicts.json")
    if not v:
        out += [f"_{MISSING}: файла {path} нет; вердикты по правилам ТЗ Этап 4 выносит "
                "`flyguard.experiments.verdicts_run` после E1/E2/E4 и E0._", ""]
    else:
        rows = []
        for name, vd in iter_verdicts(v):
            base = name.split("/")[0]
            ci_key = "ci" if vd.get("ci") is not None else "ci_envelope"
            seeds_note = (f"{num(vd.get('n_seeds'), path, f'{name}/n_seeds')} сидов" if vd.get("n_seeds") is not None else NA)
            rows.append([HYPOTHESIS_TITLES.get(base, base) + (f" ({name.split('/', 1)[1]})" if "/" in name else ""),
                         f"**{vd.get('status')}** {ref(path, f'{name}/status')}",
                         "; ".join(effect_lines(vd.get("effect"), path, f"{name}/effect")) or NA,
                         "; ".join(ci_lines(vd.get(ci_key), path, f"{name}/{ci_key}")) or NA,
                         seeds_note, f"{vd.get('reason', '')} {ref(path, f'{name}/reason')}"])
        out += ["Эффект и интервал: одного сида — его значения; нескольких сидов — средний эффект и огибающая "
                "интервалов по сидам (`ci_envelope`), вердикт по сидам агрегирован правилом `aggregate_rule` файла; "
                "вердикты каждого сида лежат в `per_seed`.", ""]
        out += table(["гипотеза", "вердикт", "эффект", "интервал", "сиды", "основание"], rows)
        status_rows = [[HYPOTHESIS_TITLES.get(name.split("/")[0], name) + (f" ({name.split('/', 1)[1]})" if "/" in name else ""),
                        # one reference per row to the whole ``n_seeds_by_status`` node: the status names are the
                        # keys and contain spaces, which a ``path#key`` reference cannot carry
                        ", ".join(f"{st}: {fmt(cnt)}" for st, cnt in sorted(vd["n_seeds_by_status"].items()))
                        + f" {ref(path, f'{name}/n_seeds_by_status')}"]
                       for name, vd in iter_verdicts(v) if isinstance(vd.get("n_seeds_by_status"), dict) and vd["n_seeds_by_status"]]
        if status_rows:
            out += ["**Вердикты по сидам**", ""] + table(["гипотеза", "сидов с каждым статусом"], status_rows)
        warnings = v.get("warnings") or []
        if warnings:
            out += ["**Предупреждения verdicts.json** (например, незамороженный power.json делает вердикты предварительными):", ""]
            out += [f"- {w} {ref(path, f'warnings/{i}')}" for i, w in enumerate(warnings)] + [""]
        meta = {k: v[k] for k in ("config_hash", "git_commit", "created_at", "smoke") if k in v and not isinstance(v[k], dict)}
        if isinstance(v.get("sources"), dict) and isinstance(v["sources"].get("power"), dict):
            meta.update({f"sources/power/{k}": val for k, val in v["sources"]["power"].items() if not isinstance(val, dict)})
        if meta:
            out += dump(meta, path, "", "Провенанс вердиктов")
    out += ["**Таблица носителей E0 (источник × метрика)**", ""]
    ppath = inp.p(R.power_path(inp.root, inp.smoke))
    carriers = (inp.power or {}).get("carriers") or {}
    if carriers:
        metrics = sorted({m for row in carriers.values() for m in row})
        out += table(["источник"] + metrics,
                     [[s] + [f"{carriers[s].get(m, NA)} {ref(ppath, f'carriers/{s}/{m}')}" for m in metrics]
                      for s in sorted(carriers, key=order_key(SOURCE_ORDER + ("macro",)))])
    else:
        out += [f"_{MISSING}: {ppath} нет._", ""]
    return out


def iter_verdicts(v: dict[str, Any]):
    for k, val in v.items():
        if isinstance(val, dict) and "status" in val:
            yield k, val
        elif isinstance(val, dict):
            for sub, sv in val.items():
                if isinstance(sv, dict) and "status" in sv:
                    yield f"{k}/{sub}", sv


def effect_lines(eff: Any, path: str, key: str) -> list[str]:
    if eff is None:
        return []
    if isinstance(eff, dict):
        out = []
        for k, val in eff.items():
            out += [f"{k}: {x}" for x in effect_lines(val, path, f"{key}/{k}")]
        return out
    return [num(eff, path, key)]


def ci_lines(ci: Any, path: str, key: str) -> list[str]:
    if ci is None:
        return []
    if isinstance(ci, dict) and ("low" in ci or "high" in ci):
        return [num(ci.get("point"), path, key, ci.get("low"), ci.get("high"))]
    if isinstance(ci, dict):
        out = []
        for k, val in ci.items():
            out += [f"{k}: {x}" for x in ci_lines(val, path, f"{key}/{k}")]
        return out
    return [num(ci, path, key)]




HARNESS_PINS = (("agentdojo", "AgentDojo"), ("agentdyn", "AgentDyn"), ("flyhash_connectome", "FlyHash-Connectome"),
                ("malecns", "MaleCNS"))


def sec_setup(inp: Inputs, setup: dict[str, Any], spath: str) -> list[str]:
    out = ["## 2. Установка", ""]
    pk = setup["packages"]
    out += [f"- Python {code(setup['python'])} {ref(spath, 'python')}; платформа {code(setup['platform'])} {ref(spath, 'platform')}.",
            "- Пакеты (версии из окружения, закреплены в `requirements.lock`): "
            + ", ".join(f"{p} {code(v)} {ref(spath, f'packages/{p}')}" for p, v in pk.items() if v) + ".",
            f"- Коммит, на котором собран отчёт: {code(setup.get('git_commit'))} {ref(spath, 'git_commit')} (коммиты, на "
            f"которых посчитаны результаты, — в таблице ниже); хеш конфига {code(setup['config_hash'])} {ref(spath, 'config_hash')}.",
            f"- Сиды: глобальные {', '.join(str(s) for s in setup['seeds']['global'])} {ref(spath, 'seeds/global')}; "
            f"дети {', '.join(setup['seeds']['children'])} {ref(spath, 'seeds/children')}."]
    conn = setup["connectome"]
    if conn.get("available"):
        out.append(f"- MaleCNS: сторона {setup['malecns'].get('side')} {ref(spath, 'malecns/side')}, minconf "
                   f"{num(setup['malecns'].get('minconf'), spath, 'malecns/minconf')}, гломерул "
                   f"{num(conn.get('glomeruli'), spath, 'connectome/glomeruli')}, клеток Кеньона "
                   f"{num(conn.get('kenyon_cells'), spath, 'connectome/kenyon_cells')}, средняя входящая степень "
                   f"{num(conn.get('indegree_mean'), spath, 'connectome/indegree_mean')} (диапазон проверки "
                   f"{', '.join(fmt(x) for x in conn.get('check_range', []))} {ref(spath, 'connectome/check_range')}, "
                   f"в диапазоне: {conn.get('in_range')} {ref(spath, 'connectome/in_range')}).")
    else:
        out.append(f"- MaleCNS: матрица недоступна ({conn.get('error', 'файла нет')}) {ref(spath, 'connectome/available')}.")
    out.append(f"- Компаратор H1a/H2: {code(setup['comparator'])} {ref(spath, 'comparator')}; промышленные детекторы: "
               + ", ".join(f"{g} (в наличии: {v['present']}) {ref(spath, f'guards/{g}/present')}" for g, v in setup["guards"].items()) + ".")
    if inp.pilot:
        ppath = inp.p(inp.root / "results" / "pilot.json")
        out.append(f"- Модель агента: {code(inp.pilot.get('chosen_model'))} {ref(ppath, 'chosen_model')}; выбрана по правилу "
                   f"пилота: {inp.pilot.get('chosen_by_rule')} {ref(ppath, 'chosen_by_rule')} (пилоты кандидатов — раздел 4"
                   + ("; отклонение от правила — DEVIATIONS D5)." if inp.pilot.get("chosen_by_rule") is False else ")."))
    else:
        out.append(f"- Модель агента и пилот: {MISSING} (results/pilot.json нет).")
    pr = (inp.paraphrases or {}).get("models") or {}
    if pr:
        prp = inp.p(inp.root / "results/paraphrases.json")
        out.append(f"- Генераторы парафраз: {', '.join(code(g) for g in pr.get('generators') or [])} {ref(prp, 'models/generators')} "
                   f"(temperature {num(pr.get('temperature_generate'), prp, 'models/temperature_generate', nd=1)}); судьи: "
                   f"{', '.join(code(j) for j in pr.get('judges') or [])} {ref(prp, 'models/judges')} (temperature "
                   f"{num(pr.get('temperature_judge'), prp, 'models/temperature_judge', nd=1)}), судей "
                   f"{num(pr.get('n_judges'), prp, 'models/n_judges')}; правило {pr.get('judge_rule')} {ref(prp, 'models/judge_rule')}.")
    else:
        pm = setup.get("paraphrase_models") or {}
        gens = ", ".join(code(g.get("model")) for g in pm.get("generators", []))
        judges = ", ".join(code(j.get("model")) for j in pm.get("judges", []))
        out.append(f"- Генераторы парафраз (конфиг; results/paraphrases.json нет): {gens or NA} {ref(spath, 'paraphrase_models/generators')}; "
                   f"судьи: {judges or NA} {ref(spath, 'paraphrase_models/judges')}.")
    out += ["", "**Харнессы и внешние репозитории** (пины `data/manifests/sources.json`, скопированы в setup.json)", ""]
    pins = setup.get("pins") or {}
    rows = []
    for name, title in HARNESS_PINS + tuple((n, n) for n in sorted(pins) if n not in dict(HARNESS_PINS)):
        pin = pins.get(name)
        if not isinstance(pin, dict):
            rows.append([title, NA, MISSING + " (пина нет)", NA])
            continue
        src = pin.get("repo") or pin.get("bucket") or NA
        commit = pin.get("commit")
        versions = [f"{k} {code(pin[k])} {ref(spath, f'pins/{name}/{k}')}" for k in ("package", "tag", "benchmark_version", "version")
                    if pin.get(k) is not None]
        rows.append([title, f"{code(src)} {ref(spath, f'pins/{name}')}",
                     f"{code(commit)} {ref(spath, f'pins/{name}/commit')}" if commit
                     else "не записан в sources.json" if pin.get("repo") else NA,   # a data bucket has no commit
                     "; ".join(versions) or NA])
    out += table(["компонент", "источник", "коммит", "версии"], rows)
    out += ["**Вычислитель: потоки при расчёте результатов** (`timing.threads` файлов результатов, ASSUMPTIONS A39, A41)", ""]
    trows = thread_rows(inp)
    out += table(["эксперимент", "сиды", "CPU", "переменные потоков", "пулы BLAS/OpenMP (потоков)"], trows) if trows \
        else [f"_{MISSING}: файлов результатов нет._", ""]
    out += ["**Коммиты, на которых посчитаны результаты** (ASSUMPTIONS A52: одинаковость кода src, scripts и configs "
            "проверяет `git diff`; то же делает `scripts/check_acceptance.py`)", ""]
    rows, commits, dirty = result_commit_rows(inp)
    out += table(["результаты", "коммит", "сиды", "незакоммиченные изменения кода"], rows) if rows else [f"_{MISSING}: файлов результатов нет._", ""]
    out += commit_warnings(inp, commits, dirty)
    return out + [""]


def thread_rows(inp: Inputs) -> list[list[str]]:
    """One row per experiment and distinct ``timing.threads`` record: the seeds sharing it, the CPU count, the thread
    environment variables and the thread pools actually loaded (ASSUMPTIONS A41: SVD and fits run single-threaded)."""
    rows: list[list[str]] = []
    for exp in EXPERIMENTS:
        s = inp.summaries.get(exp)
        if not s:
            continue
        path = inp.spath(exp)
        groups: dict[str | None, list[str]] = {}
        for seed, t in sorted((s.get("timing") or {}).items(), key=seed_sort):
            th = (t or {}).get("threads")
            groups.setdefault(json.dumps(th, sort_keys=True) if th else None, []).append(str(seed))
        for sig, seeds in groups.items():
            seeds_cell = f"{', '.join(seeds)} {ref(path, 'seeds')}"
            if sig is None:
                rows.append([exp, seeds_cell, "не записано", "не записано", "не записано"])
                continue
            th, base = json.loads(sig), f"timing/{seeds[0]}/threads"
            env = ", ".join(f"{k}={v}" for k, v in sorted((th.get("env") or {}).items()) if v is not None) or "не заданы"
            pools = sorted({(str(p.get("internal_api") or p.get("user_api")), p.get("num_threads")) for p in th.get("pools") or []},
                           key=lambda x: (x[0], str(x[1])))
            pools_cell = ", ".join(f"{a}: {fmt(n)}" for a, n in pools) or NA
            rows.append([exp, seeds_cell, f"{fmt(th.get('cpu_count'))} {ref(path, f'{base}/cpu_count')}",
                         f"{env} {ref(path, f'{base}/env')}", f"{pools_cell} {ref(path, f'{base}/pools')}"])
    return rows


def result_commit_rows(inp: Inputs) -> tuple[list[list[str]], list[str | None], list[str]]:
    """(rows, commits, dirty) of the results: the seeds of every experiment grouped by the commit recorded in
    ``summary.json#git_commits``, then power (stage 1 and frozen), contract and verdicts. ``git_dirty`` is read from
    the seed files: "нет" only when every file of the row records ``false``."""
    def dirty_cell(flags: list[Any], seeds: list[str] | None = None) -> str:
        bad = [s for s, f in zip(seeds or [NA] * len(flags), flags) if f is True]
        if bad:
            return "да" + (f": сиды {', '.join(bad)}" if seeds else "")
        return "не записано" if any(f is None for f in flags) else "нет"

    rows: list[list[str]] = []
    commits: list[str | None] = []
    dirty: list[str] = []
    for exp in EXPERIMENTS:
        s = inp.summaries.get(exp)
        if not s:
            continue
        path = inp.spath(exp)
        by_commit: dict[Any, list[str]] = {}
        for seed, c in sorted((s.get("git_commits") or {}).items(), key=seed_sort):
            by_commit.setdefault(c, []).append(str(seed))
        recorded = s.get("git_dirty") if isinstance(s.get("git_dirty"), dict) else None
        for c, seeds in by_commit.items():
            flags = [recorded.get(x) if recorded is not None else
                     (inp.json(R.result_path(exp, int(x), inp.smoke, inp.root)) or {}).get("git_dirty") for x in seeds]
            commits.append(c)
            if any(f is True for f in flags):
                dirty.append(f"{exp}, сиды {', '.join(x for x, f in zip(seeds, flags) if f is True)} {ref(path, 'seeds')}")
            rows.append([exp, (code(str(c)[:12]) if c else NA) + f" {ref(path, 'git_commits')}", f"{', '.join(seeds)} {ref(path, 'seeds')}",
                         dirty_cell(flags, seeds)])
    for p in (inp.power_stage1_path, R.power_path(inp.root, inp.smoke), inp.rdir / "contract.json", inp.rdir / "verdicts.json"):
        d = inp.json(p)
        if d is None:
            continue
        c = d.get("git_commit")
        commits.append(c)
        if d.get("git_dirty") is True:
            dirty.append(inp.p(p))
        rows.append([inp.p(p), (code(str(c)[:12]) if c else NA) + f" {ref(inp.p(p), 'git_commit')}", NA,
                     dirty_cell([d.get("git_dirty")])])
    return rows, commits, dirty


def code_differs(root: Path, a: str, b: str) -> bool | None:
    """``git diff --quiet a b -- src scripts configs`` (ASSUMPTIONS A52): False = same code, True = the code differs,
    None = cannot tell (no git, unknown commit)."""
    try:
        proc = subprocess.run(["git", "-C", str(root), "diff", "--quiet", a, b, "--", "src", "scripts", "configs"],
                              capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: False, 1: True}.get(proc.returncode)


def commit_warnings(inp: Inputs, commits: list[str | None], dirty: list[str]) -> list[str]:
    """The provenance verdict under the commits table: one line when every result shares one code state, a warning
    line for each commit whose code differs from the most frequent one (or cannot be compared), for results without
    a recorded commit and for results computed with uncommitted code changes."""
    known = [c for c in commits if c]
    out: list[str] = []
    if not commits:
        return out
    if len(known) < len(commits):
        out.append("⚠ Предупреждение: " + ("у части результатов коммит не записан" if known else "коммиты результатов не записаны")
                   + "; происхождение кода не проверяется.")
    distinct = [c for c, _ in Counter(known).most_common()]
    if len(distinct) == 1 and len(known) == len(commits):
        out.append(f"Все результаты посчитаны на одном коммите {code(distinct[0][:12])}.")
    elif len(distinct) > 1:
        base = distinct[0]
        for c in distinct[1:]:
            d = code_differs(inp.root, base, c)
            if d is True:
                out.append(f"⚠ Предупреждение: код (src, scripts, configs) коммита {code(c[:12])} отличается от "
                           f"{code(base[:12])}: результаты посчитаны разным кодом.")
            elif d is None:
                out.append(f"⚠ Предупреждение: коммиты {code(c[:12])} и {code(base[:12])} не удалось сравнить через git.")
            else:
                out.append(f"Коммиты {code(c[:12])} и {code(base[:12])} различаются только вне src, scripts и configs.")
    if dirty:
        out.append("⚠ Предупреждение: результаты посчитаны с незакоммиченными изменениями кода: " + "; ".join(dirty) + ".")
    return out + [""] if out else out


DROP_REASONS = {"positives_all_excluded": "позитивы, у которых исключены все позитивные окна",
                "no_windows_left": "документы без оставшихся окон",
                "bipia_pairs": "документы BIPIA, снятые вместе с разрушенной парой (весь контекст)",
                "bipia_contexts": "контексты BIPIA, снятые целиком"}
SPLIT_ORDER = ("train", "val", "test", "unused", "total")


def source_of(doc_id: str) -> str:
    return str(doc_id).split(":", 1)[0]


def e1_sources_rows(inp: Inputs) -> dict[str, tuple[int, dict[str, Any]]]:
    """source -> (row index, row) of the E1 table ``sources`` (post-dedup test composition), first seed."""
    return {str(r.get("source")): (i, r) for i, r in inp.first_rows("E1", "sources")}


def e1_source_cell(inp: Inputs, src: str, col: str) -> str:
    row = e1_sources_rows(inp).get(src)
    if row is None:
        return f"{NA} (E1: {MISSING})"
    return num(row[1].get(col), inp.spath("E1"), f"tables/sources/{row[0]}/{col}")


def sec_data(inp: Inputs) -> list[str]:
    out = ["## 3. Данные", ""]
    mp = inp.p(inp.manifests)
    cy = "configs/default.yaml"
    d = inp.cfg.default
    ap = inp.p(inp.audit_path)
    out += [f"Счётчики взяты из манифестов {code(mp)} (`splits.json`, `pools.json`, `dedup.json`, `contamination.json`), "
            f"из аудита {code(ap)} (ТЗ 1.2; только счётчики) и из таблицы `sources` E1 (состав теста после дедупликации)."
            + ("" if inp.audit_text else f" Аудит: {MISSING}."), ""]
    # ---- windows
    w = d["windows"]
    out += ["### 3.1. Окна (ТЗ 1.3)", "",
            f"Окно — {num(w['size'], cy, 'windows/size')} символов с шагом {num(w['stride'], cy, 'windows/stride')}; документ "
            f"короче окна образует одно окно. Окно помечено «инъекция», если покрывает не меньше "
            f"{num(w['min_span_chars'], cy, 'windows/min_span_chars')} символов спана атаки или весь спан, когда он короче; "
            "источники без спанов (deepset, NotInject, парафразы) передают окну метку документа. Схема одна для всех "
            "детекторов; окна трансформеров по токенам — только в E6. Обучение идёт на окнах, метрики — на документах "
            "через максимум по окнам.", ""]
    out += ["**Окна по источникам (audit.md)**", ""] + (md_section(inp.audit_text, "Окна", ap) or [f"_{MISSING}_", ""])
    # ---- documents by source / split / label
    out += ["### 3.2. Документы по источникам, разбиениям и меткам", ""]
    out += ["**Документы по источникам (audit.md)**", ""] + (md_section(inp.audit_text, "Документы по источникам", ap) or [f"_{MISSING}_", ""])
    out += contamination_counts(inp)
    out += e1_split_block(inp)
    # ---- BIPIA pairs
    out += ["### 3.3. Пары BIPIA (ТЗ 1.4)", ""] + bipia_block(inp)
    out += ["**Задачи BIPIA (audit.md)**", ""] + (md_section(inp.audit_text, "BIPIA", ap) or [f"_{MISSING}_", ""])
    # ---- dedup
    out += ["### 3.4. Дедупликация (ТЗ 1.7)", ""] + dedup_block(inp)
    # ---- pools and the FPR operating point
    out += ["### 3.5. Пулы негативов и рабочая точка FPR (ТЗ 1.8, Этап 0)", ""] + pools_block(inp)
    # ---- C_unl
    out += ["### 3.6. Корпус C_unl (ТЗ 1.9)", ""] + c_unl_block(inp)
    # ---- languages
    out += ["### 3.7. Языковые страты (ТЗ 1.2)", ""]
    out += md_section(inp.audit_text, "Языки", ap) or [f"_{MISSING}_", ""]
    out += ["**Состав NotInject (audit.md)**", ""] + (md_section(inp.audit_text, "Состав NotInject", ap) or [f"_{MISSING}_", ""])
    out += ["**Длины нормализованного текста (audit.md)**", ""] + (md_section(inp.audit_text, "Длины", ap) or [f"_{MISSING}_", ""])
    return out


def contamination_counts(inp: Inputs) -> list[str]:
    """Documents and windows per source x E1 split x label (``contamination.json`` counts every document and window
    of the tables, before the dedup exclusions) next to the documents the dedup dropped (``dedup.json``)."""
    out = ["**Документы и окна по источнику, разбиению E1 и метке** (все документы таблиц, до исключений дедупликации; "
           "`contamination.json` считает их для аудита ТЗ 3.2; последний столбец — документы, выпавшие при дедупликации)", ""]
    if inp.contamination_skipped():
        return out + [f"_{inp.missing('contamination_audit')}: счётчики по разбиению и метке в "
                      f"{inp.p(inp.contamination_path)} не записаны._", ""]
    cp = inp.p(inp.contamination_path)
    dp = f"{inp.p(inp.manifests)}/dedup.json"
    dl = inp.contamination.get("document_level") or {}
    wl = inp.contamination.get("window_level") or {}
    dropped = (inp.dedup or {}).get("documents_dropped_by_source_variant") or {}
    rows = []
    for sk in sorted(set(dl) | set(wl), key=order_key(("deep", "bipia", "bipia_e6", "dojo", "dyn", "para", "notinject"))):
        splits = sorted((set(dl.get(sk) or {}) | set(wl.get(sk) or {})) - {"total"}, key=order_key(SPLIT_ORDER))
        for split in splits:
            labels = sorted(set((dl.get(sk) or {}).get(split) or {}) | set((wl.get(sk) or {}).get(split) or {}))
            for lab in labels:
                dc = ((dl.get(sk) or {}).get(split) or {}).get(lab) or {}
                wc = ((wl.get(sk) or {}).get(split) or {}).get(lab) or {}
                dkey = (f"bipia/e6/{lab}" if sk == "bipia_e6" else f"{sk}/main/{lab}")
                if split == "test" and dkey in dropped:      # the dedup excludes test documents only (ТЗ 1.7)
                    drop = num(dropped[dkey], dp, f"documents_dropped_by_source_variant/{dkey}")
                elif split == "test" and dropped:
                    drop = f"нет {ref(dp, 'documents_dropped_by_source_variant')}"
                else:
                    drop = NA
                rows.append([sk, split, lab, num(dc.get("documents"), cp, f"document_level/{sk}/{split}/{lab}/documents"),
                             num(wc.get("windows"), cp, f"window_level/{sk}/{split}/{lab}/windows"), drop])
    return out + table(["источник", "разбиение", "метка", "документов", "окон", "выпало при дедупликации"], rows)


def e1_split_block(inp: Inputs) -> list[str]:
    out: list[str] = []
    if not inp.splits:
        return [f"_splits.json: {MISSING}_", ""]
    sp = f"{inp.p(inp.manifests)}/splits.json"
    c = inp.splits.get("counts") or {}
    out += ["**Разбиение E1 (документы в списках после дедупликации)**", ""]
    rows = [["train (deepset)", num(c.get("train"), sp, "counts/train")], ["val (все источники)", num(c.get("val"), sp, "counts/val")],
            ["C_unl", num(c.get("c_unl"), sp, "counts/c_unl")]]
    rows += [[f"test / {s}", num(n, sp, f"counts/test/{s}")]
             for s, n in sorted((c.get("test") or {}).items(), key=lambda kv: order_key(SOURCE_ORDER)(kv[0]))]
    rows += [[f"val / {s}", num(n, sp, f"counts/val_by_source/{s}")] for s, n in sorted((c.get("val_by_source") or {}).items())]
    out += table(["роль", "документов"], rows)
    srows = e1_sources_rows(inp)
    out += ["**Тестовые источники E1 после дедупликации** (таблица `sources` E1, первый сид; состав одинаков для всех сидов)", ""]
    if srows:
        path = inp.spath("E1")
        cols = ("n_docs", "n_pos", "n_neg", "n_clusters", "n_windows")
        out += table(["источник"] + list(cols),
                     [[src] + [num(r.get(col), path, f"tables/sources/{i}/{col}") for col in cols]
                      for src, (i, r) in sorted(srows.items(), key=lambda kv: order_key(SOURCE_ORDER)(kv[0]))])
    else:
        out += [f"_{MISSING}: E1 не выполнен._", ""]
    e3 = inp.splits.get("e3") or {}
    if e3:
        lst = lambda k: (", ".join(code(x) for x in e3[k]) if isinstance(e3.get(k), list) else fmt(e3.get(k))) + " " + ref(sp, f"e3/{k}")  # noqa: E731
        out += [f"Фолды E3: шаблонов в конфиге {lst('templates_configured')}, есть в данных {lst('templates_present')}, "
                f"не хватает {lst('templates_missing')}; пригодных фолдов: "
                + ", ".join(f"{k} {num(v, sp, f'e3/usable/{k}')}" for k, v in sorted((e3.get("usable") or {}).items())) + ".", ""]
    return out


def bipia_block(inp: Inputs) -> list[str]:
    if not inp.splits:
        return [f"_splits.json: {MISSING}_", ""]
    D = inp.derived
    sp = f"{inp.p(inp.manifests)}/splits.json"
    dp = f"{inp.p(inp.manifests)}/dedup.json"
    b = inp.splits.get("bipia") or {}
    c = inp.splits.get("counts") or {}
    rules = (inp.splits.get("rules") or {}).get("bipia") or {}
    e6 = b.get("e6_docs") or {}
    ln = lambda key, lst, node: D.num(key, len(lst), f"{sp}#{node}", "len(list)") if isinstance(lst, list) else MISSING  # noqa: E731
    out = [f"- Контексты делятся на валидацию и тест по кластерам (доля валидации {num(rules.get('val_fraction'), sp, 'rules/bipia/val_fraction')}): "
           f"контекстов валидации {ln('data/bipia/val_contexts', b.get('val_contexts'), 'bipia/val_contexts')}, теста "
           f"{ln('data/bipia/test_contexts', b.get('test_contexts'), 'bipia/test_contexts')}.",
           f"- Основная пара на контекст — чистый документ и документ с одной атакой в одной позиции (сид `subsample`): "
           f"документов в тесте E1 {num((c.get('test') or {}).get('bipia'), sp, 'counts/test/bipia')}, в валидации "
           f"{num((c.get('val_by_source') or {}).get('bipia'), sp, 'counts/val_by_source/bipia')}; в тесте после дедупликации "
           f"атакованных {e1_source_cell(inp, 'bipia', 'n_pos')}, чистых {e1_source_cell(inp, 'bipia', 'n_neg')}.",
           f"- Варианты E6 (все атаки × позиции, только E6): документов валидации {ln('data/bipia/e6_val', e6.get('val'), 'bipia/e6_docs/val')}, "
           f"теста {ln('data/bipia/e6_test', e6.get('test'), 'bipia/e6_docs/test')}."]
    dd = inp.dedup or {}
    drops = dd.get("documents_dropped") or {}
    by_var = dd.get("documents_dropped_by_source_variant") or {}
    if dd:
        parts = [f"{k} {num(v, dp, f'documents_dropped_by_source_variant/{k}')}" for k, v in sorted(by_var.items()) if k.startswith("bipia/")]
        out.append(f"- Дедупликация не рвёт пары: снято контекстов целиком {num(drops.get('bipia_contexts'), dp, 'documents_dropped/bipia_contexts')}, "
                   f"документов этих контекстов {num(drops.get('bipia_pairs'), dp, 'documents_dropped/bipia_pairs')}; выпавшие документы BIPIA "
                   f"по варианту и метке: {', '.join(parts) or NA}.")
    else:
        out.append(f"- Дедупликация BIPIA: {MISSING} (dedup.json нет).")
    return out + [""]


def dedup_block(inp: Inputs) -> list[str]:
    dd = inp.dedup
    if not dd:
        return [f"_dedup.json: {MISSING}_", ""]
    dp = f"{inp.p(inp.manifests)}/dedup.json"
    rule = dd.get("rule") or {}
    out = [f"Правило: {rule.get('scope')} {ref(dp, 'rule/scope')}; MinHash по символьным шинглам длины "
           f"{num(rule.get('shingle'), dp, 'rule/shingle')} ({num(rule.get('minhash_perm'), dp, 'rule/minhash_perm')} перестановок, "
           f"LSH b = {num(rule.get('lsh_bands'), dp, 'rule/lsh_bands')}, r = {num(rule.get('lsh_rows'), dp, 'rule/lsh_rows')}), "
           f"проверка точным Жаккаром ≥ {num(rule.get('jaccard'), dp, 'rule/jaccard')}; эталоны — окна "
           f"{', '.join(code(x) for x in rule.get('reference_splits') or [])} {ref(dp, 'rule/reference_splits')}. Исключаются только "
           "тестовые окна; пары BIPIA не разрываются.", ""]
    out += table(["величина", "значение"], [
        ["окон всего", num(dd.get("windows_total"), dp, "windows_total")],
        ["окон в тесте", num(dd.get("windows_test"), dp, "windows_test")],
        ["эталонных окон (train+val)", num(dd.get("windows_reference"), dp, "windows_reference")],
        ["исключено тестовых окон", num(dd.get("test_windows_excluded"), dp, "test_windows_excluded")],
        ["документов выпало из теста", num(dd.get("documents_dropped_total"), dp, "documents_dropped_total")]])
    by_label = dd.get("test_windows_excluded_by_label") or {}
    by_src = dd.get("test_windows_excluded_by_source") or {}
    by_pair = dd.get("test_windows_excluded_by_source_pair") or {}
    if by_pair or by_src or by_label:
        rows = [[f"пара источников {code(k)} (тест → эталон)", num(v, dp, f"test_windows_excluded_by_source_pair/{k}")] for k, v in sorted(by_pair.items())]
        rows += [[f"источник {code(k)}", num(v, dp, f"test_windows_excluded_by_source/{k}")] for k, v in sorted(by_src.items())]
        rows += [[f"метка {k}", num(v, dp, f"test_windows_excluded_by_label/{k}")] for k, v in sorted(by_label.items())]
        out += ["**Исключённые тестовые окна**", ""] + table(["срез", "окон"], rows)
    drops = dd.get("documents_dropped") or {}
    by_var = dd.get("documents_dropped_by_source_variant") or {}
    if drops or by_var:
        rows = [[DROP_REASONS.get(k, code(k)), num(v, dp, f"documents_dropped/{k}")] for k, v in sorted(drops.items())]
        rows += [[f"источник / вариант / метка {code(k)}", num(v, dp, f"documents_dropped_by_source_variant/{k}")] for k, v in sorted(by_var.items())]
        out += ["**Документы, выпавшие из теста**", ""] + table(["причина / срез", "документов"], rows)
    if "dojo/main/0" in by_var:
        pre = (((inp.contamination or {}).get("document_level") or {}).get("dojo") or {}).get("test", {}).get("0", {}).get("documents") \
            if not inp.contamination_skipped() else None
        cp = inp.p(inp.contamination_path)
        out += [f"Статичные окружения AgentDojo (ASSUMPTIONS A38, контракт §10): чистые ответы тестовых задач совпадают с "
                f"ответами валидационных задач, поэтому выпало {num(by_var['dojo/main/0'], dp, 'documents_dropped_by_source_variant/dojo/main/0')} "
                "чистых тестовых документов AgentDojo"
                + (f" из {num(pre, cp, 'document_level/dojo/test/0/documents')}" if pre is not None else "")
                + f"; в тесте E1 осталось негативов AgentDojo {e1_source_cell(inp, 'dojo', 'n_neg')}. Это свойство бенчмарка, "
                "а не детектора; угрозы валидности — раздел 9.", ""]
    return out


def pools_block(inp: Inputs) -> list[str]:
    out: list[str] = []
    cy = "configs/default.yaml"
    th = inp.cfg.default.get("thresholds") or {}
    if inp.pools:
        pp = f"{inp.p(inp.manifests)}/pools.json"
        rows = []
        for pool in ("p_val", "p_test", "notinject"):
            d = inp.pools.get(pool) or {}
            if not d:
                continue
            rows.append([pool, num(d.get("n"), pp, f"{pool}/n"),
                         ", ".join(f"{s} {num(n, pp, f'{pool}/by_source/{s}')}" for s, n in sorted((d.get("by_source") or {}).items())) or NA,
                         num(d.get("target_min_docs"), pp, f"{pool}/target_min_docs") if "target_min_docs" in d else NA,
                         f"{d.get('meets_target')} {ref(pp, f'{pool}/meets_target')}" if "meets_target" in d else NA,
                         num(d.get("shortfall"), pp, f"{pool}/shortfall") if "shortfall" in d else NA])
        out += table(["пул", "документов", "по источникам", "цель, документов", "цель достигнута", "не хватает"], rows)
        n_test = (inp.pools.get("p_test") or {}).get("n")
    else:
        out += [f"_pools.json: {MISSING}_", ""]
        n_test, pp = None, None
    pm, ft = th.get("pool_min_docs") or {}, th.get("fpr_targets") or {}
    ppath = inp.p(R.power_path(inp.root, inp.smoke))
    target = (inp.power or {}).get("fpr_target") if inp.power else None
    line = (f"Рабочая точка TPR (правило Этапа 0 по размеру P_test): при |P_test| ≥ {num(pm.get('fpr_1pct'), cy, 'thresholds/pool_min_docs/fpr_1pct')} "
            f"— FPR {num(ft.get('primary'), cy, 'thresholds/fpr_targets/primary')}, при |P_test| ≥ {num(pm.get('fpr_5pct'), cy, 'thresholds/pool_min_docs/fpr_5pct')} "
            f"— FPR {num(ft.get('fallback'), cy, 'thresholds/fpr_targets/fallback')}, при меньшем пуле — только ROC-AUC. ")
    if n_test is not None:
        line += f"Здесь |P_test| = {num(n_test, pp, 'p_test/n')}; "
    if not inp.power:
        line += f"таблица E0 {code(ppath)} — {MISSING}."
    elif target is None:
        line += f"по замороженной таблице E0 TPR при FPR не определён, метрика — только AUC {ref(ppath, 'fpr_target')}."
    else:
        small = ((inp.pools or {}).get("p_test") or {}).get("meets_target") is False
        line += f"по замороженной таблице E0 рабочая точка — FPR {num(target, ppath, 'fpr_target')}" + (" (DEVIATIONS D13)." if small else ".")
    return out + [line, ""]


def c_unl_block(inp: Inputs) -> list[str]:
    if not inp.splits or not isinstance(inp.splits.get("c_unl"), list):
        return [f"_C_unl: {MISSING}_", ""]
    D = inp.derived
    sp = f"{inp.p(inp.manifests)}/splits.json"
    c_unl = [str(x) for x in inp.splits["c_unl"]]
    e1 = inp.splits.get("e1") or {}
    train, val = set(map(str, e1.get("train") or [])), set(map(str, e1.get("val") or []))
    test = {str(x) for ids in (e1.get("test") or {}).values() for x in (ids or [])}
    by_src = Counter(source_of(x) for x in c_unl)
    parts = [f"{s} {D.num(f'data/c_unl/by_source/{s}', n, f'{sp}#c_unl', 'count of doc_id prefixes')}"
             for s, n in sorted(by_src.items(), key=lambda kv: order_key(SOURCE_ORDER)(kv[0]))]
    cset = set(c_unl)
    return [f"Тексты обучающего и валидационного разбиений E1 без меток (на них обучаются SVD, idf, центрирование и "
            f"стандартизация): документов {D.num('data/c_unl/n', len(c_unl), f'{sp}#c_unl', 'len(list)')}; по источникам: "
            + (", ".join(parts) or NA) + f"; из обучения E1 {D.num('data/c_unl/in_train', len(cset & train), f'{sp}#c_unl,e1/train', 'overlap')}, "
            f"из валидации E1 {D.num('data/c_unl/in_val', len(cset & val), f'{sp}#c_unl,e1/val', 'overlap')}, из тестовых списков E1 "
            f"{D.num('data/c_unl/in_test', len(cset & test), f'{sp}#c_unl,e1/test', 'overlap')}.", ""]


# ================================================================================================= section 4
RATE_TITLES = {"acceptance": ("Доли принятия кандидатов судьёй", "accepted", "принято"),
               "generation_refusal": ("Доли отказов генератора", "refusals", "отказов"),
               "judge_refusal": ("Доли отказов судьи", "refusals", "отказов")}
GROUP_TITLES = {"by_stratum": "страта", "by_kind": "вид базы", "by_template": "шаблон / вид базы",
                "by_generator": "генератор", "by_judge": "судья"}


def sec_generation(inp: Inputs) -> list[str]:
    out = ["## 4. Генерация трасс и парафраз", ""]
    tm = inp.traces_manifest
    tp = inp.p(inp.root / "results/shared/traces_manifest.json")
    if tm:
        out += [f"Трассы: модель агента {code(tm.get('agent_model'))} {ref(tp, 'agent_model')}, провайдер {tm.get('provider')} "
                f"{ref(tp, 'provider')}, temperature {num(tm.get('temperature'), tp, 'temperature', nd=1)}, режим размышлений "
                f"{tm.get('thinking')} {ref(tp, 'thinking')}, заморожено {tm.get('generated')} {ref(tp, 'generated')}; "
                f"sha256 каждого лога — в списке `files` {ref(tp, 'files')}.", ""]
        rows = []
        for bench, suites in sorted((tm.get("counts") or {}).items()):
            for suite, attacks in sorted(suites.items()):
                for attack, classes in sorted(attacks.items()):
                    rows.append([bench, suite, code(attack)] + [num(classes.get(c), tp, f"counts/{bench}/{suite}/{attack}/{c}")
                                                                if c in classes else NA for c in ("benign", "hijacked", "injection_ignored", "error")])
        out += ["**Эпизоды по классам (контракт §2)**", ""] + table(["бенчмарк", "сьют", "шаблон", "benign", "hijacked", "injection_ignored", "error"], rows)
    else:
        out += [f"_Манифест трасс: {MISSING}_", ""]
    out += traces_check(inp)
    out += pilot_block(inp)
    if inp.extraction:
        ep = inp.p(inp.root / "data/manifests/traces_extraction.json")
        rows = []
        for bench, d in sorted(inp.extraction.items()):
            if not isinstance(d, dict):
                continue
            rows.append([bench, num(d.get("n_logs"), ep, f"{bench}/n_logs"), num(d.get("documents"), ep, f"{bench}/documents"),
                         num(d.get("documents_positive"), ep, f"{bench}/documents_positive"),
                         num(d.get("steps_total"), ep, f"{bench}/steps_total"), num(d.get("steps_labelled"), ep, f"{bench}/steps_labelled"),
                         num((d.get("attacked_without_span") or {}).get("count"), ep, f"{bench}/attacked_without_span/count"),
                         num(d.get("errors") if not isinstance(d.get("errors"), dict) else (d.get("errors") or {}).get("count"), ep,
                             f"{bench}/errors" if not isinstance(d.get("errors"), dict) else f"{bench}/errors/count")])
        out += ["**Извлечение шагов (ТЗ 1.5)**", ""]
        out += table(["бенчмарк", "логов", "документов", "позитивов", "шагов", "шагов с меткой", "атак без спана", "ошибок"], rows)
    if inp.split_manifest:
        smp = inp.p(inp.root / "results/shared/split_manifest.json")
        c = inp.split_manifest.get("counts") or {}
        out += ["**Разбиение контракта (§6)**", ""]
        out += table(["список", "эпизодов"], [[k, num(c.get(k), smp, f"counts/{k}")]
                                              for k in ("test", "observation", "validation_clean", "validation_attacks", "train_attacks", "excluded", "error") if k in c])
    out += paraphrase_block(inp)
    sp = inp.spend
    spp = inp.p(inp.root / "results/spend.json")
    if sp:
        out += [f"**Расход API**: {num(sp.get('spent_usd'), spp, 'spent_usd')} USD из бюджета "
                f"{num(sp.get('budget_usd'), spp, 'budget_usd')} USD, вызовов {num(sp.get('n_calls'), spp, 'n_calls')}, "
                f"в бюджете: {sp.get('within_budget')} {ref(spp, 'within_budget')}.", ""]
        streams = {k: v for k, v in (sp.get("breakdown") or {}).items() if k.startswith("stream:") or k.startswith("model:")}
        out += table(["поток / модель", "вызовов", "USD"],
                     [[code(k), num(v.get("calls"), spp, f"breakdown/{k}/calls", nd=0), num(v.get("cost_usd"), spp, f"breakdown/{k}/cost_usd")]
                      for k, v in sorted(streams.items())])
    else:
        out += [f"_Расход API: {MISSING}_", ""]
    return out


def traces_check(inp: Inputs) -> list[str]:
    """ТЗ 1.5 «Сверка»: our targeted ASR and utility per suite next to the published range over the undefended models
    of the published runs (``results/traces_stats.json``); the published range is min / median / max over models."""
    out = ["**Сверка с опубликованными ASR и utility (ТЗ 1.5 «Сверка»)**", ""]
    ts = inp.traces_stats
    tsp = inp.p(inp.traces_stats_path)
    if not ts:
        return out + [f"_{MISSING}: файла {tsp} нет._", ""]
    D = inp.derived
    same = ts.get("same_model_published")
    out += [f"Опубликованные прогоны: {code(ts.get('published_source'))} {ref(tsp, 'published_source')}. Та же модель агента среди "
            f"опубликованных прогонов: {'есть' if same else 'нет'} {ref(tsp, 'same_model_published')}"
            + (". Поэтому наши числа стоят рядом с диапазоном других моделей без защиты (минимум / медиана / максимум по "
               "моделям); расхождение записывается как наблюдение, а не как проверка." if not same else "."), ""]
    ours = ts.get("ours") or {}
    pub = ts.get("published") or {}
    rows = []
    for bench in sorted(set(ours) | set(pub)):
        models = pub.get(bench) or {}
        suites = sorted(set(ours.get(bench) or {}) | {s for m in models.values() if isinstance(m, dict) for s in m},
                        key=lambda x: (x == "all", x))           # the pooled row "all" last
        for suite in suites:
            o = (ours.get(bench) or {}).get(suite) or {}
            base = f"ours/{bench}/{suite}"
            cells = [num(o.get(k), tsp, f"{base}/{k}") if o.get(k) is not None else NA
                     for k in ("n_attacked", "n_hijacked", "targeted_asr", "n_clean", "utility_clean", "utility_under_attack")]
            ranges = []
            for metric in ("targeted_asr", "utility_clean"):
                vals = [(m.get(suite) or {}).get(metric) for m in models.values() if isinstance(m, dict)]
                st, r = D.stats(f"traces/published/{bench}/{suite}/{metric}", vals, f"{tsp}#published/{bench}/*/{suite}/{metric}",
                                "min / median / max over published models")
                ranges.append(f"{fmt(st['min'])} / {fmt(st['median'])} / {fmt(st['max'])} (моделей {st['n']}) {r}" if st else NA)
            rows.append([bench, suite] + cells + ranges)
    out += table(["бенчмарк", "сьют", "атак", "угнано", "targeted ASR", "чистых", "utility", "utility под атакой",
                  "опубл. ASR: мин / медиана / макс", "опубл. utility: мин / медиана / макс"], rows)
    notes = ts.get("notes") or []
    if notes:
        out += [f"- {n} {ref(tsp, f'notes/{i}')}" for i, n in enumerate(notes)] + [""]
    return out


def pilot_block(inp: Inputs) -> list[str]:
    """The agent-model pilot of ТЗ 1.5 / Приложение B (every candidate) and the budget projection behind D6."""
    p = inp.pilot
    ppath = inp.p(inp.root / "results/pilot.json")
    out = ["**Пилот выбора модели агента (ТЗ 1.5, Приложение B)**", ""]
    if not p:
        return out + [f"_{MISSING}: файла {ppath} нет._", ""]
    rule = p.get("rule") or {}
    head = []
    if p.get("suite") is not None:
        head.append(f"сьют {p.get('suite')} {ref(ppath, 'suite')}")
    if p.get("attack") is not None:
        head.append(f"шаблон {code(p.get('attack'))} {ref(ppath, 'attack')}")
    if rule.get("asr_range"):
        head.append(f"правило: targeted ASR в [{', '.join(fmt(x) for x in rule['asr_range'])}] {ref(ppath, 'rule/asr_range')}")
    if rule.get("min_utility") is not None:
        head.append(f"utility ≥ {num(rule.get('min_utility'), ppath, 'rule/min_utility')}")
    out += [("; ".join(head) + ". " if head else "") + f"Выбрана {code(p.get('chosen_model'))} {ref(ppath, 'chosen_model')}, "
            f"по правилу: {p.get('chosen_by_rule')} {ref(ppath, 'chosen_by_rule')}.", ""]
    rows = []
    for i, c in enumerate(p.get("candidates") or []):
        b = f"candidates/{i}"
        rows.append([code(c.get("model")), num(c.get("n_attacked"), ppath, f"{b}/n_attacked"), num(c.get("n_clean"), ppath, f"{b}/n_clean"),
                     num(c.get("targeted_asr"), ppath, f"{b}/targeted_asr"), num(c.get("utility_clean"), ppath, f"{b}/utility_clean"),
                     num(c.get("pilot_cost_usd"), ppath, f"{b}/pilot_cost_usd", nd=4),
                     num(c.get("cost_per_attacked_episode_usd"), ppath, f"{b}/cost_per_attacked_episode_usd", nd=6),
                     f"{c.get('accepted')} {ref(ppath, f'{b}/accepted')}"])
    out += table(["кандидат", "атак", "чистых", "targeted ASR", "utility", "стоимость пилота, USD", "USD на атакованный эпизод", "принят"], rows)
    pr = (p.get("projection") or {}).get("priorities") or []
    if pr:
        rows = []
        for i, r in enumerate(pr):
            b = f"projection/priorities/{i}"
            rows.append([code(r.get("benchmark")), code(r.get("attack")),
                         code(r.get("tasks")), num(r.get("episodes_attacked"), ppath, f"{b}/episodes_attacked"),
                         num(r.get("episodes_clean"), ppath, f"{b}/episodes_clean"), num(r.get("est_cost_usd"), ppath, f"{b}/est_cost_usd", nd=2),
                         num(r.get("cumulative_usd"), ppath, f"{b}/cumulative_usd", nd=2),
                         f"{r.get('fits_remaining_budget')} {ref(ppath, f'{b}/fits_remaining_budget')}"])
        proj = p.get("projection") or {}
        out += [f"**Проекция стоимости по приоритетам ТЗ 1.5** (бюджет трасс {num(proj.get('trace_budget_usd'), ppath, 'projection/trace_budget_usd', nd=2)} USD; "
                "урезания — DEVIATIONS D6)", ""]
        out += table(["бенчмарк", "шаблон", "задачи", "атак", "чистых", "оценка, USD", "нарастающим итогом, USD", "в бюджете"], rows)
    return out


def paraphrase_block(inp: Inputs) -> list[str]:
    """ТЗ 1.6: volumes, then the acceptance and refusal shares by every grouping the file records, in full."""
    pr = inp.paraphrases
    prp = inp.p(inp.root / "results/paraphrases.json")
    out = ["**Парафразы (ТЗ 1.6): объёмы, принятие и отказы**", ""]
    if not pr:
        return out + [f"_Парафразы: {MISSING} (results/paraphrases.json нет)._", ""]
    out += dump(pr.get("counts") or {}, prp, "counts", limit=None)
    rates = pr.get("rates") or {}
    for kind in sorted(rates, key=order_key(tuple(RATE_TITLES))):
        title, count_key, count_title = RATE_TITLES.get(kind, (kind, "count", "count"))
        rows = []
        for grouping in sorted(rates[kind] or {}, key=order_key(tuple(GROUP_TITLES))):
            for name, rec in sorted((rates[kind][grouping] or {}).items()):
                if not isinstance(rec, dict):
                    continue
                b = f"rates/{kind}/{grouping}/{name}"
                ck = count_key if count_key in rec else next((k for k in rec if k not in ("n", "rate")), count_key)
                rows.append([GROUP_TITLES.get(grouping, grouping), code(name), num(rec.get("n"), prp, f"{b}/n"),
                             num(rec.get(ck), prp, f"{b}/{ck}"), num(rec.get("rate"), prp, f"{b}/rate", nd=4)])
        out += [f"**{title}** (все группировки файла)", ""] + table(["группировка", "группа", "кандидатов / вызовов", count_title, "доля"], rows)
    return out


# ================================================================================================= section 5
def power_block(p: dict[str, Any] | None, pp: str, title: str, full: bool) -> list[str]:
    out = [f"### {title}", ""]
    if not p:
        return out + [f"_{MISSING}: {pp} нет._", ""]
    meta = {k: p[k] for k in ("stage", "frozen", "created_at", "config_hash", "delta_rel", "alpha", "tost_level",
                              "power_target", "fpr_target") if k in p}
    out += dump(meta, pp, "", "Параметры и статус")
    if isinstance(p.get("pools"), dict) and p["pools"]:
        out += [f"Пулы, по которым выбрана рабочая точка FPR: " + ", ".join(
            f"{k} {num(v, pp, f'pools/{k}')}" for k, v in sorted(p["pools"].items())) + ".", ""]
    sizes = p.get("sizes") or {}
    out += ["**Размеры**", ""] + table(["источник", "позитивов", "негативов", "кластеров"],
                                       [[s, num(d.get("n_pos"), pp, f"sizes/{s}/n_pos"), num(d.get("n_neg"), pp, f"sizes/{s}/n_neg"),
                                         num(d.get("n_clusters"), pp, f"sizes/{s}/n_clusters")]
                                        for s, d in sorted(sizes.items(), key=lambda kv: order_key(SOURCE_ORDER + ("macro",))(kv[0]))
                                        if isinstance(d, dict)])
    carriers = p.get("carriers") or {}
    if carriers:
        metrics = sorted({m for row in carriers.values() for m in row})
        out += ["**Носители: источник × метрика**", ""]
        out += table(["источник"] + metrics, [[s] + [f"{carriers[s].get(m, NA)} {ref(pp, f'carriers/{s}/{m}')}" for m in metrics]
                                             for s in sorted(carriers, key=order_key(SOURCE_ORDER + ("macro",)))])
    planning = p.get("planning_level") or {}
    cells = p.get("cells") or {}
    rows = []
    for s, lvl in sorted(planning.items(), key=lambda kv: order_key(SOURCE_ORDER + ("macro",))(kv[0])):
        if finite(lvl) is None:
            continue
        cell = (cells.get(s) or {}).get(f"{lvl:g}") or {}
        rows.append([s, num(lvl, pp, f"planning_level/{s}")]
                    + [num(cell.get(k), pp, f"cells/{s}/{lvl:g}/{k}") if not isinstance(cell.get(k), (dict, list)) else NA
                       for k in ("mdd", "delta", "tost_power", "status")])
    if rows:
        out += ["**MDD разности AUC и мощность TOST на уровне планирования** (прочерк = не определено на этом объёме)", ""]
        out += table(["источник", "уровень AUC", "MDD", "δ", "мощность TOST", "статус"], rows)
    if full:
        for key, title2 in (("notinject", "H2: ширина ДИ разности долей на парных наблюдениях NotInject"),
                            ("spread", "Разброс на валидации: curveball и перестановки perm"), ("hypotheses", "Входы вердиктов")):
            if key in p:
                out += dump(p[key], pp, key, title2)
    return out


def sec_power(inp: Inputs) -> list[str]:
    out = ["## 5. Мощность (E0)", "",
           "Этап 0 выполняется дважды (ТЗ Этап 0, ASSUMPTIONS A44): первая стадия — на таблицах без трасс и "
           "парафраз; вторая — окончательная, после них; она замораживается до финального прогона и используется в "
           "вердиктах. Обе таблицы приведены ниже.", ""]
    out += power_block(inp.power_stage1, inp.p(inp.power_stage1_path), "Первая стадия (до трасс и парафраз)", full=False)
    out += power_block(inp.power, inp.p(R.power_path(inp.root, inp.smoke)), "Вторая стадия (окончательная, замороженная)", full=True)
    return out


# ================================================================================================= section 6
def sec_results(inp: Inputs, figs: dict[str, tuple[Path | None, str]], figrel) -> list[str]:
    out = ["## 6. Результаты E1–E6", ""]
    st = inp.cfg.default["stats"]
    out += ["Интервалы (ТЗ Этап 4 «Статистика», ASSUMPTIONS A55): кластерный бутстреп по `cluster_id` внутри источника, "
            f"уровень `1 − α` при α = {num(st['bootstrap']['alpha'], 'configs/default.yaml', 'stats/bootstrap/alpha')}, "
            f"{num(st['bootstrap'].get('n'), 'configs/default.yaml', 'stats/bootstrap/n')} повторов"
            + (f" (в смоуке {num(inp.smoke_cfg.get('bootstrap'), 'configs/default.yaml', 'smoke/bootstrap')})" if inp.smoke else "") + ". "
            "У одного сида число — его значение и его интервал. У нескольких сидов число — среднее по сидам, интервал — "
            "огибающая [мин. нижняя граница, макс. верхняя граница] бутстреп-интервалов сидов (консервативная сводка того же "
            "типа интервала), после ссылки — sd точечных оценок по сидам и число сидов. TOST — по интервалу уровня "
            f"{num(st['tost']['ci'], 'configs/default.yaml', 'stats/tost/ci')} с коридором δ = "
            f"{num(st['tost']['delta_rel'], 'configs/default.yaml', 'stats/tost/delta_rel')} от референса. macroAUC — среднее "
            "AUC по присутствующим источникам с равным весом (основа вердиктов); объединение по числу документов не используется.", ""]
    for name in ("roc", "auc", "notinject", "para"):
        path, caption = figs[name]
        out += [f"![{caption}]({figrel(path)})" if path else f"_Рисунок ({caption})._", ""]
    titles = {"E1": "основной кросс-датасетный", "E2": "кривые обучения", "E3": "перенос", "E4": "проводка",
              "E5": "сетка абляций", "E6": "чувствительность"}
    for exp in ("E1", "E2", "E3", "E4", "E5", "E6"):
        out += experiment_block(inp, exp, titles[exp])
        if exp == "E2":
            path, caption = figs["learning"]
            out += [f"![{caption}]({figrel(path)})" if path else f"_Рисунок ({caption})._", ""]
        if exp == "E4":
            path, caption = figs["curveball"]
            out += [f"![{caption}]({figrel(path)})" if path else f"_Рисунок ({caption})._", ""]
        if exp == "E5":
            out += ["**Сетка абляций (текстовая таблица)**", ""] + ablation_grid(inp)
    return out


def sec_contributions(inp: Inputs) -> list[str]:
    out = ["## 7. Разделение вкладов", ""]
    out += ["Определения (ТЗ Этап 4 «Вклады»): нос = AUC(нос, без слоя, линейный); расширение = AUC(нос, расширение, линейный) − нос; "
            "правило обучения = AUC(Bloom) − AUC(линейный) в клетке; проводка = AUC(измеренная) − среднее AUC(curveball).", ""]
    s5 = inp.summaries.get("E5")
    if s5 and any("contrib" in n for n in (s5.get("tables") or {})):
        for name in sorted(n for n in s5["tables"] if "contrib" in n):
            out += [f"**Таблица {code(name)} (E5)**", ""] + table_rows(s5, name, inp.spath("E5"))
    elif s5:
        contrib = {k: v for k, v in (s5.get("numbers") or {}).items() if "contrib" in k}
        out += generic_numbers(contrib, inp.spath("E5"), skip_prefixes=()) if contrib else [f"_Таблицы вкладов в E5 нет: {MISSING}_", ""]
    else:
        out += [f"_E5: {MISSING}_", ""]
    s4 = inp.summaries.get("E4")
    if s4:
        wiring = {k: v for k, v in (s4.get("numbers") or {}).items() if "wiring" in k or "contrib" in k or k.startswith("diff/macro_auc/real_fly_bloom-curveball")}
        out += (["**Проводка (E4)**", ""] + generic_numbers(wiring, inp.spath("E4"), skip_prefixes=())) if wiring else []
    return out


def sec_contract(inp: Inputs) -> list[str]:
    out = ["## 8. Общий датасет (контракт сравнения v3)", ""]
    shared = (inp.rdir / "shared") if inp.smoke else (inp.root / "results" / "shared")
    csv_path = shared / "flyguard.csv"
    out.append(f"Файлы в `results/shared/`: traces_manifest.json ({'есть' if inp.traces_manifest else 'нет'}), "
               f"split_manifest.json ({'есть' if inp.split_manifest else 'нет'}), flyguard.csv ({'есть' if csv_path.exists() else 'нет'}), "
               f"копия трасс ({'есть' if (shared / 'traces').exists() else 'нет'}).")
    out.append("")
    if csv_path.exists():
        try:
            from flyguard.agentdojo_io.contract import validate_csv

            problems = validate_csv(csv_path)
            out += ["Валидация CSV по схеме §8: " + ("проходит" if not problems else "не проходит: " + "; ".join(code(x) for x in problems[:5])), ""]
        except Exception as exc:  # noqa: BLE001
            out += [f"Валидация CSV не выполнена ({type(exc).__name__}).", ""]
    c = inp.contract
    cp = inp.p(inp.rdir / "contract.json")
    if not c:
        return out + [f"_Метрики контракта (§9): {MISSING} ({cp} нет)._", ""]
    numbers = c.get("numbers") or {}
    if numbers:
        out += ["**Метрики сравнения (§9) по вариантам и пороги (`contract/<метрика>/<детектор>`)**", ""]
        out += generic_numbers(numbers, cp, skip_prefixes=())
    thresholds = c.get("thresholds") or {}
    if thresholds:
        rows = [[name, num(rec.get("value"), cp, f"thresholds/{name}/value"), str(rec.get("source")), str(rec.get("target")),
                 num(rec.get("n"), cp, f"thresholds/{name}/n")] for name, rec in sorted(thresholds.items()) if isinstance(rec, dict)]
        out += ["**Пороги контракта (§7)**", ""] + table(["порог", "значение", "источник", "цель", "n"], rows)
    for name in sorted(c.get("tables") or {}):
        out += [f"**Таблица {code(name)}**", ""] + table_rows(c, name, cp)
    scalars = {k: v for k, v in c.items() if k in ("training_mode", "n_rows", "csv", "csv_valid", "metrics", "metrics_by_benchmark")}
    if scalars:
        out += dump(scalars, cp, "", "Прочее")
    notes = c.get("notes") or []
    if notes:
        out += ["Примечания:", ""] + [f"- {n} {ref(cp, f'notes/{i}')}" for i, n in enumerate(notes)] + [""]
    out += [f"Хеш конфига {code(c.get('config_hash'))} {ref(cp, 'config_hash')}.", ""]
    return out




# ================================================================================================= section 9
def sec_threats(inp: Inputs) -> list[str]:
    out = ["## 9. Угрозы валидности", ""]
    devs = inp.journal_ids("DEVIATIONS.md", "D")
    asms = inp.journal_ids("ASSUMPTIONS.md", "A")
    blks = inp.journal_ids("BLOCKERS.md", "B")
    out += [f"Отклонения: {', '.join(devs) or 'нет'} (DEVIATIONS.md); допущения: {', '.join(asms) or 'нет'} (ASSUMPTIONS.md); "
            f"блокеры: {', '.join(blks) or 'нет'} (BLOCKERS.md).", ""]
    cy = "configs/default.yaml"
    prp = inp.p(inp.root / "results/paraphrases.json")
    models = (inp.paraphrases or {}).get("models") or {}
    # ---- construct
    out += ["### Конструктная", "",
            f"- Метка окна следует спану инъекции (не меньше {num(inp.cfg.default['windows']['min_span_chars'], cy, 'windows/min_span_chars')} "
            "символов спана), а не поведению агента; классы эпизодов зависят от модели агента (контракт §10).",
            ("- Парафразы: " + (f"судей {num(models.get('n_judges'), prp, 'models/n_judges')}, правило {models.get('judge_rule')} "
                                f"{ref(prp, 'models/judge_rule')}"
                                + ("; при одном судье «двойное да» ТЗ 1.6 вырождается в «да» одного судьи (DEVIATIONS D2), отбор "
                                   "позитивов может быть мягче." if models.get("n_judges") == 1 else ".")
                                if models else f"{MISSING} (results/paraphrases.json нет)."))]
    # ---- internal
    sm = inp.smoke_cfg
    out += ["", "### Внутренняя", "",
            "- Метки только из deepset train, гиперпараметры и пороги — на валидации; тест открыт через журналируемый доступ "
            "(logs/data_access.log). Регулярки выведены из deepset train до первого чтения теста (DEVIATIONS D10 — порядок по журналу). "
            "Дедупликация исключает только тестовые окна (раздел 3).",
            f"- Смоук и тест (ASSUMPTIONS A50): смоук открывает детерминированную голову тестового разбиения "
            f"({num(sm.get('docs_per_source'), cy, 'smoke/docs_per_source')} документов на источник); по числам смоука не выбирались "
            "ни модели, ни гиперпараметры, ни регулярки, ни настройки конфига; после смоука менялись только исполнительские ключи "
            "(ASSUMPTIONS A40)."]
    # ---- external
    out += ["", "### Внешняя", ""] + external_items(inp)
    # ---- statistical
    pp = f"{inp.p(inp.manifests)}/pools.json"
    pt = (inp.pools or {}).get("p_test") or {}
    ppath = inp.p(R.power_path(inp.root, inp.smoke))
    target = (inp.power or {}).get("fpr_target")
    out += ["", "### Статистическая", "",
            "- Кластерный бутстреп внутри источника; какие источники несут разности, решает таблица носителей E0 (раздел 5).",
            (f"- Пулы негативов{' меньше цели' if pt.get('meets_target') is False else ''}: |P_test| = {num(pt.get('n'), pp, 'p_test/n')} "
             f"при цели {num(pt.get('target_min_docs'), pp, 'p_test/target_min_docs')}; "
             + (f"рабочая точка TPR — FPR {num(target, ppath, 'fpr_target')}" if target is not None else
                (f"TPR при FPR не определён {ref(ppath, 'fpr_target')}" if inp.power else f"таблица E0 — {MISSING}"))
             + (" (DEVIATIONS D13)." if pt.get("meets_target") is False else ".") if pt else f"- Пулы негативов: {MISSING}."),
            "- Сводка по сидам (ASSUMPTIONS A55): среднее и огибающая бутстреп-интервалов сидов — консервативно; вердикт выносится "
            "по каждому сиду, итог — модальный статус (ASSUMPTIONS A30, A42)."]
    # ---- generator style and refusals
    out += ["", "### Стиль генераторов и отказы провайдеров", ""]
    rates = (inp.paraphrases or {}).get("rates") or {}
    if models:
        gen = rates.get("generation_refusal", {}).get("by_generator") or {}
        jud = rates.get("judge_refusal", {}).get("by_judge") or {}
        pm = inp.cfg.operator.get("llm_api", {}).get("paraphrase") or {}
        providers = sorted({str(x.get("provider")) for k in ("generators", "judges") for x in pm.get(k) or [] if isinstance(x, dict)})
        out.append(f"- Провайдеры генераторов и судей по конфигу: {', '.join(code(x) for x in providers) or NA} "
                   f"{ref('configs/operator.yaml', 'llm_api/paraphrase')}; генераторы {', '.join(code(g) for g in models.get('generators') or [])} "
                   f"{ref(prp, 'models/generators')}, судьи {', '.join(code(j) for j in models.get('judges') or [])} {ref(prp, 'models/judges')}"
                   + (" — один провайдер (DEVIATIONS D1): стиль парафраз однороден, судья может разделять стиль генератора."
                      if len(providers) == 1 else "."))
        if gen or jud:
            out.append("- Отказы смещают состав набора: доля отказов генератора "
                       + (", ".join(f"{code(g)} {num(r.get('rate'), prp, f'rates/generation_refusal/by_generator/{g}/rate', nd=4)}" for g, r in sorted(gen.items())) or NA)
                       + "; судьи " + (", ".join(f"{code(j)} {num(r.get('rate'), prp, f'rates/judge_refusal/by_judge/{j}/rate', nd=4)}" for j, r in sorted(jud.items())) or NA)
                       + "; доли по стратам, шаблонам и генераторам — раздел 4.")
    else:
        out.append(f"- {MISSING} (results/paraphrases.json нет).")
    # ---- contamination and authorship
    out += ["", "### Пересечение с обучением промышленных детекторов и авторство (ТЗ 3.2)", ""] + contamination_items(inp)
    return out + [""]


def external_items(inp: Inputs) -> list[str]:
    out: list[str] = []
    dp = f"{inp.p(inp.manifests)}/dedup.json"
    by_var = (inp.dedup or {}).get("documents_dropped_by_source_variant") or {}
    if "dojo/main/0" in by_var:
        out.append(f"- Статичные окружения AgentDojo (контракт §10, ASSUMPTIONS A38): дедупликация сняла "
                   f"{num(by_var['dojo/main/0'], dp, 'documents_dropped_by_source_variant/dojo/main/0')} чистых тестовых документов "
                   f"AgentDojo, в тесте осталось {e1_source_cell(inp, 'dojo', 'n_neg')}; FPR на AgentDojo измерен на остатке, "
                   "незнакомые окружения даёт только AgentDyn.")
    else:
        out.append(f"- Статичные окружения AgentDojo (контракт §10, ASSUMPTIONS A38): {MISSING}.")
    p = inp.pilot
    ppath = inp.p(inp.root / "results/pilot.json")
    if p:
        chosen = p.get("chosen_model")
        idx = next((i for i, c in enumerate(p.get("candidates") or []) if c.get("model") == chosen), None)
        rng = (p.get("rule") or {}).get("asr_range")
        asr = num((p["candidates"][idx] or {}).get("targeted_asr"), ppath, f"candidates/{idx}/targeted_asr") if idx is not None else NA
        ex = inp.extraction or {}
        ep = inp.p(inp.root / "data/manifests/traces_extraction.json")
        hij = [f"{b} {num((d.get('episodes_by_class') or {}).get('hijacked'), ep, f'{b}/episodes_by_class/hijacked')}"
               for b, d in sorted(ex.items()) if isinstance(d, dict) and (d.get("episodes_by_class") or {}).get("hijacked") is not None]
        unm = [f"{b} {num(((d.get('match_counts_by_class') or {}).get('hijacked') or {}).get('unmatched'), ep, f'{b}/match_counts_by_class/hijacked/unmatched')}"
               for b, d in sorted(ex.items()) if isinstance(d, dict) and ((d.get("match_counts_by_class") or {}).get("hijacked") or {}).get("unmatched") is not None]
        out.append(f"- Модель агента{' выбрана не по правилу пилота (DEVIATIONS D5, ASSUMPTIONS A13)' if p.get('chosen_by_rule') is False else ''}: "
                   f"{code(chosen)} {ref(ppath, 'chosen_model')}, "
                   f"по правилу: {p.get('chosen_by_rule')} {ref(ppath, 'chosen_by_rule')}, targeted ASR на пилоте {asr}"
                   + (f" при коридоре [{', '.join(fmt(x) for x in rng)}] {ref(ppath, 'rule/asr_range')}" if rng else "")
                   + ". Угнанных эпизодов: "
                   + (", ".join(hij) or NA) + "; из них без совпадения с эталонным вызовом (исключены из метрики остановки, "
                   "ASSUMPTIONS A20): " + (", ".join(unm) or NA) + ".")
    else:
        out.append(f"- Модель агента: {MISSING} (results/pilot.json нет).")
    tm = inp.traces_manifest or {}
    tp = inp.p(inp.root / "results/shared/traces_manifest.json")
    generated = sorted({a for suites in (tm.get("counts") or {}).values() for attacks in suites.values() for a in attacks if a != "none"})
    configured = list(((tm.get("harnesses") or {}).get("agentdojo") or {}).get("attacks") or [])
    missing = [a for a in configured if a not in generated]
    proj = ((p or {}).get("projection") or {}).get("priorities") or []
    cut = [f"{r.get('benchmark')}/{r.get('attack')}/{r.get('tasks')}" for r in proj if r.get("fits_remaining_budget") is False]
    sp, spp = inp.spend or {}, inp.p(inp.root / "results/spend.json")
    if tm:
        out.append(f"- Объём трасс{' урезан по бюджету (DEVIATIONS D6)' if missing or cut else ''}: сгенерированы шаблоны {', '.join(code(a) for a in generated) or NA} "
                   f"{ref(tp, 'counts')}"
                   + (f" из {', '.join(code(a) for a in configured)} {ref(tp, 'harnesses/agentdojo/attacks')}; не сгенерированы "
                      f"{', '.join(code(a) for a in missing) or 'нет'}" if configured else "")
                   + (f"; приоритеты вне бюджета: {', '.join(code(c) for c in cut) or 'нет'} "
                      f"{ref(inp.p(inp.root / 'results/pilot.json'), 'projection/priorities')}" if proj else "") + "; "
                   + (f"расход {num(sp.get('spent_usd'), spp, 'spent_usd')} из {num(sp.get('budget_usd'), spp, 'budget_usd')} USD." if sp else f"расход: {MISSING}."))
    else:
        out.append(f"- Урезание объёма трасс (DEVIATIONS D6): {MISSING} (манифест трасс нет).")
    e3 = (inp.splits or {}).get("e3") or {}
    sp3 = f"{inp.p(inp.manifests)}/splits.json"
    if e3:
        one = len(e3.get("templates_present") or []) == 1
        out.append(f"- Перенос E3{' при одном шаблоне (DEVIATIONS D12)' if one else ''}: шаблоны в данных {', '.join(code(x) for x in e3.get('templates_present') or []) or NA} "
                   f"{ref(sp3, 'e3/templates_present')}; пригодных фолдов "
                   + ", ".join(f"{k} {num(v, sp3, f'e3/usable/{k}')}" for k, v in sorted((e3.get("usable") or {}).items()))
                   + ("; кросс-шаблонные и двойные фолды получают «не хватило данных»." if one else "."))
    out.append("- Языки: немецкая половина deepset и мультиязычные подмножества NotInject ограничивают перенос (языковые страты — раздел 3).")
    return out


def contamination_items(inp: Inputs) -> list[str]:
    c = inp.contamination
    cp = inp.p(inp.contamination_path)
    out: list[str] = []
    if c and not inp.contamination_skipped():
        D = inp.derived
        dl, wl = c.get("document_level") or {}, c.get("window_level") or {}
        rows, sums = [], []
        for sk in sorted(dl, key=order_key(("deep", "bipia", "bipia_e6", "dojo", "dyn", "para", "notinject"))):
            for split in sorted((dl.get(sk) or {}), key=order_key(SPLIT_ORDER)):
                if split == "total":
                    continue
                labels = (dl[sk][split] or {})
                for lab, dc in sorted(labels.items()):
                    wc = ((wl.get(sk) or {}).get(split) or {}).get(lab) or {}
                    b = f"document_level/{sk}/{split}/{lab}"
                    wb = f"window_level/{sk}/{split}/{lab}"
                    rows.append([sk, split, lab, num(dc.get("documents_matched"), cp, f"{b}/documents_matched"),
                                 num(dc.get("documents"), cp, f"{b}/documents"), num(dc.get("share"), cp, f"{b}/share", nd=4),
                                 num(wc.get("windows_matched"), cp, f"{wb}/windows_matched") if wc else NA,
                                 num(wc.get("share"), cp, f"{wb}/share", nd=4) if wc else NA])
                if len(labels) > 1:
                    m = sum(int(v.get("documents_matched") or 0) for v in labels.values())
                    n = sum(int(v.get("documents") or 0) for v in labels.values())
                    src = f"{cp}#document_level/{sk}/{split}"
                    sums.append(f"{sk}/{split}: {D.num(f'contamination/{sk}/{split}/documents_matched', m, src, 'sum over labels')} из "
                                f"{D.num(f'contamination/{sk}/{split}/documents', n, src, 'sum over labels')}")
        out += ["Почти-дубликаты наших документов и окон в открытом обучающем наборе PIGuard (Жаккар по символьным шинглам, "
                f"правило `contamination.json → rule`; документ целиком и окна ТЗ 1.3):", ""]
        out += table(["источник", "разбиение", "метка", "документов совпало", "документов", "доля", "окон совпало", "доля окон"], rows)
        if sums:
            out += ["Итого по разбиениям (обе метки): " + "; ".join(sums) + ".", ""]
        tc = c.get("targeted_containment") or {}
        trows = []
        for sk, d in sorted(tc.items(), key=lambda kv: order_key(SOURCE_ORDER)(kv[0])):
            for split, s in sorted((d.get("by_split") or {}).items(), key=lambda kv: order_key(SPLIT_ORDER)(kv[0])):
                b = f"targeted_containment/{sk}/by_split/{split}"
                trows.append([sk, split, ", ".join(code(t) for t in d.get("tags") or []), num(s.get("documents"), cp, f"{b}/documents"),
                              num(s.get("ours_in_piguard"), cp, f"{b}/ours_in_piguard"), num(s.get("piguard_in_ours"), cp, f"{b}/piguard_in_ours")])
        if trows:
            out += ["**Точечное вхождение по тегам PIGuard** (наш документ целиком внутри записи PIGuard и наоборот)", ""]
            out += table(["источник", "разбиение", "тег PIGuard", "документов", "наших внутри PIGuard", "PIGuard внутри наших"], trows)
    else:
        out += [f"- Пересечение с обучающим набором PIGuard: {inp.missing('contamination_audit')}"
                + (f" ({c.get('note')}) {ref(cp, 'note')}" if c and c.get("note") else "") + ".", ""]
    cards = (c or {}).get("model_cards") or {}
    for name, card in sorted(cards.items()):
        if not isinstance(card, dict):
            continue
        ds = ", ".join(code(x) for x in card.get("training_datasets_on_card") or []) or NA
        out.append(f"- Карточка {code(card.get('model') or name)}: обучающие наборы по карточке — {ds} "
                   f"{ref(cp, f'model_cards/{name}/training_datasets_on_card')}; пересечение с нашими источниками: "
                   f"{card.get('named_overlap_with_our_sources')} {ref(cp, f'model_cards/{name}/named_overlap_with_our_sources')}.")
    for i, t in enumerate((c or {}).get("threats_to_validity") or []):
        out.append(f"- {t} {ref(cp, f'threats_to_validity/{i}')}")
    if not c:
        out.append(f"- Авторство (ТЗ 3.2) и карточки моделей: {MISSING} ({cp} нет).")
    return out + [""]


# ================================================================================================= section 10
def sec_reproduce(inp: Inputs, spath: str, setup: dict[str, Any]) -> list[str]:
    out = ["## 10. Воспроизведение", "",
           "Команды выполняются из корня репозитория; интерпретатор — `.venv/bin/python`, системный `python` в PATH не нужен.", "",
           "```sh",
           "scripts/setup_env.sh                          # три виртуальных окружения (.venv, .venv-agentdyn, .venv-flypath)",
           "scripts/fetch.sh                              # данные, модели, пины с sha256 -> data/manifests/sources.json",
           "scripts/gen_traces.sh pilot && scripts/gen_traces.sh run && scripts/gen_traces.sh freeze   # трассы (API)",
           "scripts/gen_paraphrases.sh all                # парафразы (API)",
           "scripts/smoke.sh                              # смоук: results/smoke/REPORT.md",
           "scripts/run_all.sh --jobs 4                   # финальный прогон: сиды E1-E6 параллельно (ASSUMPTIONS A39, A51)",
           "scripts/run_all.sh --jobs 4                   # тот же вызов продолжает прерванный прогон",
           ".venv/bin/python scripts/make_report.py       # REPORT.md и results/figures/",
           ".venv/bin/python scripts/check_acceptance.py  # критерии приёмки",
           "```", ""]
    out += [f"Коммит отчёта {code(setup.get('git_commit'))} {ref(spath, 'git_commit')} (коммиты результатов — раздел 2), хеш конфига "
            f"{code(setup['config_hash'])} {ref(spath, 'config_hash')}; сиды {', '.join(str(s) for s in setup['seeds']['global'])} "
            f"{ref(spath, 'seeds/global')} (смоук: первые {num(setup['seeds']['smoke_seeds'], spath, 'seeds/smoke_seeds')}). "
            "`run_all.sh` пропускает эксперимент, чей results/<E>/<seed>.json существует с текущим хешем конфига, и продолжает "
            "прерванный прогон; журнал стадий — `logs/run_all.log`.", ""]
    rows = []
    for exp in EXPERIMENTS:
        s = inp.summaries.get(exp)
        if not s:
            rows.append([exp, MISSING, NA, NA])
            continue
        for seed, t in sorted((s.get("timing") or {}).items(), key=seed_sort):
            rows.append([exp, f"{seed} {ref(inp.spath(exp), 'seeds')}", num(t.get("seconds"), inp.spath(exp), f"timing/{seed}/seconds", nd=1),
                         ", ".join(str(x) for x in (t.get("test_reads") or [])) or NA])
    out += ["**Время экспериментов по сидам и открытые тестовые источники**", ""] + table(["эксперимент", "сид", "секунд", "тестовые чтения"], rows)
    return out


# ================================================================================================= verifier
REF_RE = re.compile(r"\((?P<path>[\w./+-]+\.(?:json|ya?ml|md|txt|lock|log|csv))(?:#(?P<key>[^\s()]+))?\)")
# a number token: not glued to a word, a path or a key, and not the head of a dotted version string ("3.11.14")
NUM_RE = re.compile(r"(?<![\w/.:#=_+-])[-+]?\d+(?:\.\d+)?(?![\w/]|\.\d)")
CODE_RE = re.compile(r"`[^`]*`")
FENCE_RE = re.compile(r"```.*?```", re.S)
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?")
CITATION_RE = re.compile(r"(?:ТЗ|§|Этап[а-я]*|раздел[а-я]*|контракт[а-я]*|Приложени[а-я]*|п\.)\s?[A-Z]?\d+(?:\.\d+)*[a-z]?")


class Resolver:
    """Loads referenced files once and walks ``path#key`` references (greedy longest-key match)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cache: dict[str, Any] = {}

    def load(self, path: str) -> Any:
        if path not in self.cache:
            p = self.root / path
            if not p.exists():
                self.cache[path] = None
            elif p.suffix == ".json":
                self.cache[path] = json.loads(p.read_text(encoding="utf-8"))
            elif p.suffix in (".yaml", ".yml"):
                self.cache[path] = yaml.safe_load(p.read_text(encoding="utf-8"))
            else:
                self.cache[path] = p.read_text(encoding="utf-8", errors="replace")
        return self.cache[path]

    def resolve(self, path: str, key: str | None) -> tuple[bool, Any, list[str]]:
        """(found, node, key segments)."""
        node = self.load(path)
        if node is None:
            return False, None, []
        if not key:
            return True, node, []
        segs = [s for s in key.split("/") if s != ""]
        used: list[str] = []
        i = 0
        while i < len(segs):
            if isinstance(node, dict):
                for j in range(len(segs), i, -1):
                    cand = "/".join(segs[i:j])
                    if cand in node:
                        node = node[cand]
                        used.append(cand)
                        i = j
                        break
                else:
                    return False, None, used
            elif isinstance(node, list) and segs[i].isdigit() and int(segs[i]) < len(node):
                node = node[int(segs[i])]
                used.append(segs[i])
                i += 1
            else:
                return False, None, used
        return True, node, used


def leaves(node: Any, out: list | None = None, depth: int = 0) -> list:
    out = [] if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(str(k))
            if depth < 8:
                leaves(v, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if depth < 8:
                leaves(v, out, depth + 1)
    else:
        out.append(node)
    return out


def token_matches(tok: str, candidates: Iterable[Any]) -> bool:
    """Exact match of one numeric token of the report against the candidates of its references.

    A number of a JSON/YAML node matches when it *renders* to the token: an integer token needs an integral candidate
    of the same value (``3`` never matches ``2.6`` or ``0.9``), a token with ``d`` decimals needs a candidate whose
    ``f"{c:.{d}f}"`` equals it (the report's own rounding rule, :func:`fmt`). A string candidate (a text file such as
    ``audit.md``, a reason string of ``verdicts.json``, an environment variable) matches when it contains the token as
    a whole number: not preceded by a digit or ``.`` and not followed by a digit or a decimal part."""
    plain = tok.lstrip("+")
    dec = len(plain.split(".")[1]) if "." in plain else 0
    pattern = re.compile(rf"(?<![\d.]){re.escape(plain)}(?!\d|\.\d)")
    for c in candidates:
        if isinstance(c, bool) or c is None:
            continue
        if isinstance(c, (int, float)):
            f = float(c)
            if not math.isfinite(f):
                continue
            if dec == 0:
                if f.is_integer() and int(f) == int(plain):
                    return True
            elif float(f"{f:.{dec}f}") == float(plain):     # "-0.000" and "0.000" are the same rendering
                return True
        elif isinstance(c, str) and pattern.search(c):
            return True
    return False


def cells_of(md_text: str) -> list[tuple[str, list[str]]]:
    """(cell text, row refs) for every checkable cell: table cells (header and separator rows skipped) and prose lines."""
    text = FENCE_RE.sub("", md_text)
    out: list[tuple[str, list[str]]] = []
    in_table = False
    for raw in text.splitlines():
        line = CODE_RE.sub("", raw).strip()
        if not line or line.startswith("#") or line.startswith("!["):
            in_table = False
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:            # header row
                in_table = True
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            row_refs = [m.group(0) for m in REF_RE.finditer(line)]
            out += [(c, row_refs) for c in cells]
        else:
            in_table = False
            out.append((line, [m.group(0) for m in REF_RE.finditer(line)]))
    return out


def trace_numbers(md_text: str, root: Path) -> tuple[int, list[str]]:
    """Verify that every numeric token of the report is found in the referenced JSON/YAML node (or text file).

    Returns ``(n_tokens_checked, problems)``; ``problems`` is empty when every number traces back. Headings, code,
    image links, table header/separator rows, reference tags themselves and section citations ("ТЗ 1.7", "§6",
    "Этап 4", "раздел 3") are not numbers of the report; ISO timestamps are matched as whole strings. A cell
    without a reference borrows the references of its table row.
    """
    res = Resolver(root)
    problems: list[str] = []
    checked = 0
    for cell, row_refs in cells_of(md_text):
        own = [m.group(0) for m in REF_RE.finditer(cell)]
        refs = own or row_refs
        body = CITATION_RE.sub(" ", REF_RE.sub(" ", cell))
        stamps = ISO_RE.findall(body)
        body = ISO_RE.sub(" ", body)
        tokens = NUM_RE.findall(body) + stamps
        if not tokens:
            continue
        if not refs:
            problems.append(f"number without reference: {cell[:120]!r}")
            continue
        cands: list = []
        for r in refs:
            m = REF_RE.fullmatch(r)
            ok, node, used = res.resolve(m.group("path"), m.group("key"))
            if not ok:
                problems.append(f"unresolved reference {r}")
                continue
            cands += leaves(node) + used
        for tok in tokens:
            checked += 1
            if tok in stamps:
                if not any(isinstance(c, str) and tok in c for c in cands):
                    problems.append(f"timestamp {tok} not in {refs}")
            elif not token_matches(tok, cands):
                problems.append(f"number {tok} not found in {refs} (cell {cell[:80]!r})")
    return checked, problems




# ================================================================================================= main
def render(root: Path = ROOT, smoke: bool = False, figures: bool = True) -> tuple[str, Path]:
    inp = Inputs(root, smoke)
    setup, spath = write_setup(inp)
    figdir = inp.rdir / "figures"
    out_path = (inp.rdir / "REPORT.md") if smoke else (inp.root / "REPORT.md")
    figs: dict[str, tuple[Path | None, str]] = {}
    for name, fn in (("roc", fig_roc), ("auc", fig_auc_by_source), ("learning", fig_learning_curves),
                     ("curveball", fig_curveball), ("notinject", fig_notinject), ("para", fig_para_strata)):
        if not figures:
            figs[name] = (None, "рисунки отключены")
            continue
        try:
            figs[name] = fn(inp, figdir)
        except Exception as exc:  # noqa: BLE001 - a figure must never block the report
            figs[name] = (None, f"рисунок не построен ({type(exc).__name__})")

    def figrel(p: Path | None) -> str:
        return rel(p, out_path.parent) if p else ""

    lines = [f"# FlyGuard — отчёт{' (смоук)' if smoke else ''}", "",
             f"Сгенерировано `scripts/make_report.py` {setup['created_at']} {ref(spath, 'created_at')}; "
             f"коммит отчёта {code(setup.get('git_commit'))} {ref(spath, 'git_commit')} (коммиты, на которых посчитаны результаты, — "
             f"раздел 2); хеш конфига {code(setup['config_hash'])} {ref(spath, 'config_hash')}. "
             "Каждое число записано как `значение [нижняя, верхняя] (файл#ключ)`; ссылка ведёт к узлу JSON/YAML, из которого число "
             "прочитано (ключи результатов содержат `/`, разбор жадный), и число совпадает с ним точно после округления вывода. "
             "Десятичный разделитель — точка. Один сид: интервал — кластерный бутстреп этого сида; несколько сидов: среднее по "
             "сидам, интервал — огибающая бутстреп-интервалов сидов, после ссылки — sd по сидам и число сидов (ASSUMPTIONS A55). "
             f"Производные счётчики (длины списков манифестов, суммы, медианы) лежат в {code(inp.derived.path)}. "
             f"Незавершённые этапы помечены «{MISSING}», части, которые смоук пропускает, — «{MISSING_SMOKE}».", ""]
    lines += sec_verdicts(inp)
    lines += sec_setup(inp, setup, spath)
    lines += sec_data(inp)
    lines += sec_generation(inp)
    lines += sec_power(inp)
    lines += sec_results(inp, figs, figrel)
    lines += sec_contributions(inp)
    lines += sec_contract(inp)
    lines += sec_threats(inp)
    lines += sec_reproduce(inp, spath, setup)
    text = "\n".join(lines).rstrip() + "\n"
    inp.derived.write({"config_hash": setup["config_hash"], "git_commit": setup.get("git_commit"), "smoke": smoke})
    # (written before the report: every reference of the text must resolve)
    atomic_write_text(out_path, text)
    return text, out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render REPORT.md from results/*.json (ТЗ Этап 6).")
    ap.add_argument("--smoke", action="store_true", help="results/smoke -> results/smoke/REPORT.md")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the traceability pass over the rendered text")
    args = ap.parse_args(argv)
    text, path = render(Path(args.root), args.smoke, figures=not args.no_figures)
    print(f"report: {path} ({len(text.splitlines())} lines)")
    if not args.no_verify:
        n, problems = trace_numbers(text, Path(args.root))
        print(f"traceability: {n} numbers checked, {len(problems)} problems")
        for p in problems[:30]:
            print("  -", p)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
