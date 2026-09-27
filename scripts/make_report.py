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
  rule; ``scripts/check_acceptance.py`` and ``tests/experiments/test_report.py`` run it over the rendered text.
  Text files (``requirements.lock``, ``audit.md``) are referenced as ``(path)`` and checked by substring.
* Per-seed summaries (``results/<E>/summary.json``, :mod:`flyguard.experiments.results`): with one seed a number is
  that seed's ``value [ci_low, ci_high]`` (cluster bootstrap); with several seeds it is the mean over seeds with the
  t-interval of the seed mean (``seed_ci_low/high``), because percentile bounds of different seeds must not be
  averaged. The column headers say which one is shown; the per-seed bootstrap intervals stay in the file.
* An experiment that has not run renders as "не выполнено" cells; the report is complete at every stage of
  ``run_all.sh`` (smoke criterion: every section present).
* Facts of the setup (versions, KC count and in-degree, seeds, comparator) are written first to
  ``results/setup.json`` through ``flyguard.io`` (config_hash + git_commit) so that they are referenced like every
  other number; the snapshot contains no data text.
* Prose carries no bare numerals: constants come with a config reference, and the decimal separator is a dot.
* No text of any dataset example is read or printed: only counts, statistics, ids of experiments and journals.

Figures (matplotlib, Agg) under ``results/figures/``: AUC by source with intervals (ROC curves need per-document
scores, which the results files do not carry; the caption says so), learning curves with bands (E2), the Curveball
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from flyguard.config import ROOT, config_hash, git_commit, load_configs  # noqa: E402
from flyguard.data.build import output_dirs  # noqa: E402
from flyguard.experiments import results as R  # noqa: E402
from flyguard.io import atomic_write_json, atomic_write_text  # noqa: E402

EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
SOURCE_ORDER = ("deep", "bipia", "dojo", "dyn", "para", "para_deep", "para_shallow", "notinject")
DETECTOR_ORDER = ("regex", "tfidf_lr", "knn1", "knn5", "centroid", "lr_svd", "real_fly_bloom", "real_fly_linear",
                  "flyhash_bloom", "flyhash_linear", "protectai_v2", "piguard", "prompt_guard_2")
MISSING = "не выполнено"
NA = "—"
MAX_ROWS = 60          # rows of a results table shown inline (the file keeps the rest)
NUMBER_ROWS = 400      # rows of the "remaining numbers" table of an experiment
PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "torch", "transformers", "xxhash", "datasketch",
            "langdetect", "matplotlib", "agentdojo")
HYPOTHESIS_TITLES = {"H1a": "H1a, бенчмарки", "H1b": "H1b, алгоритм мухи", "H2": "H2, слова-триггеры",
                     "H3": "H3, проводка"}


# ================================================================================================= references
def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ref(path: str, key: str | None = None) -> str:
    return f"({path}#{key})" if key else f"({path})"


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


def rec_triple(rec: dict[str, Any] | None) -> tuple[Any, Any, Any]:
    """(value, low, high) of a summary record: one seed -> its bootstrap CI, several seeds -> mean and seed t-CI."""
    if not rec:
        return None, None, None
    per_seed = rec.get("per_seed") or {}
    if rec.get("n_seeds", len(per_seed)) <= 1 and per_seed:
        one = next(iter(per_seed.values()))
        return one.get("value"), one.get("ci_low"), one.get("ci_high")
    if "mean" in rec:
        return rec.get("mean"), rec.get("seed_ci_low"), rec.get("seed_ci_high")
    return rec.get("value"), rec.get("ci_low"), rec.get("ci_high")


def rec_num(summary: dict[str, Any] | None, path: str, key: str, nd: int = 3) -> str:
    rec = ((summary or {}).get("numbers") or {}).get(key)
    if rec is None:
        return NA
    v, lo, hi = rec_triple(rec)
    return num(v, path, f"numbers/{key}", lo, hi, nd)


def rec_p(rec: dict[str, Any] | None, path: str, key: str) -> str:
    """The bootstrap p of a paired difference: one seed -> ``p``, several -> its range over seeds."""
    ps = [(s, r.get("p")) for s, r in ((rec or {}).get("per_seed") or {}).items() if r.get("p") is not None]
    if not ps:
        return NA
    if len(ps) == 1:
        return num(ps[0][1], path, f"numbers/{key}/per_seed/{ps[0][0]}/p")
    vals = [p for _, p in ps]
    lo, hi = min(ps, key=lambda x: x[1]), max(ps, key=lambda x: x[1])
    return (f"{fmt(min(vals))}–{fmt(max(vals))} {ref(path, f'numbers/{key}/per_seed/{lo[0]}/p')}"
            f" {ref(path, f'numbers/{key}/per_seed/{hi[0]}/p')}")


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
        self.verdicts = self.json(self.rdir / "verdicts.json")
        self.contract = self.json(self.rdir / "contract.json")
        self.spend = self.json(self.root / "results" / "spend.json")
        self.pilot = self.json(self.root / "results" / "pilot.json")
        self.paraphrases = self.json(self.root / "results" / "paraphrases.json")
        self.splits = self.json(self.manifests / "splits.json")
        self.pools = self.json(self.manifests / "pools.json")
        self.dedup = self.json(self.manifests / "dedup.json")
        self.sources = self.json(self.root / "data" / "manifests" / "sources.json")
        self.traces_manifest = self.json(self.root / "results" / "shared" / "traces_manifest.json")
        self.split_manifest = self.json(self.root / "results" / "shared" / "split_manifest.json")
        self.extraction = self.json(self.root / "data" / "manifests" / "traces_extraction.json")
        self.contamination = self.json(self.root / "data" / "manifests" / "contamination.json")
        self.para_manifest = self.json(self.root / "data" / "paraphrases" / "paraphrases_manifest.json")

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
        "compute": dict(inp.cfg.operator.get("compute") or {}),
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
def scalar_rows(obj: Any, path: str, key: str, rows: list, depth: int = 0, max_depth: int = 4) -> None:
    """Flatten scalar leaves of ``obj`` into ``(key, value (ref))`` rows (lists of scalars stay on one row)."""
    sub = (lambda k: f"{key}/{k}") if key else (lambda k: str(k))
    if isinstance(obj, dict):
        for k in sorted(obj):
            if depth < max_depth:
                scalar_rows(obj[k], path, sub(k), rows, depth + 1, max_depth)
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            rows.append((key, ", ".join(fmt(x) for x in obj[:20]) + (" …" if len(obj) > 20 else "") + " " + ref(path, key)))
        else:
            for i, x in enumerate(obj[:MAX_ROWS]):
                if depth < max_depth:
                    scalar_rows(x, path, sub(i), rows, depth + 1, max_depth)
    else:
        rows.append((key, f"{fmt(obj)} {ref(path, key)}"))


def dump(obj: Any, path: str, key: str, title: str | None = None) -> list[str]:
    rows: list = []
    scalar_rows(obj, path, key, rows)
    note: list[str] = []
    out = [f"**{title}**", ""] if title else []
    return out + table(["ключ", "значение"], truncated(rows, note)) + note


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


def pivot_detectors(numbers: dict[str, Any], prefixes: Sequence[str], path: str) -> list[str]:
    """``<prefix>/<detector>[/<param>]`` -> detector rows with one column per prefix (and parameter)."""
    cols: dict[str, dict[str, str]] = {}
    for key in numbers:
        parts = split_key(key)
        if len(parts) >= 2 and parts[0] in prefixes:
            col = parts[0] + ("/" + "/".join(parts[2:]) if len(parts) > 2 else "")
            cols.setdefault(col, {})[parts[1]] = key
    if not cols:
        return [f"_{MISSING}_", ""]
    names = sorted({d for m in cols.values() for d in m}, key=order_key(DETECTOR_ORDER))
    colnames = sorted(cols, key=order_key(list(prefixes)))
    rows = [[d] + [rec_num({"numbers": numbers}, path, cols[c][d], nd=(1 if c.startswith("latency") else 3))
                   if d in cols[c] else NA for c in colnames] for d in names]
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
        out += [f"Порог `{tau}`:", ""] + table(["детектор"] + subsets, rows)
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
        rows.append([metric, pair.replace("-", " − ", 1),
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
    rows = [[k, rec_num({"numbers": numbers}, path, k),
             fmt(numbers[k].get("n")) + (" " + ref(path, f"numbers/{k}/n") if numbers[k].get("n") is not None else ""),
             str(numbers[k].get("note") or "") + (" " + ref(path, f"numbers/{k}/note") if numbers[k].get("note") else "")]
            for k in truncated(keys, note, NUMBER_ROWS)]
    return table(["ключ", "значение [интервал]", "n", "примечание"], rows) + note


def experiment_block(inp: Inputs, exp: str, title: str) -> list[str]:
    """Section-6 block of one experiment: known pivots, remaining numbers, tables, thresholds, notes."""
    s = inp.summaries.get(exp)
    path = inp.spath(exp)
    out = [f"### {exp}. {title}", ""]
    if not s:
        return out + [f"_{MISSING}: файла {path} нет._", ""]
    numbers = s.get("numbers") or {}
    seeds = s.get("seeds") or []
    out += [f"Сиды: {', '.join(str(x) for x in seeds)} {ref(path, 'seeds')}; интервалы: "
            + ("кластерный бутстреп одного сида" if len(seeds) <= 1 else "t-интервал среднего по сидам")
            + f" (см. правила в шапке); хеш конфига {s.get('config_hash')} {ref(path, 'config_hash')}.", ""]
    if s.get("warnings"):
        out += ["Предупреждения сводки: " + "; ".join(str(w) for w in s["warnings"]) + " " + ref(path, "warnings"), ""]
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
        out += ["**Гиперпараметры, задержка на документ (мс) и размер состояния (байт)**", ""]
        out += pivot_detectors(numbers, ("hyper", "latency_ms", "state_bytes"), path)
    rest = generic_numbers(numbers, path)
    if rest:
        out += ["**Остальные числа эксперимента**", ""] + rest
    for name in sorted(s.get("tables") or {}):
        out += [f"**Таблица `{name}`**", ""] + table_rows(s, name, path)
    thresholds = s.get("thresholds") or {}
    if thresholds:
        first = sorted(thresholds, key=lambda x: int(x))[0]
        rows = [[name, num(rec.get("value"), path, f"thresholds/{first}/{name}/value"), str(rec.get("source")),
                 str(rec.get("target")), num(rec.get("n"), path, f"thresholds/{first}/{name}/n")]
                for name, rec in sorted((thresholds.get(first) or {}).items()) if isinstance(rec, dict)]
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


def fig_auc_by_source(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    numbers = inp.numbers("E1")
    cells: dict[str, dict[str, tuple]] = {}
    for key, rec in numbers.items():
        parts = split_key(key)
        if parts[0] == "auc" and len(parts) == 3:
            cells.setdefault(parts[2], {})[parts[1]] = rec_triple(rec)
    if not cells:
        return None, "ROC/AUC по источникам: " + MISSING
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
    return path, ("AUC по источникам с интервалами (ROC-кривые требуют оценок документов, которых в файлах "
                  "результатов нет; столбики заменяют их)")


def shots_of(key: str) -> tuple[str | None, str | None, str | None]:
    """(detector, shots, metric) from an E2 key that carries a ``shots``-like segment; None when it does not."""
    parts = split_key(key)
    for i, p in enumerate(parts):
        m = re.fullmatch(r"(?:shots[=:_-]?)?(\d+|full)(?:[_-]?shots?)?", p)
        if m and (("shot" in p) or any("shot" in q or q in ("learning", "curve") for q in parts[:i])):
            shots = m.group(1)
            rest = [q for j, q in enumerate(parts) if j != i and q not in ("shots", "learning", "curve")]
            det = next((q for q in rest if q in DETECTOR_ORDER), None)
            metric = "/".join(q for q in rest if q != det) or "macro_auc"
            return det, shots, metric
    return None, None, None


def fig_learning_curves(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    numbers = inp.numbers("E2")
    series: dict[tuple[str, str], dict[str, tuple]] = {}
    for key, rec in numbers.items():
        det, shots, metric = shots_of(key)
        if det and shots and "macro" in (metric or ""):
            series.setdefault((metric, det), {})[shots] = rec_triple(rec)
    if not series:
        return None, "Кривые обучения: " + MISSING
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (metric, det), pts in sorted(series.items()):
        order = sorted(pts, key=lambda s: (s == "full", int(s) if s.isdigit() else 0))
        x = list(range(len(order))); y = [pts[s][0] for s in order]
        lo = [pts[s][1] if pts[s][1] is not None else pts[s][0] for s in order]
        hi = [pts[s][2] if pts[s][2] is not None else pts[s][0] for s in order]
        ax.plot(x, y, marker="o", label=det); ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_xlabel("примеров на класс"); ax.set_ylabel("macroAUC"); ax.set_title("E2: кривые обучения с полосами"); ax.legend(fontsize=7)
    path = save(fig, figdir, "e2_learning_curves.png"); plt.close(fig)
    return path, "кривые обучения с полосами интервалов"


def curveball_values(inp: Inputs) -> tuple[list[float], float | None, str]:
    """Per-null macroAUC values of E4 from a table with a curveball column, else from ``curveball``-keyed numbers."""
    s = inp.summaries.get("E4") or {}
    for name, rows in (s.get("tables") or {}).items():
        if "curveball" in name or any("curveball" in str(r.get("matrix", r.get("null", ""))) for r in rows[:5]):
            col = next((c for c in ("macro_auc", "value", "auc") if rows and c in rows[0]), None)
            if col:
                vals = [float(r[col]) for r in rows if r.get(col) is not None
                        and ("curveball" in str(r.get("matrix", r.get("null", "curveball"))))]
                if vals:
                    return vals, None, f"tables/{name}"
    vals = []
    for key, rec in (s.get("numbers") or {}).items():
        parts = split_key(key)
        if any(p.startswith("curveball") for p in parts) and any(p.isdigit() for p in parts) and "macro" in key:
            v = rec_triple(rec)[0]
            if v is not None:
                vals.append(float(v))
    return vals, None, "numbers"


def fig_curveball(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    vals, _, _ = curveball_values(inp)
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
    return path, "гистограмма macroAUC по curveball-перемешиваниям с отметкой измеренной M"


def fig_notinject(inp: Inputs, figdir: Path) -> tuple[Path | None, str]:
    numbers = inp.numbers("E1")
    cells: dict[str, dict[str, float]] = {}
    for key, rec in numbers.items():
        parts = split_key(key)
        if parts[0] == "fpr_notinject" and parts[1] == "tau90_deep" and len(parts) in (3, 4):
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


def sec_setup(inp: Inputs, setup: dict[str, Any], spath: str) -> list[str]:
    out = ["## 2. Установка", ""]
    pk = setup["packages"]
    out += [f"- Python {setup['python']} {ref(spath, 'python')}; платформа {setup['platform']} {ref(spath, 'platform')}.",
            "- Пакеты (версии из окружения, закреплены в `requirements.lock`): " + ", ".join(f"{p} {v} {ref(spath, f'packages/{p}')}" for p, v in pk.items() if v) + ".",
            f"- Коммит репозитория: {setup.get('git_commit')} {ref(spath, 'git_commit')}; хеш конфига "
            f"{setup['config_hash']} {ref(spath, 'config_hash')}.",
            f"- Сиды: глобальные {', '.join(str(s) for s in setup['seeds']['global'])} {ref(spath, 'seeds/global')}; "
            f"дети {', '.join(setup['seeds']['children'])} {ref(spath, 'seeds/children')}."]
    pins = setup.get("pins") or {}
    for name, pin in sorted(pins.items()):
        out.append(f"- Пин `{name}`: " + ", ".join(f"{k} {v} {ref(spath, f'pins/{name}/{k}')}" for k, v in sorted(pin.items())) + ".")
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
    out.append(f"- Компаратор H1a/H2: `{setup['comparator']}` {ref(spath, 'comparator')}; промышленные детекторы: "
               + ", ".join(f"{g} (в наличии: {v['present']}) {ref(spath, f'guards/{g}/present')}" for g, v in setup["guards"].items()) + ".")
    if inp.pilot:
        ppath = inp.p(inp.root / "results" / "pilot.json")
        out.append(f"- Модель агента: `{inp.pilot.get('chosen_model')}` {ref(ppath, 'chosen_model')}, выбрана по правилу пилота: "
                   f"{inp.pilot.get('chosen_by_rule')} {ref(ppath, 'chosen_by_rule')} (иначе см. DEVIATIONS D5); пилоты кандидатов:")
        for i, c in enumerate(inp.pilot.get("candidates") or []):
            out.append(f"  - `{c.get('model')}`: ASR {num(c.get('targeted_asr'), ppath, f'candidates/{i}/targeted_asr')}, "
                       f"utility {num(c.get('utility_clean'), ppath, f'candidates/{i}/utility_clean')}, стоимость пилота USD "
                       f"{num(c.get('pilot_cost_usd'), ppath, f'candidates/{i}/pilot_cost_usd')}, принят: {c.get('accepted')} "
                       f"{ref(ppath, f'candidates/{i}/accepted')}.")
    else:
        out.append(f"- Модель агента и пилот: {MISSING} (results/pilot.json нет).")
    pm = setup.get("paraphrase_models") or {}
    if pm:
        gens = ", ".join(f"{g.get('model')} ({g.get('provider')})" for g in pm.get("generators", []))
        judges = ", ".join(f"{j.get('model')} ({j.get('provider')})" for j in pm.get("judges", []))
        out.append(f"- Генераторы парафраз: {gens} {ref(spath, 'paraphrase_models/generators')}; судьи: {judges} "
                   f"{ref(spath, 'paraphrase_models/judges')} (один судья: DEVIATIONS D2).")
    out.append(f"- Вычислитель: " + ", ".join(f"{k} {fmt(v)} {ref(spath, f'compute/{k}')}" for k, v in setup.get("compute", {}).items()) + ".")
    return out + [""]


def sec_data(inp: Inputs) -> list[str]:
    out = ["## 3. Данные", ""]
    mp = inp.p(inp.manifests)
    audit = inp.manifests / "audit.md"
    out.append(f"Аудит (ТЗ 1.2) лежит в `{mp}/audit.md`" + ("" if audit.exists() else f" — {MISSING}") + "; ниже счётчики манифестов.")
    out.append("")
    if inp.splits:
        sp = f"{mp}/splits.json"
        c = inp.splits.get("counts") or {}
        out += ["**Разбиение E1 (документы)**", ""]
        rows = [["train (deepset)", num(c.get("train"), sp, "counts/train")], ["val (все источники)", num(c.get("val"), sp, "counts/val")],
                ["C_unl", num(c.get("c_unl"), sp, "counts/c_unl")]]
        rows += [[f"test / {s}", num(n, sp, f"counts/test/{s}")] for s, n in sorted((c.get("test") or {}).items(), key=lambda kv: order_key(SOURCE_ORDER)(kv[0]))]
        rows += [[f"val / {s}", num(n, sp, f"counts/val_by_source/{s}")] for s, n in sorted((c.get("val_by_source") or {}).items())]
        out += table(["роль", "документов"], rows)
        e3 = inp.splits.get("e3") or {}
        lst = lambda k: (", ".join(str(x) for x in e3[k]) if isinstance(e3.get(k), list) else fmt(e3.get(k))) + " " + ref(sp, f"e3/{k}")  # noqa: E731
        out += [f"Фолды E3: шаблонов в конфиге {lst('templates_configured')}, есть в данных {lst('templates_present')}, "
                f"не хватает {lst('templates_missing')}; пригодных фолдов: "
                + ", ".join(f"{k} {num(v, sp, f'e3/usable/{k}')}" for k, v in sorted((e3.get("usable") or {}).items())) + ".", ""]
    else:
        out += [f"_splits.json: {MISSING}_", ""]
    if inp.pools:
        pp = f"{mp}/pools.json"
        rows = []
        for pool in ("p_val", "p_test", "notinject"):
            d = inp.pools.get(pool) or {}
            rows.append([pool, num(d.get("n"), pp, f"{pool}/n"),
                         ", ".join(f"{s} {num(n, pp, f'{pool}/by_source/{s}')}" for s, n in sorted((d.get("by_source") or {}).items())) or NA,
                         f"{d.get('meets_target')} {ref(pp, f'{pool}/meets_target')}" if "meets_target" in d else NA])
        out += ["**Пулы негативов (ТЗ 1.8)**", ""] + table(["пул", "документов", "по источникам", "достигнута цель"], rows)
    else:
        out += [f"_pools.json: {MISSING}_", ""]
    if inp.dedup:
        dp = f"{mp}/dedup.json"
        d = inp.dedup
        out += ["**Дедупликация (ТЗ 1.7)**", ""]
        out += table(["величина", "значение"], [
            ["окон всего", num(d.get("windows_total"), dp, "windows_total")],
            ["окон в тесте", num(d.get("windows_test"), dp, "windows_test")],
            ["эталонных окон (train+val)", num(d.get("windows_reference"), dp, "windows_reference")],
            ["исключено тестовых окон", num(d.get("test_windows_excluded"), dp, "test_windows_excluded")],
            ["документов выпало", num(d.get("documents_dropped_total"), dp, "documents_dropped_total")],
            ["правило", f"{(d.get('rule') or {}).get('scope')} {ref(dp, 'rule/scope')}; Жаккар ≥ {num((d.get('rule') or {}).get('jaccard'), dp, 'rule/jaccard')}"],
        ])
        by_src = d.get("test_windows_excluded_by_source") or {}
        if by_src:
            out += ["Исключено по источникам: " + ", ".join(f"{s} {num(n, dp, f'test_windows_excluded_by_source/{s}')}" for s, n in sorted(by_src.items())) + ".", ""]
    else:
        out += [f"_dedup.json: {MISSING}_", ""]
    if inp.contamination:
        cp = inp.p(inp.root / "data/manifests/contamination.json")
        out += dump({k: inp.contamination[k] for k in ("document_level", "window_level") if k in inp.contamination}, cp, "",
                    "Пересечение с открытым обучающим набором PIGuard (ТЗ 3.2)")
    return out


def sec_generation(inp: Inputs) -> list[str]:
    out = ["## 4. Генерация трасс и парафраз", ""]
    tm = inp.traces_manifest
    tp = inp.p(inp.root / "results/shared/traces_manifest.json")
    if tm:
        out += [f"Трассы: модель агента `{tm.get('agent_model')}` {ref(tp, 'agent_model')}, провайдер {tm.get('provider')} "
                f"{ref(tp, 'provider')}, temperature {num(tm.get('temperature'), tp, 'temperature')}, режим размышлений "
                f"{tm.get('thinking')} {ref(tp, 'thinking')}, заморожено {tm.get('generated')} {ref(tp, 'generated')}; "
                f"логов в манифесте: см. `files` {ref(tp, 'files')}.", ""]
        rows = []
        for bench, suites in sorted((tm.get("counts") or {}).items()):
            for suite, attacks in sorted(suites.items()):
                for attack, classes in sorted(attacks.items()):
                    rows.append([bench, suite, attack] + [num(classes.get(c), tp, f"counts/{bench}/{suite}/{attack}/{c}")
                                                          if c in classes else NA for c in ("benign", "hijacked", "injection_ignored", "error")])
        out += ["**Эпизоды по классам (контракт §2)**", ""] + table(["бенчмарк", "сьют", "шаблон", "benign", "hijacked", "injection_ignored", "error"], rows)
        out += ["Сверка с опубликованными ASR и utility: опубликованные значения для этой модели агента в репозитории не "
                "зафиксированы, сверка не выполнена (DEVIATIONS D5 описывает выбор модели по пилоту).", ""]
    else:
        out += [f"_Манифест трасс: {MISSING}_", ""]
    if inp.extraction:
        ep = inp.p(inp.root / "data/manifests/traces_extraction.json")
        rows = []
        for bench, d in sorted(inp.extraction.items()):
            rows.append([bench, num(d.get("n_logs"), ep, f"{bench}/n_logs"), num(d.get("documents"), ep, f"{bench}/documents"),
                         num(d.get("documents_positive"), ep, f"{bench}/documents_positive"),
                         num(d.get("steps_total"), ep, f"{bench}/steps_total"), num(d.get("steps_labelled"), ep, f"{bench}/steps_labelled"),
                         num((d.get("attacked_without_span") or {}).get("count"), ep, f"{bench}/attacked_without_span/count"),
                         num(d.get("errors") if not isinstance(d.get("errors"), dict) else (d.get("errors") or {}).get("count"), ep, f"{bench}/errors")])
        out += ["**Извлечение шагов (ТЗ 1.5)**", ""]
        out += table(["бенчмарк", "логов", "документов", "позитивов", "шагов", "шагов с меткой", "атак без спана", "ошибок"], rows)
    if inp.split_manifest:
        smp = inp.p(inp.root / "results/shared/split_manifest.json")
        c = inp.split_manifest.get("counts") or {}
        out += ["**Разбиение контракта (§6)**", ""]
        out += table(["список", "эпизодов"], [[k, num(c.get(k), smp, f"counts/{k}")]
                                              for k in ("test", "observation", "validation_clean", "validation_attacks", "train_attacks", "excluded", "error") if k in c])
    pr = inp.paraphrases
    prp = inp.p(inp.root / "results/paraphrases.json")
    if pr:
        out += ["**Парафразы (ТЗ 1.6): объёмы, принятие и отказы**", ""]
        out += dump(pr.get("counts") or {}, prp, "counts")
        out += dump(pr.get("rates") or {}, prp, "rates")
    else:
        out += [f"_Парафразы: {MISSING} (results/paraphrases.json нет)._", ""]
    sp = inp.spend
    spp = inp.p(inp.root / "results/spend.json")
    if sp:
        out += [f"**Расход API**: {num(sp.get('spent_usd'), spp, 'spent_usd')} USD из бюджета "
                f"{num(sp.get('budget_usd'), spp, 'budget_usd')} USD, вызовов {num(sp.get('n_calls'), spp, 'n_calls')}, "
                f"в бюджете: {sp.get('within_budget')} {ref(spp, 'within_budget')}.", ""]
        streams = {k: v for k, v in (sp.get("breakdown") or {}).items() if k.startswith("stream:") or k.startswith("model:")}
        out += table(["поток / модель", "вызовов", "USD"],
                     [[k, num(v.get("calls"), spp, f"breakdown/{k}/calls", nd=0), num(v.get("cost_usd"), spp, f"breakdown/{k}/cost_usd")]
                      for k, v in sorted(streams.items())])
    else:
        out += [f"_Расход API: {MISSING}_", ""]
    return out


def sec_power(inp: Inputs) -> list[str]:
    out = ["## 5. Мощность (E0)", ""]
    p = inp.power
    pp = inp.p(R.power_path(inp.root, inp.smoke))
    if not p:
        return out + [f"_{MISSING}: {pp} нет._", ""]
    meta = {k: p[k] for k in ("stage", "frozen", "created_at", "config_hash", "delta_rel", "alpha", "tost_level", "power_target", "fpr_target") if k in p}
    out += dump(meta, pp, "", "Параметры и статус")
    sizes = p.get("sizes") or {}
    out += ["**Размеры**", ""] + table(["источник", "позитивов", "негативов", "кластеров"],
                                       [[s, num(d.get("n_pos"), pp, f"sizes/{s}/n_pos"), num(d.get("n_neg"), pp, f"sizes/{s}/n_neg"),
                                         num(d.get("n_clusters"), pp, f"sizes/{s}/n_clusters")] for s, d in sorted(sizes.items(), key=lambda kv: order_key(SOURCE_ORDER + ("macro",))(kv[0]))])
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
        cell = (cells.get(s) or {}).get(f"{lvl:g}") or {}
        rows.append([s, num(lvl, pp, f"planning_level/{s}")]
                    + [num(cell.get(k), pp, f"cells/{s}/{lvl:g}/{k}") if not isinstance(cell.get(k), (dict, list)) else NA
                       for k in ("mdd", "delta", "tost_power", "status")])
    if rows:
        out += ["**MDD разности AUC и мощность TOST на уровне планирования** (прочерк = не определено на этом объёме)", ""]
        out += table(["источник", "уровень AUC", "MDD", "δ", "мощность TOST", "статус"], rows)
    for key, title in (("notinject", "H2: ширина ДИ разности долей на парных наблюдениях NotInject"),
                       ("spread", "Разброс на валидации: curveball и перестановки perm"), ("hypotheses", "Входы вердиктов")):
        if key in p:
            out += dump(p[key], pp, key, title)
    return out


def sec_results(inp: Inputs, figs: dict[str, tuple[Path | None, str]], figrel) -> list[str]:
    out = ["## 6. Результаты E1–E6", ""]
    out += ["Интервалы: кластерный бутстреп по `cluster_id` внутри источника, уровень 1 − α при α = "
            f"{num(inp.cfg.default['stats']['bootstrap']['alpha'], 'configs/default.yaml', 'stats/bootstrap/alpha')}; TOST по интервалу "
            f"уровня {num(inp.cfg.default['stats']['tost']['ci'], 'configs/default.yaml', 'stats/tost/ci')} с коридором δ = "
            f"{num(inp.cfg.default['stats']['tost']['delta_rel'], 'configs/default.yaml', 'stats/tost/delta_rel')} от референса. "
            "macroAUC — среднее AUC по присутствующим источникам с равным весом (основа вердиктов); объединение по числу "
            "документов не используется.", ""]
    for name in ("auc", "notinject", "para"):
        path, caption = figs[name]
        out.append(f"![{caption}]({figrel(path)})" if path else f"_Рисунок ({caption})._")
        out.append("")
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
            out += [f"**Таблица `{name}` (E5)**", ""] + table_rows(s5, name, inp.spath("E5"))
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
            out += ["Валидация CSV по схеме §8: " + ("проходит" if not problems else "не проходит: " + "; ".join(problems[:5])), ""]
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
        out += [f"**Таблица `{name}`**", ""] + table_rows(c, name, cp)
    scalars = {k: v for k, v in c.items() if k in ("training_mode", "n_rows", "csv", "csv_valid", "metrics", "metrics_by_benchmark")}
    if scalars:
        out += dump(scalars, cp, "", "Прочее")
    notes = c.get("notes") or []
    if notes:
        out += ["Примечания:", ""] + [f"- {n} {ref(cp, f'notes/{i}')}" for i, n in enumerate(notes)] + [""]
    out += [f"Хеш конфига {c.get('config_hash')} {ref(cp, 'config_hash')}.", ""]
    return out


def sec_threats(inp: Inputs) -> list[str]:
    out = ["## 9. Угрозы валидности", ""]
    devs = inp.journal_ids("DEVIATIONS.md", "D")
    asms = inp.journal_ids("ASSUMPTIONS.md", "A")
    blks = inp.journal_ids("BLOCKERS.md", "B")
    out += [f"Отклонения: {', '.join(devs) or 'нет'} (DEVIATIONS.md); допущения: {', '.join(asms) or 'нет'} (ASSUMPTIONS.md); "
            f"блокеры: {', '.join(blks) or 'нет'} (BLOCKERS.md).", ""]
    out += ["- **Конструктная.** Метка окна следует спану инъекции, а не поведению агента; классы эпизодов зависят от модели агента "
            "(контракт §10). Парафразы описывают целевое действие атаки, судья один (D2), поэтому отбор позитивов может быть мягче.",
            "- **Внутренняя.** Метки только из deepset train, гиперпараметры и пороги на валидации; тест открыт один раз через "
            "журналируемый доступ (logs/data_access.log). Регулярки выведены из deepset train до первого чтения теста (D10 описывает "
            "порядок по журналу). Дедупликация исключает только тестовые окна.",
            "- **Внешняя.** Окружения AgentDojo статичны, негативы тестовых задач совпадают с обучающими (контракт §10); AgentDyn "
            "даёт честный FPR. Немецкая половина deepset и мультиязычные подмножества NotInject ограничивают перенос на другие языки.",
            "- **Статистическая.** Кластерный бутстреп внутри источника; таблица мощности E0 определяет, какие источники несут разности; "
            "малые пулы негативов сдвигают TPR@FPR к режиму «только AUC». Числа сидов усредняются t-интервалом, бутстреп-интервалы "
            "сидов не смешиваются.",
            "- **Стиль генераторов и отказы провайдеров.** Один провайдер (D1) и одно семейство моделей для генерации и судейства: "
            "стиль парафраз однороден; отказы генератора и судьи смещают состав набора — доли отказов по стратам, шаблонам и "
            "генераторам приведены в разделе 4 (файл `results/paraphrases.json`)."]
    pr = inp.paraphrases or {}
    prp = inp.p(inp.root / "results/paraphrases.json")
    rates = (pr.get("rates") or {}).get("generation_refusal") or {}
    if rates:
        out += ["", "Доли отказов генератора по видам баз: " + ", ".join(
            f"{k} {num(v.get('rate'), prp, f'rates/generation_refusal/by_kind/{k}/rate')}" for k, v in sorted((rates.get("by_kind") or {}).items())) + "."]
    return out + [""]


def sec_reproduce(inp: Inputs, spath: str, setup: dict[str, Any]) -> list[str]:
    out = ["## 10. Воспроизведение", ""]
    out += ["```", "scripts/setup_env.sh", "scripts/fetch.sh", "scripts/gen_traces.sh pilot && scripts/gen_traces.sh run && scripts/gen_traces.sh freeze",
            "scripts/gen_paraphrases.sh all", "scripts/smoke.sh", "scripts/run_all.sh", "scripts/check_acceptance.py", "```", ""]
    out += [f"Коммит {setup.get('git_commit')} {ref(spath, 'git_commit')}, хеш конфига {setup['config_hash']} {ref(spath, 'config_hash')}; "
            f"сиды {', '.join(str(s) for s in setup['seeds']['global'])} {ref(spath, 'seeds/global')} "
            f"(смоук: первые {num(setup['seeds']['smoke_seeds'], spath, 'seeds/smoke_seeds')}). "
            "`run_all.sh` пропускает эксперимент, чей results/<E>/<seed>.json существует с текущим хешем конфига, "
            "и продолжает прерванный прогон; журнал стадий — logs/run_all.log.", ""]
    rows = []
    for exp in EXPERIMENTS:
        s = inp.summaries.get(exp)
        if not s:
            rows.append([exp, MISSING, NA, NA])
            continue
        for seed, t in sorted((s.get("timing") or {}).items(), key=lambda kv: int(kv[0])):
            rows.append([exp, seed, num(t.get("seconds"), inp.spath(exp), f"timing/{seed}/seconds", nd=1),
                         ", ".join(str(x) for x in (t.get("test_reads") or [])) or NA])
    out += ["**Время экспериментов по сидам и открытые тестовые источники**", ""] + table(["эксперимент", "сид", "секунд", "тестовые чтения"], rows)
    return out


# ================================================================================================= verifier
REF_RE = re.compile(r"\((?P<path>[\w./+-]+\.(?:json|ya?ml|md|txt|lock|log|csv))(?:#(?P<key>[^\s()]+))?\)")
NUM_RE = re.compile(r"(?<![\w/.:#=_+-])[-+]?\d+(?:\.\d+)?(?![\w/])")
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
    val = float(tok)
    dec = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    plain = tok.lstrip("+")
    for c in candidates:
        if isinstance(c, bool) or c is None:
            continue
        if isinstance(c, (int, float)):
            if math.isfinite(c) and abs(float(c) - val) <= tol:
                return True
        elif isinstance(c, str) and plain in c:
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
    for name, fn in (("auc", fig_auc_by_source), ("learning", fig_learning_curves), ("curveball", fig_curveball),
                     ("notinject", fig_notinject), ("para", fig_para_strata)):
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
             f"коммит {setup.get('git_commit')} {ref(spath, 'git_commit')}; хеш конфига {setup['config_hash']} {ref(spath, 'config_hash')}. "
             "Каждое число записано как `значение [нижняя, верхняя] (файл#ключ)`; ссылка ведёт к узлу JSON/YAML, из которого число "
             "прочитано (ключи результатов содержат `/`, разбор жадный). Десятичный разделитель — точка. Один сид: интервал — "
             "кластерный бутстреп; несколько сидов: среднее по сидам и t-интервал среднего (бутстреп-интервалы сидов в файлах). "
             f"Незавершённые этапы помечены «{MISSING}».", ""]
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
