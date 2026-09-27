#!/usr/bin/env python
"""scripts/check_acceptance.py — machine checks of ТЗ «Критерии приёмки» (docs/design_experiments.md §4).

Usage::

    .venv/bin/python scripts/check_acceptance.py                 # real results; runs pytest; exit 1 on any ❌
    .venv/bin/python scripts/check_acceptance.py --smoke         # results/smoke, data/manifests/smoke, results/smoke/REPORT.md
    .venv/bin/python scripts/check_acceptance.py --pytest log    # take the last pytest verdict from logs/pytest.log
    .venv/bin/python scripts/check_acceptance.py --pytest skip   # do not run pytest (reported as ⚠)
    .venv/bin/python scripts/check_acceptance.py --only c_network,c_regex

Prints one line per check, ``✅`` / ``❌``, and ``⚠`` for the parts of a criterion that cannot be decided on this
machine, with the reason (the second team's copy of the traces and manifests, ``run_all.sh`` on a clean machine) or
whose evidence the results do not record yet (the fitted Bloom balance); exits non-zero when any ❌ is present.
Every check compares values, never the mere presence of a key or substring, and every test file it opens (windows,
documents, paraphrase texts, judge verdicts) is journaled through ``log_data_access(split="test")``; texts are read
by code only and never printed (only ids and counts are).

The ТЗ list item by item (criterion in quotes -> what is recomputed):

* "fetch.sh скачал всё с проверкой sha256, включая flypath build" — every file of ``data/manifests/sources.json`` is
  re-hashed (missing only the raw MaleCNS files deleted after ``flypath build``, ASSUMPTIONS A10); ``malecns_R.npz``
  and ``.json`` come from ``flypath build``; the four pins (agentdojo, agentdyn, FlyHash-Connectome, MaleCNS).
* "network.log содержит только скачивания и два разрешённых потока" — every line of ``logs/network.log`` is a
  GET/HEAD/git download from a host of ``sources.json`` or the toolchain, or a POST to ``/chat/completions`` of a
  provider of ``configs/operator.yaml`` whose purpose is flow 1 (agent model calls of the trace harness) or flow 2
  (paraphrase generator/judge).
* "Модель агента выбрана пилотом по правилу 1.5; пилоты всех кандидатов записаны" — the Appendix B rule recomputed
  over ``llm_api.agent_models`` from ``results/pilot.json`` (40 + 10 cases each); the chosen model, the
  ``chosen_by_rule`` flag and the traces' agent model must agree; an off-rule choice needs its DEVIATIONS entry.
* "Трассы заморожены ... копия у второй команды" — every log of ``traces_manifest.json`` re-hashed in the trace
  directory *and* in the copy ``shared.traces_out_dir`` (no extra file in the copy); the hand-over is ⚠.
* "Парафразы заморожены" — manifest models/dates/prompt hashes (equal to the current ``configs/prompts`` files) and
  acceptance/refusal rates; every CSV row joined to its candidate (``calls.jsonl``) and to the last decisive record of
  every configured judge in ``judgements.jsonl`` (single judge, D2) with an explicit yes (A.6: positives
  ``same_action`` and ``still_instruction``; negatives ``meaning_preserved`` and not ``contains_instruction_to_ai``);
  Jaccard with the base recomputed (≤ ``jaccard_max``, deep ≤ ``deep_max``, equal to the CSV column); no word of the
  final banned list in a deep positive.
* "spend.json ≤ budget; урезания записаны в DEVIATIONS.md" — ``spent_usd`` equals the ledger ``results/spend/*.jsonl``
  and is within ``llm_api.budget_usd``; every priority skipped in ``results/traces_run.json`` is named (benchmark and
  template) in one DEVIATIONS entry.
* "smoke.sh за 15 минут со всеми разделами отчёта" — ``logs/run_all.log`` (ASSUMPTIONS A36): the sum over the stages
  of ТЗ steps 2–13 (``build_stage1`` … ``report``, ``prescore`` included) of each stage's most recent smoke ``done``
  duration, printed, ≤ 900 s, and the measured ``prescore`` started from an empty guard-score cache
  (``scores_cache_rows=0``); the ten report sections present.
* "run_all.sh на чистой машине ... и продолжает прерванный прогон" — ``run_all.sh --dry-run`` (same mode, the
  ``--skip``/``--seeds`` of the last full run) on the tree plans nothing but ``report`` and ``check``; the clean
  machine is ⚠ (only a run there can show it).
* "pytest проходит, включая все тесты 2.6" — the suite (``--pytest run``) or its last log; skips are listed.
* "Манифесты полны; ... тест не пересекается ...; ни один кластер не содержит обе метки; пары BIPIA целы" — the
  manifests exist and agree with the tables (every main-variant test document of ``documents.parquet`` is in
  ``splits.e1.test`` or dropped by dedup and vice versa, train/val lists equal the tables, pools sized as listed,
  every present source has test documents); dedup invariants over ``windows.parquet`` (clusters, exact hashes, dedup
  groups label-pure and pointing into train/val, one clean + one attacked document per BIPIA test context).
* "Средняя степень KC измеренной M в 4–8; число KC записано" — ``flyguard.connectome.indegree_stats`` and setup.json.
* "power.json записан до финального прогона и использован в вердиктах; финальная версия после трасс и парафраз" —
  stage 2, frozen, current hash, ``created_at`` before every E1 seed file; its sources cover every E1 test source;
  ``verdicts.json`` references exactly this file (hash, time, stage) and every carrier flag of H1a/H1b/H3 equals the
  E0 table's cell.
* "Одна схема окон; 512-токенные окна только в E6" — every window of ``windows.parquet`` starts on the stride and is
  at most ``windows.size`` long; the 512-token flag only in ``E6.yaml``; no 512-token key outside E6 results.
* "У каждого порога записаны источник, целевая точка и число примеров" — every threshold record (seed files,
  contract) has value/source/target/n; every E1 file has threshold records.
* "macroAUC основа вердиктов; объединение по числу документов не является основой" — every ``verdicts.json`` input
  key of H1b/H3 is a macroAUC key, no key is pooled, every key resolves in the E1/E2/E4 summaries, and the per-seed
  effects of H1b/H3 equal the summaries' values; statuses and preconditions (H2, H3) recorded.
* "Перестановка π разыграна от сида и входит в двухступенчатый бутстреп H3" — the H3 record and table of every E4
  file carry ``n_perms == len(seeds.global)``, ``n_null`` = the configured curveball nulls and ``n_boot`` = the
  configured draws; the π seeds of ``perm_grid`` are the ``perm`` children of the global seeds.
* "Bloom обучен на сбалансированных классах" — ``readout.bloom.balance`` and the fitted class counts of the Bloom
  detectors in the results (⚠ while the results do not record them).
* "Компаратор ProtectAI v2 назначен в конфиге до теста" — the config value, the git date of its introduction before
  the first journaled test read, and the comparator of the H1a/H2 verdict inputs.
* "Хеш конфига в тестовых результатах совпадает; DEVIATIONS.md перечисляет каждое отклонение" — every results file,
  summary and verdict source against ``config_hash``; DEVIATIONS entries numbered without gaps, each with a date and
  an effect statement, every cited ``D<n>`` exists, every deviation detectable from the artefacts is journaled.
* "Каждое число REPORT.md прослеживается" — ``make_report.trace_numbers``; the ten sections.
* "Контракт: CSV проходит схему; split_manifest по crc32 mod 3 = 2; important_instructions вне обучения; манифесты" —
  ``validate_csv``; every variant has exactly one row per test episode of the split manifest; the split rule
  recomputed; no validation episode of the contract run is a test task or ``important_instructions``; the second
  team's copy is ⚠.
* "Коммит с regex_patterns.txt старше первого чтения тестовых файлов" — git date of the last commit touching the file
  against the first ``test`` line of ``logs/data_access.log``; retroactive entries (D10) are shown and accepted only
  when DEVIATIONS explains them.
* Beyond the list: every experiment has all configured seeds; all results were computed by one code version (A52).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flyguard.config import ROOT, config_hash, load_configs, seeds_for  # noqa: E402
from flyguard.data.build import output_dirs  # noqa: E402
from flyguard.experiments import results as R  # noqa: E402

OK, FAIL, INFO = "✅", "❌", "⚠"
STATUSES = ("подтверждена", "опровергнута", "не хватило данных", "предусловие не выполнено")
TOOLCHAIN_HOSTS = {"pypi.org", "files.pythonhosted.org", "download.pytorch.org", "github.com", "raw.githubusercontent.com",
                   "huggingface.co", "storage.googleapis.com"}   # setup_env.sh / fetch.sh downloads (MaleCNS via flypath build)
DOWNLOAD_METHODS = {"GET", "HEAD", "GIT"}
FLOW_RES = {"traces": re.compile(r"agent model call|availability check|stream 1 of 2", re.I),
            "paraphrases": re.compile(r"paraphrase|stream 2 of 2", re.I)}
"""The two permitted outgoing LLM flows (CLAUDE.md): the purposes written by ``gen/harness_run.py`` /
``gen/traces.py`` (flow 1) and ``gen/paraphrases.py`` (flow 2)."""
A10_MISSING_OK = ("data/ext/FlyHash-Connectome/data/raw/",)
REQUIRED_PINS = ("agentdojo", "agentdyn", "flyhash_connectome", "malecns")
FLYPATH_OUTPUTS = ("data/processed/connectome/malecns_R.npz", "data/processed/connectome/malecns_R.json")
SMOKE_LIMIT_S = 15 * 60
RETRO_RE = re.compile(r"retroactiv|after the fact|задним числом", re.I)
TOK512_RE = re.compile(r"512[_-]?tok|tok(?:ens?)?[_-]?512|win(?:dows?)?[_-]?512", re.I)
POOLED_RE = re.compile(r"pooled|micro_auc|all_docs", re.I)
H3_PRIMARY = "diff90/macro_auc/real_fly_bloom-curveball_mean"
ALWAYS_RUN = ("report", "check")
"""Stages without a done predicate (a report render and this check): a dry run over a finished tree still lists them."""


@dataclass
class Check:
    name: str
    ok: bool | None          # None = ⚠ (cannot be decided here; the detail says why)
    detail: str = ""

    @property
    def mark(self) -> str:
        return INFO if self.ok is None else (OK if self.ok else FAIL)


def _json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
    except OSError:
        pass
    return out


def _utc(ts: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return None
    return float(x)


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    fa, fb = _num(a), _num(b)
    return fa is not None and fb is not None and abs(fa - fb) <= tol * (1.0 + abs(fb))


def _few(items: Sequence[Any], n: int = 4) -> str:
    items = list(items)
    return ", ".join(str(x) for x in items[:n]) + (f" и ещё {len(items) - n}" if len(items) > n else "")


def sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ================================================================================================= pure helpers
def hosts_outside(lines: Iterable[str], allowed: set[str]) -> dict[str, int]:
    """Hosts of ``network.log`` lines (``ts\\thost\\tmethod\\turl\\tpurpose``) that are not allowed, with counts."""
    bad: dict[str, int] = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[1]:
            continue
        host = parts[1].lower()
        if host not in allowed:
            bad[host] = bad.get(host, 0) + 1
    return bad


def network_audit(lines: Iterable[str], download_hosts: set[str], provider_hosts: set[str]) -> dict[str, Any]:
    """Classify every ``network.log`` line: a download (GET/HEAD/git to a download host), flow 1 or flow 2 (POST to
    ``/chat/completions`` of a configured provider with the purpose of that flow), or a violation (anything else,
    malformed lines included)."""
    bad: Counter = Counter()
    flows = {k: 0 for k in FLOW_RES}
    downloads = 0
    for line in lines:
        if not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or not p[1]:
            bad["строка без хоста/метода"] += 1
            continue
        host, method = p[1].lower(), p[2].upper()
        url, purpose = (p[3] if len(p) > 3 else ""), (p[4] if len(p) > 4 else "")
        if host in provider_hosts:
            flow = next((k for k, rx in FLOW_RES.items() if rx.search(purpose)), None)
            if flow is None:
                bad[f"{host} {method}: цель не относится ни к одному из двух потоков"] += 1
            elif method != "POST" or not urlsplit(url).path.endswith("/chat/completions"):
                bad[f"{host} {method} {urlsplit(url).path}: не POST chat/completions"] += 1
            else:
                flows[flow] += 1
        elif host in download_hosts:
            if method in DOWNLOAD_METHODS:
                downloads += 1
            else:
                bad[f"{host} {method}: не скачивание"] += 1
        else:
            bad[f"{host}: посторонний хост"] += 1
    return {"bad": dict(bad), "flows": flows, "downloads": downloads}


def rehash_sources(files: Sequence[dict[str, Any]], root: Path, missing_ok: tuple[str, ...] = A10_MISSING_OK) -> dict[str, list[str]]:
    """Re-hash every file listed in ``sources.json`` (streamed: the guard weights are ~2.7 GB)."""
    out: dict[str, list[str]] = {"ok": [], "no_sha": [], "mismatch": [], "missing": [], "missing_a10": []}
    for f in files:
        rel, want = str(f.get("path", "")), str(f.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", want):
            out["no_sha"].append(rel)
            continue
        p = Path(rel) if Path(rel).is_absolute() else root / rel
        if not p.is_file():
            out["missing_a10" if rel.startswith(missing_ok) else "missing"].append(rel)
            continue
        out["ok" if sha256_file(p) == want else "mismatch"].append(rel)
    return out


def pilot_rule_choice(candidates: Sequence[dict[str, Any]], order: Sequence[str], asr_range: Sequence[float],
                      min_utility: float, target: float = 0.40) -> tuple[str | None, bool]:
    """ТЗ Appendix B: the first candidate (in ``llm_api.agent_models`` order) with targeted ASR in ``asr_range`` and
    clean utility ≥ ``min_utility``; none accepted -> the candidate whose ASR is closest to 0.40 (order breaks ties).
    Returns ``(model, chosen_by_rule)``."""
    by = {c.get("model"): c for c in candidates}
    present = [m for m in order if m in by and _num(by[m].get("targeted_asr")) is not None
               and _num(by[m].get("utility_clean")) is not None]
    lo, hi = float(asr_range[0]), float(asr_range[1])
    for m in present:
        if lo <= float(by[m]["targeted_asr"]) <= hi and float(by[m]["utility_clean"]) >= float(min_utility):
            return m, True
    if not present:
        return None, False
    return min(present, key=lambda m: (abs(float(by[m]["targeted_asr"]) - target), list(order).index(m))), False


def traces_rehash(entries: Sequence[dict[str, Any]], logdir: Path, copydir: Path) -> dict[str, list[str]]:
    """Re-hash every manifest log in the trace directory and in the second team's copy (``<dir>/<benchmark>/<path>``);
    files of the copy that the manifest does not list are reported as extra."""
    out: dict[str, list[str]] = {"src_missing": [], "src_changed": [], "copy_missing": [], "copy_changed": [], "copy_extra": []}
    listed = set()
    for e in entries:
        rel = f"{e.get('benchmark')}/{e.get('path')}"
        listed.add(rel)
        for kind, base in (("src", logdir), ("copy", copydir)):
            p = base / rel
            if not p.is_file():
                out[f"{kind}_missing"].append(rel)
            elif sha256_file(p) != e.get("sha256"):
                out[f"{kind}_changed"].append(rel)
    if copydir.is_dir():
        out["copy_extra"] = sorted(r for r in (str(p.relative_to(copydir)) for p in copydir.rglob("*.json")) if r not in listed)
    return out


def judge_yes(rec: dict[str, Any] | None, label: str) -> bool:
    """An explicit yes of one judge (ТЗ A.6), never a proxy such as a non-empty confidence."""
    if not rec or rec.get("status") != "ok" or not isinstance(rec.get("verdict"), dict):
        return False
    v = rec["verdict"]
    if str(label) == "1":
        return v.get("same_action") is True and v.get("still_instruction") is True
    return v.get("meaning_preserved") is True and v.get("contains_instruction_to_ai") is False


def last_decisive(judgements: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """``cand_id -> judge -> last ok/refusal record`` (the rule of ``gen.paraphrases.load_judgements``)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in judgements:
        if rec.get("status") in ("ok", "refusal") and rec.get("cand_id") and rec.get("judge"):
            out.setdefault(str(rec["cand_id"]), {})[str(rec["judge"])] = rec
    return out


def candidate_index(calls: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[str]]:
    """``(base_id, generator, text) -> [cand_id]`` over the ok generator calls (``gen.paraphrases.candidates_of_call``)."""
    idx: dict[tuple[str, str, str], list[str]] = {}
    for rec in calls:
        if rec.get("status") != "ok":
            continue
        for it in rec.get("candidates") or []:
            cid = f"{rec['base_id']}|{rec['generator']}|{int(rec['call_index'])}|{int(it['k'])}"
            idx.setdefault((str(rec["base_id"]), str(rec["generator"]), str(it.get("text", ""))), []).append(cid)
    return idx


def paraphrase_audit(rows: Sequence[dict[str, str]], base_text: dict[str, str], cands: dict[tuple[str, str, str], list[str]],
                     judged: dict[str, dict[str, dict[str, Any]]], judges: Sequence[str], filters: dict[str, Any],
                     jaccard: Callable[[str, str], float], banned: Callable[[str], list[str]]) -> dict[str, list[str]]:
    """Per CSV row (ids only in the output): its base exists; Jaccard with the base recomputed ≤ ``jaccard_max``
    and equal to the CSV column (±1e-3); deep rows ≤ ``deep_max``; deep positives without a banned word; the row is a
    generated candidate and every configured judge said an explicit yes to it."""
    out: dict[str, list[str]] = {k: [] for k in ("bad_label", "no_base", "jaccard_high", "jaccard_column", "deep_jaccard",
                                                   "deep_banned", "no_candidate", "judge_not_yes")}
    jmax, dmax = float(filters["jaccard_max"]), float(filters["deep_max"])
    for r in rows:
        pid, label, text = r.get("para_id"), str(r.get("label")), str(r.get("text") or "")
        if label not in ("0", "1"):
            out["bad_label"].append(pid)
            continue
        base = base_text.get(str(r.get("base_id")))
        if base is None:
            out["no_base"].append(pid)
        else:
            j = jaccard(base, text)
            if j > jmax:
                out["jaccard_high"].append(pid)
            col = _num(float(r["jaccard_to_base"])) if re.fullmatch(r"-?\d+(\.\d+)?", str(r.get("jaccard_to_base", ""))) else None
            if col is None or abs(col - j) > 1e-3:
                out["jaccard_column"].append(pid)
            if r.get("stratum") == "deep" and j > dmax:
                out["deep_jaccard"].append(pid)
        if label == "1" and r.get("stratum") == "deep" and banned(text):
            out["deep_banned"].append(pid)
        ids = cands.get((str(r.get("base_id")), str(r.get("generator")), text), [])
        if not ids:
            out["no_candidate"].append(pid)
        elif not judges or not any(all(judge_yes(judged.get(c, {}).get(m), label) for m in judges) for c in ids):
            out["judge_not_yes"].append(pid)
    return out


def ledger_total(spend_dir: Path) -> tuple[float, int]:
    """Sum of ``cost_usd`` over the per-call records of ``results/spend/*.jsonl`` (``gen.spend.load_calls``)."""
    total, n = 0.0, 0
    for f in sorted(spend_dir.glob("*.jsonl")) if spend_dir.is_dir() else []:
        for rec in _jsonl(f):
            if "cost_usd" in rec:
                total += float(rec["cost_usd"] or 0.0)
                n += 1
    return total, n


DEV_RE = re.compile(r"^- \*\*(D\d+)\b.*?(?=^- \*\*D\d+\b|\Z)", re.M | re.S)


def deviation_entries(text: str) -> dict[str, str]:
    """``D<n> -> the entry's text`` of a DEVIATIONS.md with ``- **D<n> (date). ...`` bullets."""
    return {m.group(1): m.group(0) for m in DEV_RE.finditer(text)}


def journal_problems(entries: dict[str, str]) -> list[str]:
    """CLAUDE.md: a deviation is recorded with item, change, reason, effect on hypotheses and date. Checked
    mechanically: numbering without gaps, a ``(YYYY-MM-DD)`` date and an effect statement (``Влияние`` / ``влияет``)."""
    nums = sorted(int(k[1:]) for k in entries)
    probs = [f"нет записей D{n}" for n in range(1, (nums[-1] if nums else 0) + 1) if n not in nums]
    for k, body in sorted(entries.items(), key=lambda kv: int(kv[0][1:])):
        if not re.search(r"\(\d{4}-\d{2}-\d{2}\)", body.splitlines()[0]):
            probs.append(f"{k}: нет даты")
        if not re.search(r"[Вв]лия", body):
            probs.append(f"{k}: не сказано о влиянии на гипотезы")
    return probs


def journaled(entries: dict[str, str], *patterns: str) -> bool:
    """Some single DEVIATIONS entry matches every pattern (case-insensitive)."""
    return any(all(re.search(p, body, re.I) for p in patterns) for body in entries.values())


def unjournaled_cuts(items: Sequence[dict[str, Any]], entries: dict[str, str]) -> list[str]:
    """Priorities of ``traces_run.json`` skipped by the budget whose benchmark and template no DEVIATIONS entry names."""
    return [f"{it.get('benchmark')}/{it.get('attack')}/{it.get('tasks')}" for it in items if it.get("skipped")
            and not journaled(entries, re.escape(str(it.get("benchmark"))), re.escape(str(it.get("attack"))))]


def plan_from_dry_run(text: str) -> list[str]:
    """Stages a ``run_all.sh --dry-run`` would run (its ``would run <stage>`` lines)."""
    return re.findall(r"would run (\S+)", text)


def last_run_args(lines: Iterable[str], mode: str) -> dict[str, str]:
    """``skip`` and ``seeds`` of the most recent full ``RUN start`` of ``mode`` (no --from/--only) in run_all.log."""
    found: dict[str, str] = {}
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[1] == mode and p[2] == "RUN" and p[3] == "start":
            args = dict(re.findall(r"\b(smoke|from|only|skip|seeds)=(\S*)", p[5]))
            if not args.get("from") and not args.get("only"):
                found = args
    return found


def parse_pytest(text: str) -> dict[str, int]:
    """Counts of the last pytest summary line (``3 failed, 450 passed, 2 skipped in 90s``)."""
    lines = [ln for ln in text.splitlines() if re.search(r"\b(passed|failed|error|errors|skipped|no tests ran)\b", ln)]
    last = lines[-1] if lines else ""
    out = {k: 0 for k in ("passed", "failed", "errors", "skipped")}
    for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)", last):
        out["errors" if k.startswith("error") else k] += int(n)
    return out


def _dig(node: Any, path: Sequence[str]) -> Any:
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _seed_rows(node: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Per-seed verdict rows (``per_seed``) or the verdict itself as one row (single-seed layout)."""
    ps = node.get("per_seed") if isinstance(node, dict) else None
    if isinstance(ps, dict) and ps:
        return [(str(s), r) for s, r in ps.items() if isinstance(r, dict)]
    return [("-", node)] if isinstance(node, dict) else []


def key_owner(key: str) -> str:
    """The experiment whose summary holds a verdict input key: E2 few-shot levels, E4 null comparisons, else E1."""
    if re.search(r"(^|/)(shots\d+|fewshot)(/|$)", key):
        return "E2"
    return "E4" if "curveball" in key else "E1"


def verdict_basis_problems(verdicts: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> list[str]:
    """macroAUC is the basis: the H1b/H3 input keys are macroAUC keys, no input key is a pooled metric, every key
    resolves in its experiment's summary (a key of an experiment cut by the ТЗ order, whose summary is absent, is not
    a failure: its verdict part is "не хватило данных"), and the per-seed H1b/H3 effects are the summaries' values of
    those keys. ``summaries`` maps E1/E2/E4 to their ``summary.json``."""
    inputs = verdicts.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return ["verdicts.json без inputs (ключи результатов, по которым вынесены вердикты)"]
    numbers: dict[str, Any] = {}
    for s in summaries.values():
        for k, rec in ((s or {}).get("numbers") or {}).items():
            numbers.setdefault(k, rec)
    probs: list[str] = []
    for need in ("H1a", "H1b", "H2", "H3"):
        if not any(h == need or h.startswith(need + "/") for h in inputs):
            probs.append(f"нет inputs {need}")
    for hyp, group in inputs.items():
        for name, key in (group or {}).items() if isinstance(group, dict) else []:
            key = str(key)
            if POOLED_RE.search(key):
                probs.append(f"{hyp}/{name}: объединённая метрика {key}")
            if hyp.split("/")[0] in ("H1b", "H3") and not {"macro_auc", "val_macro_auc"} & set(key.split("/")):
                probs.append(f"{hyp}/{name}: не macroAUC ({key})")
            if key not in numbers and key_owner(key) in summaries:
                probs.append(f"{hyp}/{name}: {key} нет в summary.json {key_owner(key)}")
    for name, key in (inputs.get("H1b") or {}).items():
        variant, *rest = str(name).split("/")
        for s, row in _seed_rows(((verdicts.get("H1b") or {}).get(variant)) or {}):
            want = _dig((numbers.get(key) or {}).get("per_seed"), [s, "value"]) if s != "-" else (numbers.get(key) or {}).get("mean")
            got = _dig(row.get("effect"), rest)
            if not _close(got, want):
                probs.append(f"H1b/{name} сид {s}: эффект вердикта {got} ≠ {key} = {want}")
    key = (inputs.get("H3") or {}).get("primary")
    for s, row in _seed_rows(verdicts.get("H3") or {}):
        want = _dig((numbers.get(key) or {}).get("per_seed"), [s, "value"]) if s != "-" else (numbers.get(key) or {}).get("mean")
        if not _close(row.get("effect"), want):
            probs.append(f"H3 сид {s}: эффект вердикта {row.get('effect')} ≠ {key} = {want}")
    return probs


def _hyp_nodes(verdicts: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = [("H1a", verdicts.get("H1a")), ("H3", verdicts.get("H3"))]
    out += [(f"H1b/{k}", v) for k, v in (verdicts.get("H1b") or {}).items() if isinstance(v, dict)]
    return [(n, v) for n, v in out if isinstance(v, dict)]


def carriers_missing(verdicts: dict[str, Any]) -> list[str]:
    """H1a, H1b and H3 verdicts whose inputs carry no boolean E0 carrier flag (the E0 gate is mandatory)."""
    def has_flag(node: Any) -> bool:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "carrier" and (isinstance(v, bool) or (isinstance(v, dict) and any(isinstance(x, bool) for x in v.values()))):
                    return True
                if has_flag(v):
                    return True
        elif isinstance(node, list):
            return any(has_flag(x) for x in node)
        return False
    names = [n for n, _ in _hyp_nodes(verdicts)]
    missing = [n for n in ("H1a", "H3") if n not in names] + ([] if any(n.startswith("H1b/") for n in names) else ["H1b"])
    return missing + [n for n, node in _hyp_nodes(verdicts) if not has_flag(node)]


def carrier_mismatches(verdicts: dict[str, Any], power: dict[str, Any]) -> list[str]:
    """Every carrier flag of the verdicts equals the cell of the E0 table it was read from: H1a template sources
    ``carriers[src].auc_diff``, H1b ``carriers.macro.auc_diff``, H3 ``carriers.macro.<metric>`` (``auc_diff_h3``
    when the table has it)."""
    from flyguard.eval.power import CARRIES

    table = power.get("carriers") or {}
    carries = lambda src, metric: (table.get(src) or {}).get(metric) == CARRIES  # noqa: E731
    h3_metric = "auc_diff_h3" if "auc_diff_h3" in (table.get("macro") or {}) else "auc_diff"
    out = []
    for name, node in _hyp_nodes(verdicts):
        for s, row in _seed_rows(node):
            inp = row.get("inputs") or {}
            if name == "H1a":
                for src, t in (inp.get("template") or {}).items():
                    if isinstance(t, dict) and isinstance(t.get("carrier"), bool) and t["carrier"] != carries(src, "auc_diff"):
                        out.append(f"H1a {src} сид {s}")
                continue
            c = inp.get("carrier")
            flag, metric = (c.get("macro"), c.get("metric")) if isinstance(c, dict) else (c, None)
            metric = metric or (h3_metric if name == "H3" else "auc_diff")
            if not isinstance(flag, bool) or flag != carries("macro", metric):
                out.append(f"{name} сид {s} ({metric})")
    return out


def h3_perm_problems(e4_files: dict[str, dict[str, Any]], n_perms: int, perm_seeds: set[int], n_null: int,
                     n_boot: int) -> list[str]:
    """The H3 record and ``h3`` table row of every E4 file carry ``n_perms`` π permutations and ``n_null`` curveball
    nulls with ``n_boot`` two-stage draws; the π seeds of ``perm_grid`` are the ``perm`` children of the global
    seeds."""
    if not e4_files:
        return ["результатов E4 нет"]
    probs = []
    for name, res in sorted(e4_files.items()):
        rec = (res.get("numbers") or {}).get(H3_PRIMARY)
        if not isinstance(rec, dict):
            probs.append(f"{name}: нет {H3_PRIMARY}")
            continue
        for k, want in (("n_perms", n_perms), ("n_null", n_null), ("n_boot", n_boot)):
            if rec.get(k) != want:
                probs.append(f"{name}: {k}={rec.get(k)} ≠ {want}")
        row = next((r for r in (res.get("tables") or {}).get("h3") or [] if r.get("primary")), None)
        if not row or row.get("n_perms") != n_perms or row.get("n_null") != n_null:
            probs.append(f"{name}: строка h3 основного выхода без n_perms={n_perms}/n_null={n_null}")
        got = {int(r["perm_seed"]) for r in (res.get("tables") or {}).get("perm_grid") or [] if r.get("perm_seed") is not None}
        if got != perm_seeds:
            probs.append(f"{name}: сиды π в perm_grid не совпадают с дочерними perm глобальных сидов "
                         f"(совпало {len(got & perm_seeds)} из {len(perm_seeds)}, лишних {len(got - perm_seeds)})")
    return probs


def bloom_balance_records(seed_files: dict[str, dict[str, Any]]) -> list[tuple[str, int, int]]:
    """``(file:detector, n0, n1)`` of the fitted Bloom readouts recorded in the ``detectors`` tables (``balanced_counts``
    list or ``balanced_n0``/``balanced_n1`` scalars)."""
    out = []
    for name, res in seed_files.items():
        for row in (res.get("tables") or {}).get("detectors") or []:
            c = row.get("balanced_counts")
            pair = (c[0], c[1]) if isinstance(c, (list, tuple)) and len(c) == 2 else (row.get("balanced_n0"), row.get("balanced_n1"))
            if all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
                out.append((f"{name}:{row.get('detector')}", int(pair[0]), int(pair[1])))
    return out


def contract_problems(rows: Sequence[dict[str, str]], manifest: dict[str, Any], variants: Iterable[str],
                      validation_ids: Iterable[str], mod: int, rem: int, test_attack: str) -> list[str]:
    """Every submitted variant has exactly one CSV row per test episode of the split manifest and none else; no
    validation episode of the contract run is a test task or ``test_attack`` (contract §6-§8)."""
    from flyguard.agentdojo_io.parse import split_episode_id

    test = set(manifest.get("test") or [])
    probs = []
    by: dict[str, Counter] = {v: Counter() for v in variants}
    for r in rows:
        by.setdefault(r.get("variant", ""), Counter())[r.get("episode_id", "")] += 1
    for v, cnt in sorted(by.items()):
        missing, extra, dup = test - set(cnt), set(cnt) - test, [e for e, n in cnt.items() if n > 1]
        if missing or extra or dup:
            probs.append(f"{v}: нет {len(missing)} тестовых эпизодов, лишних {len(extra)}, повторов {len(dup)}")
    for eid in validation_ids:
        try:
            p = split_episode_id(str(eid))
        except ValueError:
            probs.append(f"валидационный эпизод с неверным id: {eid}")
            continue
        if zlib.crc32(str(p["user_task"]).encode()) % mod == rem or p.get("attack") == test_attack:
            probs.append(f"валидационный эпизод контракта из теста или {test_attack}: {eid}")
    return probs


def manifest_problems(splits: dict[str, Any], pools: dict[str, Any], docs: Any, present: set[str]) -> list[str]:
    """The manifests agree with ``documents.parquet``: every main-variant test document is listed in
    ``splits.e1.test`` or dropped by dedup (and only those), train/val lists equal the tables, the counts equal the
    lists, pools are sized as listed, every present source has test documents."""
    probs: list[str] = []
    e1 = splits.get("e1") or {}
    test = e1.get("test") or {}
    for part in ("train", "val"):
        if not e1.get(part):
            probs.append(f"splits.e1.{part} пуст")
    for s in sorted(present):
        if not test.get(s):
            probs.append(f"splits.e1.test.{s} пуст")
    counts = (splits.get("counts") or {}).get("test") or {}
    probs += [f"counts.test.{s}={counts[s]} ≠ {len(ids)}" for s, ids in test.items() if s in counts and counts[s] != len(ids)]
    if not splits.get("c_unl"):
        probs.append("c_unl пуст")
    for name in ("p_val", "p_test"):
        p = pools.get(name) or {}
        if not p.get("doc_ids") or p.get("n") != len(p["doc_ids"]):
            probs.append(f"pools.{name}: n={p.get('n')} при {len(p.get('doc_ids') or [])} документах")
    if docs is not None and len(docs):
        variant = docs["meta_json"].map(lambda s: (json.loads(s) if isinstance(s, str) and s else {}).get("variant", "main"))
        main = docs[variant != "e6"]
        listed = {d for ids in test.values() for d in ids}
        dropped = set(splits.get("dropped_by_dedup") or [])
        tdocs = set(main.loc[main["split"] == "test", "doc_id"])
        for what, bad in (("тестовых документов нет в splits.json", tdocs - listed - dropped),
                          ("в splits.e1.test документов не из теста", listed - tdocs),
                          ("документов и в e1.test, и в dropped_by_dedup", listed & dropped)):
            if bad:
                probs.append(f"{what}: {len(bad)}")
        for part in ("train", "val"):
            if set(e1.get(part) or []) != set(main.loc[main["split"] == part, "doc_id"]):
                probs.append(f"splits.e1.{part} не совпадает с documents.parquet")
        known = set(docs["doc_id"])
        for name in ("p_val", "p_test"):
            unknown = set((pools.get(name) or {}).get("doc_ids") or []) - known
            if unknown:
                probs.append(f"pools.{name}: {len(unknown)} документов нет в таблице")
    return probs


# ------------------------------------------------------------------------------------------ helpers kept from v1
SMOKE_STAGES = ("build_stage1", "e0_stage1", "build_full", "e0_stage2", "prescore", "e1", "e4", "e5", "e3", "e2", "e6",
                "contract", "verdicts", "report")
"""``run_all.sh`` stages of ТЗ "Бюджет времени" steps 2–13, in run order (the ``check`` stage is this script and is
not a step). ``prescore`` is the guard scoring of steps 5/7 moved ahead of the seeds: once it has filled the cache,
E1/E6/contract only read it, so leaving it out would drop the most expensive work of the smoke from the sum."""
CACHE_NOTE_RE = re.compile(r"scores_cache_rows=(\d+|unknown)")
CODE_PATHS = ("src", "scripts", "configs")
"""Paths whose content decides the numbers of a results file (``git diff`` between the commits of two files)."""
SEED_EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")


def parse_access_log(lines: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        ts = _utc(parts[0])
        if ts is None:
            continue
        purpose = parts[3] if len(parts) > 3 else ""
        out.append({"ts": ts, "split": parts[1], "path": parts[2], "purpose": purpose, "retro": bool(RETRO_RE.search(purpose))})
    return out


def smoke_timing(lines: Iterable[str]) -> dict[str, Any]:
    """Reconstruct the duration of a clean smoke run from ``logs/run_all.log`` lines
    (``ts\\tmode\\tstage\\tstatus\\tseconds\\tnote``).

    Returns ``{"last_total": (ts, seconds, note) | None, "stages": {stage: (ts, seconds)} (the most recent smoke
    ``done`` of each stage), "sum": seconds over the measured stages, "missing": stages never measured, "failed":
    stages whose most recent smoke event is ``fail``, "prescore_cache_rows": rows of the guard score cache when the
    measured ``prescore`` started (``None``: not recorded or unknown)}``. Skipped (``skip``) and interrupted
    (``start`` without ``done``) events measure nothing; only ``mode == smoke`` lines are read.
    """
    stages: dict[str, tuple[str, float]] = {}
    latest: dict[str, str] = {}
    start_note: dict[str, str] = {}
    cache_rows: int | None = None
    last_total = None
    last_start = None          # the most recent smoke "RUN start": a run whose TOTAL is not written yet is unfinished
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 5 or p[1] != "smoke":
            continue
        ts, stage, status, secs, note = p[0], p[2], p[3], p[4], (p[5] if len(p) > 5 else "")
        try:
            seconds = float(secs or 0)
        except ValueError:
            continue
        if stage == "RUN" and status == "TOTAL":
            last_total = (ts, seconds, note)
        elif stage == "RUN" and status == "start":
            last_start = ts
        elif stage in SMOKE_STAGES:
            if status == "start":
                start_note[stage] = note
            if status in ("done", "fail"):
                latest[stage] = status
            if status == "done":
                stages[stage] = (ts, seconds)
                if stage == "prescore":
                    m = CACHE_NOTE_RE.search(start_note.get(stage, ""))
                    cache_rows = int(m.group(1)) if m and m.group(1).isdigit() else None
    return {"last_total": last_total, "last_start": last_start, "stages": stages,
            "sum": float(sum(v[1] for v in stages.values())),
            "missing": [s for s in SMOKE_STAGES if s not in stages],
            "failed": [s for s in SMOKE_STAGES if latest.get(s) == "fail"],
            "prescore_cache_rows": cache_rows}


def seed_coverage(seed_files: dict[str, dict[str, Any]], expected: Sequence[int],
                  experiments: Sequence[str] = SEED_EXPERIMENTS, required: Sequence[str] = ("E1", "E4")) -> Check:
    """Every experiment with seed files has exactly the ``expected`` seeds (keys are ``.../<E>/<seed>.json`` paths):
    a missing seed (a failed or killed seed process) or an extra one (a run with other ``--seeds``) makes the summary
    average another seed set than the configured one. Experiments without files are listed; only ``required`` ones
    fail then (ТЗ cutting order: E6, E2, E3, E5 may be cut first, E1 and E4 carry H1/H3)."""
    want = sorted({int(s) for s in expected})
    have: dict[str, set[int]] = {e: set() for e in experiments}
    for name in seed_files:
        p = Path(name)
        if p.parent.name in have and p.stem.isdigit():
            have[p.parent.name].add(int(p.stem))
    bad = []
    for e in experiments:
        if not have[e]:
            continue
        missing, extra = sorted(set(want) - have[e]), sorted(have[e] - set(want))
        if missing or extra:
            bad.append(f"{e}: " + ", ".join(x for x in (f"нет сидов {missing}" if missing else "",
                                                          f"лишние сиды {extra}" if extra else "") if x))
    absent = [e for e in experiments if not have[e]]
    absent_required = [e for e in absent if e in required]
    detail = (f"ожидаются сиды {want}; " + ("; ".join(bad) if bad else "у выполненных экспериментов все сиды")
              + (f"; не выполнены: {absent}" if absent else "")
              + (f" (из них обязательны {absent_required})" if absent_required else ""))
    return Check("у каждого выполненного эксперимента E1–E6 все сиды конфига",
                 bool(want) and not bad and not absent_required, detail)


def code_provenance(files: dict[str, dict[str, Any] | None], same_code: Callable[[str, str], bool | None],
                    head: str | None = None) -> Check:
    """The results files were computed by one version of the code (A52): every file records a ``git_commit``;
    distinct commits have the same ``src``/``scripts``/``configs`` (``same_code(a, b)``: True same, False differs,
    None unknown commit); no file has ``git_dirty: true``. Files without the dirty flag are counted in the detail
    only. ``head`` (the commit being checked out now) is compared for information."""
    name = "результаты посчитаны одним кодом (коммит без незакоммиченных изменений src/scripts/configs)"
    files = {k: v for k, v in files.items() if v is not None}
    if not files:
        return Check(name, False, "файлов результатов нет")
    commits = {k: v.get("git_commit") for k, v in files.items()}
    no_commit = sorted(k for k, c in commits.items() if not c)
    counts: dict[str, int] = {}
    for c in commits.values():
        if c:
            counts[c] = counts.get(c, 0) + 1
    ref = max(counts, key=lambda c: (counts[c], c)) if counts else None
    differs, unknown = [], []
    for c in sorted(counts):
        if c == ref:
            continue
        same = same_code(ref, c)
        (differs if same is False else unknown if same is None else []).append(c[:12])
    dirty = sorted(k for k, v in files.items() if v.get("git_dirty") is True)
    unflagged = sum(1 for v in files.values() if "git_dirty" not in v)
    detail = (f"файлов {len(files)}; коммиты: " + ", ".join(f"{c[:12]} ({n})" for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))
              if counts else f"файлов {len(files)}; коммиты не записаны")
    if differs:
        detail += f"; код (src/scripts/configs) отличается от {ref[:12]} в: {differs}"
    if unknown:
        detail += f"; коммиты не найдены в истории: {unknown}"
    if no_commit:
        detail += f"; без git_commit: {no_commit[:6]}" + (f" и ещё {len(no_commit) - 6}" if len(no_commit) > 6 else "")
    if dirty:
        detail += f"; с незакоммиченными изменениями кода (git_dirty): {dirty[:6]}"
    if unflagged:
        detail += f"; флаг git_dirty не записан в {unflagged} файлах (чистота дерева при расчёте не известна)"
    if ref and head and head != ref:
        same_head = same_code(ref, head)
        detail += ("; код текущего HEAD тот же" if same_head is True
                   else "; код текущего HEAD отличается от кода результатов" if same_head is False else "")
    return Check(name, bool(counts) and not no_commit and not differs and not unknown and not dirty, detail)


def regex_order(commit_time: datetime | None, dirty: bool, entries: Sequence[dict[str, Any]], journal_text: str) -> Check:
    """The regex criterion: the file's last commit precedes the first journaled test read; retroactive entries are
    shown as the real order and accepted only when the journal (DEVIATIONS) explains them (D10)."""
    name = "regex_patterns.txt закоммичен до первого чтения теста"
    tests = sorted((e for e in entries if e["split"] == "test"), key=lambda e: e["ts"])
    if commit_time is None:
        return Check(name, False, "git-дата коммита файла недоступна")
    if dirty:
        return Check(name, False, "файл изменён и не закоммичен")
    if not tests:
        return Check(name, True, f"коммит {commit_time.isoformat()}, чтений теста в журнале ещё нет")
    first = tests[0]
    formal = commit_time < first["ts"]
    retro = [e for e in tests if e["retro"]]
    detail = f"коммит {commit_time.strftime('%Y-%m-%dT%H:%M:%SZ')} < первое чтение теста {first['ts'].strftime('%Y-%m-%dT%H:%M:%SZ')} ({first['path']})"
    if not formal:
        detail = detail.replace(" < ", " ≥ ")
    if retro:
        explained = bool(RETRO_RE.search(journal_text)) or "D10" in journal_text
        detail += (f"; по существу: {len(retro)} записей журнала помечены как ретроспективные (чтения до коммита, "
                   f"первая помечена {retro[0]['ts'].strftime('%Y-%m-%dT%H:%M:%SZ')} {retro[0]['path']}); "
                   + ("объяснение в DEVIATIONS (D10)" if explained else "объяснения в DEVIATIONS нет"))
        return Check(name, formal and explained, detail)
    return Check(name, formal, detail)


def thresholds_complete(results: dict[str, dict[str, Any]]) -> Check:
    """Every threshold record of every results file carries value, source, target and n (ТЗ 2.5)."""
    bad: list[str] = []
    n = 0
    for name, res in results.items():
        for key, rec in ((res or {}).get("thresholds") or {}).items():
            n += 1
            if not isinstance(rec, dict):
                bad.append(f"{name}:{key}")
                continue
            v = rec.get("value")
            ok = (isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
                  and isinstance(rec.get("source"), str) and rec["source"]
                  and rec.get("target") not in (None, "")
                  and isinstance(rec.get("n"), int) and not isinstance(rec.get("n"), bool) and rec["n"] > 0)
            if not ok:
                bad.append(f"{name}:{key}")
    return Check("у каждого порога записаны значение, источник, цель и n", not bad,
                 f"порогов {n}; неполные: {', '.join(bad[:8])}" if bad else f"порогов {n}")


def config_hash_mismatches(files: dict[str, dict[str, Any] | None], current: str) -> list[str]:
    return [name for name, d in files.items() if not d or d.get("config_hash") != current]


def split_rule_violations(manifest: dict[str, Any], mod: int, rem: int, test_attack: str) -> list[str]:
    """Contract §6 recomputed: test ids are test tasks with clean or ``test_attack`` runs; the other lists hold
    non-test tasks and never the ``test_attack`` template."""
    from flyguard.agentdojo_io.parse import split_episode_id

    problems: list[str] = []

    def parts(eid: str) -> dict[str, Any] | None:
        try:
            return split_episode_id(str(eid))
        except ValueError:
            problems.append(f"malformed id {eid!r}")
            return None

    for eid in manifest.get("test") or []:
        p = parts(eid)
        if p is None:
            continue
        if zlib.crc32(str(p["user_task"]).encode()) % mod != rem:
            problems.append(f"test list holds a non-test task: {eid}")
        if p.get("attack") not in (None, "none", test_attack):
            problems.append(f"test list holds another template: {eid}")
    for lst in ("observation", "validation_clean", "validation_attacks", "train_attacks"):
        for eid in manifest.get(lst) or []:
            p = parts(eid)
            if p is None:
                continue
            if zlib.crc32(str(p["user_task"]).encode()) % mod == rem:
                problems.append(f"{lst} holds a test task: {eid}")
            if p.get("attack") == test_attack:
                problems.append(f"{lst} holds {test_attack}: {eid}")
    return problems


def power_before_e1(power: dict[str, Any] | None, e1_files: dict[str, dict[str, Any]], current: str) -> Check:
    name = "power.json заморожен (стадия 2) с текущим хешем и записан раньше результатов E1"
    if not power:
        return Check(name, False, "power.json нет")
    frozen = power.get("stage") == 2 and power.get("frozen") is not False
    hash_ok = power.get("config_hash") == current
    stamp = next((power.get(k) for k in ("created_at", "frozen_at", "generated", "written_at") if power.get(k)), None)
    p_ts = _utc(str(stamp)) if stamp else None
    e1_ts = [_utc(str(d.get("created_at"))) for d in e1_files.values() if d.get("created_at")]
    e1_ts = [t for t in e1_ts if t is not None]
    detail = f"stage={power.get('stage')} frozen={power.get('frozen')} hash_ok={hash_ok} created_at={stamp}"
    if not e1_files:
        detail += "; результатов E1 нет"
        order_ok = False
    elif p_ts is None or len(e1_ts) < len(e1_files):
        detail += "; нет меток времени для сравнения"
        order_ok = False
    else:
        order_ok = all(p_ts <= t for t in e1_ts)
        detail += f"; первый E1 {min(e1_ts).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    return Check(name, bool(frozen and hash_ok and order_ok), detail)


# ================================================================================================= the checks
class Checker:
    def __init__(self, root: Path, smoke: bool, pytest_mode: str = "run",
                 access_log: Callable[[Path, str, str], None] | None = None, only: Sequence[str] | None = None,
                 dry_run: Callable[[Path, Sequence[str]], tuple[int, str]] | None = None) -> None:
        self.root, self.smoke, self.pytest_mode = Path(root), bool(smoke), pytest_mode
        self.only = set(only or ())
        self.cfg = load_configs(self.root)
        self.rdir = R.results_dir(self.root, self.smoke)
        self.processed, self.manifests = output_dirs(self.root, self.smoke)
        self.shared = (self.rdir / "shared") if self.smoke else (self.root / "results/shared")   # contract_run.shared_dir
        self.current = config_hash(self.root)
        self.checks: list[Check] = []
        self._access_log = access_log
        self._dry_run = dry_run
        self.seed_files: dict[str, dict[str, Any]] = {}
        for exp in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
            for p in R.list_results(exp, self.smoke, self.root):
                d = _json(p)
                if d is not None:
                    self.seed_files[str(p.relative_to(self.root))] = d
        self.deviations = self._read(self.root / "DEVIATIONS.md")
        self.dev_entries = deviation_entries(self.deviations)

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def log_access(self, path: Path, purpose: str) -> None:
        if self._access_log is not None:
            self._access_log(path, "test", purpose)
        else:
            from flyguard.netlog import log_data_access

            log_data_access(path, split="test", purpose=purpose)

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    def summaries(self) -> dict[str, dict[str, Any]]:
        out = {}
        for e in SEED_EXPERIMENTS:
            s = _json(R.summary_path(e, self.smoke, self.root))
            if s:
                out[e] = s
        return out

    def present_sources(self) -> set[str]:
        have = {"deep", "bipia", "notinject"}
        if (self.root / "results/shared/traces_manifest.json").exists():
            have |= {"dojo", "dyn"}
        if (self.root / "data/paraphrases/paraphrases.csv").exists():
            have.add("para")
        return have

    def run(self) -> list[Check]:
        for fn in (self.c_sources, self.c_network, self.c_pilot, self.c_traces, self.c_paraphrases, self.c_spend,
                   self.c_smoke, self.c_resumable, self.c_pytest, self.c_manifests, self.c_dedup, self.c_kc,
                   self.c_power, self.c_windows, self.c_thresholds, self.c_verdicts, self.c_h3, self.c_bloom,
                   self.c_comparator, self.c_config_hash, self.c_seeds, self.c_provenance, self.c_deviations,
                   self.c_report, self.c_contract, self.c_regex):
            if self.only and fn.__name__ not in self.only:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check, not a crashed checklist
                self.add((fn.__doc__ or fn.__name__).strip(), False, f"ошибка проверки: {type(exc).__name__}: {exc}")
        return self.checks

    # --------------------------------------------------------------------------------------------------- data
    def c_sources(self) -> None:
        """fetch.sh скачал всё с проверкой sha256 (повторный хеш), включая flypath build"""
        name = self.c_sources.__doc__
        src = _json(self.root / "data/manifests/sources.json")
        if not src or not src.get("files"):
            self.add(name, False, "data/manifests/sources.json нет или пуст")
            return
        files = src["files"]
        h = rehash_sources(files, self.root)
        by_path = {str(f.get("path")): f for f in files}
        flypath = [p for p in FLYPATH_OUTPUTS if "flypath build" not in str((by_path.get(p) or {}).get("url", ""))]
        pins = [k for k in REQUIRED_PINS if not (src.get("pins") or {}).get(k)]
        ok = not h["no_sha"] and not h["mismatch"] and not h["missing"] and not flypath and not pins
        detail = f"файлов {len(files)}, хеш совпал у {len(h['ok'])}"
        for key, what in (("mismatch", "хеш не совпал"), ("missing", "нет на диске"), ("no_sha", "без sha256")):
            if h[key]:
                detail += f"; {what}: {_few(h[key])}"
        if h["missing_a10"]:
            detail += f"; удалены после flypath build (A10): {len(h['missing_a10'])}"
        detail += ("; flypath build: " + (f"нет записи для {flypath}" if flypath else "malecns_R.npz/.json")
                   + ("; пины: все" if not pins else f"; нет пинов {pins}"))
        self.add(name, ok, detail)

    def c_network(self) -> None:
        """network.log: только скачивания и два разрешённых потока к провайдерам из конфига"""
        src = _json(self.root / "data/manifests/sources.json") or {}
        downloads = set(TOOLCHAIN_HOSTS)
        for f in src.get("files") or []:
            host = urlsplit(str(f.get("url", ""))).hostname
            if host:
                downloads.add(host.lower())
        providers = (self.cfg.operator.get("llm_api") or {}).get("providers") or []
        prov = {h.lower() for h in (urlsplit(str(p.get("base_url", ""))).hostname for p in providers) if h}
        log = self.root / "logs/network.log"
        if not log.exists():
            self.add(self.c_network.__doc__, False, "logs/network.log нет")
            return
        with open(log, encoding="utf-8", errors="replace") as fh:
            a = network_audit(fh, downloads - prov, prov)
        detail = (f"скачиваний {a['downloads']}; поток 1 (трассы) {a['flows']['traces']}, поток 2 (парафразы) "
                  f"{a['flows']['paraphrases']} к {sorted(prov)}")
        if a["bad"]:
            detail += "; нарушения: " + "; ".join(f"{k} ({n})" for k, n in sorted(a["bad"].items()))
        self.add(self.c_network.__doc__, bool(prov) and not a["bad"], detail)

    def c_pilot(self) -> None:
        """модель агента выбрана пилотом по правилу Приложения B; пилоты всех кандидатов записаны"""
        name = self.c_pilot.__doc__
        pilot = _json(self.root / "results/pilot.json")
        order = list((self.cfg.operator.get("llm_api") or {}).get("agent_models") or [])
        rule = self.cfg.default["traces"]["pilot"]
        if not pilot:
            self.add(name, False, "results/pilot.json нет")
            return
        cands = pilot.get("candidates") or []
        by = {c.get("model"): c for c in cands}
        probs = [f"нет пилота {m}" for m in order if m not in by]
        probs += [f"{m}: {c.get('n_attacked')}+{c.get('n_clean')} кейсов вместо {rule['n_attacked']}+{rule['n_clean']}"
                  for m, c in by.items() if m in order and (c.get("n_attacked") != rule["n_attacked"] or c.get("n_clean") != rule["n_clean"])]
        chosen, by_rule = pilot_rule_choice(cands, order, rule["asr_range"], rule["min_utility"])
        if pilot.get("chosen_model") != chosen:
            probs.append(f"выбрана {pilot.get('chosen_model')}, по правилу выходит {chosen}")
        if bool(pilot.get("chosen_by_rule")) != by_rule:
            probs.append(f"chosen_by_rule={pilot.get('chosen_by_rule')}, по правилу {by_rule}")
        if chosen and not by_rule and not journaled(self.dev_entries, r"пилот", re.escape(chosen)):
            probs.append(f"выбор вне коридора ASR не описан в DEVIATIONS (запись с «пилот» и {chosen})")
        tm = _json(self.root / "results/shared/traces_manifest.json") or {}
        if tm and tm.get("agent_model") != pilot.get("chosen_model"):
            probs.append(f"трассы сгенерированы моделью {tm.get('agent_model')}")
        rates = ", ".join(f"{m}: ASR {by[m].get('targeted_asr')}, utility {by[m].get('utility_clean')}" for m in order if m in by)
        self.add(name, not probs, f"выбрана {pilot.get('chosen_model')} ({'по правилу' if by_rule else 'ASR ближе к 0,40'}); {rates}"
                 + (f"; {'; '.join(probs)}" if probs else ""))

    def c_traces(self) -> None:
        """трассы заморожены: повторный хеш каждого лога и копии results/shared/traces по traces_manifest.json"""
        name = self.c_traces.__doc__
        m = _json(self.root / "results/shared/traces_manifest.json")
        if not m or not m.get("files"):
            self.add(name, False, "results/shared/traces_manifest.json нет или без файлов")
        else:
            shared_dir = (self.cfg.operator.get("shared") or {}).get("traces_dir")
            logdir = Path(shared_dir) if shared_dir else Path(self.cfg.default["traces"]["logdir"])
            logdir = logdir if logdir.is_absolute() else self.root / logdir
            out_dir = Path(str((self.cfg.operator.get("shared") or {}).get("traces_out_dir") or "results/shared/traces"))
            out_dir = out_dir if out_dir.is_absolute() else self.root / out_dir
            h = traces_rehash(m["files"], logdir, out_dir)
            bad = {k: v for k, v in h.items() if v}
            self.add(name, not bad, f"логов в манифесте {len(m['files'])}; каталог {self._rel(logdir)}, копия {self._rel(out_dir)}"
                     + ("; " + "; ".join(f"{k}: {len(v)} ({_few(v, 2)})" for k, v in bad.items()) if bad
                        else "; все хеши совпали, лишних файлов в копии нет"))
        self.add("копия трасс у второй команды", None,
                 "передача копии второй команде проверяется только ею (здесь проверена копия results/shared/traces)")

    def c_paraphrases(self) -> None:
        """парафразы заморожены: манифест, явное «да» судьи, Жаккар с базой ≤ 0,5, нет запрещённых слов в глубокой страте"""
        name = self.c_paraphrases.__doc__
        pdir = self.root / "data/paraphrases"
        m = _json(pdir / "paraphrases_manifest.json")
        paths = {k: pdir / f for k, f in (("csv", "paraphrases.csv"), ("calls", "calls.jsonl"),
                                          ("judgements", "judgements.jsonl"), ("bases", "bases.jsonl"))}
        missing = [str(p.name) for p in paths.values() if not p.exists()]
        if not m or missing:
            self.add(name, False, f"нет paraphrases_manifest.json или {missing}")
            return
        from flyguard.gen.paraphrases import BannedMatcher, jaccard_texts

        pc = self.cfg.default["paraphrase"]
        f = pc["filters"]
        probs: list[str] = []
        keys = [k for k in ("models", "dates", "prompts_sha256", "rates", "counts", "banned_words") if not m.get(k)]
        if keys:
            probs.append(f"в манифесте нет {keys}")
        prompt_files = {**{k: self.root / v for k, v in (pc.get("prompts") or {}).items()},
                        "banned_words": self.root / pc["banned_words_file"]}
        stale = [k for k, p in prompt_files.items() if not p.exists() or (m.get("prompts_sha256") or {}).get(k) != sha256_file(p)]
        if stale:
            probs.append(f"хеши промптов не совпадают с configs/prompts: {stale}")
        rates = m.get("rates") or {}
        if not ((rates.get("acceptance") or {}).get("by_stratum") and rates.get("generation_refusal") and rates.get("judge_refusal")):
            probs.append("нет долей принятия по стратам или долей отказов генератора/судьи")
        if not (m.get("dates") or {}).get("first_call"):
            probs.append("нет дат генерации")
        judges = [str(j.get("model")) for j in ((self.cfg.operator.get("llm_api") or {}).get("paraphrase") or {}).get("judges") or []]
        in_manifest = [str(j.get("model")) for j in (m.get("models") or {}).get("judges") or []]
        if sorted(judges) != sorted(in_manifest) or not judges:
            probs.append(f"судьи манифеста {in_manifest} ≠ конфигу {judges}")
        final = (m.get("banned_words") or {}).get("final") or (_json(pdir / "banned_words_final.json") or {}).get("final") or []
        if not final:
            probs.append("нет итогового списка запрещённых слов")
        purpose = "check_acceptance: paraphrase freeze audit (judge verdicts, Jaccard, banned words; texts read by code only, never printed)"
        for p in paths.values():
            self.log_access(p, purpose)
        with open(paths["csv"], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        bases = {str(b.get("base_id")): str(b.get("text", "")) for b in _jsonl(paths["bases"])}
        matcher = BannedMatcher(list(final))
        k = int(f.get("shingle", 5))
        a = paraphrase_audit(rows, bases, candidate_index(_jsonl(paths["calls"])), last_decisive(_jsonl(paths["judgements"])),
                             judges, f, lambda base, text: jaccard_texts(base, text, k), matcher.hits)
        what = {"bad_label": "метка не 0/1", "no_base": "нет базы", "jaccard_high": f"Жаккар > {f['jaccard_max']}",
                "jaccard_column": "Жаккар в CSV не совпадает с пересчитанным", "deep_jaccard": f"глубокая страта с Жаккаром > {f['deep_max']}",
                "deep_banned": "запрещённое слово в глубоком позитиве", "no_candidate": "строки нет среди кандидатов генератора",
                "judge_not_yes": "нет явного «да» каждого судьи"}
        probs += [f"{what[k2]}: {len(v)} ({_few(v, 2)})" for k2, v in a.items() if v]
        pos = sum(1 for r in rows if str(r.get("label")) == "1")
        deep = sum(1 for r in rows if r.get("stratum") == "deep")
        self.add(name, not probs and bool(rows),
                 f"строк {len(rows)} (позитивов {pos}, глубоких {deep}); судьи {judges} ({(m.get('models') or {}).get('judge_rule')}); "
                 f"запрещённых слов {len(final)}" + (f"; {'; '.join(probs)}" if probs else "; всё пересчитано"))

    def c_spend(self) -> None:
        """расход API ≤ бюджета (по журналу вызовов); урезания записаны в DEVIATIONS.md"""
        name = self.c_spend.__doc__
        sp = _json(self.root / "results/spend.json")
        budget = float((self.cfg.operator.get("llm_api") or {}).get("budget_usd") or 0)
        if not sp:
            self.add(name, False, "results/spend.json нет")
            return
        spent = float(sp.get("spent_usd") or 0)
        ledger, n = ledger_total(self.root / str(self.cfg.default["traces"]["spend_dir"]))
        probs = []
        if spent > budget or sp.get("within_budget") is not True:
            probs.append(f"бюджет превышен ({spent:.4f} > {budget:.2f})")
        if abs(ledger - spent) > 1e-3 or ledger > budget:
            probs.append(f"журнал вызовов даёт {ledger:.4f} USD ({n} вызовов), spend.json {spent:.4f}")
        run = _json(self.root / "results/traces_run.json") or {}
        cuts = [it for it in run.get("items") or [] if it.get("skipped")]
        unj = unjournaled_cuts(cuts, self.dev_entries)
        if unj:
            probs.append(f"урезания без записи в DEVIATIONS: {unj}")
        partial = ((_json(self.root / "data/paraphrases/paraphrases_manifest.json") or {}).get("counts") or {}).get("bases_partial") or []
        if partial and not journaled(self.dev_entries, r"парафраз", r"бюджет|срез|урез"):
            probs.append(f"{len(partial)} баз парафраз урезаны бюджетом без записи в DEVIATIONS")
        self.add(name, not probs, f"потрачено {spent:.4f} USD из {budget:.2f} (журнал {ledger:.4f}, вызовов {n}); "
                 f"урезаний трасс {len(cuts)}, все в DEVIATIONS: {not unj}" + (f"; {'; '.join(probs)}" if probs else ""))

    # --------------------------------------------------------------------------------------------------- runs
    def c_smoke(self) -> None:
        """smoke.sh проходит за 15 минут со всеми разделами отчёта"""
        log = self.root / "logs/run_all.log"
        t = smoke_timing(log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else [])
        report = self.root / "results/smoke/REPORT.md"
        sections = 0
        if report.exists():
            text = report.read_text(encoding="utf-8")
            sections = sum(1 for i in range(1, 11) if re.search(rf"^## {i}\. ", text, re.M))
        if not t["stages"]:
            self.add(self.c_smoke.__doc__, False, "в logs/run_all.log нет ни одной завершённой стадии смоука")
            return
        rows = t["prescore_cache_rows"]
        cold = "prescore" not in t["stages"] or rows == 0      # a missing prescore already fails as "не измерены"
        ok = not t["missing"] and not t["failed"] and t["sum"] <= SMOKE_LIMIT_S and sections == 10 and cold
        newest = max(ts for ts, _ in t["stages"].values())
        detail = (f"чистый прогон по сумме последних измерений стадий {t['sum']:.0f} с (предел {SMOKE_LIMIT_S} с; "
                  f"последнее измерение {newest}): "
                  + ", ".join(f"{s} {t['stages'][s][1]:.0f}" for s in SMOKE_STAGES if s in t["stages"]))
        if t["last_start"] and (not t["last_total"] or t["last_start"] > t["last_total"][0]):
            # run_all.sh calls this script as its last stage, before it writes the RUN TOTAL line
            detail += f"; текущий запуск {t['last_start']} ещё не завершён (TOTAL не записан)"
        elif t["last_total"]:
            ts, secs, note = t["last_total"]
            detail += f"; последний запуск {ts}: {secs:.0f} с ({note})"
        if t["missing"]:
            detail += f"; не измерены: {t['missing']}"
        if t["failed"]:
            detail += f"; упали: {t['failed']}"
        if "prescore" in t["stages"]:
            cdir = str(self.cfg.default["baselines"]["transformers"]["cache_dir"])
            detail += ("; prescore измерен с пустого кеша оценок" if rows == 0 else
                       f"; prescore измерен при {rows} строках в кеше оценок {cdir}: это не чистая машина" if rows else
                       f"; состояние кеша оценок {cdir} при замере prescore не записано (прогон до этой проверки)")
            if not cold:
                detail += (f" — чистый смоук не измерен: перенесите {cdir} в сторону и повторите scripts/smoke.sh "
                           f"(признак — число строк кеша, консервативно: любые строки считаются тёплым кешем)")
        detail += f"; разделов в results/smoke/REPORT.md: {sections}"
        self.add(self.c_smoke.__doc__, ok, detail)

    def _default_dry_run(self, root: Path, args: Sequence[str]) -> tuple[int, str]:
        script = root / "scripts/run_all.sh"
        if not script.exists():
            return 127, f"{self._rel(script)} нет"
        proc = subprocess.run(["bash", str(script), *args], cwd=root, capture_output=True, text=True, timeout=1800)
        return proc.returncode, proc.stdout + proc.stderr

    def c_resumable(self) -> None:
        """run_all.sh продолжает прерванный прогон: dry-run по готовому дереву ничего не планирует"""
        mode = "smoke" if self.smoke else "real"
        last = last_run_args(self._read(self.root / "logs/run_all.log").splitlines(), mode)
        args = (["--smoke"] if self.smoke else []) + ["--dry-run"]
        if last.get("skip"):
            args += ["--skip", last["skip"]]
        if last.get("seeds") and last["seeds"] != "config":
            args += ["--seeds", last["seeds"]]
        rc, out = (self._dry_run or self._default_dry_run)(self.root, args)
        planned = plan_from_dry_run(out)
        extra = [s for s in planned if s not in ALWAYS_RUN]
        self.add(self.c_resumable.__doc__, rc == 0 and not extra,
                 f"run_all.sh {' '.join(args)}: код {rc}; " + (f"запланированы бы {extra}" if extra else "ничего, кроме "
                                                                f"{[s for s in planned if s in ALWAYS_RUN]}")
                 + ("" if rc == 0 else f"; вывод: {out.strip().splitlines()[-1][:160] if out.strip() else ''}"))
        self.add("run_all.sh на чистой машине проходит без ручных шагов, кроме чек-листа оператора", None,
                 "проверяется только прогоном на чистой машине; здесь проверено продолжение прерванного прогона (строка выше)")

    def c_pytest(self) -> None:
        """pytest проходит"""
        log = self.root / "logs/pytest.log"
        if self.pytest_mode == "skip":
            self.add(self.c_pytest.__doc__, None, "не запускался (--pytest skip)")
            return
        if self.pytest_mode == "log":
            text = self._read(log)
            m = re.search(r"^exit=(\d+)", text, re.M)
            rc = int(m.group(1)) if m else None
        else:
            # addopts cleared: pyproject's "-q" plus ours would be -qq, which drops the "N passed" line parsed below
            proc = subprocess.run([str(self.root / ".venv/bin/python"), "-m", "pytest", "-o", "addopts=", "-q", "-rs",
                                   "-p", "no:cacheprovider"],
                                  cwd=self.root, capture_output=True, text=True, timeout=3600)
            text = proc.stdout + proc.stderr + f"\nexit={proc.returncode}\n"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(text, encoding="utf-8")
            rc = proc.returncode
        c = parse_pytest(text)
        skips = re.findall(r"^SKIPPED \[\d+\] (\S+?):\d+", text, re.M)
        ok = rc == 0 and c["passed"] > 0 and c["failed"] == 0 and c["errors"] == 0
        self.add(self.c_pytest.__doc__, ok, (f"{'по logs/pytest.log: ' if self.pytest_mode == 'log' else ''}код {rc}; "
                                             f"прошло {c['passed']}, упало {c['failed']}, ошибок {c['errors']}, пропущено {c['skipped']}")
                 + (f" ({_few(sorted(set(skips)), 3)})" if skips else ""))

    # ----------------------------------------------------------------------------------------------- manifests
    def c_manifests(self) -> None:
        """манифесты полны и согласованы с таблицами"""
        present = self.present_sources()
        need = ["splits.json", "pools.json", "dedup.json", "audit.md", "contamination.json"]
        if {"dojo", "dyn"} & present:
            need.append("traces_extraction.json")
        missing = [f"{self._rel(self.manifests)}/{n}" for n in need if not (self.manifests / n).exists()]
        missing += [n for n in ("data/manifests/sources.json", "results/shared/traces_manifest.json",
                                "data/paraphrases/paraphrases_manifest.json") if not (self.root / n).exists()]
        if not (self.shared / "split_manifest.json").exists():
            missing.append(self._rel(self.shared / "split_manifest.json"))
        docs = None
        dpath = self.processed / "documents.parquet"
        if dpath.exists():
            import pyarrow.parquet as pq

            self.log_access(dpath, "check_acceptance: manifest completeness (doc_id/source/split/meta columns only, no text)")
            cols = [c for c in ("doc_id", "source", "split", "meta_json") if c in pq.read_schema(dpath).names]
            docs = pq.read_table(dpath, columns=cols).to_pandas()
        else:
            missing.append(self._rel(dpath))
        probs = manifest_problems(_json(self.manifests / "splits.json") or {}, _json(self.manifests / "pools.json") or {},
                                  docs, present)
        self.add(self.c_manifests.__doc__, not missing and not probs,
                 f"источники {sorted(present)}" + (f"; нет файлов {missing}" if missing else "")
                 + (f"; {'; '.join(probs)}" if probs else "; списки совпадают с documents.parquet"))

    def c_dedup(self) -> None:
        """после дедупликации тест не пересекается с обучением/валидацией; группы дублей однометочны; пары BIPIA целы"""
        import pyarrow.parquet as pq

        wpath, dpath = self.processed / "windows.parquet", self.processed / "documents.parquet"
        if not wpath.exists() or not dpath.exists():
            self.add(self.c_dedup.__doc__, False, "windows.parquet / documents.parquet нет")
            return
        purpose = "check_acceptance: dedup invariants (columns id/split/label/cluster/hash/meta only, no text)"
        self.log_access(wpath, purpose)
        self.log_access(dpath, purpose)
        w = pq.read_table(wpath, columns=["window_id", "doc_id", "source", "split", "label", "cluster_id", "text_hash",
                                          "dedup_excluded", "dup_of"]).to_pandas()
        dcols = [c for c in ("doc_id", "source", "split", "label", "cluster_id", "meta_json", "dedup_dropped")
                 if c in pq.read_schema(dpath).names]
        d = pq.read_table(dpath, columns=dcols).to_pandas()
        problems: list[str] = []
        test_docs = d[d["split"] == "test"]
        ref_docs = d[d["split"].isin(["train", "val"])]
        overlap = set(test_docs["cluster_id"]) & set(ref_docs["cluster_id"])
        if overlap:
            problems.append(f"кластеров и в тесте, и в train/val: {len(overlap)}")
        ref_w = w[w["split"].isin(["train", "val"])]
        test_w = w[(w["split"] == "test") & (~w["dedup_excluded"].astype(bool))]
        ref_hash = set(zip(ref_w["text_hash"], ref_w["label"]))
        collisions = sum(1 for h, l in zip(test_w["text_hash"], test_w["label"]) if (h, l) in ref_hash)
        if collisions:
            problems.append(f"точных совпадений text_hash тест↔train/val в одном классе: {collisions}")
        label_of = dict(zip(w["window_id"], w["label"]))
        split_of = dict(zip(w["window_id"], w["split"]))
        excl = w[w["dedup_excluded"].astype(bool)]
        bad_groups = sum(1 for wid, ref in zip(excl["window_id"], excl["dup_of"])
                         if not isinstance(ref, str) or split_of.get(ref) not in ("train", "val") or label_of.get(ref) != label_of[wid])
        if bad_groups:
            problems.append(f"групп дублей с чужой меткой или эталоном вне train/val: {bad_groups}")
        if (excl["split"] != "test").any():
            problems.append("исключены окна вне теста")
        dj = _json(self.manifests / "dedup.json") or {}
        if dj.get("test_windows_excluded") is not None and int(dj["test_windows_excluded"]) != int(len(excl)):
            problems.append(f"dedup.json: исключено {dj['test_windows_excluded']}, в таблице {len(excl)}")
        bip = test_docs[test_docs["source"] == "bipia"]
        if "dedup_dropped" in bip.columns:
            bip = bip[~bip["dedup_dropped"].astype(bool)]
        if "meta_json" in bip.columns and len(bip):
            variant = bip["meta_json"].map(lambda s: (json.loads(s) if isinstance(s, str) and s else {}).get("variant", "main"))
            main = bip[variant == "main"]
            broken = sum(1 for _, g in main.groupby("cluster_id") if sorted(g["label"].tolist()) != [0, 1])
            if broken:
                problems.append(f"контекстов BIPIA без ровно одной чистой и одной атакованной версии: {broken}")
        self.add(self.c_dedup.__doc__, not problems,
                 "; ".join(problems) if problems else f"тестовых окон {len(w[w['split'] == 'test'])}, исключено {len(excl)}, "
                                                      f"кластеров теста {test_docs['cluster_id'].nunique()}")

    def c_kc(self) -> None:
        """средняя входящая степень KC измеренной M в 4–8; число KC записано"""
        from flyguard.connectome import indegree_stats, load_malecns

        path = self.root / "data/processed/connectome/malecns_R.npz"
        if not path.exists():
            self.add(self.c_kc.__doc__, False, "malecns_R.npz нет")
            return
        M, _ = load_malecns(path)
        st = indegree_stats(M)
        setup = _json(self.rdir / "setup.json") or {}
        recorded = (setup.get("connectome") or {}).get("kenyon_cells") == st["n_cells"]
        self.add(self.c_kc.__doc__, bool(st["in_range"]) and recorded,
                 f"KC {st['n_cells']}, входов {st['n_inputs']}, средняя степень {st['indegree_mean']:.3f} в {st['check_range']}: "
                 f"{st['in_range']}; записано в setup.json: {recorded}")

    # ---------------------------------------------------------------------------------------------- E0 / verdicts
    def c_power(self) -> None:
        """power.json заморожен до E1, финальный и использован в вердиктах"""
        power = _json(R.power_path(self.root, self.smoke))
        e1 = {k: v for k, v in self.seed_files.items() if "/E1/" in k}
        self.checks.append(power_before_e1(power, e1, self.current))
        v = _json(self.rdir / "verdicts.json")
        missing = carriers_missing(v) if v else ["verdicts.json нет"]
        self.add("вердикты используют таблицу носителей E0", not missing,
                 "флаг носителя E0 у H1a, H1b и H3" if not missing else f"без флага носителя: {missing}")
        name = "финальный power.json (после трасс и парафраз) — тот, по которому вынесены вердикты"
        if not power or not v:
            self.add(name, False, "power.json или verdicts.json нет")
            return
        probs = []
        sp = _json(self.manifests / "splits.json") or {}
        test_sources = {s for s, ids in ((sp.get("e1") or {}).get("test") or {}).items() if ids}
        if not test_sources or not test_sources <= set(power.get("sources") or []):
            probs.append(f"источники power.json {sorted(power.get('sources') or [])} не покрывают тест E1 {sorted(test_sources)}")
        ref = (v.get("sources") or {}).get("power") or {}
        for k in ("config_hash", "created_at", "stage"):
            if ref.get(k) != power.get(k):
                probs.append(f"verdicts.sources.power.{k}={ref.get(k)} ≠ power.json {power.get(k)}")
        if v.get("power_frozen") is not True or v.get("power_stage") != 2:
            probs.append(f"verdicts: power_frozen={v.get('power_frozen')}, power_stage={v.get('power_stage')}")
        mism = carrier_mismatches(v, power)
        if mism:
            probs.append(f"флаги носителей не совпадают с таблицей E0: {_few(mism)}")
        self.add(name, not probs, (f"источники {sorted(power.get('sources') or [])}; ссылка verdicts→power.json и флаги носителей совпадают"
                                   if not probs else "; ".join(probs)))

    def c_windows(self) -> None:
        """одна схема окон для всех детекторов; 512-токенные окна только в E6"""
        win = self.cfg.default.get("windows") or {}
        size, stride = win.get("size"), win.get("stride")
        probs = []
        if not (isinstance(size, int) and isinstance(stride, int) and 0 < stride <= size):
            probs.append(f"windows.size/stride {size}/{stride}")
        wpath = self.processed / "windows.parquet"
        n = 0
        if wpath.exists() and not probs:
            import pyarrow.parquet as pq

            self.log_access(wpath, "check_acceptance: window scheme (start/end columns only, no text)")
            t = pq.read_table(wpath, columns=["start", "end"]).to_pandas()
            n = len(t)
            off = int(((t["start"] % stride != 0) | (t["end"] - t["start"] > size) | (t["end"] < t["start"])).sum())
            if off:
                probs.append(f"окон вне схемы {size}/{stride}: {off}")
        elif not wpath.exists():
            probs.append("windows.parquet нет")
        flags = {e: bool(c.get("transformer_windows_512_tokens")) for e, c in self.cfg.experiments.items()}
        if any(v for e, v in flags.items() if e != "E6"):
            probs.append(f"флаг 512 токенов вне E6: {flags}")
        offenders = []
        for rname, res in self.seed_files.items():
            if "/E6/" in rname:
                continue
            texts = (list((res.get("numbers") or {}).keys()) + list((res.get("tables") or {}).keys())
                     + [str(x) for x in res.get("notes") or []])
            if any(TOK512_RE.search(t) for t in texts):
                offenders.append(rname)
        if offenders:
            probs.append(f"512-токенные ключи вне E6: {offenders}")
        self.add(self.c_windows.__doc__, not probs, f"окна {size}/{stride}, проверено {n}; флаг 512 токенов только в E6"
                 + (f"; {'; '.join(probs)}" if probs else ""))

    def c_thresholds(self) -> None:
        """пороги полны"""
        files: dict[str, dict[str, Any]] = dict(self.seed_files)
        c = _json(self.rdir / "contract.json")
        if c:
            files[self._rel(self.rdir / "contract.json")] = c
        if not self.seed_files:
            self.add("у каждого порога записаны значение, источник, цель и n", False, "результатов нет")
            return
        chk = thresholds_complete(files)
        empty = sorted(k for k, v in self.seed_files.items() if "/E1/" in k and not v.get("thresholds"))
        if empty:
            chk = Check(chk.name, False, chk.detail + f"; файлы E1 без порогов: {empty}")
        self.checks.append(chk)

    def c_verdicts(self) -> None:
        """вердикты по правилам этапа 4 с предусловиями; macroAUC — основа вердиктов"""
        v = _json(self.rdir / "verdicts.json")
        if not v:
            self.add(self.c_verdicts.__doc__, False, "verdicts.json нет")
            return
        import make_report as mr

        items = list(mr.iter_verdicts(v))
        names = {n.split("/")[0] for n, _ in items}
        probs = [f"статус {n}={d.get('status')}" for n, d in items if d.get("status") not in STATUSES]
        probs += [f"нет вердикта {h}" for h in ("H1a", "H1b", "H2", "H3") if h not in names]
        for hyp, keys in (("H2", ("precondition", "val_auc_deep")), ("H3", ("precondition", "val_macro_auc"))):
            for s, row in _seed_rows(v.get(hyp) or {}):
                if any((row.get("inputs") or {}).get(k) is None for k in keys):
                    probs.append(f"{hyp} сид {s}: предусловие не записано ({keys})")
        probs += verdict_basis_problems(v, self.summaries())
        self.add(self.c_verdicts.__doc__, not probs,
                 "статусы: " + ", ".join(f"{n}={d.get('status')}" for n, d in items)
                 + (f"; {'; '.join(probs[:6])}" + (f" и ещё {len(probs) - 6}" if len(probs) > 6 else "") if probs
                    else "; входы H1b/H3 — ключи macroAUC, эффекты совпадают со сводками"))

    def c_h3(self) -> None:
        """перестановка π разыграна от сида и входит в двухступенчатый бутстреп H3"""
        d = self.cfg.default
        seeds = [int(s) for s in d["seeds"]["global"]]
        n_null = int(d["smoke"]["n_null"]) if self.smoke else int(d["expansion"]["curveball"]["n_null"])
        n_boot = int(d["smoke"]["bootstrap"]) if self.smoke else int(d["stats"]["bootstrap"]["n"])
        perm = {int(seeds_for(self.cfg, s)["perm"]) for s in seeds}
        e4 = {k: v for k, v in self.seed_files.items() if "/E4/" in k}
        probs = h3_perm_problems(e4, len(seeds), perm, n_null, n_boot)
        yaml_n = (self.cfg.experiments.get("E4") or {}).get("n_curveball")
        if yaml_n is not None and int(yaml_n) != int(d["expansion"]["curveball"]["n_null"]):
            probs.append(f"E4.yaml n_curveball={yaml_n} ≠ expansion.curveball.n_null")
        v = _json(self.rdir / "verdicts.json") or {}
        if ((v.get("inputs") or {}).get("H3") or {}).get("primary") != H3_PRIMARY:
            probs.append(f"вердикт H3 вынесен не по {H3_PRIMARY}")
        self.add(self.c_h3.__doc__, not probs, f"файлов E4 {len(e4)}; ожидается π {len(seeds)} (дочерние perm), нулей {n_null}, "
                 f"розыгрышей {n_boot}" + (f"; {'; '.join(probs[:5])}" if probs else "; всё совпадает"))

    def c_bloom(self) -> None:
        """Bloom обучен на сбалансированных классах"""
        balance = ((self.cfg.default.get("readout") or {}).get("bloom") or {}).get("balance")
        recs = bloom_balance_records(self.seed_files)
        unequal = [r for r in recs if r[1] != r[2] or r[1] <= 0]
        detail = f"readout.bloom.balance={balance}"
        if balance != "subsample_to_minority" or unequal:
            self.add(self.c_bloom.__doc__, False, detail + (f"; неравные классы: {_few([f'{a} {b}/{c}' for a, b, c in unequal])}" if unequal else ""))
        elif not recs:
            self.add(self.c_bloom.__doc__, None, detail + "; фактические числа классов подгонки Bloom в результатах не записаны "
                     "(таблица detectors без balanced_n0/balanced_n1): подтверждено только конфигом")
        else:
            self.add(self.c_bloom.__doc__, True, detail + f"; подгонок Bloom {len(recs)}, у всех классы поровну")

    def c_comparator(self) -> None:
        """компаратор ProtectAI v2 назначен в конфиге до теста"""
        comp = (self.cfg.default.get("baselines") or {}).get("transformers", {}).get("comparator")
        first = git(self.root, "log", "--reverse", "--format=%cI", "-S", "comparator: protectai_v2", "--", "configs/default.yaml")
        first_ts = _utc(first.splitlines()[0]) if first else None
        entries = parse_access_log(self._read(self.root / "logs/data_access.log").splitlines())
        tests = min((e["ts"] for e in entries if e["split"] == "test"), default=None)
        before = first_ts is not None and (tests is None or first_ts < tests)
        probs = []
        v = _json(self.rdir / "verdicts.json")
        if v:
            if v.get("comparator") != comp:
                probs.append(f"verdicts.comparator={v.get('comparator')}")
            inputs = v.get("inputs") or {}
            used = [str(k) for h, g in inputs.items() if h.split("/")[0] in ("H1a", "H2") and isinstance(g, dict) for k in g.values()]
            if not any(comp and comp in k for k in used):
                probs.append("входы H1a/H2 не используют компаратор")
        self.add(self.c_comparator.__doc__, comp == "protectai_v2" and before and not probs,
                 f"comparator={comp}; назначен в коммите от {first_ts.strftime('%Y-%m-%dT%H:%M:%SZ') if first_ts else 'н/д'}"
                 + (f", первое чтение теста {tests.strftime('%Y-%m-%dT%H:%M:%SZ')}" if tests else "")
                 + (f"; {'; '.join(probs)}" if probs else ""))

    def c_config_hash(self) -> None:
        """хеш конфига в результатах совпадает с текущим"""
        files: dict[str, dict[str, Any] | None] = dict(self.seed_files)
        for name in ("power.json", "contract.json", "verdicts.json", "setup.json"):
            p = self.rdir / name
            if p.exists():
                files[self._rel(p)] = _json(p)
        for e, s in self.summaries().items():
            files[self._rel(R.summary_path(e, self.smoke, self.root))] = s
        v = files.get(self._rel(self.rdir / "verdicts.json")) or {}
        for k, src in (v.get("sources") or {}).items():
            if isinstance(src, dict) and "config_hash" in src:
                files[f"verdicts.json#sources/{k}"] = src
        bad = config_hash_mismatches(files, self.current)
        self.add(self.c_config_hash.__doc__, bool(files) and not bad,
                 f"файлов {len(files)}, расходятся: {bad[:6]}" if bad else f"файлов {len(files)}, хеш {self.current[:12]}…" if files else "файлов результатов нет")

    def c_seeds(self) -> None:
        """у каждого выполненного эксперимента E1–E6 все сиды конфига"""
        seeds = [int(s) for s in self.cfg.default["seeds"]["global"]]
        if self.smoke:
            seeds = seeds[: int(self.cfg.default["smoke"]["seeds"])]
        self.checks.append(seed_coverage(self.seed_files, seeds))

    def _same_code(self, a: str, b: str) -> bool | None:
        try:
            rc = subprocess.run(["git", "diff", "--quiet", a, b, "--", *CODE_PATHS], cwd=self.root,
                                capture_output=True).returncode
        except FileNotFoundError:
            return None
        return {0: True, 1: False}.get(rc)

    def c_provenance(self) -> None:
        """результаты посчитаны одним кодом"""
        files: dict[str, dict[str, Any] | None] = dict(self.seed_files)
        for name in ("power.json", "contract.json", "verdicts.json"):
            p = self.rdir / name
            if p.exists():
                files[self._rel(p)] = _json(p)
        self.checks.append(code_provenance(files, self._same_code, git(self.root, "rev-parse", "HEAD")))

    def c_deviations(self) -> None:
        """DEVIATIONS.md перечисляет каждое отклонение"""
        name = self.c_deviations.__doc__
        entries = self.dev_entries
        probs = journal_problems(entries) if entries else ["DEVIATIONS.md без записей"]
        cited: set[str] = set()
        docs = [json.dumps(r.get("notes") or [], ensure_ascii=False) + str(r.get("training_mode", "")) for r in self.seed_files.values()]
        for fname in ("verdicts.json", "contract.json", "power.json"):
            docs.append(json.dumps(_json(self.rdir / fname) or {}, ensure_ascii=False))
        for text in docs:
            cited |= set(re.findall(r"\bD(\d+)\b", text))
        probs += [f"D{c} упомянут в результатах, но не описан" for c in sorted(cited, key=int) if f"D{c}" not in entries]
        # deviations visible in the artefacts; each must have a journal entry matching the patterns
        op = self.cfg.operator.get("llm_api") or {}
        para = op.get("paraphrase") or {}
        pilot = _json(self.root / "results/pilot.json") or {}
        pools = _json(self.manifests / "pools.json") or {}
        splits = _json(self.manifests / "splits.json") or {}
        run = _json(self.root / "results/traces_run.json") or {}
        detected = [
            ("один провайдер LLM вместо двух", len(op.get("providers") or []) < 2, (r"провайдер",)),
            ("меньше двух генераторов/судей парафраз", len(para.get("judges") or []) < 2 or len(para.get("generators") or []) < 2,
             (r"судь|судей", r"генератор")),
            ("модель агента выбрана не по правилу пилота", bool(pilot) and pilot.get("chosen_by_rule") is False, (r"пилот",)),
            ("срезы трасс по бюджету", any(it.get("skipped") for it in run.get("items") or []), (r"срез|урез", r"бюджет")),
            ("пулы негативов меньше цели", any((pools.get(p) or {}).get("meets_target") is False for p in ("p_val", "p_test")),
             (r"пул", r"цел")),
            ("фолды E3 без шаблонов", bool((splits.get("e3") or {}).get("templates_missing")), (r"E3",)),
            ("перезаморозка power.json (A44)", not self.smoke and any((self.root / "results/E0").glob("power_frozen_*.json")),
             (r"power\.json|перезамор",)),
        ]
        ran = {Path(k).parent.name for k in self.seed_files}
        detected += [(f"{e} не выполнен (порядок отрезания ТЗ)", bool(ran) and e not in ran, (rf"\b{e}\b", r"отрез|срез|не выполн"))
                     for e in ("E2", "E3", "E5", "E6")]
        for what, present, pats in detected:
            if present and not journaled(entries, *pats):
                probs.append(f"не записано: {what}")
        self.add(name, not probs, f"записей {len(entries)}; обнаруживаемых отклонений {sum(1 for _, p, _ in detected if p)}"
                 + (f"; {'; '.join(probs)}" if probs else ", все записаны; у каждой записи дата и влияние"))

    def c_report(self) -> None:
        """каждое число REPORT.md прослеживается до results/; разделы 1–10 на месте"""
        import make_report as mr

        report = (self.rdir / "REPORT.md") if self.smoke else (self.root / "REPORT.md")
        if not report.exists():
            self.add(self.c_report.__doc__, False, f"{self._rel(report)} нет")
            return
        text = report.read_text(encoding="utf-8")
        sections = [i for i in range(1, 11) if not re.search(rf"^## {i}\. ", text, re.M)]
        n, problems = mr.trace_numbers(text, self.root)
        self.add(self.c_report.__doc__, not problems and not sections and n > 0,
                 f"чисел проверено {n}, не прослеживаются {len(problems)}" + (f": {problems[:3]}" if problems else "")
                 + (f"; нет разделов {sections}" if sections else ""))

    def c_contract(self) -> None:
        """контракт: CSV по схеме, строка на каждый тестовый эпизод, split_manifest по crc32 mod 3 = 2, important_instructions вне обучения, манифесты"""
        from flyguard.agentdojo_io.contract import VARIANTS, read_csv, validate_csv, validate_split_manifest

        csv_path = self.shared / "flyguard.csv"
        rule = self.cfg.default["splits"]["contract"]
        mod, rem, attack = int(rule["mod"]), int(rule["rem"]), str(rule["test_attack"])
        probs = validate_csv(csv_path) if csv_path.exists() else ["flyguard.csv нет"]
        sm = _json(self.shared / "split_manifest.json")
        if sm:
            probs += validate_split_manifest(sm) + split_rule_violations(sm, mod, rem, attack)
        else:
            probs.append("split_manifest.json нет")
        c = _json(self.rdir / "contract.json")
        if not c:
            probs.append("contract.json нет")
        if sm and csv_path.exists():
            val_ids = sorted({str(e.get("episode_id")) for e in (c or {}).get("validation_episodes") or [] if isinstance(e, dict)})
            probs += contract_problems(read_csv(csv_path), sm, VARIANTS, val_ids, mod, rem, attack)
        if not (self.root / "results/shared/traces_manifest.json").exists():
            probs.append("traces_manifest.json нет")
        self.add(self.c_contract.__doc__, not probs,
                 f"тестовых эпизодов {len((sm or {}).get('test') or [])}, варианты {list(VARIANTS)}"
                 + (f"; {'; '.join(str(p) for p in probs[:5])}" if probs else "; всё сходится"))
        self.add("манифесты совпадают у обеих команд", None, "проверяется второй командой по копии в results/shared")

    def c_regex(self) -> None:
        """regex_patterns.txt закоммичен до первого чтения теста"""
        rel = "configs/regex_patterns.txt"
        stamp = git(self.root, "log", "-1", "--format=%cI", "--", rel)
        commit_time = _utc(stamp) if stamp else None
        dirty = git(self.root, "status", "--porcelain", "--", rel)
        dirty = bool(dirty) if dirty is not None else True
        entries = parse_access_log(self._read(self.root / "logs/data_access.log").splitlines())
        self.checks.append(regex_order(commit_time, dirty, entries, self.deviations))


# ================================================================================================= main
def run_checks(root: Path = ROOT, smoke: bool = False, pytest_mode: str = "run",
               access_log: Callable[[Path, str, str], None] | None = None, only: Sequence[str] | None = None,
               dry_run: Callable[[Path, Sequence[str]], tuple[int, str]] | None = None) -> list[Check]:
    return Checker(root, smoke, pytest_mode, access_log, only, dry_run).run()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Machine checks of ТЗ «Критерии приёмки».")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--pytest", choices=("run", "log", "skip"), default="run")
    ap.add_argument("--only", default="", help="comma list of check methods (c_network, c_regex, ...) to run alone")
    args = ap.parse_args(argv)
    checks = run_checks(Path(args.root), args.smoke, args.pytest, only=[x for x in args.only.split(",") if x])
    failed = [c for c in checks if c.ok is False]
    print(f"Критерии приёмки ({'смоук' if args.smoke else 'финальный прогон'}), {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    for c in checks:
        print(f"{c.mark} {c.name}" + (f" — {c.detail}" if c.detail else ""))
    print(f"\nитог: {len(checks) - len(failed) - sum(1 for c in checks if c.ok is None)} ✅, {len(failed)} ❌, "
          f"{sum(1 for c in checks if c.ok is None)} ⚠")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
