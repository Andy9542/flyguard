#!/usr/bin/env python
"""scripts/check_acceptance.py — machine checks of ТЗ «Критерии приёмки» (docs/design_experiments.md §4).

Usage::

    .venv/bin/python scripts/check_acceptance.py                 # real results; runs pytest; exit 1 on any ❌
    .venv/bin/python scripts/check_acceptance.py --smoke         # results/smoke, data/manifests/smoke, results/smoke/REPORT.md
    .venv/bin/python scripts/check_acceptance.py --pytest log    # take the last pytest verdict from logs/pytest.log
    .venv/bin/python scripts/check_acceptance.py --pytest skip   # do not run pytest (reported as ⚠)

Prints one line per criterion, ``✅`` / ``❌`` (``⚠`` for information that is not a criterion), and exits non-zero
when any ❌ is present. The checks are the machine-checkable parts of the list; what cannot be checked here (the
second team's copy of the manifests, "run_all.sh on a clean machine") is printed as ⚠ with the reason.

What each check reads and why (criterion in quotes):

* "fetch.sh скачал всё с проверкой sha256, включая flypath build" — ``data/manifests/sources.json``: a sha256 per
  file, the ``malecns_R.npz`` entry built by ``flypath build``, the pins.
* "network.log содержит только скачивания и два разрешённых потока" — every host of ``logs/network.log`` is a
  download host (URLs of ``sources.json``, the package indexes and code hosts of ``setup_env.sh``/``fetch.sh``) or a
  provider host from ``configs/operator.yaml`` (``llm_api.providers[].base_url``).
* "Модель агента выбрана пилотом; пилоты всех кандидатов записаны" — ``results/pilot.json`` has every
  ``llm_api.agent_models`` candidate; a choice outside the rule needs its DEVIATIONS entry (D5).
* "Трассы заморожены" — ``flyguard.gen.traces.verify_frozen`` re-hashes every log of ``traces_manifest.json``;
  the copy for the second team exists under ``shared.traces_out_dir``.
* "Парафразы заморожены" — ``paraphrases_manifest.json`` (models, dates, prompt hashes, rates); every positive of
  ``paraphrases.csv`` has Jaccard ≤ ``paraphrase.filters.jaccard_max`` and a judge verdict (single judge: D2); no
  banned word of ``banned_words_final.json`` in the deep stratum. The CSV texts are read by code only for the banned
  word test, never printed, and the read is journaled through ``log_data_access`` (the paraphrases are test material).
* "spend.json ≤ budget" — ``results/spend.json`` against ``llm_api.budget_usd``.
* "smoke.sh за 15 минут со всеми разделами" — ``logs/run_all.log``: the time of a *clean* smoke run is the sum,
  over the stages of ТЗ steps 2–13 (``build_stage1`` … ``report``), of the most recent smoke ``done`` duration of
  each stage (:func:`smoke_timing`); ``run_all.sh`` is idempotent, so the ``TOTAL`` of the last run (also printed)
  understates a clean run whenever a stage was skipped as done, and an interrupted run has no ``TOTAL`` at all. A
  stage never measured or whose latest smoke event is ``fail`` fails the criterion; the ten section headings of
  ``results/smoke/REPORT.md`` must be present.
* "pytest проходит" — the suite is run here (``--pytest run``, default) or its last log is read.
* "Манифесты полны; тест не пересекается с обучением; ни один кластер не содержит обе метки; пары BIPIA целы" —
  manifests present; from the parquet tables (id, split, label, cluster, hash and meta columns only, journaled as a
  test read): test clusters are disjoint from train/val clusters, no non-excluded test window shares a ``text_hash``
  with a train/val window of the same class, every dedup group (an excluded window and its ``dup_of``) is label-pure
  and points into train/val (the reading of "no cluster contains both labels" fixed by ``tests/data/test_dedup.py``:
  BIPIA clusters hold a clean and an attacked document by design), and every main-variant BIPIA test context keeps
  exactly one clean and one attacked document.
* "Средняя степень KC измеренной M в 4–8; число KC записано" — ``flyguard.connectome.indegree_stats``.
* "power.json записан до финального прогона и использован в вердиктах" — ``results/power.json`` frozen (stage 2)
  with the current config hash, its ``created_at`` earlier than every ``results/E1/<seed>.json``; ``verdicts.json``
  carries the carrier flags.
* "Одна схема окон; 512-токенные окна только в E6" — one ``windows`` block in the config, the 512-token flag only in
  ``E6.yaml``, no 512-token key in the E1–E5 results.
* "У каждого порога записаны источник, целевая точка и число примеров" — every ``thresholds`` record of every result.
* "macroAUC основа вердиктов" and "π в двухступенчатом бутстрепе H3; Bloom на сбалансированных классах" —
  ``verdicts.json`` inputs, the E4 keys, ``readout.bloom.balance``.
* "Компаратор ProtectAI v2 назначен в конфиге до теста" — the config value and the git date of its introduction.
* "Хеш конфига в результатах совпадает; DEVIATIONS перечисляет отклонения" — every results file against
  ``flyguard.config.config_hash``; the journal has entries and every ``D<n>`` cited in results exists.
* "Каждое число REPORT.md прослеживается; вердикты с предусловиями" — ``make_report.trace_numbers`` over the
  rendered report; the statuses and precondition inputs in ``verdicts.json``.
* "Контракт" — ``agentdojo_io.contract.validate_csv``; ``split_manifest.json`` re-checked against
  ``crc32(user_task) mod 3 == 2`` and the ``important_instructions`` exclusion; both manifests present.
* "Коммит с regex_patterns.txt старше первого чтения тестовых файлов" — ``git log`` date of the last commit touching
  the file (UTC) against the first ``test`` line of ``logs/data_access.log``; an uncommitted change fails. Entries
  marked "retroactive" / "after the fact" (DEVIATIONS D10) are listed so the real order is visible next to the formal
  one; they are accepted only when the journal explains them.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from flyguard.config import ROOT, config_hash, load_configs  # noqa: E402
from flyguard.data.build import output_dirs  # noqa: E402
from flyguard.experiments import results as R  # noqa: E402

OK, FAIL, INFO = "✅", "❌", "⚠"
STATUSES = ("подтверждена", "опровергнута", "не хватило данных", "предусловие не выполнено")
TOOLCHAIN_HOSTS = {"pypi.org", "files.pythonhosted.org", "download.pytorch.org", "github.com", "raw.githubusercontent.com",
                   "huggingface.co", "storage.googleapis.com"}   # setup_env.sh / fetch.sh downloads (MaleCNS via flypath build)
SMOKE_LIMIT_S = 15 * 60
RETRO_RE = re.compile(r"retroactiv|after the fact|задним числом", re.I)
TOK512_RE = re.compile(r"512[_-]?tok|tok(?:ens?)?[_-]?512|win(?:dows?)?[_-]?512", re.I)


@dataclass
class Check:
    name: str
    ok: bool | None          # None = informational
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


def _utc(ts: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


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


SMOKE_STAGES = ("build_stage1", "e0_stage1", "build_full", "e0_stage2", "e1", "e4", "e5", "e3", "e2", "e6",
                "contract", "verdicts", "report")
"""``run_all.sh`` stages of ТЗ "Бюджет времени" steps 2–13 (the ``check`` stage is this script and is not a step)."""


def smoke_timing(lines: Iterable[str]) -> dict[str, Any]:
    """Reconstruct the duration of a clean smoke run from ``logs/run_all.log`` lines
    (``ts\\tmode\\tstage\\tstatus\\tseconds\\tnote``).

    Returns ``{"last_total": (ts, seconds, note) | None, "stages": {stage: (ts, seconds)} (the most recent smoke
    ``done`` of each stage), "sum": seconds over the measured stages, "missing": stages never measured, "failed":
    stages whose most recent smoke event is ``fail``}``. Skipped (``skip``) and interrupted (``start`` without
    ``done``) events measure nothing; only ``mode == smoke`` lines are read.
    """
    stages: dict[str, tuple[str, float]] = {}
    latest: dict[str, str] = {}
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
            if status in ("done", "fail"):
                latest[stage] = status
            if status == "done":
                stages[stage] = (ts, seconds)
    return {"last_total": last_total, "last_start": last_start, "stages": stages,
            "sum": float(sum(v[1] for v in stages.values())),
            "missing": [s for s in SMOKE_STAGES if s not in stages],
            "failed": [s for s in SMOKE_STAGES if latest.get(s) == "fail"]}


def regex_order(commit_time: datetime | None, dirty: bool, entries: Sequence[dict[str, Any]], journal_text: str) -> Check:
    """The regex criterion: the file's last commit precedes the first journaled test read; retroactive entries are
    shown as the real order and accepted only when the journal (DEVIATIONS) explains them."""
    tests = sorted((e for e in entries if e["split"] == "test"), key=lambda e: e["ts"])
    if commit_time is None:
        return Check("regex_patterns.txt закоммичен до первого чтения теста", False, "git-дата коммита файла недоступна")
    if dirty:
        return Check("regex_patterns.txt закоммичен до первого чтения теста", False, "файл изменён и не закоммичен")
    if not tests:
        return Check("regex_patterns.txt закоммичен до первого чтения теста", True,
                     f"коммит {commit_time.isoformat()}, чтений теста в журнале ещё нет")
    first = tests[0]
    formal = commit_time < first["ts"]
    retro = [e for e in tests if e["retro"]]
    detail = f"коммит {commit_time.strftime('%Y-%m-%dT%H:%M:%SZ')} < первое чтение теста {first['ts'].strftime('%Y-%m-%dT%H:%M:%SZ')} ({first['path']})"
    if retro:
        explained = bool(RETRO_RE.search(journal_text)) or "D10" in journal_text
        detail += (f"; по существу: {len(retro)} записей журнала помечены как ретроспективные (чтения до коммита, "
                   f"первая помечена {retro[0]['ts'].strftime('%Y-%m-%dT%H:%M:%SZ')} {retro[0]['path']}); "
                   + ("объяснение в DEVIATIONS (D10)" if explained else "объяснения в DEVIATIONS нет"))
        return Check("regex_patterns.txt закоммичен до первого чтения теста", formal and explained, detail)
    return Check("regex_patterns.txt закоммичен до первого чтения теста", formal, detail)


def thresholds_complete(results: dict[str, dict[str, Any]]) -> Check:
    """Every threshold record of every results file carries value, source, target and n (ТЗ 2.5)."""
    bad: list[str] = []
    n = 0
    for name, res in results.items():
        for key, rec in (res.get("thresholds") or {}).items():
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
    frozen = power.get("stage") == 2 or power.get("frozen") is True
    hash_ok = power.get("config_hash") == current
    stamp = next((power.get(k) for k in ("created_at", "frozen_at", "generated", "written_at") if power.get(k)), None)
    p_ts = _utc(str(stamp)) if stamp else None
    e1_ts = [_utc(str(d.get("created_at"))) for d in e1_files.values() if d.get("created_at")]
    e1_ts = [t for t in e1_ts if t is not None]
    order_ok = True
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
                 access_log: Callable[[Path, str, str], None] | None = None, only: Sequence[str] | None = None) -> None:
        self.root, self.smoke, self.pytest_mode = Path(root), bool(smoke), pytest_mode
        self.only = set(only or ())
        self.cfg = load_configs(self.root)
        self.rdir = R.results_dir(self.root, self.smoke)
        self.processed, self.manifests = output_dirs(self.root, self.smoke)
        self.current = config_hash(self.root)
        self.checks: list[Check] = []
        self._access_log = access_log
        self.seed_files: dict[str, dict[str, Any]] = {}
        for exp in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
            for p in R.list_results(exp, self.smoke, self.root):
                d = _json(p)
                if d is not None:
                    self.seed_files[str(p.relative_to(self.root))] = d
        self.deviations = self._read(self.root / "DEVIATIONS.md")

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def log_access(self, path: Path, purpose: str) -> None:
        if self._access_log is not None:
            self._access_log(path, "test", purpose)
        else:
            from flyguard.netlog import log_data_access

            log_data_access(path, split="test", purpose=purpose)

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    def run(self) -> list[Check]:
        for fn in (self.c_sources, self.c_network, self.c_pilot, self.c_traces, self.c_paraphrases, self.c_spend,
                   self.c_smoke, self.c_pytest, self.c_manifests, self.c_dedup, self.c_kc, self.c_power,
                   self.c_windows, self.c_thresholds, self.c_verdicts, self.c_h3, self.c_comparator,
                   self.c_config_hash, self.c_deviations, self.c_report, self.c_contract, self.c_regex):
            if self.only and fn.__name__ not in self.only:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check, not a crashed checklist
                self.add(fn.__doc__ or fn.__name__, False, f"ошибка проверки: {type(exc).__name__}: {exc}")
        return self.checks

    # ---------------------------------------------------------------------------------------------------------
    def c_sources(self) -> None:
        """sources.json: sha256 у каждого файла, MaleCNS через flypath build"""
        src = _json(self.root / "data/manifests/sources.json")
        if not src:
            self.add(self.c_sources.__doc__, False, "data/manifests/sources.json нет")
            return
        files = src.get("files") or []
        no_sha = [f.get("path") for f in files if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", "")))]
        malecns = [f for f in files if str(f.get("path", "")).endswith("malecns_R.npz")]
        flypath = any("flypath build" in str(f.get("url", "")) for f in malecns)
        self.add(self.c_sources.__doc__, not no_sha and bool(files) and flypath and bool(src.get("pins")),
                 f"файлов {len(files)}, без sha256 {len(no_sha)}, malecns через flypath build: {flypath}, pins: {sorted((src.get('pins') or {}))}")

    def c_network(self) -> None:
        """network.log: только скачивания и провайдеры из конфига"""
        src = _json(self.root / "data/manifests/sources.json") or {}
        allowed = set(TOOLCHAIN_HOSTS)
        for f in src.get("files") or []:
            host = urlsplit(str(f.get("url", ""))).hostname
            if host:
                allowed.add(host.lower())
        providers = (self.cfg.operator.get("llm_api") or {}).get("providers") or []
        prov_hosts = {urlsplit(str(p.get("base_url", ""))).hostname for p in providers} - {None}
        allowed |= {h.lower() for h in prov_hosts}
        log = self.root / "logs/network.log"
        if not log.exists():
            self.add(self.c_network.__doc__, False, "logs/network.log нет")
            return
        with open(log, encoding="utf-8", errors="replace") as fh:
            bad = hosts_outside(fh, allowed)
        self.add(self.c_network.__doc__, not bad,
                 ("посторонние хосты: " + ", ".join(f"{h} ({n})" for h, n in sorted(bad.items()))) if bad
                 else f"провайдеры: {sorted(h for h in prov_hosts)}; хостов скачивания разрешено {len(allowed) - len(prov_hosts)}")

    def c_pilot(self) -> None:
        """пилот: все кандидаты модели агента записаны, выбор по правилу или через DEVIATIONS"""
        pilot = _json(self.root / "results/pilot.json")
        wanted = list((self.cfg.operator.get("llm_api") or {}).get("agent_models") or [])
        if not pilot:
            self.add(self.c_pilot.__doc__, False, "results/pilot.json нет")
            return
        have = {c.get("model") for c in pilot.get("candidates") or []}
        missing = [m for m in wanted if m not in have]
        by_rule = bool(pilot.get("chosen_by_rule"))
        justified = by_rule or ("D5" in self.deviations and "пилот" in self.deviations.lower())
        self.add(self.c_pilot.__doc__, not missing and justified and bool(pilot.get("chosen_model")),
                 f"выбрана {pilot.get('chosen_model')}, по правилу: {by_rule}" + (f", нет пилотов: {missing}" if missing else "")
                 + ("" if by_rule else ", объяснение: DEVIATIONS D5" if justified else ", объяснения в DEVIATIONS нет"))

    def c_traces(self) -> None:
        """трассы заморожены: манифест, повторный хеш, копия для второй команды"""
        mpath = self.root / "results/shared/traces_manifest.json"
        m = _json(mpath)
        if not m:
            self.add(self.c_traces.__doc__, False, f"{mpath.relative_to(self.root)} нет")
            return
        from flyguard.gen.traces import verify_frozen

        bad = verify_frozen(self.cfg)
        out_dir = self.root / str((self.cfg.operator.get("shared") or {}).get("traces_out_dir") or "results/shared/traces")
        n_copy = sum(1 for _ in out_dir.rglob("*.json")) if out_dir.exists() else 0
        n_files = len(m.get("files") or [])
        self.add(self.c_traces.__doc__, not bad and n_files > 0 and n_copy >= n_files,
                 f"логов {n_files}, расхождений хеша {len(bad)}, копий в {out_dir.relative_to(self.root)}: {n_copy}")

    def c_paraphrases(self) -> None:
        """парафразы заморожены: манифест, Жаккар ≤ порога, судья, нет запрещённых слов в глубокой страте"""
        pdir = self.root / "data/paraphrases"
        m = _json(pdir / "paraphrases_manifest.json")
        csv_path = pdir / "paraphrases.csv"
        if not m or not csv_path.exists():
            self.add(self.c_paraphrases.__doc__, False, "paraphrases_manifest.json или paraphrases.csv нет")
            return
        import csv as csvmod

        f = self.cfg.default["paraphrase"]["filters"]
        jmax, dmax = float(f["jaccard_max"]), float(f["deep_max"])
        self.log_access(csv_path, "check_acceptance: Jaccard/stratum/judge columns and banned-word scan of the deep "
                                  "stratum (texts read by code only, never printed)")
        with open(csv_path, encoding="utf-8", newline="") as fh:
            rows = list(csvmod.DictReader(fh))
        pos = [r for r in rows if str(r.get("label")) == "1"]
        j_bad = [r["para_id"] for r in pos if float(r.get("jaccard_to_base") or 1.0) > jmax]
        deep = [r for r in rows if r.get("stratum") == "deep"]
        d_bad = [r["para_id"] for r in deep if float(r.get("jaccard_to_base") or 1.0) > dmax]
        judged = all(str(r.get("judge_confidence", "")).strip() != "" for r in pos)
        banned_hits = 0
        final = (_json(pdir / "banned_words_final.json") or {}).get("final") or (m.get("banned_words") or {}).get("final") or []
        if final:
            from flyguard.gen.paraphrases import BannedMatcher

            matcher = BannedMatcher(list(final))
            banned_hits = sum(1 for r in deep if str(r.get("label")) == "1" and matcher.hits(str(r.get("text", ""))))
        models = m.get("models") or {}
        keys_ok = all(k in m for k in ("models", "dates", "prompts_sha256", "rates", "counts"))
        self.add(self.c_paraphrases.__doc__, keys_ok and not j_bad and not d_bad and judged and banned_hits == 0 and bool(final),
                 f"строк {len(rows)}, позитивов {len(pos)}, J>{jmax}: {len(j_bad)}, глубоких {len(deep)} с J>{dmax}: {len(d_bad)}, "
                 f"судей {models.get('n_judges')} ({models.get('judge_rule')}), запрещённых слов в глубокой страте: {banned_hits}, "
                 f"манифест полон: {keys_ok}")

    def c_spend(self) -> None:
        """расход API ≤ бюджета"""
        sp = _json(self.root / "results/spend.json")
        budget = float((self.cfg.operator.get("llm_api") or {}).get("budget_usd") or 0)
        if not sp:
            self.add(self.c_spend.__doc__, False, "results/spend.json нет")
            return
        spent = float(sp.get("spent_usd") or 0)
        self.add(self.c_spend.__doc__, spent <= budget and bool(sp.get("within_budget")),
                 f"потрачено {spent:.4f} USD из {budget:.2f} USD")

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
        ok = not t["missing"] and not t["failed"] and t["sum"] <= SMOKE_LIMIT_S and sections == 10
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
        detail += f"; разделов в results/smoke/REPORT.md: {sections}"
        self.add(self.c_smoke.__doc__, ok, detail)

    def c_pytest(self) -> None:
        """pytest проходит"""
        log = self.root / "logs/pytest.log"
        if self.pytest_mode == "skip":
            self.add(self.c_pytest.__doc__, None, "не запускался (--pytest skip)")
            return
        if self.pytest_mode == "log":
            text = self._read(log)
            m = re.search(r"^exit=(\d+)", text, re.M)
            self.add(self.c_pytest.__doc__, bool(m) and m.group(1) == "0",
                     f"по logs/pytest.log: {text.strip().splitlines()[-2] if m and len(text.strip().splitlines()) > 1 else 'записи нет'}")
            return
        proc = subprocess.run([str(self.root / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=self.root, capture_output=True, text=True, timeout=1800)
        tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + proc.stderr + f"\nexit={proc.returncode}\n", encoding="utf-8")
        self.add(self.c_pytest.__doc__, proc.returncode == 0, tail.replace("\n", " | "))

    def c_manifests(self) -> None:
        """манифесты полны"""
        need = ["splits.json", "pools.json", "dedup.json", "audit.md"]
        missing = [n for n in need if not (self.manifests / n).exists()]
        glob_missing = [n for n in ("data/manifests/sources.json", "data/manifests/contamination.json",
                                    "results/shared/traces_manifest.json", "results/shared/split_manifest.json")
                        if not (self.root / n).exists()]
        detail = (f"{self.manifests.relative_to(self.root)}: нет {missing}" if missing else "") + (f"; нет {glob_missing}" if glob_missing else "")
        self.add(self.c_manifests.__doc__, not missing and not glob_missing, detail.strip("; "))

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

    def c_power(self) -> None:
        """power.json заморожен до финального прогона и использован в вердиктах"""
        power = _json(R.power_path(self.root, self.smoke))
        e1 = {k: v for k, v in self.seed_files.items() if "/E1/" in k}
        self.checks.append(power_before_e1(power, e1, self.current))
        v = _json(self.rdir / "verdicts.json")
        used = bool(v) and "carrier" in json.dumps(v, ensure_ascii=False)
        self.add("вердикты используют таблицу носителей E0", used, "" if used else "verdicts.json нет или без флагов носителей")

    def c_windows(self) -> None:
        """одна схема окон для всех детекторов; 512-токенные окна только в E6"""
        win = self.cfg.default.get("windows") or {}
        one_scheme = isinstance(win.get("size"), int) and isinstance(win.get("stride"), int)
        flags = {e: bool(c.get("transformer_windows_512_tokens")) for e, c in self.cfg.experiments.items()}
        only_e6 = flags.get("E6", False) and not any(v for e, v in flags.items() if e != "E6")
        offenders = []
        for name, res in self.seed_files.items():
            if "/E6/" in name:
                continue
            texts = list((res.get("numbers") or {}).keys()) + [str(n) for n in res.get("notes") or []]
            if any(TOK512_RE.search(t) for t in texts):
                offenders.append(name)
        self.add(self.c_windows.__doc__, one_scheme and only_e6 and not offenders,
                 f"окна {win.get('size')}/{win.get('stride')}; флаг 512 токенов: {flags}"
                 + (f"; 512-токенные ключи вне E6: {offenders}" if offenders else ""))

    def c_thresholds(self) -> None:
        """пороги полны"""
        if not self.seed_files:
            self.add("у каждого порога записаны значение, источник, цель и n", False, "результатов нет")
            return
        self.checks.append(thresholds_complete(self.seed_files))

    def c_verdicts(self) -> None:
        """вердикты по правилам этапа 4 с предусловиями; macroAUC — основа"""
        v = _json(self.rdir / "verdicts.json")
        if not v:
            self.add(self.c_verdicts.__doc__, False, "verdicts.json нет")
            return
        import make_report as mr

        items = list(mr.iter_verdicts(v))
        names = {n.split("/")[0] for n, _ in items}
        bad_status = [n for n, d in items if d.get("status") not in STATUSES]
        pre_ok = all("precondition" in json.dumps(d, ensure_ascii=False) for n, d in items if n.startswith(("H2", "H3")))
        macro_ok = all("macro" in json.dumps(d, ensure_ascii=False) for n, d in items if n.startswith(("H1b", "H3")))
        self.add(self.c_verdicts.__doc__, {"H1a", "H1b", "H2", "H3"} <= names and not bad_status and pre_ok and macro_ok,
                 f"гипотезы {sorted(names)}; статусы: " + ", ".join(f"{n}={d.get('status')}" for n, d in items)
                 + ("" if pre_ok else "; предусловия H2/H3 не записаны") + ("" if macro_ok else "; H1b/H3 без macroAUC"))

    def c_h3(self) -> None:
        """перестановка π в двухступенчатом бутстрепе H3; Bloom на сбалансированных классах"""
        balance = (self.cfg.default.get("readout") or {}).get("bloom", {}).get("balance")
        e4 = {k: v for k, v in self.seed_files.items() if "/E4/" in k}
        blob = json.dumps({k: [list((v.get("numbers") or {}).keys()), v.get("notes"), v.get("numbers")] for k, v in e4.items()}, ensure_ascii=False)
        two_stage = ("two_stage" in blob) and ("n_perms" in blob or "perm" in blob)
        self.add(self.c_h3.__doc__, balance == "subsample_to_minority" and bool(e4) and two_stage,
                 f"readout.bloom.balance={balance}; файлов E4 {len(e4)}; ключи двухступенчатого бутстрепа/π: {two_stage}")

    def c_comparator(self) -> None:
        """компаратор ProtectAI v2 назначен в конфиге до теста"""
        comp = (self.cfg.default.get("baselines") or {}).get("transformers", {}).get("comparator")
        first = git(self.root, "log", "--reverse", "--format=%cI", "-S", "comparator: protectai_v2", "--", "configs/default.yaml")
        first_ts = _utc(first.splitlines()[0]) if first else None
        entries = parse_access_log(self._read(self.root / "logs/data_access.log").splitlines())
        tests = min((e["ts"] for e in entries if e["split"] == "test"), default=None)
        before = first_ts is not None and (tests is None or first_ts < tests)
        self.add(self.c_comparator.__doc__, comp == "protectai_v2" and before,
                 f"comparator={comp}; назначен в коммите от {first_ts.strftime('%Y-%m-%dT%H:%M:%SZ') if first_ts else 'н/д'}"
                 + (f", первое чтение теста {tests.strftime('%Y-%m-%dT%H:%M:%SZ')}" if tests else ""))

    def c_config_hash(self) -> None:
        """хеш конфига в результатах совпадает с текущим"""
        files: dict[str, dict[str, Any] | None] = dict(self.seed_files)
        for name in ("power.json", "contract.json", "verdicts.json", "setup.json"):
            p = self.rdir / name
            if p.exists():
                files[str(p.relative_to(self.root))] = _json(p)
        bad = config_hash_mismatches(files, self.current)
        self.add(self.c_config_hash.__doc__, bool(files) and not bad,
                 f"файлов {len(files)}, расходятся: {bad[:6]}" if bad else f"файлов {len(files)}, хеш {self.current[:12]}…" if files else "файлов результатов нет")

    def c_deviations(self) -> None:
        """DEVIATIONS.md перечисляет отклонения; каждый упомянутый D<n> существует"""
        ids = set(re.findall(r"\*\*(D\d+)", self.deviations))
        cited: set[str] = set()
        for res in self.seed_files.values():
            cited |= set(re.findall(r"\bD(\d+)\b", " ".join(str(n) for n in res.get("notes") or [])))
        for name in ("verdicts.json", "contract.json"):
            cited |= set(re.findall(r"\bD(\d+)\b", json.dumps(_json(self.rdir / name) or {}, ensure_ascii=False)))
        missing = sorted(f"D{c}" for c in cited if f"D{c}" not in ids)
        self.add(self.c_deviations.__doc__, bool(ids) and not missing, f"записей {len(ids)}" + (f"; упомянуты, но не описаны: {missing}" if missing else ""))

    def c_report(self) -> None:
        """каждое число REPORT.md прослеживается до results/; разделы 1–10 на месте"""
        import make_report as mr

        report = (self.rdir / "REPORT.md") if self.smoke else (self.root / "REPORT.md")
        if not report.exists():
            self.add(self.c_report.__doc__, False, f"{report.relative_to(self.root)} нет")
            return
        text = report.read_text(encoding="utf-8")
        sections = [i for i in range(1, 11) if not re.search(rf"^## {i}\. ", text, re.M)]
        n, problems = mr.trace_numbers(text, self.root)
        self.add(self.c_report.__doc__, not problems and not sections,
                 f"чисел проверено {n}, не прослеживаются {len(problems)}" + (f": {problems[:3]}" if problems else "")
                 + (f"; нет разделов {sections}" if sections else ""))

    def c_contract(self) -> None:
        """контракт: CSV по схеме, split_manifest по crc32 mod 3 = 2, important_instructions вне обучения, оба манифеста"""
        from flyguard.agentdojo_io.contract import validate_csv, validate_split_manifest

        shared = (self.rdir / "shared") if self.smoke else (self.root / "results/shared")   # contract_run.shared_dir
        csv_path = shared / "flyguard.csv"
        csv_problems = validate_csv(csv_path) if csv_path.exists() else ["flyguard.csv нет"]
        sm = _json(shared / "split_manifest.json") or _json(self.root / "results/shared/split_manifest.json")
        rule = self.cfg.default["splits"]["contract"]
        sm_problems = (validate_split_manifest(sm) + split_rule_violations(sm, int(rule["mod"]), int(rule["rem"]), str(rule["test_attack"]))
                       if sm else ["split_manifest.json нет"])
        both = (self.root / "results/shared/traces_manifest.json").exists() and sm is not None
        self.add(self.c_contract.__doc__, not csv_problems and not sm_problems and both,
                 f"CSV: {'ок' if not csv_problems else csv_problems[:3]}; split_manifest: {'ок' if not sm_problems else sm_problems[:3]}; оба манифеста: {both}")
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
               access_log: Callable[[Path, str, str], None] | None = None, only: Sequence[str] | None = None) -> list[Check]:
    return Checker(root, smoke, pytest_mode, access_log, only).run()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Machine checks of ТЗ «Критерии приёмки».")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--pytest", choices=("run", "log", "skip"), default="run")
    ap.add_argument("--only", default="", help="comma list of check names (c_network, c_regex, ...) to run alone")
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
