#!/usr/bin/env python
"""Standalone AgentDojo / AgentDyn episode runner for an OpenAI-compatible provider (DeepSeek), ТЗ 1.5.

Imports only agentdojo, openai and the standard library, so the very same file runs in `.venv` (agentdojo
0.1.35) and in `.venv-agentdyn` (the AgentDyn fork, same package name). The harness's own TraceLogger writes
<logdir>/<model>/<suite>/<user_task>/<attack>/<injection_task>.json; existing logs are never re-run
(idempotent per episode). Every API call is metered from the `usage` fields into a spend JSONL and appended to
logs/network.log.

Provider adaptations (all recorded in ASSUMPTIONS.md):
  * role `developer` -> `system` (DeepSeek rejects `developer`); system content blocks -> plain string;
  * `thinking: disabled` (non-thinking mode: temperature applies; thinking mode would need reasoning_content
    round-trips that agentdojo does not implement);
  * temperature is forced explicitly (agentdojo drops temperature 0.0 because `0.0 or NOT_GIVEN` is NOT_GIVEN);
  * the prose model name used by the attacks' `{model}` placeholder is registered in agentdojo.models.MODEL_NAMES.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import openai

PROSE_NAMES = {"deepseek-flash": "DeepSeek", "deepseek-v4-pro": "DeepSeek"}
PEAK_WINDOWS_UTC = ((1, 4), (6, 10))  # Mon-Fri, hours [start, end)


class BudgetExceeded(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def is_peak(t: dt.datetime) -> bool:
    return t.weekday() < 5 and any(a <= t.hour < b for a, b in PEAK_WINDOWS_UTC)


def seconds_to_offpeak(t: dt.datetime) -> float:
    for a, b in PEAK_WINDOWS_UTC:
        if a <= t.hour < b:
            end = t.replace(hour=b, minute=0, second=0, microsecond=0)
            return (end - t).total_seconds() + 1
    return 0.0


def usage_cost(prices: dict, model: str, usage: dict, peak: bool) -> tuple[float, int, int]:
    p = prices[model]
    i = 1 if peak else 0
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = max(prompt - hit, 0)
    out = usage.get("completion_tokens") or 0
    cost = (hit * p["input_hit"][i] + miss * p["input_miss"][i] + out * p["output"][i]) / 1e6
    return cost, int(hit), int(miss)


def fix_messages(messages):
    fixed = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("role") == "system" and isinstance(m.get("content"), list):
            m["content"] = "".join(part.get("text", "") for part in m["content"])
        fixed.append(m)
    return fixed


class Meter:
    """Patches client.chat.completions.create: message fixes, provider options, retries, usage accounting."""

    def __init__(self, args, benchmark: str):
        self.model = args.model
        self.spend_path = Path(args.spend)
        self.netlog_path = Path(args.netlog)
        self.budget = float(args.budget_usd)
        self.prices = json.loads(args.prices_json)
        self.thinking = args.thinking
        self.temperature = float(args.temperature)
        self.avoid_peak = bool(args.avoid_peak)
        self.base_url = args.base_url.rstrip("/")
        self.benchmark = benchmark
        self.context: dict[str, str | None] = {}
        self.calls = 0
        self.cost = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self._last_total = 0.0
        self.spend_path.parent.mkdir(parents=True, exist_ok=True)
        self.netlog_path.parent.mkdir(parents=True, exist_ok=True)

    def total_spend(self) -> float:
        total = 0.0
        for f in self.spend_path.parent.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if '"cost_usd"' in line:
                        try:
                            total += json.loads(line)["cost_usd"]
                        except (ValueError, KeyError):
                            pass
        self._last_total = total
        return total

    def check_budget(self) -> None:
        if self.total_spend() >= self.budget:
            raise BudgetExceeded(f"spend {self._last_total:.4f} USD >= budget {self.budget:.2f} USD")

    def wait_offpeak(self) -> None:
        if not self.avoid_peak:
            return
        now = utc_now()
        if is_peak(now):
            s = seconds_to_offpeak(now)
            print(f"[meter] peak hours (prices x2): sleeping {s/60:.0f} min", file=sys.stderr, flush=True)
            time.sleep(s)

    def record(self, response, latency: float) -> None:
        usage = response.usage.model_dump() if response.usage is not None else {}
        now = utc_now()
        peak = is_peak(now)
        cost, hit, miss = usage_cost(self.prices, self.model, usage, peak)
        self.calls += 1
        self.cost += cost
        self.tokens_in += int(usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("completion_tokens") or 0)
        rec = {"ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "stream": "traces", "benchmark": self.benchmark,
               "model": self.model, "peak": peak, "prompt_tokens": usage.get("prompt_tokens"),
               "cache_hit": hit, "cache_miss": miss, "completion_tokens": usage.get("completion_tokens"),
               "cost_usd": round(cost, 8), "latency_s": round(latency, 3), **self.context}
        with open(self.spend_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        ctx = "/".join(str(self.context.get(k) or "-") for k in ("suite", "attack", "user_task", "injection_task"))
        with open(self.netlog_path, "a", encoding="utf-8") as fh:
            fh.write(f"{rec['ts']}\tapi.deepseek.com\tPOST\t{self.base_url}/chat/completions\t"
                     f"agent model call ({self.benchmark} {ctx}); stream 1 of 2 permitted outgoing flows\n")

    def install(self, client: openai.OpenAI) -> None:
        original = client.chat.completions.create
        meter = self

        def create(**kw):
            kw["messages"] = fix_messages(kw["messages"])
            extra = dict(kw.pop("extra_body", None) or {})
            extra["thinking"] = {"type": meter.thinking}
            kw["extra_body"] = extra
            kw["temperature"] = meter.temperature
            meter.wait_offpeak()
            delay = 2.0
            for attempt in range(8):
                t0 = time.time()
                try:
                    resp = original(**kw)
                except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                        openai.InternalServerError) as err:
                    if attempt == 7:
                        raise
                    sleep = min(90.0, delay * (2 ** attempt)) * (0.5 + random.random())
                    print(f"[meter] {type(err).__name__}: retry in {sleep:.0f}s", file=sys.stderr, flush=True)
                    time.sleep(sleep)
                    continue
                meter.record(resp, time.time() - t0)
                return resp
            raise RuntimeError("unreachable")

        client.chat.completions.create = create  # type: ignore[assignment]


def build_pipeline(args, meter: Meter):
    from agentdojo import models as ad_models
    from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    ad_models.MODEL_NAMES.setdefault(args.model, PROSE_NAMES.get(args.model, "DeepSeek"))
    api_key = os.environ.get(args.key_env)
    if not api_key:
        sys.exit(f"{args.key_env} is not set (source scripts/env.sh)")
    client = openai.OpenAI(api_key=api_key, base_url=args.base_url, timeout=600, max_retries=0)
    meter.install(client)
    llm = OpenAILLM(client, args.model, temperature=meter.temperature)
    llm.name = args.model
    pipeline = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None,
                                                        system_message_name=None, system_message=None))
    pipeline.name = args.model
    return pipeline


def dump_meta(suite, args) -> dict:
    """Benchmark metadata used offline by flyguard.agentdojo_io (no API calls): tools, prompts, goals, ground truth."""
    env = suite.load_and_inject_default_environment({})
    tools = [{"name": t.name, "description": t.description,
              "parameters": sorted(getattr(t.parameters, "model_fields", {}).keys())} for t in suite.tools]
    user_tasks = {tid: {"prompt": t.PROMPT, "difficulty": str(getattr(t, "DIFFICULTY", ""))}
                  for tid, t in sorted(suite.user_tasks.items(), key=lambda kv: task_number(kv[0]))}
    injection_tasks = {}
    for tid, t in sorted(suite.injection_tasks.items(), key=lambda kv: task_number(kv[0])):
        try:
            gt = [{"function": c.function, "args": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                                                 for k, v in (c.args or {}).items()},
                   "placeholder_args": {k: str(v) for k, v in (getattr(c, "placeholder_args", None) or {}).items()}}
                  for c in t.ground_truth(env)]
        except Exception as err:  # noqa: BLE001
            gt = [{"error": f"{type(err).__name__}: {err}"}]
        injection_tasks[tid] = {"goal": t.GOAL, "ground_truth": gt}
    return {"benchmark": args.benchmark, "suite": args.suite, "benchmark_version": args.benchmark_version,
            "tools": tools, "user_tasks": user_tasks, "injection_tasks": injection_tasks}


def task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def select_user_tasks(suite, args) -> list[str]:
    ids = sorted(suite.user_tasks.keys(), key=task_number)
    if args.user_tasks:
        wanted = set(args.user_tasks.split(","))
        ids = [i for i in ids if i in wanted]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        ids = [i for idx, i in enumerate(ids) if idx % n == k]
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", required=True, choices=["agentdojo", "agentdyn"])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--attack", default="none", help="attack name or 'none' for clean runs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--spend", required=True, help="spend JSONL of this process (dir shared by all processes)")
    ap.add_argument("--netlog", default="logs/network.log")
    ap.add_argument("--budget-usd", required=True, type=float)
    ap.add_argument("--prices-json", required=True)
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled"])
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--avoid-peak", action="store_true")
    ap.add_argument("--user-tasks", default="", help="comma list; default all")
    ap.add_argument("--injection-tasks", default="", help="comma list; default all")
    ap.add_argument("--pairs", default="", help="comma list of user_task:injection_task (overrides the two above)")
    ap.add_argument("--shard", default="", help="k/n: take user tasks with index %% n == k")
    ap.add_argument("--list", action="store_true", help="print the suite's user and injection tasks and exit")
    ap.add_argument("--dump-meta", action="store_true",
                    help="print suite metadata (tools, user-task prompts, injection-task goals and ground-truth calls) and exit")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--summary", default="", help="write the JSON summary here as well")
    args = ap.parse_args(argv)

    from agentdojo.task_suite.load_suites import get_suite
    suite = get_suite(args.benchmark_version, args.suite)
    if args.list:
        print(json.dumps({"suite": args.suite, "benchmark_version": args.benchmark_version,
                          "user_tasks": sorted(suite.user_tasks, key=task_number),
                          "injection_tasks": sorted(suite.injection_tasks, key=task_number)}))
        return 0

    if args.dump_meta:
        print(json.dumps(dump_meta(suite, args)))
        return 0

    from agentdojo.attacks import load_attack
    from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
    from agentdojo.logging import Logger

    class QuietLogger(Logger):
        """Silent delegate for TraceLogger: pushed on the logger stack (unlike NullLogger) and carries `logdir`."""

        def __init__(self, logdir: str) -> None:
            self.logdir = logdir
            self.messages = []

        def log(self, *args, **kwargs):
            pass

        def log_error(self, message: str):
            print(f"[harness] {message}", file=sys.stderr, flush=True)

    meter = Meter(args, args.benchmark)
    pipeline = build_pipeline(args, meter)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    attack = None if args.attack == "none" else load_attack(args.attack, suite, pipeline)

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    else:
        its = sorted(suite.injection_tasks, key=task_number)
        if args.injection_tasks:
            wanted = set(args.injection_tasks.split(","))
            its = [i for i in its if i in wanted]
        uts = select_user_tasks(suite, args)
        pairs = [(u, i) for u in uts for i in its] if attack is not None else [(u, None) for u in uts]

    summary = {"benchmark": args.benchmark, "suite": args.suite, "attack": args.attack, "model": args.model,
               "planned": len(pairs), "run": 0, "existing": 0, "not_injectable": 0, "errors": 0,
               "utility_true": 0, "security_true": 0, "stopped": None}
    outcomes = []
    t_start = time.time()
    stop_reason = None
    for idx, (ut_id, it_id) in enumerate(pairs):
        meter.context = {"suite": args.suite, "attack": args.attack, "user_task": ut_id, "injection_task": it_id}
        log_path = logdir / args.model / args.suite / ut_id / (args.attack if it_id else "none") / f"{it_id or 'none'}.json"
        existed = log_path.exists()
        try:
            meter.check_budget()
            user_task = suite.get_user_task_by_id(ut_id)
            with QuietLogger(str(logdir)):
                if attack is None:
                    utility, security = run_task_without_injection_tasks(
                        suite, pipeline, user_task, logdir, args.force_rerun, benchmark_version=args.benchmark_version)
                else:
                    u, s = run_task_with_injection_tasks(
                        suite, pipeline, user_task, attack, logdir, args.force_rerun,
                        injection_tasks=[it_id], benchmark_version=args.benchmark_version)
                    utility, security = u[(ut_id, it_id)], s[(ut_id, it_id)]
        except BudgetExceeded as err:
            stop_reason = str(err)
            break
        except ValueError as err:
            if "not injectable" in str(err):
                summary["not_injectable"] += 1
                continue
            summary["errors"] += 1
            print(f"[runner] {ut_id}/{it_id}: {err}", file=sys.stderr, flush=True)
            continue
        except Exception as err:  # noqa: BLE001 - one bad episode must not kill the shard
            summary["errors"] += 1
            print(f"[runner] {ut_id}/{it_id}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
            traceback.print_exc(limit=3)
            continue
        summary["existing" if existed else "run"] += 1
        summary["utility_true"] += int(bool(utility))
        summary["security_true"] += int(bool(security))
        outcomes.append({"user_task": ut_id, "injection_task": it_id, "utility": bool(utility),
                         "security": bool(security), "existing": existed})
        if (idx + 1) % 10 == 0:
            print(f"[runner] {args.suite}/{args.attack}: {idx + 1}/{len(pairs)} episodes, "
                  f"{meter.calls} calls, {meter.cost:.4f} USD this process", file=sys.stderr, flush=True)

    summary.update({"stopped": stop_reason, "calls": meter.calls, "cost_usd": round(meter.cost, 6),
                    "tokens_in": meter.tokens_in, "tokens_out": meter.tokens_out,
                    "elapsed_s": round(time.time() - t_start, 1), "outcomes": outcomes})
    text = json.dumps(summary)
    print(text)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text + "\n")
    return 3 if stop_reason else 0


if __name__ == "__main__":
    sys.exit(main())
