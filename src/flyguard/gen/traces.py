"""Trace generation orchestration (ТЗ 1.5, Appendix B): pilot, model choice, prioritised full run, freeze.

The harness itself runs in `src/flyguard/gen/harness_run.py` inside the harness venv (.venv for AgentDojo,
.venv-agentdyn for AgentDyn). This module only selects work, launches shards, reads the harness's own JSON
logs for utility/security and the spend files for cost, and writes results/pilot.json, results/spend.json and
results/shared/traces_manifest.json. Tool outputs inside the logs are never printed here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from flyguard.config import ROOT, Configs, load_configs, seeds_for
from flyguard.gen import spend as spend_mod
from flyguard.io import atomic_write_json, read_json, sha256_file

RUNNER = ROOT / "src" / "flyguard" / "gen" / "harness_run.py"


def _venv_python(cfg: Configs, benchmark: str) -> str:
    return str(ROOT / cfg.default["traces"][benchmark]["venv"] / "bin" / "python")


def _provider(cfg: Configs) -> dict[str, Any]:
    return cfg.operator["llm_api"]["providers"][0]


def _prices_json(cfg: Configs) -> str:
    return json.dumps(cfg.operator["llm_api"]["prices_usd_per_million"])


def _logdir(cfg: Configs, benchmark: str) -> Path:
    return ROOT / cfg.default["traces"]["logdir"] / benchmark


def runner_cmd(cfg: Configs, benchmark: str, suite: str, attack: str, model: str, spend_name: str,
               **opts: str) -> list[str]:
    llm = cfg.operator["llm_api"]
    cmd = [_venv_python(cfg, benchmark), str(RUNNER), "--benchmark", benchmark, "--suite", suite,
           "--attack", attack, "--model", model, "--base-url", _provider(cfg)["base_url"],
           "--key-env", _provider(cfg)["key_env"],
           "--benchmark-version", cfg.default["traces"][benchmark]["benchmark_version"],
           "--logdir", str(_logdir(cfg, benchmark)),
           "--spend", str(spend_mod.spend_dir(cfg) / f"{spend_name}.jsonl"),
           "--netlog", str(ROOT / "logs" / "network.log"),
           "--budget-usd", str(llm["budget_usd"]), "--prices-json", _prices_json(cfg),
           "--thinking", llm.get("agent_thinking", "disabled"),
           "--temperature", str(cfg.default["traces"]["temperature"])]
    if llm.get("avoid_peak_hours"):
        cmd.append("--avoid-peak")
    for k, v in opts.items():
        if v:
            cmd += [f"--{k.replace('_', '-')}", v]
    return cmd


def list_suite(cfg: Configs, benchmark: str, suite: str) -> dict[str, Any]:
    cmd = runner_cmd(cfg, benchmark, suite, "none", "x", "list") + ["--list"]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    return json.loads(out.strip().splitlines()[-1])


def _key_ok() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY not set: run `source scripts/env.sh` first")


# ------------------------------------------------------------------------------------------- pilot

def pilot_selection(cfg: Configs, seed_subsample: int) -> tuple[list[tuple[str, str]], list[str]]:
    """40 injectable (user_task, injection_task) pairs of the pilot suite/attack + 10 clean tasks (Appendix B)."""
    p = cfg.default["traces"]["pilot"]
    info = list_suite(cfg, "agentdojo", p["suite"])
    rng = np.random.default_rng(seed_subsample)
    pairs = [(u, i) for u in info["user_tasks"] for i in info["injection_tasks"]]
    order = rng.permutation(len(pairs))
    chosen = [pairs[j] for j in order[: p["n_attacked"]]]
    clean = [info["user_tasks"][j] for j in rng.permutation(len(info["user_tasks"]))[: p["n_clean"]]]
    return chosen, clean


def availability_check(cfg: Configs, model: str) -> dict[str, Any]:
    """One short call (ТЗ 1.5) through the same OpenAI-compatible client; metered like everything else."""
    import openai
    from flyguard.netlog import log_request
    prov = _provider(cfg)
    client = openai.OpenAI(api_key=os.environ[prov["key_env"]], base_url=prov["base_url"], timeout=120)
    t0 = time.time()
    log_request("POST", prov["base_url"] + "/chat/completions", f"availability check of {model} (pilot)")
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply: ok"}],
                                           max_tokens=4, temperature=0,
                                           extra_body={"thinking": {"type": "disabled"}})
        usage = r.usage.model_dump() if r.usage else {}
        return {"model": model, "available": True, "latency_s": round(time.time() - t0, 2), "usage": usage}
    except Exception as err:  # noqa: BLE001
        return {"model": model, "available": False, "error": f"{type(err).__name__}: {str(err)[:200]}"}


def _run(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    summary = json.loads(lines[-1]) if lines else {"error": f"no summary, exit {proc.returncode}"}
    summary["returncode"] = proc.returncode
    return summary


def run_pilot(cfg: Configs, global_seed: int = 0) -> dict[str, Any]:
    _key_ok()
    p = cfg.default["traces"]["pilot"]
    seeds = seeds_for(cfg, global_seed)
    chosen, clean = pilot_selection(cfg, seeds["subsample"])
    pairs_arg = ",".join(f"{u}:{i}" for u, i in chosen)
    results: dict[str, Any] = {"seed": global_seed, "seed_subsample": seeds["subsample"], "suite": p["suite"],
                               "attack": p["attack"], "pairs": [f"{u}/{i}" for u, i in chosen], "clean": clean,
                               "rule": {"asr_range": p["asr_range"], "min_utility": p["min_utility"]},
                               "candidates": [], "chosen_model": None, "chosen_by_rule": None}
    for model in cfg.operator["llm_api"]["agent_models"]:
        cand: dict[str, Any] = {"model": model, "availability": availability_check(cfg, model)}
        if not cand["availability"]["available"]:
            results["candidates"].append(cand)
            continue
        spent_before = spend_mod.total_usd(cfg)
        att = _run(runner_cmd(cfg, "agentdojo", p["suite"], p["attack"], model, f"pilot_{model}",
                              pairs=pairs_arg), ROOT / "logs" / f"pilot_{model}.log")
        cln = _run(runner_cmd(cfg, "agentdojo", p["suite"], "none", model, f"pilot_{model}",
                              user_tasks=",".join(clean)), ROOT / "logs" / f"pilot_{model}.log")
        n_att = att.get("run", 0) + att.get("existing", 0)
        n_cln = cln.get("run", 0) + cln.get("existing", 0)
        asr = att.get("security_true", 0) / n_att if n_att else None
        utility = cln.get("utility_true", 0) / n_cln if n_cln else None
        cost = spend_mod.total_usd(cfg) - spent_before
        accepted = (asr is not None and utility is not None and p["asr_range"][0] <= asr <= p["asr_range"][1]
                    and utility >= p["min_utility"])
        cand.update({"n_attacked": n_att, "not_injectable": att.get("not_injectable", 0), "n_clean": n_cln,
                     "targeted_asr": asr, "utility_clean": utility, "pilot_cost_usd": round(cost, 5),
                     "cost_per_attacked_episode_usd": round(att.get("cost_usd", 0) / max(att.get("run", 1), 1), 6),
                     "cost_per_clean_episode_usd": round(cln.get("cost_usd", 0) / max(cln.get("run", 1), 1), 6),
                     "errors": att.get("errors", 0) + cln.get("errors", 0), "accepted": accepted,
                     "attacked_summary": {k: v for k, v in att.items() if k != "outcomes"},
                     "clean_summary": {k: v for k, v in cln.items() if k != "outcomes"}})
        results["candidates"].append(cand)
        if accepted:
            results["chosen_model"], results["chosen_by_rule"] = model, True
            break
    if results["chosen_model"] is None:
        scored = [c for c in results["candidates"] if c.get("targeted_asr") is not None]
        if scored:
            best = min(scored, key=lambda c: abs(c["targeted_asr"] - 0.40))
            results["chosen_model"], results["chosen_by_rule"] = best["model"], False
    results["projection"] = project_cost(cfg, results)
    atomic_write_json(ROOT / "results" / "pilot.json", results)
    spend_mod.write_spend_json(cfg, {"pilot": {"chosen_model": results["chosen_model"],
                                              "chosen_by_rule": results["chosen_by_rule"],
                                              "projection": results["projection"]}})
    return results


def episode_counts(cfg: Configs) -> dict[str, dict[str, int]]:
    """Attacked pairs and clean tasks per benchmark from the installed suites (A6)."""
    out: dict[str, dict[str, int]] = {}
    for bench in ("agentdojo", "agentdyn"):
        pairs = clean = 0
        for suite in cfg.default["traces"][bench]["suites"]:
            info = list_suite(cfg, bench, suite)
            pairs += len(info["user_tasks"]) * len(info["injection_tasks"])
            clean += len(info["user_tasks"])
        out[bench] = {"pairs_per_attack": pairs, "clean": clean}
    return out


def project_cost(cfg: Configs, pilot: dict[str, Any]) -> dict[str, Any]:
    """Extrapolate the pilot's per-episode cost to the priority list (ТЗ 1.5) and mark what fits the budget."""
    model = pilot.get("chosen_model")
    cand = next((c for c in pilot["candidates"] if c["model"] == model), None)
    if not cand or cand.get("cost_per_attacked_episode_usd") is None:
        return {"model": model, "note": "no accepted candidate with cost data"}
    counts = episode_counts(cfg)
    ca, cc = cand["cost_per_attacked_episode_usd"], cand["cost_per_clean_episode_usd"]
    dyn_factor = 2.5  # AgentDyn episodes are longer (~7 steps, 3 apps); refined after its first shard
    budget = float(cfg.operator["llm_api"]["budget_usd"])
    spent = spend_mod.total_usd(cfg)
    remaining = budget - spent
    items = []
    cum = 0.0
    for pr in cfg.default["traces"]["priority"]:
        bench = pr["benchmark"]
        n_att = counts[bench]["pairs_per_attack"]
        n_cln = counts[bench]["clean"] if pr.get("clean") else 0
        f = dyn_factor if bench == "agentdyn" else 1.0
        est = f * (n_att * ca + n_cln * cc)
        cum += est
        items.append({**pr, "episodes_attacked": n_att, "episodes_clean": n_cln, "est_cost_usd": round(est, 4),
                      "cumulative_usd": round(cum, 4), "fits_remaining_budget": cum <= remaining})
    return {"model": model, "cost_per_attacked_episode_usd": ca, "cost_per_clean_episode_usd": cc,
            "agentdyn_length_factor": dyn_factor, "budget_usd": budget, "spent_usd": round(spent, 5),
            "remaining_usd": round(remaining, 5), "priorities": items,
            "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


# ------------------------------------------------------------------------------------------- full run

def run_priority(cfg: Configs, model: str, item: dict[str, Any], max_workers: int) -> list[dict[str, Any]]:
    """One priority item = (benchmark, attack[, clean]) over all its suites; shards run in parallel processes.

    Idempotent: the harness skips episodes whose log exists. Returns the shard summaries."""
    _key_ok()
    bench = item["benchmark"]
    suites = cfg.default["traces"][bench]["suites"]
    jobs: list[tuple[str, str, str]] = []  # (suite, attack, shard)
    for suite in suites:
        if item.get("clean"):
            jobs.append((suite, "none", ""))
    n_shards = max(1, max_workers // len(suites))
    for suite in suites:
        for k in range(n_shards):
            jobs.append((suite, item["attack"], f"{k}/{n_shards}"))
    running: list[tuple[subprocess.Popen, tuple[str, str, str], Path]] = []
    summaries = []
    pending = list(jobs)
    while pending or running:
        while pending and len(running) < max_workers:
            suite, attack, shard = pending.pop(0)
            name = f"{bench}_{suite}_{attack}" + (f"_s{shard.replace('/', 'of')}" if shard else "")
            summary_path = ROOT / "results" / "spend" / "summaries" / f"{name}.json"
            cmd = runner_cmd(cfg, bench, suite, attack, model, name, shard=shard, summary=str(summary_path))
            log = open(ROOT / "logs" / f"gen_{name}.log", "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=log, text=True)
            running.append((proc, (suite, attack, shard), summary_path))
        time.sleep(5)
        still = []
        for proc, job, summary_path in running:
            if proc.poll() is None:
                still.append((proc, job, summary_path))
                continue
            s = read_json(summary_path) if summary_path.exists() else {"error": "no summary"}
            s["returncode"] = proc.returncode
            s.pop("outcomes", None)
            summaries.append(s)
            print(f"[traces] done {job}: rc={proc.returncode} run={s.get('run')} existing={s.get('existing')} "
                  f"errors={s.get('errors')} cost={s.get('cost_usd')} stopped={s.get('stopped')}", flush=True)
        running = still
        if any(s.get("stopped") for s in summaries):
            for proc, _, _ in running:
                proc.terminate()
            pending.clear()
    return summaries


def run_all(cfg: Configs, only: list[int] | None = None) -> dict[str, Any]:
    pilot = read_json(ROOT / "results" / "pilot.json")
    model = pilot["chosen_model"]
    if not model:
        sys.exit("no agent model chosen: run the pilot first")
    out: dict[str, Any] = {"model": model, "items": []}
    for idx, item in enumerate(cfg.default["traces"]["priority"]):
        if only is not None and idx not in only:
            continue
        proj = pilot["projection"]["priorities"][idx]
        remaining = float(cfg.operator["llm_api"]["budget_usd"]) - spend_mod.total_usd(cfg)
        if proj["est_cost_usd"] > remaining:
            out["items"].append({**item, "skipped": "projected cost exceeds remaining budget",
                                 "est_cost_usd": proj["est_cost_usd"], "remaining_usd": round(remaining, 4)})
            print(f"[traces] skip priority {idx}: {item} est {proj['est_cost_usd']} > remaining {remaining:.3f}")
            continue
        mw = cfg.default["traces"][item["benchmark"]]["max_workers"]
        summaries = run_priority(cfg, model, item, mw)
        out["items"].append({**item, "shards": summaries})
        spend_mod.write_spend_json(cfg)
        if any(s.get("stopped") for s in summaries):
            out["stopped"] = "budget exhausted"
            break
    atomic_write_json(ROOT / "results" / "traces_run.json", out)
    spend_mod.write_spend_json(cfg)
    return out


# ------------------------------------------------------------------------------------------- freeze

def freeze(cfg: Configs) -> dict[str, Any]:
    """results/shared/traces_manifest.json with sha256 of every log + copy for the second team (ТЗ 1.5)."""
    pilot = read_json(ROOT / "results" / "pilot.json")
    model = pilot["chosen_model"]
    src_json = read_json(ROOT / "data" / "manifests" / "sources.json") if (ROOT / "data/manifests/sources.json").exists() else {}
    manifest: dict[str, Any] = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent_model": model, "provider": _provider(cfg)["name"], "thinking": cfg.operator["llm_api"].get("agent_thinking"),
        "temperature": cfg.default["traces"]["temperature"], "pilot": {k: pilot[k] for k in ("chosen_by_rule", "rule")},
        "harnesses": {"agentdojo": {"repo": "https://github.com/ethz-spylab/agentdojo", "package": "agentdojo==0.1.35",
                                    "benchmark_version": cfg.default["traces"]["agentdojo"]["benchmark_version"],
                                    "attacks": cfg.default["traces"]["agentdojo"]["attacks"]},
                      "agentdyn": {**src_json.get("pins", {}).get("agentdyn", {}),
                                   "benchmark_version": cfg.default["traces"]["agentdyn"]["benchmark_version"],
                                   "attacks": cfg.default["traces"]["agentdyn"]["attacks"]}},
        "counts": {}, "files": []}
    out_dir = ROOT / cfg.operator["shared"]["traces_out_dir"]
    for bench in ("agentdojo", "agentdyn"):
        root = _logdir(cfg, bench) / model
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.json")):
            rel = f.relative_to(_logdir(cfg, bench))
            suite, user_task, attack, fname = rel.parts[1], rel.parts[2], rel.parts[3], rel.parts[4]
            if user_task.startswith("injection_task_"):
                continue  # utility checks of injection tasks run as user tasks: not episodes of the dataset
            d = read_json(f)
            cls = ("benign" if d.get("injection_task_id") is None
                   else ("hijacked" if d.get("security") else "injection_ignored"))
            if d.get("error"):
                cls = "error"
            manifest["files"].append({"benchmark": bench, "path": str(rel), "sha256": sha256_file(f), "suite": suite,
                                      "user_task": user_task, "attack": attack, "injection_task": fname[:-5],
                                      "episode_class": cls, "utility": d.get("utility"), "security": d.get("security"),
                                      "duration_s": d.get("duration")})
            c = manifest["counts"].setdefault(bench, {}).setdefault(suite, {}).setdefault(attack, {})
            c[cls] = c.get(cls, 0) + 1
            dest = out_dir / bench / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
    manifest["spend"] = spend_mod.summarize(cfg)
    atomic_write_json(ROOT / "results" / "shared" / "traces_manifest.json", manifest)
    return manifest


def verify_frozen(cfg: Configs) -> list[str]:
    """Re-hash every log listed in the manifest; a mismatch means a trace changed after the freeze."""
    m = read_json(ROOT / "results" / "shared" / "traces_manifest.json")
    bad = []
    for e in m["files"]:
        f = _logdir(cfg, e["benchmark"]) / e["path"]
        if not f.exists() or sha256_file(f) != e["sha256"]:
            bad.append(e["path"])
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run"); r.add_argument("--only", default="", help="comma list of priority indices")
    sub.add_parser("counts"); sub.add_parser("freeze"); sub.add_parser("verify"); sub.add_parser("spend")
    args = ap.parse_args(argv)
    cfg = load_configs()
    if args.cmd == "pilot":
        res = run_pilot(cfg, args.seed)
        print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "clean")}, indent=1, default=str))
    elif args.cmd == "run":
        only = [int(x) for x in args.only.split(",")] if args.only else None
        print(json.dumps(run_all(cfg, only), indent=1, default=str)[:4000])
    elif args.cmd == "counts":
        print(json.dumps(episode_counts(cfg), indent=1))
    elif args.cmd == "freeze":
        m = freeze(cfg)
        print(json.dumps({"files": len(m["files"]), "counts": m["counts"]}, indent=1))
    elif args.cmd == "verify":
        bad = verify_frozen(cfg)
        print("frozen traces intact" if not bad else f"{len(bad)} changed: {bad[:5]}")
        return 1 if bad else 0
    elif args.cmd == "spend":
        print(json.dumps(spend_mod.write_spend_json(cfg), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
