"""API spend accounting (ТЗ "Бюджет API"): aggregate per-call usage records into results/spend.json."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from flyguard.config import ROOT, Configs
from flyguard.io import atomic_write_json, read_json, read_jsonl


def spend_dir(cfg: Configs) -> Path:
    return ROOT / cfg.default["traces"]["spend_dir"]


def load_calls(cfg: Configs) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for f in sorted(spend_dir(cfg).glob("*.jsonl")):
        calls.extend(c for c in read_jsonl(f) if "cost_usd" in c)
    return calls


def total_usd(cfg: Configs) -> float:
    return float(sum(c["cost_usd"] for c in load_calls(cfg)))


def summarize(cfg: Configs) -> dict[str, Any]:
    calls = load_calls(cfg)
    by: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for c in calls:
        for key in (f"stream:{c.get('stream')}", f"model:{c.get('model')}",
                    f"benchmark:{c.get('benchmark')}", f"suite:{c.get('benchmark')}/{c.get('suite')}",
                    f"attack:{c.get('benchmark')}/{c.get('attack')}"):
            d = by[key]
            d["calls"] += 1
            d["cost_usd"] += c["cost_usd"]
            d["prompt_tokens"] += c.get("prompt_tokens") or 0
            d["cache_hit"] += c.get("cache_hit") or 0
            d["completion_tokens"] += c.get("completion_tokens") or 0
    budget = float(cfg.operator["llm_api"]["budget_usd"])
    total = float(sum(c["cost_usd"] for c in calls))
    return {"budget_usd": budget, "spent_usd": round(total, 6), "remaining_usd": round(budget - total, 6),
            "within_budget": total <= budget, "n_calls": len(calls),
            "breakdown": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in sorted(by.items())}}


def write_spend_json(cfg: Configs, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """results/spend.json: totals, breakdown, and any projections/decisions passed in `extra` (kept across calls)."""
    path = ROOT / "results" / "spend.json"
    prev = read_json(path) if path.exists() else {}
    out = {**prev, **summarize(cfg)}
    if extra:
        out.update(extra)
    atomic_write_json(path, out)
    return out
