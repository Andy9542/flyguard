"""Assemble the verdict inputs of ТЗ Этап 4 "Правила вердиктов" from the result summaries and write
``results/verdicts.json`` (docs/design_experiments.md §5)::

    python -m flyguard.experiments.verdicts_run [--smoke] [--root DIR]

Inputs (per global seed, from ``results/<E>/summary.json`` -> ``numbers[key].per_seed[seed]``):

* **E1** (required): ``diff90/auc/<deep|dojo>/tfidf_lr-protectai_v2`` + ``auc/<src>/protectai_v2`` (H1a template
  half), ``diff/auc/<para_deep|bipia|dyn>/protectai_v2-tfidf_lr`` with its bootstrap ``p`` (H1a semantic half; the
  ТЗ's ``para`` is the deep paraphrase stratum), ``diff90/macro_auc/real_fly_linear-lr_svd`` + ``macro_auc/lr_svd``
  and ``diff90/macro_auc/flyhash_linear-tfidf_lr`` + ``macro_auc/tfidf_lr`` (H1b i), ``diff/macro_auc/<fly>_bloom-
  tfidf_lr`` (H1b ii, full training), ``val_auc/deep/real_fly_bloom`` and ``diff/fpr_notinject/tau90_deep/
  real_fly_bloom-protectai_v2`` / ``.../protectai_v2-piguard`` (H2; ``real_fly_linear`` reported as secondary),
  ``val_macro_auc/real_fly_bloom`` (H3 precondition).
* **E2** (optional, H1b ii few-shot): ``diff/macro_auc/shots<k>/<fly>_bloom-knn1`` for k in {1, 10} -- the 95 %
  paired interval of macroAUC(Bloom fly at k examples per class) − macroAUC(kNN(1) at k examples). Other spellings
  of the shots segment (``shot1``, ``1shot``, ``shots=1``, ``n1``, a bare ``1``, before or after the pair) are found
  by :func:`find_pair_key`.
* **E4** (optional, H3): ``diff90/macro_auc/<det>-curveball_mean`` for ``det`` in ``real_fly_bloom`` (primary),
  ``real_fly_linear`` and ``real_fly_bloom_10shot`` (secondary) -- the 90 % two-stage-bootstrap interval of
  macroAUC(measured M) − mean macroAUC(Curveball nulls); the null mean (the corridor's reference) is taken from the
  record's ``reference`` extra, else from ``macro_auc/curveball_mean/<det>``; the randomisation p from the record's
  ``p_randomization`` extra, else from ``p_randomization/macro_auc/<det>``. Any ``<b>`` segment containing
  ``curveball`` is accepted.
* ``results/power.json`` (required; the E0 gate of :mod:`flyguard.eval.verdicts` is mandatory). A file that is not
  frozen (stage 1) only adds a warning: the verdicts are then preliminary.

Keys the summaries do not have produce "не хватило данных" naming the key, never an exception, so a partial run
(E1 only) still yields a complete ``verdicts.json``.

Seeds. The verdict functions take one interval each; the experiments run ten global seeds and :mod:`results`
deliberately offers no averaged interval (a mean of percentile bounds is not an interval). The primitive is
therefore the *per-seed verdict*, and the aggregate status is the modal status over seeds, ties broken towards the
more conservative status (не хватило данных < предусловие не выполнено < опровергнута < подтверждена). Beside it the
file keeps every per-seed verdict, the count of seeds per status, the seed-mean effect and the *envelope* of the
per-seed intervals (min low, max high; descriptive only, no verdict is computed from it). This rule is not in the
ТЗ and is recorded as an assumption (see the report of this module's author).
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from flyguard.config import ROOT, Configs, config_hash, git_commit, load_configs
from flyguard.eval.verdicts import (CONFIRMED, INSUFFICIENT, PRECONDITION, REFUTED, Verdict, verdict_h1a,
                                    verdict_h1b, verdict_h2, verdict_h3)
from flyguard.experiments import results as results_mod
from flyguard.io import atomic_write_json, read_json

STATUS_RANK = {INSUFFICIENT: 0, PRECONDITION: 1, REFUTED: 2, CONFIRMED: 3}   # lower = more conservative
AGGREGATE_RULE = ("modal status over the global seeds; ties -> the more conservative status; effect = seed mean; "
                  "ci_envelope = [min low, max high] over seeds (descriptive only)")
LEXICAL = "tfidf_lr"
H1A_TEMPLATE = ("deep", "dojo")
H1A_SEMANTIC = {"para": "para_deep", "bipia": "bipia", "dyn": "dyn"}
H1B_VARIANTS = {"real_fly": ("real_fly_linear", "real_fly_bloom", "lr_svd"),
                "flyhash": ("flyhash_linear", "flyhash_bloom", LEXICAL)}
H2_FLIES = ("real_fly_bloom", "real_fly_linear")
H3_PRIMARY = "real_fly_bloom"
H3_SECONDARY = {"linear": "real_fly_linear", "bloom_10shot": "real_fly_bloom_10shot"}
SHOT_TOKENS = ("shots{k}", "shot{k}", "{k}shot", "{k}shots", "shots={k}", "n{k}", "{k}")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------------------------
class Summary:
    """Read access to one ``summary.json``: ``rec(key, seed)`` gives the per-seed number record."""

    def __init__(self, path: Path, data: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.data = dict(data)
        self.numbers: dict[str, Any] = dict(data.get("numbers") or {})
        self.seeds: list[int] = [int(s) for s in data.get("seeds") or []]
        self.config_hash = data.get("config_hash")

    @classmethod
    def load(cls, experiment: str, root: Path, smoke: bool) -> "Summary | None":
        path = results_mod.summary_path(experiment, smoke, root)
        if not path.exists():
            return None
        return cls(path, read_json(path))

    def rec(self, key: str | None, seed: int) -> dict[str, Any] | None:
        if key is None:
            return None
        return (self.numbers.get(key) or {}).get("per_seed", {}).get(str(int(seed)))

    def value(self, key: str | None, seed: int) -> float | None:
        r = self.rec(key, seed)
        return None if r is None else r.get("value")

    def keys(self) -> list[str]:
        return sorted(self.numbers)


def ci_from_record(rec: Mapping[str, Any] | None, level: float | None = None) -> dict[str, Any] | None:
    """A results number -> the CI dict of :class:`flyguard.eval.bootstrap.CI` (``None`` without bounds)."""
    if rec is None or rec.get("value") is None or rec.get("ci_low") is None or rec.get("ci_high") is None:
        return None
    out = {"point": float(rec["value"]), "low": float(rec["ci_low"]), "high": float(rec["ci_high"]),
           "level": float(rec.get("level") or level or 0.95), "n_boot": int(rec.get("n_boot") or 0),
           "n": int(rec.get("n") or 0)}
    for k in ("n_valid", "n_clusters"):
        if rec.get(k) is not None:
            out[k] = int(rec[k])
    return out


def find_pair_key(keys: Sequence[str], prefix: str, metric: str, a: str, b: str | None,
                  extra_tokens: Sequence[str] = ()) -> str | None:
    """The first key ``<prefix>/.../<a>-<b>`` whose segments contain ``metric`` and, when given, every token of
    ``extra_tokens`` (any one spelling per token list entry, ``|``-separated); ``b=None`` accepts any ``<b>`` and a
    ``b`` starting with ``~`` any partner *containing* the rest (``~curveball``)."""
    want_tokens = [set(t.split("|")) for t in extra_tokens]
    for key in sorted(keys):
        segs = key.split("/")
        if segs[0] != prefix or metric not in segs:
            continue
        for i, seg in enumerate(segs[1:], start=1):   # the pair segment may sit anywhere after the prefix
            if "-" not in seg:
                continue
            ka, kb = seg.split("-", 1)
            if ka != a:
                continue
            if b is not None and ((b.startswith("~") and b[1:] not in kb) or (not b.startswith("~") and kb != b)):
                continue
            others = set(segs[1:i] + segs[i + 1:])
            if all(others & toks for toks in want_tokens):
                return key
    return None


# ----------------------------------------------------------------------------------------------------------------
# Per-seed inputs
# ----------------------------------------------------------------------------------------------------------------
def h1a_inputs(e1: Summary, seed: int, comparator: str) -> dict[str, Any]:
    template: dict[str, Any] = {}
    keys: dict[str, str | None] = {}
    for s in H1A_TEMPLATE:
        dk, rk = f"diff90/auc/{s}/{LEXICAL}-{comparator}", f"auc/{s}/{comparator}"
        template[s] = {"diff_ci90": ci_from_record(e1.rec(dk, seed), 0.90), "reference": e1.value(rk, seed)}
        keys[f"template/{s}"] = dk
    semantic: dict[str, Any] = {}
    for s, src in H1A_SEMANTIC.items():
        dk = f"diff/auc/{src}/{comparator}-{LEXICAL}"
        rec = e1.rec(dk, seed)
        keys[f"semantic/{s}"] = dk
        if rec is not None:
            semantic[s] = {"diff_ci95": ci_from_record(rec), "p": rec.get("p")}
    return {"template": template, "semantic": semantic, "keys": keys}


def h1b_inputs(e1: Summary, e2: Summary | None, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"keys": {}}
    for variant, (linear, bloom, ref) in H1B_VARIANTS.items():
        ek, rk, fk = f"diff90/macro_auc/{linear}-{ref}", f"macro_auc/{ref}", f"diff/macro_auc/{bloom}-{LEXICAL}"
        fewshot: dict[str, Any] = {}
        for k in ("1", "10"):
            key = None
            if e2 is not None:
                key = find_pair_key(e2.keys(), "diff", "macro_auc", bloom, "knn1",
                                    ["|".join(t.format(k=k) for t in SHOT_TOKENS)])
            fewshot[k] = {"diff_ci95": ci_from_record(e2.rec(key, seed)) if (e2 is not None and key) else None}
            out["keys"][f"{variant}/fewshot/{k}"] = key or f"E2: diff/macro_auc/shots{k}/{bloom}-knn1 (missing)"
        out[variant] = {"equiv": {"diff_ci90": ci_from_record(e1.rec(ek, seed), 0.90), "reference": e1.value(rk, seed)},
                        "fewshot": fewshot, "full": {"diff_ci95": ci_from_record(e1.rec(fk, seed))}}
        out["keys"][f"{variant}/equiv"], out["keys"][f"{variant}/full"] = ek, fk
    return out


def h2_inputs(e1: Summary, seed: int, fly: str, comparator: str) -> dict[str, Any]:
    k1 = f"diff/fpr_notinject/tau90_deep/{fly}-{comparator}"
    k2 = f"diff/fpr_notinject/tau90_deep/{comparator}-piguard"
    return {"val_auc_deep": e1.value(f"val_auc/deep/{H3_PRIMARY}", seed),
            "fly_vs_protectai": {"diff_ci95": ci_from_record(e1.rec(k1, seed))},
            "protectai_vs_piguard": {"diff_ci95": ci_from_record(e1.rec(k2, seed))},
            "keys": {"fly_vs_protectai": k1, "protectai_vs_piguard": k2, "val_auc_deep": f"val_auc/deep/{H3_PRIMARY}"}}


def _h3_row(e4: Summary | None, det: str, seed: int) -> tuple[dict[str, Any] | None, str]:
    if e4 is None:
        return None, f"E4: diff90/macro_auc/{det}-curveball_mean (E4 summary missing)"
    key = find_pair_key(e4.keys(), "diff90", "macro_auc", det, "~curveball")
    if key is None:
        return None, f"E4: diff90/macro_auc/{det}-curveball_mean (missing)"
    rec = e4.rec(key, seed)
    if rec is None:
        return None, f"{key} (no seed {seed})"
    ref = rec.get("reference")
    if ref is None:
        ref = rec.get("delta_reference", rec.get("null_mean"))
    if ref is None:
        ref = e4.value(f"macro_auc/curveball_mean/{det}", seed)
    p = rec.get("p_randomization", rec.get("p_rand"))
    if p is None:
        p = e4.value(f"p_randomization/macro_auc/{det}", seed)
    return {"diff_ci90": ci_from_record(rec, 0.90), "reference": ref, "p_randomization": p}, key


def h3_inputs(e1: Summary, e4: Summary | None, seed: int) -> dict[str, Any]:
    val = e1.value(f"val_macro_auc/{H3_PRIMARY}", seed)
    if val is None and e4 is not None:
        val = e4.value(f"val_macro_auc/{H3_PRIMARY}", seed)
    primary, pk = _h3_row(e4, H3_PRIMARY, seed)
    secondary: dict[str, Any] = {}
    keys = {"primary": pk, "val_macro_auc": f"val_macro_auc/{H3_PRIMARY}"}
    pvals: dict[str, float] = {}
    for name, det in H3_SECONDARY.items():
        row, key = _h3_row(e4, det, seed)
        keys[f"secondary/{name}"] = key
        if row is not None:
            secondary[name] = row
            if row.get("p_randomization") is not None and math.isfinite(float(row["p_randomization"])):
                pvals[name] = float(row["p_randomization"])
    return {"val_macro_auc": val, "primary": primary or {}, "secondary": secondary, "p_values": pvals, "keys": keys}


# ----------------------------------------------------------------------------------------------------------------
# Aggregation over seeds
# ----------------------------------------------------------------------------------------------------------------
def _mean_tree(trees: Sequence[Any]) -> Any:
    """Element-wise mean of nested dicts of numbers (``None`` leaves are skipped)."""
    trees = [t for t in trees if t is not None]
    if not trees:
        return None
    if isinstance(trees[0], Mapping):
        keys = sorted({k for t in trees if isinstance(t, Mapping) for k in t})
        return {k: _mean_tree([t.get(k) for t in trees if isinstance(t, Mapping)]) for k in keys}
    vals = [float(t) for t in trees if isinstance(t, (int, float)) and math.isfinite(float(t))]
    return sum(vals) / len(vals) if vals else None


def _envelope_tree(trees: Sequence[Any]) -> Any:
    """Nested CI dicts -> ``{point: mean, low: min, high: max, n_seeds}``; other dicts recurse."""
    trees = [t for t in trees if t is not None]
    if not trees:
        return None
    if all(isinstance(t, Mapping) and "low" in t and "high" in t for t in trees):
        lows = [float(t["low"]) for t in trees if t.get("low") is not None]
        highs = [float(t["high"]) for t in trees if t.get("high") is not None]
        pts = [float(t["point"]) for t in trees if t.get("point") is not None]
        return {"point": (sum(pts) / len(pts)) if pts else None, "low": min(lows) if lows else None,
                "high": max(highs) if highs else None, "n_seeds": len(trees),
                "level": trees[0].get("level"), "descriptive_only": True}
    if isinstance(trees[0], Mapping):
        keys = sorted({k for t in trees if isinstance(t, Mapping) for k in t})
        return {k: _envelope_tree([t.get(k) for t in trees if isinstance(t, Mapping)]) for k in keys}
    return None


def aggregate(per_seed: Mapping[int, Verdict], hypothesis: str) -> dict[str, Any]:
    """The aggregate entry of ``verdicts.json`` for one hypothesis (module docstring, "Seeds")."""
    if not per_seed:
        return {"hypothesis": hypothesis, "status": INSUFFICIENT, "reason": "нет ни одного сида с результатами",
                "n_seeds": 0, "n_seeds_by_status": {}, "per_seed": {}, "aggregate_rule": AGGREGATE_RULE}
    counts = Counter(v.status for v in per_seed.values())
    top = max(counts.values())
    status = min((s for s, c in counts.items() if c == top), key=lambda s: STATUS_RANK.get(s, -1))
    with_status = sorted(s for s, v in per_seed.items() if v.status == status)
    rep = with_status[len(with_status) // 2]
    reason = (f"{status} в {counts[status]} из {len(per_seed)} сидов ({dict(sorted(counts.items()))}); "
              f"сид {rep}: {per_seed[rep].reason}")
    return {
        "hypothesis": hypothesis, "status": status, "reason": reason,
        "effect": _mean_tree([v.effect for v in per_seed.values()]),
        "ci_envelope": _envelope_tree([v.ci for v in per_seed.values()]),
        "n_seeds": len(per_seed), "n_seeds_by_status": {s: int(c) for s, c in sorted(counts.items())},
        "representative_seed": int(rep), "aggregate_rule": AGGREGATE_RULE,
        "per_seed": {str(s): per_seed[s].to_dict() for s in sorted(per_seed)},
    }


# ----------------------------------------------------------------------------------------------------------------
# Build and write
# ----------------------------------------------------------------------------------------------------------------
def _per_seed(seeds: Sequence[int], fn: Callable[[int], Verdict]) -> dict[int, Verdict]:
    return {int(s): fn(int(s)) for s in seeds}


def build_verdicts(root: Path = ROOT, smoke: bool = False, cfg: Configs | None = None) -> dict[str, Any]:
    """Read the summaries and ``power.json`` under ``root`` and return the ``verdicts.json`` payload."""
    root = Path(root)
    cfg = cfg or load_configs(root)
    ppath = results_mod.power_path(root, smoke)
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} is missing: run E0 first (the E0 gate of the verdict rules is mandatory)")
    power = read_json(ppath)
    e1 = Summary.load("E1", root, smoke)
    if e1 is None:
        raise FileNotFoundError(f"{results_mod.summary_path('E1', smoke, root)} is missing: run E1 first")
    e2, e4 = Summary.load("E2", root, smoke), Summary.load("E4", root, smoke)
    comparator = str(cfg.default["baselines"]["transformers"]["comparator"])
    seeds = e1.seeds
    warnings: list[str] = []
    if not power.get("frozen"):
        warnings.append("power.json не заморожен (стадия 1 E0): вердикты предварительные")
    current = config_hash(root)
    for name, h in (("power.json", power.get("config_hash")), ("E1", e1.config_hash),
                    ("E2", e2.config_hash if e2 else None), ("E4", e4.config_hash if e4 else None)):
        if h is not None and h != current:
            warnings.append(f"config_hash of {name} differs from the current configs")
    if e2 is None:
        warnings.append("нет results/E2/summary.json: H1b(ii) few-shot -> не хватило данных")
    if e4 is None:
        warnings.append("нет results/E4/summary.json: H3 -> не хватило данных")

    inputs_used: dict[str, Any] = {}

    def _h1a(seed: int) -> Verdict:
        inp = h1a_inputs(e1, seed, comparator)
        inputs_used.setdefault("H1a", inp["keys"])
        return verdict_h1a(inp, cfg, power)

    def _h1b(variant: str) -> Callable[[int], Verdict]:
        def fn(seed: int) -> Verdict:
            inp = h1b_inputs(e1, e2, seed)
            inputs_used.setdefault("H1b", inp["keys"])
            return verdict_h1b(inp, variant, cfg, power)
        return fn

    def _h2(fly: str) -> Callable[[int], Verdict]:
        def fn(seed: int) -> Verdict:
            inp = h2_inputs(e1, seed, fly, comparator)
            inputs_used.setdefault(f"H2/{fly}", inp["keys"])
            return verdict_h2(inp, cfg)
        return fn

    def _h3(seed: int) -> Verdict:
        inp = h3_inputs(e1, e4, seed)
        inputs_used.setdefault("H3", inp["keys"])
        return verdict_h3(inp, cfg, power)

    # The hypotheses sit at the top level (``H1b`` / ``H2_secondary`` hold one entry per variant): make_report's
    # ``iter_verdicts`` and check_acceptance read exactly this shape.
    verdicts = {
        "H1a": aggregate(_per_seed(seeds, _h1a), "H1a"),
        "H1b": {v: aggregate(_per_seed(seeds, _h1b(v)), "H1b") for v in H1B_VARIANTS},
        "H2": aggregate(_per_seed(seeds, _h2(H2_FLIES[0])), "H2"),
        "H2_secondary": {f: aggregate(_per_seed(seeds, _h2(f)), "H2") for f in H2_FLIES[1:]},
        "H3": aggregate(_per_seed(seeds, _h3), "H3"),
    }
    overview = {"H1a": verdicts["H1a"]["status"], "H1b": {v: verdicts["H1b"][v]["status"] for v in H1B_VARIANTS},
                "H2": verdicts["H2"]["status"], "H3": verdicts["H3"]["status"]}
    return {
        **verdicts,
        "created_at": _stamp(), "config_hash": current, "git_commit": git_commit(root), "smoke": bool(smoke),
        "seeds": seeds, "aggregate_rule": AGGREGATE_RULE, "comparator": comparator, "overview": overview,
        "power_config_hash": power.get("config_hash"), "power_frozen": bool(power.get("frozen")),
        "power_stage": power.get("stage"), "inputs": inputs_used,
        "sources": {"power": {"path": str(ppath), "frozen": bool(power.get("frozen")), "stage": power.get("stage"),
                              "created_at": power.get("created_at"), "config_hash": power.get("config_hash")},
                    "E1": {"path": str(e1.path), "seeds": e1.seeds, "config_hash": e1.config_hash},
                    "E2": None if e2 is None else {"path": str(e2.path), "seeds": e2.seeds, "config_hash": e2.config_hash},
                    "E4": None if e4 is None else {"path": str(e4.path), "seeds": e4.seeds, "config_hash": e4.config_hash}},
        "warnings": warnings,
    }


def verdicts_path(root: Path = ROOT, smoke: bool = False) -> Path:
    return results_mod.results_dir(root, smoke) / "verdicts.json"


def write_verdicts(root: Path = ROOT, smoke: bool = False, cfg: Configs | None = None) -> Path:
    payload = build_verdicts(root, smoke, cfg)
    path = verdicts_path(root, smoke)
    atomic_write_json(path, payload)
    return path


def main(argv: Sequence[str] | None = None, log: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="python -m flyguard.experiments.verdicts_run",
                                 description="results/verdicts.json from the summaries and power.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    path = write_verdicts(args.root, args.smoke)
    data = read_json(path)
    log(f"verdicts: {path}")
    for h, st in data["overview"].items():
        log(f"  {h}: {st}")
    for w in data["warnings"]:
        log(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
