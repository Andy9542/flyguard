"""Verdict rules of ТЗ Этап 4 "Правила вердиктов", applied literally to plain numbers and intervals.

Every function takes a dict of numbers / CI dicts (as stored in ``results/*.json``) and returns a
:class:`Verdict` with one of the four statuses of ТЗ 0: ``подтверждена`` (the rule's conditions all hold),
``опровергнута`` (data suffice by E0 and the conditions fail), ``не хватило данных`` (E0 says the source cannot
carry the metric, or an input the rule needs is missing), ``предусловие не выполнено`` (H2/H3 validation gate).
Nothing here reads data files; the experiments layer assembles the inputs and the report prints the outputs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from flyguard.config import Configs, load_configs
from flyguard.eval.bootstrap import CI, as_ci
from flyguard.eval.tost import bootstrap_p, equivalence_margin, holm, p_from_ci, tost, tost_equivalent

CONFIRMED = "подтверждена"
REFUTED = "опровергнута"
INSUFFICIENT = "не хватило данных"
PRECONDITION = "предусловие не выполнено"
STATUSES = (CONFIRMED, REFUTED, INSUFFICIENT, PRECONDITION)


@dataclass
class Verdict:
    """Status, effect (a number or a per-source dict), CI (dict or per-source dicts), the reason in Russian and the
    inputs the decision used (so the report can show every number behind the verdict)."""

    status: str
    effect: Any
    ci: Any
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_stats(cfg: Configs | None) -> dict[str, Any]:
    return (cfg or load_configs()).default["stats"]


def _ci(x: Any) -> CI | None:
    if x is None:
        return None
    c = as_ci(x)
    return c if math.isfinite(c.low) and math.isfinite(c.high) else None


def _d(c: CI | None) -> dict[str, Any] | None:
    return None if c is None else c.to_dict()


def _fmt(x: float | None) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:+.4f}"


def carriers_from_power(power: Mapping[str, Any], metric: str = "auc_diff") -> dict[str, bool]:
    """Read the E0 table (``power_table``) into ``{source: carries}`` for one metric (default: the AUC difference)."""
    from flyguard.eval.power import CARRIES

    return {s: (row.get(metric) == CARRIES) for s, row in (power.get("carriers") or {}).items()}


# ----------------------------------------------------------------------------------------------------------------
# H1a
# ----------------------------------------------------------------------------------------------------------------
def verdict_h1a(results: Mapping[str, Any], cfg: Configs | None = None) -> Verdict:
    """H1a (ТЗ Этап 4). Template half: on each of {deep, dojo} that carries the AUC difference by E0, the 90 % CI of
    AUC(TF-IDF) − AUC(ProtectAI v2) lies inside ±δ, δ = delta_rel × AUC(ProtectAI v2). Semantic half: on each present
    source of {para (deep stratum), bipia, dyn} the lower bound of the 95 % CI of AUC(ProtectAI v2) − AUC(TF-IDF) is
    above zero, with Holm over the sources. Confirmed when both halves hold.

    ``results = {"template": {src: {"diff_ci90", "reference", "carrier"}}, "semantic": {src: {"diff_ci95", "p"}},
    "delta_rel"?, "alpha"?}``. ``p`` is the two-sided percentile-bootstrap p-value of the difference (from the same
    draws as the CI); when absent it is approximated from the CI (flagged). Holm is applied to these p-values and a
    source passes when its adjusted p <= α *and* its 95 % lower bound is above zero (the literal wording keeps the
    interval condition; Holm adds the family-wise control). No carrying template source or no present semantic
    source -> "не хватило данных".
    """
    st = _cfg_stats(cfg)
    delta_rel = float(results.get("delta_rel", st["tost"]["delta_rel"]))
    alpha = float(results.get("alpha", st["bootstrap"]["alpha"]))
    template = results.get("template") or {}
    semantic = results.get("semantic") or {}
    inputs: dict[str, Any] = {"delta_rel": delta_rel, "alpha": alpha, "template": {}, "semantic": {}}
    effect: dict[str, Any] = {"template": {}, "semantic": {}}
    cis: dict[str, Any] = {"template": {}, "semantic": {}}

    carriers = []
    for s in ("deep", "dojo"):
        row = template.get(s)
        if not row:
            inputs["template"][s] = {"status": "нет данных"}
            continue
        ci = _ci(row.get("diff_ci90"))
        if not row.get("carrier", True) or ci is None or row.get("reference") is None:
            inputs["template"][s] = {"status": "не несёт по E0" if not row.get("carrier", True) else "нет ДИ"}
            continue
        delta = equivalence_margin(float(row["reference"]), delta_rel)
        t = tost(ci, delta)
        inputs["template"][s] = {**t, "reference": float(row["reference"]), "carrier": True}
        effect["template"][s], cis["template"][s] = ci.point, ci.to_dict()
        carriers.append((s, t["equivalent"]))

    present = []
    pvals: dict[str, float] = {}
    approx: list[str] = []
    for s in ("para", "bipia", "dyn"):
        row = semantic.get(s)
        ci = _ci(row.get("diff_ci95")) if row else None
        if ci is None:
            inputs["semantic"][s] = {"status": "источник отсутствует"}
            continue
        p = row.get("p")
        if p is None:
            p = p_from_ci(ci)
            approx.append(s)
        pvals[s] = float(p)
        present.append((s, ci))
        effect["semantic"][s], cis["semantic"][s] = ci.point, ci.to_dict()
    adj = holm(pvals)
    semantic_ok = True
    for s, ci in present:
        ok = ci.low > 0 and adj.get(s, 1.0) <= alpha
        inputs["semantic"][s] = {"point": ci.point, "low": ci.low, "high": ci.high, "p": pvals[s],
                                 "p_holm": adj.get(s), "passes": bool(ok), "p_approximate": s in approx}
        semantic_ok &= bool(ok)

    if not carriers:
        return Verdict(INSUFFICIENT, effect, cis, "ни один шаблонный источник {deep, dojo} не несёт разность AUC по "
                       "таблице мощности E0 (или нет ДИ)", inputs, "H1a")
    if not present:
        return Verdict(INSUFFICIENT, effect, cis, "нет ни одного семантического источника {para, bipia, dyn} с ДИ "
                       "разности", inputs, "H1a")
    template_ok = all(eq for _, eq in carriers)
    t_txt = ", ".join(f"{s}: {'эквивалентны' if eq else 'не эквивалентны'} (90% ДИ {_fmt(effect['template'][s])} "
                      f"[{cis['template'][s]['low']:+.4f}, {cis['template'][s]['high']:+.4f}], δ="
                      f"{inputs['template'][s]['delta']:.4f})" for s, eq in carriers)
    s_txt = ", ".join(f"{s}: {'выше нуля' if inputs['semantic'][s]['passes'] else 'не выше нуля'} (95% ДИ "
                      f"[{ci.low:+.4f}, {ci.high:+.4f}], p_Holm={inputs['semantic'][s]['p_holm']:.3g})"
                      for s, ci in present)
    reason = f"шаблонная половина: {t_txt}; семантическая половина: {s_txt}"
    if approx:
        reason += f"; p для {', '.join(approx)} приближены по ДИ (нормальная аппроксимация)"
    status = CONFIRMED if template_ok and semantic_ok else REFUTED
    return Verdict(status, effect, cis, reason, inputs, "H1a")


# ----------------------------------------------------------------------------------------------------------------
# H1b
# ----------------------------------------------------------------------------------------------------------------
def verdict_h1b(results: Mapping[str, Any], variant: str = "real_fly", cfg: Configs | None = None) -> Verdict:
    """H1b (ТЗ Этап 4) for one fly variant (``real_fly`` or ``flyhash``; the ТЗ reads them separately).
    (i) the 90 % CI of macroAUC(fly, linear readout) − macroAUC(reference) lies inside ±δ, δ = delta_rel × the
    reference macroAUC (LR on N51-svd for the real fly, TF-IDF on N16k for FlyHash); (ii) at 1 and 10 examples per
    class the 95 % CI of macroAUC(Bloom fly) − macroAUC(kNN(1)) contains zero or lies above it, and on full training
    the 95 % CI of macroAUC(Bloom fly) − macroAUC(TF-IDF) lies below zero. Confirmed when (i) and all of (ii) hold.

    ``results[variant] = {"equiv": {"diff_ci90", "reference", "carrier"?}, "fewshot": {"1": {"diff_ci95"},
    "10": {"diff_ci95"}}, "full": {"diff_ci95"}}``; a missing interval -> "не хватило данных".
    """
    st = _cfg_stats(cfg)
    delta_rel = float(results.get("delta_rel", st["tost"]["delta_rel"]))
    r = results.get(variant) or {}
    inputs: dict[str, Any] = {"variant": variant, "delta_rel": delta_rel}
    equiv = r.get("equiv") or {}
    ci_eq = _ci(equiv.get("diff_ci90"))
    shots = {k: _ci((r.get("fewshot") or {}).get(k, {}).get("diff_ci95")) for k in ("1", "10")}
    ci_full = _ci((r.get("full") or {}).get("diff_ci95"))
    effect = {"equiv": ci_eq.point if ci_eq else None, "fewshot": {k: (c.point if c else None) for k, c in shots.items()},
              "full": ci_full.point if ci_full else None}
    cis = {"equiv": _d(ci_eq), "fewshot": {k: _d(c) for k, c in shots.items()}, "full": _d(ci_full)}
    if not equiv.get("carrier", True):
        return Verdict(INSUFFICIENT, effect, cis, "macroAUC не несёт разность AUC по таблице мощности E0", inputs,
                       "H1b")
    missing = [n for n, c in (("(i) equiv", ci_eq), ("(ii) 1-shot", shots["1"]), ("(ii) 10-shot", shots["10"]),
                              ("(ii) full", ci_full)) if c is None]
    if missing or equiv.get("reference") is None:
        return Verdict(INSUFFICIENT, effect, cis, "нет ДИ для: " + ", ".join(missing or ["reference"]), inputs,
                       "H1b")
    delta = equivalence_margin(float(equiv["reference"]), delta_rel)
    t = tost(ci_eq, delta)
    few_ok = {k: bool(c.contains(0.0) or c.low > 0) for k, c in shots.items()}
    full_ok = bool(ci_full.high < 0)
    inputs.update({"i": {**t, "reference": float(equiv["reference"])},
                   "ii": {"fewshot_not_worse_than_knn1": few_ok, "full_below_tfidf": full_ok}})
    parts = [f"(i) {'эквивалентны' if t['equivalent'] else 'не эквивалентны'} (90% ДИ [{ci_eq.low:+.4f}, "
             f"{ci_eq.high:+.4f}], δ={delta:.4f})"]
    for k, c in shots.items():
        parts.append(f"(ii) {k} прим.: {'не хуже kNN(1)' if few_ok[k] else 'хуже kNN(1)'} "
                     f"(95% ДИ [{c.low:+.4f}, {c.high:+.4f}])")
    parts.append(f"(ii) полное обучение: {'уступает TF-IDF' if full_ok else 'не уступает TF-IDF'} "
                 f"(95% ДИ [{ci_full.low:+.4f}, {ci_full.high:+.4f}])")
    status = CONFIRMED if t["equivalent"] and all(few_ok.values()) and full_ok else REFUTED
    return Verdict(status, effect, cis, f"{variant}: " + "; ".join(parts), inputs, "H1b")


def verdicts_h1b(results: Mapping[str, Any], cfg: Configs | None = None) -> dict[str, Verdict]:
    """Both H1b verdicts, real fly and FlyHash separately."""
    return {v: verdict_h1b(results, v, cfg) for v in ("real_fly", "flyhash")}


# ----------------------------------------------------------------------------------------------------------------
# H2
# ----------------------------------------------------------------------------------------------------------------
def verdict_h2(results: Mapping[str, Any], cfg: Configs | None = None) -> Verdict:
    """H2 (ТЗ Этап 4). Precondition: AUC of the real fly on deepset validation >= 0.75
    (``stats.preconditions.h2_val_auc_deep``). Confirmed when the 95 % CI of FPR_NotInject(fly) −
    FPR_NotInject(ProtectAI v2) at τ_90(deep) lies inside ±10 pp (``stats.h2_margin_pp``) and the lower bound of
    the 95 % CI of FPR(ProtectAI v2) − FPR(PIGuard) is above zero.

    ``results = {"val_auc_deep", "fly_vs_protectai": {"diff_ci95"}, "protectai_vs_piguard": {"diff_ci95"},
    "precondition"?, "margin_pp"?}``; FPRs are shares in [0, 1]. A missing interval (e.g. PIGuard unavailable) ->
    "не хватило данных".
    """
    st = _cfg_stats(cfg)
    pre = float(results.get("precondition", st["preconditions"]["h2_val_auc_deep"]))
    margin = float(results.get("margin_pp", st["h2_margin_pp"])) / 100.0
    val = results.get("val_auc_deep")
    ci1 = _ci((results.get("fly_vs_protectai") or {}).get("diff_ci95"))
    ci2 = _ci((results.get("protectai_vs_piguard") or {}).get("diff_ci95"))
    effect = {"fly_minus_protectai": ci1.point if ci1 else None, "protectai_minus_piguard": ci2.point if ci2 else None}
    cis = {"fly_minus_protectai": _d(ci1), "protectai_minus_piguard": _d(ci2)}
    inputs = {"val_auc_deep": val, "precondition": pre, "margin": margin}
    if val is None or not math.isfinite(float(val)):
        return Verdict(INSUFFICIENT, effect, cis, "нет AUC настоящей мухи на валидации deepset", inputs, "H2")
    if float(val) < pre:
        return Verdict(PRECONDITION, effect, cis, f"AUC мухи на валидации deepset {float(val):.3f} < {pre:g}",
                       inputs, "H2")
    if ci1 is None or ci2 is None:
        miss = [n for n, c in (("муха − ProtectAI v2", ci1), ("ProtectAI v2 − PIGuard", ci2)) if c is None]
        return Verdict(INSUFFICIENT, effect, cis, "нет ДИ разности FPR: " + ", ".join(miss), inputs, "H2")
    in_corridor = ci1.inside(-margin, margin)
    piguard_lower = ci2.low > 0
    inputs.update({"fly_vs_protectai_inside_corridor": in_corridor, "piguard_lower_bound_positive": piguard_lower})
    reason = (f"FPR(муха) − FPR(ProtectAI v2) при τ_90(deep): {ci1.point:+.4f} [{ci1.low:+.4f}, {ci1.high:+.4f}] "
              f"{'внутри' if in_corridor else 'не внутри'} ±{margin:.2f}; FPR(ProtectAI v2) − FPR(PIGuard): "
              f"{ci2.point:+.4f} [{ci2.low:+.4f}, {ci2.high:+.4f}], нижняя граница "
              f"{'выше' if piguard_lower else 'не выше'} нуля")
    return Verdict(CONFIRMED if in_corridor and piguard_lower else REFUTED, effect, cis, reason, inputs, "H2")


# ----------------------------------------------------------------------------------------------------------------
# H3
# ----------------------------------------------------------------------------------------------------------------
def _h3_part(row: Mapping[str, Any] | None, delta_rel: float, alpha: float) -> dict[str, Any] | None:
    if not row:
        return None
    ci = _ci(row.get("diff_ci90"))
    ref = row.get("reference")
    if ci is None or ref is None:
        return {"status": "нет ДИ"}
    t = tost(ci, equivalence_margin(float(ref), delta_rel))
    p = row.get("p_randomization")
    p = None if p is None else float(p)
    if t["equivalent"]:
        reading = ("эквивалентна в коридоре, разность значимо отлична от нуля" if (p is not None and p <= alpha)
                   or t["differs_from_zero"] else "эквивалентна в коридоре и не отличается от нуля")
    else:
        reading = "различие больше коридора" if t["outside_corridor"] else "ДИ шире коридора: эквивалентность не установлена"
    return {**t, "reference": float(ref), "p_randomization": p, "reading": reading,
            "sign": "measured>null" if ci.point > 0 else ("measured<null" if ci.point < 0 else "0")}


def verdict_h3(results: Mapping[str, Any], cfg: Configs | None = None) -> Verdict:
    """H3 (ТЗ Этап 4). Precondition: macroAUC of the real fly (Bloom, full training) on validation >= 0.75
    (``stats.preconditions.h3_val_macro_auc``). Confirmed when TOST on test macroAUC holds for the primary readout:
    the 90 % two-stage-bootstrap CI of macroAUC(measured M) − mean macroAUC(curveball nulls) lies inside ±δ, δ =
    delta_rel × the null mean. Secondary readouts (linear, Bloom at 10 examples) and the randomisation p-values are
    reported alongside and do not change the status; the reason distinguishes "эквивалентна в коридоре" (possibly
    with a small significant difference) from "не отличается" (p above α), as the ТЗ requires.

    ``results = {"val_macro_auc", "primary": {"diff_ci90", "reference", "p_randomization", "carrier"?},
    "secondary": {name: {...same...}}, "p_values": {name: p}? (Holm over secondary sources/metrics),
    "precondition"?, "delta_rel"?, "alpha"?}``.
    """
    st = _cfg_stats(cfg)
    pre = float(results.get("precondition", st["preconditions"]["h3_val_macro_auc"]))
    delta_rel = float(results.get("delta_rel", st["tost"]["delta_rel"]))
    alpha = float(results.get("alpha", st["bootstrap"]["alpha"]))
    val = results.get("val_macro_auc")
    primary = results.get("primary") or {}
    part = _h3_part(primary, delta_rel, alpha)
    secondary = {k: _h3_part(v, delta_rel, alpha) for k, v in (results.get("secondary") or {}).items()}
    pv = {k: float(v) for k, v in (results.get("p_values") or {}).items() if v is not None}
    ci = _ci(primary.get("diff_ci90"))
    effect = ci.point if ci else None
    cis = _d(ci)
    inputs = {"val_macro_auc": val, "precondition": pre, "delta_rel": delta_rel, "primary": part,
              "secondary": secondary, "holm": holm(pv) if pv else {}}
    if val is None or not math.isfinite(float(val)):
        return Verdict(INSUFFICIENT, effect, cis, "нет macroAUC настоящей мухи на валидации", inputs, "H3")
    if float(val) < pre:
        return Verdict(PRECONDITION, effect, cis, f"macroAUC мухи на валидации {float(val):.3f} < {pre:g}", inputs,
                       "H3")
    if not primary.get("carrier", True):
        return Verdict(INSUFFICIENT, effect, cis, "macroAUC не несёт разность AUC по таблице мощности E0", inputs,
                       "H3")
    if part is None or "equivalent" not in part:
        return Verdict(INSUFFICIENT, effect, cis, "нет 90% ДИ разности macroAUC(измеренная) − среднее по нулям",
                       inputs, "H3")
    p_txt = "—" if part["p_randomization"] is None else f"{part['p_randomization']:.3g}"
    reason = (f"основной выход Bloom: {part['reading']} (90% ДИ {ci.point:+.4f} [{ci.low:+.4f}, {ci.high:+.4f}], "
              f"δ={part['delta']:.4f}, p рандомизации={p_txt})")
    sec_txt = [f"{k}: {v['reading']}" if v and "reading" in v else f"{k}: нет ДИ" for k, v in secondary.items()]
    if sec_txt:
        reason += "; вторичные: " + "; ".join(sec_txt)
    return Verdict(CONFIRMED if part["equivalent"] else REFUTED, effect, cis, reason, inputs, "H3")
