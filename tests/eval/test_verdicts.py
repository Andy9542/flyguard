"""verdicts.py: each of the four rules on hand-built inputs, reaching all four statuses; the E0 gate is mandatory."""
from __future__ import annotations

import pytest

from flyguard.eval.verdicts import (CONFIRMED, INSUFFICIENT, PRECONDITION, REFUTED, Verdict, carriers_from_power,
                                    verdict_h1a, verdict_h1b, verdict_h2, verdict_h3, verdicts_h1b)


def _power(**rows):
    """A minimal E0 table: ``_power(deep=True, dojo=False, macro=True)``."""
    return {"carriers": {s: {"auc_diff": "несёт" if v else "не хватило данных"} for s, v in rows.items()}}


def _h1a(ci_dict, template_ok=True, semantic_ok=True, carriers=("deep", "dojo"), semantic=("para", "bipia", "dyn")):
    t_ci = ci_dict(0.0, -0.02, 0.02, 0.9) if template_ok else ci_dict(-0.06, -0.09, -0.03, 0.9)
    s_ci = ci_dict(0.08, 0.03, 0.13) if semantic_ok else ci_dict(0.02, -0.02, 0.06)
    return {
        "template": {s: {"diff_ci90": t_ci, "reference": 0.9, "carrier": s in carriers} for s in ("deep", "dojo")},
        "semantic": {s: {"diff_ci95": s_ci, "p": 0.002 if semantic_ok else 0.4} for s in semantic},
    }


def test_h1a_all_statuses(ci_dict):
    v = verdict_h1a(_h1a(ci_dict))
    assert isinstance(v, Verdict) and v.status == CONFIRMED and v.hypothesis == "H1a"
    assert v.effect["template"]["deep"] == 0.0 and v.ci["semantic"]["para"]["low"] == 0.03
    assert v.inputs["template"]["deep"]["delta"] == pytest.approx(0.045)
    assert verdict_h1a(_h1a(ci_dict, template_ok=False)).status == REFUTED
    assert verdict_h1a(_h1a(ci_dict, semantic_ok=False)).status == REFUTED
    # only dojo carries: deep is ignored even though it is not equivalent
    r = _h1a(ci_dict, carriers=("dojo",))
    r["template"]["deep"]["diff_ci90"] = ci_dict(-0.2, -0.3, -0.1, 0.9)
    assert verdict_h1a(r).status == CONFIRMED
    assert verdict_h1a(_h1a(ci_dict, carriers=())).status == INSUFFICIENT
    assert verdict_h1a(_h1a(ci_dict, semantic=())).status == INSUFFICIENT
    # Holm: raw p-values 0.002 / 0.03 / 0.04 all pass alone, but 0.03 x 2 = 0.06 fails and drags 0.04 with it
    r = _h1a(ci_dict)
    r["semantic"]["bipia"] = {"diff_ci95": ci_dict(0.03, 0.001, 0.06), "p": 0.03}
    r["semantic"]["dyn"] = {"diff_ci95": ci_dict(0.03, 0.001, 0.06), "p": 0.04}
    v = verdict_h1a(r)
    assert v.status == REFUTED
    assert v.inputs["semantic"]["bipia"]["p_holm"] == pytest.approx(0.06)
    assert v.inputs["semantic"]["dyn"]["p_holm"] == pytest.approx(0.06) and not v.inputs["semantic"]["dyn"]["passes"]
    r["semantic"]["bipia"]["p"] = 0.01  # 0.01 x 2 = 0.02, 0.04 x 1 = 0.04 -> all pass
    assert verdict_h1a(r).status == CONFIRMED
    # a missing p is approximated from the CI and flagged in the reason
    r = _h1a(ci_dict)
    del r["semantic"]["para"]["p"]
    v = verdict_h1a(r)
    assert v.status == CONFIRMED and v.inputs["semantic"]["para"]["p_approximate"] and "приближены" in v.reason
    assert set(v.to_dict()) == {"status", "effect", "ci", "reason", "inputs", "hypothesis"}


def test_h1a_carrier_without_interval_is_missing_data(ci_dict):
    """The rule reads "на каждом из {deep, dojo}, несущем разность": a carrier with no CI cannot be skipped."""
    r = _h1a(ci_dict)
    r["template"]["deep"]["diff_ci90"] = None
    v = verdict_h1a(r)
    assert v.status == INSUFFICIENT and "deep" in v.reason and "несущего" in v.reason
    assert v.inputs["template"]["deep"] == {"status": "нет ДИ", "carrier": True, "e0": "несёт по E0"}
    r = _h1a(ci_dict)
    r["template"]["deep"]["reference"] = None
    assert verdict_h1a(r).status == INSUFFICIENT
    r = _h1a(ci_dict)
    del r["template"]["deep"]  # row absent entirely
    v = verdict_h1a(r)
    assert v.status == INSUFFICIENT and v.inputs["template"]["deep"]["status"] == "нет данных"
    # a non-carrier without an interval is legitimately skipped
    r = _h1a(ci_dict, carriers=("dojo",))
    r["template"]["deep"] = {"carrier": False}
    assert verdict_h1a(r).status == CONFIRMED


def test_h1a_e0_gate_is_mandatory(ci_dict):
    r = _h1a(ci_dict)
    del r["template"]["deep"]["carrier"]
    with pytest.raises(ValueError, match="E0"):
        verdict_h1a(r)  # data present, carrier status unknown, no power table: never a silent "carries"
    # the E0 table supersedes the per-row flags
    assert verdict_h1a(r, power=_power(deep=True, dojo=True)).status == CONFIRMED
    bad = _h1a(ci_dict)
    bad["template"]["deep"]["diff_ci90"] = ci_dict(-0.2, -0.3, -0.1, 0.9)
    assert verdict_h1a(bad, power=_power(deep=False, dojo=True)).status == CONFIRMED  # row flag True is overridden
    assert verdict_h1a(bad, power=_power(deep=True, dojo=True)).status == REFUTED
    v = verdict_h1a(bad, power=_power(dojo=True))  # deep absent from the E0 table: did not go through E0
    assert v.status == CONFIRMED and v.inputs["template"]["deep"]["status"] == "нет в таблице E0"
    assert verdict_h1a(bad, power=_power(macro=True)).status == INSUFFICIENT


def test_h1a_nan_p_is_missing_input(ci_dict):
    r = _h1a(ci_dict)
    r["semantic"]["para"]["p"] = float("nan")
    v = verdict_h1a(r)  # no TypeError from formatting a dropped Holm entry
    assert v.status == INSUFFICIENT and "para" in v.reason and v.inputs["semantic"]["para"]["status"] == "нет p"
    r = _h1a(ci_dict)
    r["semantic"]["bipia"] = {"diff_ci95": ci_dict(0.05, 0.05, 0.05)}  # degenerate interval, no p
    v = verdict_h1a(r)
    assert v.status == INSUFFICIENT and v.inputs["semantic"]["bipia"]["p_approximate"]


def _h1b(ci_dict, equiv=True, few=True, full=True, carrier=True):
    def part():
        return {
            "equiv": {"diff_ci90": ci_dict(0.0, -0.02, 0.02, 0.9) if equiv else ci_dict(-0.05, -0.08, -0.02, 0.9),
                      "reference": 0.8, "carrier": carrier},
            "fewshot": {"1": {"diff_ci95": ci_dict(0.02, -0.01, 0.05) if few else ci_dict(-0.05, -0.08, -0.02)},
                        "10": {"diff_ci95": ci_dict(0.05, 0.02, 0.08)}},
            "full": {"diff_ci95": ci_dict(-0.05, -0.08, -0.02) if full else ci_dict(0.0, -0.02, 0.02)},
        }

    return {"real_fly": part(), "flyhash": part()}


def test_h1b_all_statuses(ci_dict):
    v = verdict_h1b(_h1b(ci_dict), "real_fly")
    assert v.status == CONFIRMED and v.effect["equiv"] == 0.0 and v.inputs["i"]["delta"] == pytest.approx(0.04)
    assert verdict_h1b(_h1b(ci_dict, equiv=False), "real_fly").status == REFUTED
    assert verdict_h1b(_h1b(ci_dict, few=False), "flyhash").status == REFUTED
    assert verdict_h1b(_h1b(ci_dict, full=False), "flyhash").status == REFUTED
    r = _h1b(ci_dict)
    del r["flyhash"]["fewshot"]["10"]
    both = verdicts_h1b(r)
    assert both["real_fly"].status == CONFIRMED and both["flyhash"].status == INSUFFICIENT
    assert "10-shot" in both["flyhash"].reason
    assert verdict_h1b(_h1b(ci_dict, carrier=False), "real_fly").status == INSUFFICIENT
    assert verdict_h1b({}, "real_fly").status == INSUFFICIENT
    # E0 gate: complete data without a known carrier status raises; the power table supersedes the flag
    r = _h1b(ci_dict)
    del r["real_fly"]["equiv"]["carrier"]
    with pytest.raises(ValueError, match="E0"):
        verdict_h1b(r, "real_fly")
    assert verdict_h1b(r, "real_fly", power=_power(macro=True)).status == CONFIRMED
    assert verdict_h1b(_h1b(ci_dict), "real_fly", power=_power(macro=False)).status == INSUFFICIENT
    assert verdicts_h1b(r, power=_power(macro=True))["flyhash"].status == CONFIRMED


def _h2(ci_dict, val=0.9, corridor=True, piguard=True):
    return {
        "val_auc_deep": val,
        "fly_vs_protectai": {"diff_ci95": ci_dict(0.02, -0.05, 0.09) if corridor else ci_dict(0.08, 0.01, 0.15)},
        "protectai_vs_piguard": {"diff_ci95": ci_dict(0.10, 0.04, 0.16) if piguard else ci_dict(0.02, -0.03, 0.07)},
    }


def test_h2_all_statuses(ci_dict):
    v = verdict_h2(_h2(ci_dict))
    assert v.status == CONFIRMED and v.inputs["margin"] == pytest.approx(0.10) and v.inputs["precondition"] == 0.75
    assert verdict_h2(_h2(ci_dict, corridor=False)).status == REFUTED
    assert verdict_h2(_h2(ci_dict, piguard=False)).status == REFUTED
    assert verdict_h2(_h2(ci_dict, val=0.74)).status == PRECONDITION
    assert verdict_h2(_h2(ci_dict, val=0.75)).status == CONFIRMED  # inclusive bound
    r = _h2(ci_dict)
    r["protectai_vs_piguard"] = {"diff_ci95": None}  # PIGuard unavailable
    assert verdict_h2(r).status == INSUFFICIENT
    assert verdict_h2({"val_auc_deep": None}).status == INSUFFICIENT
    # exactly on the corridor edge counts as inside; margin override in pp
    r = _h2(ci_dict)
    r["fly_vs_protectai"]["diff_ci95"] = ci_dict(0.0, -0.10, 0.10)
    assert verdict_h2(r).status == CONFIRMED
    assert verdict_h2({**r, "margin_pp": 5}).status == REFUTED


def _h3(ci_dict, val=0.9, equiv=True, p=0.3, carrier=True):
    return {
        "val_macro_auc": val,
        "primary": {"diff_ci90": ci_dict(-0.01, -0.03, 0.01, 0.9) if equiv else ci_dict(-0.08, -0.11, -0.05, 0.9),
                    "reference": 0.8, "p_randomization": p, "carrier": carrier},
        "secondary": {"linear": {"diff_ci90": ci_dict(0.0, -0.02, 0.02, 0.9), "reference": 0.78,
                                 "p_randomization": 0.5},
                      "bloom_10shot": {"diff_ci90": None, "reference": 0.7}},
        "p_values": {"linear/deep": 0.01, "linear/dojo": 0.2},
    }


def test_h3_all_statuses(ci_dict):
    v = verdict_h3(_h3(ci_dict))
    assert v.status == CONFIRMED and v.effect == -0.01 and v.ci["level"] == 0.9
    assert "не отличается от нуля" in v.reason and "вторичные" in v.reason
    assert v.inputs["secondary"]["bloom_10shot"] == {"status": "нет ДИ"}
    assert v.inputs["holm"]["linear/deep"] == pytest.approx(0.02)
    small_real = verdict_h3(_h3(ci_dict, p=0.02))
    assert small_real.status == CONFIRMED and "значимо отлична" in small_real.reason
    ref = verdict_h3(_h3(ci_dict, equiv=False))
    assert ref.status == REFUTED and "больше коридора" in ref.reason
    assert verdict_h3(_h3(ci_dict, val=0.6)).status == PRECONDITION
    assert verdict_h3(_h3(ci_dict, carrier=False)).status == INSUFFICIENT
    r = _h3(ci_dict)
    r["primary"]["diff_ci90"] = None
    assert verdict_h3(r).status == INSUFFICIENT
    assert verdict_h3({"val_macro_auc": None}).status == INSUFFICIENT
    wide = _h3(ci_dict)
    wide["primary"]["diff_ci90"] = ci_dict(0.0, -0.1, 0.1, 0.9)
    v = verdict_h3(wide)
    assert v.status == REFUTED and "шире коридора" in v.reason
    # E0 gate
    r = _h3(ci_dict)
    del r["primary"]["carrier"]
    with pytest.raises(ValueError, match="E0"):
        verdict_h3(r)
    assert verdict_h3(r, power=_power(macro=True)).status == CONFIRMED
    assert verdict_h3(_h3(ci_dict), power=_power(macro=False)).status == INSUFFICIENT
    assert verdict_h3(_h3(ci_dict, p=None), power=_power(macro=True)).status == CONFIRMED  # None p formats as "—"



def test_h3_zero_difference_is_decided_at_alpha_not_by_the_90_interval(ci_dict):
    """The 90 % TOST interval excluding 0 is a 10 % test; "значимо отлична" needs p <= α or the 95 % interval."""
    r = _h3(ci_dict, p=0.3)
    r["primary"]["diff_ci90"] = ci_dict(0.01, 0.002, 0.018, 0.9)  # inside ±0.04, excludes 0 at 90 %
    v = verdict_h3(r)
    part = v.inputs["primary"]
    assert v.status == CONFIRMED and "не отличается от нуля" in v.reason and "значимо" not in v.reason
    assert part["differs_from_zero"] is True and part["differs_from_zero_alpha"] is False
    assert part["differs_basis"] == "p_randomization"
    # no p: the 95 % interval of the same draws decides
    r["primary"]["p_randomization"] = None
    r["primary"]["diff_ci95"] = ci_dict(0.01, -0.001, 0.021)
    part = verdict_h3(r).inputs["primary"]
    assert part["differs_from_zero_alpha"] is False and part["differs_basis"] == "ci95"
    r["primary"]["diff_ci95"] = ci_dict(0.01, 0.001, 0.019)
    assert "значимо отлична" in verdict_h3(r).reason
    # neither p (NaN counts as missing) nor a 95 % interval: equivalence stands, the zero test is not claimed
    r["primary"]["p_randomization"] = float("nan")
    del r["primary"]["diff_ci95"]
    v = verdict_h3(r)
    assert v.status == CONFIRMED and "не проверено" in v.reason and v.inputs["primary"]["differs_from_zero_alpha"] is None



def test_h3_gate_reads_the_two_stage_column_of_e0(ci_dict):
    """E0 folds the null-matrix and π spreads into the H3 power (``auc_diff_h3``); H3 must gate on it."""
    r = _h3(ci_dict)
    del r["primary"]["carrier"]
    power = {"carriers": {"macro": {"auc_diff": "несёт", "auc_diff_h3": "не хватило данных"}}}
    v = verdict_h3(r, power=power)
    assert v.status == INSUFFICIENT and v.inputs["carrier"]["metric"] == "auc_diff_h3"
    power["carriers"]["macro"]["auc_diff_h3"] = "несёт"
    assert verdict_h3(r, power=power).status == CONFIRMED
    # an older table without the column: the plain macro cell
    assert verdict_h3(r, power=_power(macro=True)).inputs["carrier"]["metric"] == "auc_diff"


def test_carriers_from_power():
    power = {"carriers": {"deep": {"auc_diff": "не хватило данных", "tpr_at_fpr": "только 5%"},
                          "dojo": {"auc_diff": "несёт", "tpr_at_fpr": "только 5%"}}}
    assert carriers_from_power(power) == {"deep": False, "dojo": True}
    assert carriers_from_power(power, "tpr_at_fpr") == {"deep": False, "dojo": False}
    assert carriers_from_power({}) == {}
