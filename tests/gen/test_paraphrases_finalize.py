"""End to end with the fake transport, then finalize/manifest properties (ТЗ 1.6 "Объём", A.6, design §7):
selection determinism, both strata, judge unanimity and refusals, CSV round trip, manifest shape and rates."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np
import pytest

from flyguard.config import seeds_for
from flyguard.gen import paraphrases as P
from flyguard.io import read_jsonl

TMPL0 = "tmpl:important_instructions:alpha:injection_task_0"


def rows_by_base(rt):
    with open(rt.paths.csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = defaultdict(list)
    for r in rows:
        by[r["base_id"]].append(r)
    return rows, by


def test_end_to_end_with_fake_transport(make_rt, T):
    rt = make_rt(smoke=True)
    out = P.run_all(rt, smoke=True)
    assert out["stopped"] is None and out["processed"] == 3
    rows, by = rows_by_base(rt)
    assert list(rows[0]) == P.CSV_COLUMNS and len(rows) == 8
    inj, ben, tmpl = by["deep_inj:0"], by["deep_ben:1"], by[TMPL0]
    assert {r["text"] for r in inj} == {T.INJ_SHALLOW, T.INJ_DEEP, T.INJ_DEEP_BANNED}
    assert {r["stratum"] for r in inj} == {"shallow", "deep"} and {r["label"] for r in inj} == {"1"}
    assert {r["text"] for r in ben} == {T.BEN_SHALLOW, T.BEN_DEEP} and {r["label"] for r in ben} == {"0"}
    assert {r["stratum"] for r in ben} == {"shallow", "deep"}
    assert {r["text"] for r in tmpl} == {T.T_SHALLOW, T.T_DEEP, T.T_DEEP2}      # newline + quotes survive the CSV
    for base_id, rs in by.items():
        assert [r["para_id"] for r in rs] == [f"para:{base_id}:{k}" for k in range(len(rs))]
        assert rs[0]["stratum"] == "deep"                                         # deep first in the round robin
        assert all(float(r["jaccard_to_base"]) <= 0.5 and r["generator"] == T.GEN for r in rs)
        assert all(float(r["judge_confidence"]) == 0.9 for r in rs)
    spend = read_jsonl(rt.paths.spend)
    assert Counter(r["role"] for r in spend) == {"generator": 6, "judge": 8}
    assert not rt.paths.netlog.exists()
    filtered = read_jsonl(rt.paths.filtered)
    # call 2 repeats call 1: its survivors are duplicates (5 + 2 + 3); language/length reject on both calls
    assert Counter(f["reason"] for f in filtered if not f["passed"]) == {"duplicate": 10, "language": 2, "length": 2}
    assert sum(f["demoted"] for f in filtered) == 1
    csv_bytes = rt.paths.csv.read_bytes()
    rt2 = make_rt(smoke=True, transport=T.FakeTransport())
    out2 = P.run_all(rt2, smoke=True)
    assert rt2.transport.calls == [] and out2["processed"] == 0 and rt2.paths.csv.read_bytes() == csv_bytes


def test_selection_is_deterministic_and_represents_strata_and_generators():
    """A.6 / ТЗ 2.6: same seed -> same choice; up to 5 per base with both strata and both generators."""
    acc = []
    for i, (s, g) in enumerate([("deep", "gen-a")] * 3 + [("deep", "gen-b")] * 2 + [("shallow", "gen-a")] * 2
                               + [("shallow", "gen-b")]):
        acc.append({"cand_id": f"b|{g}|0|{i}", "stratum": s, "generator": g, "text": str(i)})
    pick = lambda seed: [c["cand_id"] for c in P.select_for_base(acc, 5, np.random.default_rng(seed), ["gen-a", "gen-b"])]  # noqa: E731
    first = pick(123)
    assert first == pick(123) and len(first) == 5 == len(set(first))
    chosen = [c for c in acc if c["cand_id"] in first]
    assert {c["stratum"] for c in chosen} == {"deep", "shallow"} and {c["generator"] for c in chosen} == {"gen-a", "gen-b"}
    assert [c["stratum"] for c in P.select_for_base(acc, 5, np.random.default_rng(1), ["gen-a", "gen-b"])][:4] == \
           ["deep", "deep", "shallow", "shallow"]
    assert len({tuple(pick(s)) for s in range(8)}) > 1                       # the seed does matter
    assert P.select_for_base([], 5, np.random.default_rng(0), []) == []
    only_deep = [c for c in acc if c["stratum"] == "deep"]
    assert len(P.select_for_base(only_deep, 5, np.random.default_rng(0), ["gen-a", "gen-b"])) == 5


def test_finalize_twice_same_seed_identical_and_seed_recorded(make_rt, T):
    rt = make_rt(smoke=True)
    P.run_all(rt, smoke=True)
    a = rt.paths.csv.read_bytes()
    st = P.finalize_stage(rt)
    assert rt.paths.csv.read_bytes() == a and st["seed_paraphrase"] == seeds_for(rt.cfg, 0)["paraphrase"]
    rt1 = make_rt(smoke=True, seed=1, transport=T.FakeTransport())
    st1 = P.finalize_stage(rt1)
    assert st1["selected"] == 8 and st1["seed_paraphrase"] == seeds_for(rt1.cfg, 1)["paraphrase"]
    assert {r["text"] for r in rows_by_base(rt1)[0]} == {r["text"] for r in rows_by_base(rt)[0]}   # ≤ 5 accepted: same set


def test_two_judges_require_unanimity(make_rt, T):
    """ТЗ 1.6: disagreement between judges is recorded in judge_agreement and drops the candidate."""
    reject = P.normalize_text(T.INJ_DEEP)
    judge = lambda base, cand, model: {"accept": not (model == T.JUDGE_B and cand == reject), "confidence": 0.8}  # noqa: E731
    judges = [{"provider": "fake", "model": T.JUDGE}, {"provider": "fake", "model": T.JUDGE_B}]
    rt = make_rt(smoke=True, judges=judges, transport=T.FakeTransport(judge=judge))
    P.run_all(rt, smoke=True)
    rows, by = rows_by_base(rt)
    assert {r["text"] for r in by["deep_inj:0"]} == {T.INJ_SHALLOW, T.INJ_DEEP_BANNED} and len(rows) == 7
    m = json.loads(rt.paths.manifest.read_text(encoding="utf-8"))
    assert m["judge_agreement"] == {"n_judges": 2, "n_candidates_fully_judged": 8, "n_agree": 7, "n_disagree": 1}
    assert Counter(j["judge"] for j in read_jsonl(rt.paths.judgements)) == {T.JUDGE: 8, T.JUDGE_B: 8}
    assert m["rates"]["acceptance"]["by_stratum"]["deep"]["accepted"] == 3
    assert all(float(r["judge_confidence"]) == 0.8 for r in rows)


def test_judge_refusals_are_counted_not_accepted(make_rt, T):
    prose = P.normalize_text(T.INJ_SHALLOW)
    typed = P.normalize_text(T.BEN_DEEP)
    def judge(base, cand, model):
        if cand == prose:
            return {"raw": "Sure! same_action: true, still_instruction: true"}
        if cand == typed:
            return {"raw": '{"meaning_preserved": "true", "contains_instruction_to_ai": false, "confidence": 0.9}'}
        return {"accept": True, "confidence": 0.7}
    rt = make_rt(smoke=True, transport=T.FakeTransport(judge=judge))
    P.run_all(rt, smoke=True)
    rows, by = rows_by_base(rt)
    assert T.INJ_SHALLOW not in {r["text"] for r in rows} and T.BEN_DEEP not in {r["text"] for r in rows}
    assert len(rows) == 6 and {r["stratum"] for r in by["deep_ben:1"]} == {"shallow"}
    m = json.loads(rt.paths.manifest.read_text(encoding="utf-8"))
    assert m["counts"]["judge_refusal_reasons"] == {"json_invalid": 1, "schema": 1}
    assert m["rates"]["judge_refusal"]["by_judge"][T.JUDGE] == {"n": 8, "refusals": 2, "rate": 0.25}
    rt2 = make_rt(smoke=True, transport=T.FakeTransport())
    P.run_all(rt2, smoke=True)
    assert rt2.transport.calls == []                                            # refusals are final, not retried


def test_manifest_shape_and_rates(make_rt, T):
    rt = make_rt(smoke=True)
    P.run_all(rt, smoke=True)
    m = json.loads(rt.paths.manifest.read_text(encoding="utf-8"))
    assert {"generated", "seed", "provider", "models", "dates", "prompts_sha256", "banned_words", "fill_strings",
            "filters", "counts", "rates", "judge_agreement", "spend", "files"} <= set(m)
    assert m["models"]["generators"][0]["model"] == T.GEN and m["models"]["judges"][0]["model"] == T.JUDGE
    assert m["models"]["temperature_generate"] == 0.9 and m["models"]["temperature_judge"] == 0.0
    assert m["dates"]["first_call"] == "2026-09-26T12:00:00Z" == m["dates"]["last_call"]
    assert set(m["prompts_sha256"]) == {"generator_injection", "generator_benign", "judge_injection", "judge_benign", "banned_words"}
    assert all(len(v) == 64 for v in m["prompts_sha256"].values()) and "key_env" not in m["provider"]
    bw = m["banned_words"]
    assert bw["final"] == bw["starter"] + bw["chi2_extra"] and len(bw["chi2_extra"]) == 3 and bw["n_final"] == len(bw["final"])
    assert m["fill_strings"] == P.FILL and m["seed"] == {"global": 0, "paraphrase": seeds_for(rt.cfg, 0)["paraphrase"]}
    c = m["counts"]
    assert c["bases"] == {"template": 1, "deepset_injection": 1, "deepset_benign": 1}
    assert c["bases_by_template"] == {"important_instructions": 1} and c["bases_by_origin"]["trace_log"] == 1
    assert c["template_verification"] == {"composed_equals_log": 1, "composed_differs_from_log": 0, "from_log": 1, "composed": 7}
    assert c["generation_calls"] == {"ok": 6} and c["candidates"] == 22 and c["filtered_passed"] == 8
    assert c["selected"] == 8 and c["selected_by_label"] == {"1": 6, "0": 2} and c["accepted_candidates"] == 8
    assert c["selected_by_stratum"] == {"deep": 4, "shallow": 4} and c["demoted_deep_to_shallow"] == 1
    r = m["rates"]
    assert r["generation_refusal"]["by_generator"][T.GEN] == {"n": 6, "refusals": 0, "rate": 0.0}
    assert set(r["generation_refusal"]["by_template"]) == {"important_instructions", "deepset_injection", "deepset_benign"}
    assert r["acceptance"]["by_stratum"]["deep"]["rate"] == 1.0 and set(r["acceptance"]["by_kind"]) == set(P.KINDS)
    assert m["spend"]["n_calls"] == 14 and m["spend"]["cost_usd"] == pytest.approx(14 * T.COST_PER_CALL)
    assert m["spend"]["by_role"]["judge"]["n_calls"] == 8 and m["spend"]["budget_usd"] == 1.0
    assert set(m["files"]) == {"bases.jsonl", "candidates.jsonl", "judgements.jsonl", "paraphrases.csv"}


def test_refusal_rates_by_template_over_all_bases(make_rt, T):
    """ТЗ 1.6: refusal shares by generator and by template are visible even when whole templates are refused."""
    rt = make_rt()
    bases, _ = P.build_bases(rt)
    out = P.generate_stage(rt, bases)
    assert out["ok"] == 6 and out["refusal"] == 18
    P.filter_stage(rt, bases)
    P.judge_stage(rt, bases)
    P.finalize_stage(rt, bases)
    m = P.write_manifest(rt, bases)
    bt = m["rates"]["generation_refusal"]["by_template"]
    assert bt["ignore_previous"] == {"n": 4, "refusals": 4, "rate": 1.0}
    assert bt["important_instructions"] == {"n": 4, "refusals": 2, "rate": 0.5}
    assert bt["deepset_benign"] == {"n": 4, "refusals": 2, "rate": 0.5}
    assert m["rates"]["generation_refusal"]["by_generator"][T.GEN]["rate"] == pytest.approx(0.75)
    assert m["counts"]["selected"] == 8 and m["counts"]["bases_with_output"] == 3
