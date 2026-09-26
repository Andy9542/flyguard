"""Generation and judging plumbing (ТЗ 1.6, A.2-A.5, "Бюджет API"): prompt filling, strict parsing, refusals,
metering, the budget stop and idempotency, all through a fake transport (never the network)."""
from __future__ import annotations

import json

import pytest

from flyguard.gen import paraphrases as P
from flyguard.io import read_jsonl

SPEND_FIELDS = {"ts", "stream", "model", "peak", "prompt_tokens", "cache_hit", "cache_miss", "completion_tokens",
                "cost_usd", "latency_s", "base_id", "role"}


def inj_base(T) -> P.Base:
    return P.Base("deep_inj:0", "deep:test:0", P.KIND_INJ, T.INJ_TEXT, origin="deepset_test")


def test_prompt_fill_keeps_json_braces_and_counts(make_rt, T):
    """A.2/A.3: only the named placeholders are replaced; the JSON schema braces in the user part survive."""
    rt = make_rt()
    sys_, user = (m["content"] for m in P.generator_messages(rt, inj_base(T), ["ignore", "zebraword"]))
    assert "Produce 2 SHALLOW" in sys_ and "Produce 2 DEEP" in sys_ and "any inflection: ignore, zebraword." in sys_
    assert "{n_shallow}" not in sys_ and "{banned_words}" not in sys_ and "{base_text}" not in user
    assert '{"paraphrases": [{"stratum": "shallow" | "deep", "text": "..."}]}' in user and T.INJ_TEXT in user
    ben = P.Base("deep_ben:1", "deep:test:1", P.KIND_BEN, T.BEN_TEXT)
    sys_b, user_b = (m["content"] for m in P.generator_messages(rt, ben, []))
    assert "Produce 4 paraphrases" in sys_b and '{"paraphrases": [{"text": "..."}]}' in user_b
    jsys, juser = (m["content"] for m in P.judge_messages(rt, inj_base(T), T.INJ_DEEP))
    assert "same_action" in jsys and T.INJ_TEXT in juser and T.INJ_DEEP in juser and "{candidate_text}" not in juser
    assert "shallow" not in juser.lower().replace(T.INJ_TEXT.lower(), "")   # stratum never shown to the judge


def test_generator_reply_parsing_is_strict():
    good = '{"paraphrases": [{"stratum": "deep", "text": " abc "}, {"stratum": "shallow", "text": "d", "note": 1}]}'
    items, why = P.parse_generator_reply(good, P.KIND_INJ)
    assert why is None and [i["text"] for i in items] == ["abc", "d"] and items[0]["declared_stratum"] == "deep"
    bad = {None: "empty_content", "": "empty_content", "Sure, here you go": "json_invalid",
           '{"paraphrases": []}': "schema", '{"paraphrases": {"text": "x"}}': "schema", '["x"]': "schema",
           '{"paraphrases": [{"text": "x"}]}': "schema", '{"paraphrases": [{"stratum": "mid", "text": "x"}]}': "schema",
           '{"paraphrases": [{"stratum": "deep", "text": ""}]}': "schema",
           '{"paraphrases": [{"stratum": "deep", "text": 5}]}': "schema"}
    for text, reason in bad.items():
        assert P.parse_generator_reply(text, P.KIND_INJ) == (None, reason), text
    items, why = P.parse_generator_reply('{"paraphrases": [{"text": "x"}]}', P.KIND_BEN)
    assert why is None and items[0]["declared_stratum"] is None


def test_judge_json_outside_schema_counts_as_refusal():
    """ТЗ 2.6 / A.6: a judge reply that is not exactly the schema (string booleans, missing confidence, prose)
    is a refusal, never an acceptance."""
    ok, why = P.parse_judge_reply('{"same_action": true, "still_instruction": true, "confidence": 0.8, "reason": "r"}', P.KIND_INJ)
    assert why is None and ok["confidence"] == 0.8 and P.accept_verdict(P.KIND_INJ, ok["verdict"])
    bad = {'{"same_action": "true", "still_instruction": true, "confidence": 0.9}': "schema",
           '{"same_action": true, "confidence": 0.9}': "schema",
           '{"same_action": true, "still_instruction": true}': "schema",
           '{"same_action": true, "still_instruction": true, "confidence": 1.5}': "schema",
           '{"same_action": true, "still_instruction": true, "confidence": true}': "schema",
           '{"same_action": true, "still_instruction": true, "confidence": 0.5, "reason": 3}': "schema",
           '{"meaning_preserved": true, "contains_instruction_to_ai": false, "confidence": 0.5}': "schema",
           'same_action: true': "json_invalid", "": "empty_content", "[]": "schema"}
    for text, reason in bad.items():
        assert P.parse_judge_reply(text, P.KIND_INJ) == (None, reason), text
    ok, _ = P.parse_judge_reply('{"meaning_preserved": true, "contains_instruction_to_ai": false, "confidence": 1}', P.KIND_BEN)
    assert P.accept_verdict(P.KIND_BEN, ok["verdict"])
    assert not P.accept_verdict(P.KIND_BEN, {"meaning_preserved": True, "contains_instruction_to_ai": True})
    assert not P.accept_verdict(P.KIND_INJ, {"same_action": True, "still_instruction": False})


def test_budget_stop_with_fake_transport_and_spend_records(make_rt, T):
    """ТЗ 2.6 / "Бюджет API": every call is metered with the trace runner's record shape; generation stops when
    the sum over results/spend/*.jsonl (all streams) reaches budget_usd, and a rerun resumes the missing calls."""
    rt = make_rt(budget=0.005)
    bases = P.deepset_bases(rt)
    out = P.generate_stage(rt, bases)
    assert out["stopped"] and "budget" in out["stopped"] and out["ok"] == 3
    recs = read_jsonl(rt.paths.spend)
    assert len(recs) == 3 == len(rt.transport.calls)
    for r in recs:
        assert SPEND_FIELDS <= set(r) and r["stream"] == "paraphrases" and r["role"] == "generator"
        assert r["peak"] is False and r["cost_usd"] == pytest.approx(T.COST_PER_CALL) and r["model"] == T.GEN
        assert r["cache_miss"] == 1000 and r["completion_tokens"] == 500 and r["base_id"].startswith("deep_")
    assert P.spend_total(rt.paths.spend_dir) == pytest.approx(3 * T.COST_PER_CALL)
    (rt.paths.spend_dir / "traces.jsonl").write_text(json.dumps({"cost_usd": 1.0, "stream": "traces"}) + "\n")
    rt2 = make_rt(budget=1.005, transport=T.FakeTransport())
    assert P.generate_stage(rt2, bases)["stopped"] and rt2.transport.calls == []
    rt3 = make_rt(budget=5.0, transport=T.FakeTransport())
    out3 = P.generate_stage(rt3, bases)
    assert out3["stopped"] is None and len(rt3.transport.calls) == 8 - 3 and out3["skipped"] == 3


def test_generate_is_idempotent_and_counts_refusals(make_rt, T):
    rt = make_rt()
    bases = P.deepset_bases(rt)
    out = P.generate_stage(rt, bases)
    assert out["ok"] == 4 and out["refusal"] == 4 and out["candidates"] == 16 and out["stopped"] is None
    calls = read_jsonl(rt.paths.calls)
    assert len(calls) == 8 and {c["refusal_reason"] for c in calls if c["status"] == "refusal"} == {"json_invalid"}
    assert all(c["raw_sha256"] for c in calls) and all("text" not in c for c in calls)
    rt2 = make_rt(transport=T.FakeTransport())
    out2 = P.generate_stage(rt2, bases)
    assert rt2.transport.calls == [] and out2["skipped"] == 8 and out2["processed"] == 0
    cands = read_jsonl(rt.paths.candidates)
    assert len(cands) == 16 and len({c["cand_id"] for c in cands}) == 16
    assert {c["declared_stratum"] for c in cands if c["kind"] == P.KIND_BEN} == {None}


def test_api_error_is_retried_next_run_but_provider_error_is_a_refusal(make_rt, T):
    def fail(messages, model, n):
        if n == 1:
            raise P.TransportError("boom", retryable=True)
        if n == 2:
            raise P.TransportError("policy 400", retryable=False)
        return None
    rt = make_rt(transport=T.FakeTransport(fail=fail))
    st = P.generate_base(rt, inj_base(T), [], P.load_calls(rt))
    assert st == {"api_error": 1, "refusal": 1}
    recs = read_jsonl(rt.paths.calls)
    assert [r["status"] for r in recs] == ["api_error", "refusal"] and recs[1]["refusal_reason"] == "provider_error"
    assert not rt.paths.spend.exists()                       # no usage -> no spend record
    rt2 = make_rt(transport=T.FakeTransport())
    st2 = P.generate_base(rt2, inj_base(T), [], P.load_calls(rt2))
    assert st2 == {"ok": 1, "candidates": 4, "skipped": 1}   # call 0 redone, call 1 (refusal) kept


def test_peak_hours_wait_and_price(make_rt, T):
    """A2: in a peak window the runner sleeps until off-peak and the record carries peak=True, prices x2."""
    rt = make_rt(now=T.PEAK)
    P.generate_base(rt, inj_base(T), [], {})
    assert rt.sleeps and rt.sleeps[0] > 0
    rec = read_jsonl(rt.paths.spend)[0]
    assert rec["peak"] is True and rec["cost_usd"] == pytest.approx(2 * T.COST_PER_CALL)
    rt_off = make_rt(now=T.OFFPEAK)
    P.generate_base(rt_off, inj_base(T), [], {})
    assert rt_off.sleeps == []


def test_transport_receives_the_configured_options(make_rt, T):
    rt = make_rt()
    P.generate_base(rt, inj_base(T), [], {})
    call = rt.transport.calls[0]
    assert call["temperature"] == 0.9 and call["model"] == T.GEN
    assert call["options"]["response_format"] == {"type": "json_object"} and call["options"]["max_tokens"] == 512
    assert call["options"]["extra_body"] == {"thinking": {"type": "disabled"}}
    cands = read_jsonl(rt.paths.candidates)
    P.judge_candidate(rt, inj_base(T), cands[0], {})
    jcall = rt.transport.calls[-1]
    assert jcall["temperature"] == 0.0 and jcall["model"] == T.JUDGE
    rec = read_jsonl(rt.paths.judgements)[0]
    assert rec["status"] == "ok" and rec["accept"] is True and rec["judge"] == T.JUDGE
    assert read_jsonl(rt.paths.spend)[-1]["role"] == "judge"
    assert not rt.paths.netlog.exists()                      # the fake transport never logs a request
