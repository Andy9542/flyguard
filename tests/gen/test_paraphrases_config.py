"""The frozen config keys of configs/default.yaml govern behaviour (review: paraphrase.chi2, filters.shallow_range
and judge_reason_max_chars were dead config): changing each key changes the output. Plus the single-line
generation ledger (call + candidates in one record)."""
from __future__ import annotations

import json

import pytest

from flyguard.gen import paraphrases as P
from flyguard.io import read_jsonl


def inj_base(T) -> P.Base:
    return P.Base("deep_inj:0", "deep:test:0", P.KIND_INJ, T.INJ_TEXT, origin="deepset_test")


def cand(text: str, declared: str | None = None, k: int = 0) -> dict:
    return {"cand_id": f"x|gen-a|0|{k}", "generator": "gen-a", "call_index": 0, "k": k, "text": text,
            "declared_stratum": declared}


# ------------------------------------------------------------------------------------------------ chi2 flags

INJ_DOCS = [f"the ok zebra poem {i}" for i in range(6)]                   # "the" in every injection document
BEN_DOCS = [f"pancake {i}" + (" the" if i < 3 else "") for i in range(6)]   # "the" in half of the benign ones
LABELS = [1] * 6 + [0] * 6


def test_chi2_min_token_len_flag():
    """min_token_len 3 drops "ok"; 2 admits it (same χ² as poem/zebra, alphabetical tie-break)."""
    assert P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 2) == ["poem", "zebra"]
    assert P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 3, min_token_len=2) == ["ok", "poem", "zebra"]


def test_chi2_stopword_flag():
    """"the" is injection-enriched here (6/6 vs 3/6) but an English stop word: excluded unless the flag is off."""
    assert "the" not in P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 5)
    assert P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 3, exclude_english_stopwords=False) == ["poem", "zebra", "the"]


def test_chi2_injection_enriched_flag():
    """"pancake" has the top χ² but is benign-enriched: excluded unless injection_enriched_only is off."""
    assert "pancake" not in P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 5)
    assert P.chi2_extension(INJ_DOCS + BEN_DOCS, LABELS, [], 3, injection_enriched_only=False) == ["pancake", "poem", "zebra"]


def test_final_banned_list_reads_the_chi2_config(make_rt, T):
    """final_banned_list passes paraphrase.chi2 through and records the rule beside the list."""
    rt = make_rt()
    d = P.final_banned_list(rt)
    assert d["chi2_extra"] == ["poem", "topic", "write"] and d["partial"] is False
    assert d["chi2_rule"] == {"min_token_len": 3, "exclude_english_stopwords": True, "injection_enriched_only": True}
    rt6 = make_rt(cfg=T.make_cfg(chi2={"min_token_len": 6, "exclude_english_stopwords": True, "injection_enriched_only": True}))
    rt6.paths.banned.unlink()
    assert P.final_banned_list(rt6)["chi2_extra"] == ["zebraword"]      # only tokens of >= 6 letters remain


# ------------------------------------------------------------------------------------------------ shallow_range

def test_shallow_range_governs_the_stratum_and_keeps_the_a6_demotion(T):
    """filters.shallow_range: a candidate between deep_max and the range's lower bound is dropped
    ("stratum_gap"); a declared-deep injection candidate failing only the word rule is still demoted to shallow
    below the range (A.6); with the frozen [0.3, 0.5] the gap is empty."""
    j = P.jaccard_texts(T.INJ_TEXT, T.INJ_SHALLOW, 5)
    assert 0.3 < j < 0.5
    frozen = T.make_cfg().default["paraphrase"]["filters"]
    gap = {**frozen, "shallow_range": [round(j + 0.02, 3), 0.5]}
    banned = P.BannedMatcher(["reveal"])
    r = P.filter_candidate(inj_base(T), cand(T.INJ_SHALLOW, "shallow"), gap, banned)
    assert not r.passed and r.reason == "stratum_gap" and r.stratum is None
    r = P.filter_candidate(inj_base(T), cand(T.INJ_SHALLOW, "shallow"), frozen, banned)
    assert r.passed and r.stratum == "shallow"
    assert P.assign_stratum(P.KIND_INJ, 0.2, ["reveal"], gap) == "shallow"        # A.6 demotion below the range
    assert P.assign_stratum(P.KIND_INJ, 0.2, [], gap) == "deep"
    assert P.assign_stratum(P.KIND_BEN, 0.35, [], gap) is None and P.assign_stratum(P.KIND_BEN, 0.35, [], frozen) == "shallow"
    assert P.assign_stratum(P.KIND_INJ, 0.55, [], frozen) is None


# ------------------------------------------------------------------------------------------------ judge reason

def test_judge_reason_max_chars_from_config(make_rt, T):
    long_reason = "r" * 400
    text = json.dumps({"same_action": True, "still_instruction": True, "confidence": 0.8, "reason": long_reason})
    assert len(P.parse_judge_reply(text, P.KIND_INJ)[0]["reason"]) == 300
    assert len(P.parse_judge_reply(text, P.KIND_INJ, 50)[0]["reason"]) == 50
    judge = lambda base, cand, model: {"raw": text}  # noqa: E731
    rt = make_rt(cfg=T.make_cfg(judge_reason_max_chars=20), transport=T.FakeTransport(judge=judge))
    P.generate_base(rt, inj_base(T), [], {})
    P.judge_candidate(rt, inj_base(T), P.load_candidates(rt)["deep_inj:0"][0], {})
    rec = read_jsonl(rt.paths.judgements)[0]
    assert rec["status"] == "ok" and rec["reason"] == "r" * 20


# ------------------------------------------------------------------------------------------------ ledger

def test_one_ledger_line_per_call_survives_a_crash_and_a_rerun(make_rt, T):
    """Review: candidates and their call record are one appended line, so a crash between calls leaves no
    orphaned or duplicated candidates; the rerun performs only the missing call."""
    def crash(messages, model, n):
        if n == 2:
            raise RuntimeError("simulated crash after the first call")
        return None
    rt = make_rt(transport=T.FakeTransport(fail=crash))
    with pytest.raises(RuntimeError):
        P.generate_base(rt, inj_base(T), [], P.load_calls(rt))
    lines = read_jsonl(rt.paths.calls)
    assert len(lines) == 1 and lines[0]["status"] == "ok" and len(lines[0]["candidates"]) == 4
    assert len(read_jsonl(rt.paths.spend)) == 1                                  # the crashed call was never metered
    rt2 = make_rt(transport=T.FakeTransport())
    st = P.generate_base(rt2, inj_base(T), [], P.load_calls(rt2))
    assert st == {"skipped": 1, "ok": 1, "candidates": 4} and len(rt2.transport.calls) == 1
    cands = P.load_candidates(rt2)["deep_inj:0"]
    ids = [c["cand_id"] for c in cands]
    assert len(ids) == 8 == len(set(ids)) and [c["call_index"] for c in cands] == [0] * 4 + [1] * 4
    assert all(c["cand_id"] == f"deep_inj:0|gen-a|{c['call_index']}|{c['k']}" for c in cands)
    assert [c["n_expected"] for c in read_jsonl(rt2.paths.calls)] == [4, 4]
