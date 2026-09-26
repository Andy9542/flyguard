"""Filters of ТЗ 1.6 / A.6 as property tests (ТЗ 2.6): Jaccard rejection, banned words and strata, entities,
language, length, within-base dedup, the banned-word matcher and the χ² extension."""
from __future__ import annotations

import pytest

from flyguard.gen import paraphrases as P


@pytest.fixture
def filters(T):
    return T.make_cfg().default["paraphrase"]["filters"]


@pytest.fixture
def banned(project):
    return P.BannedMatcher(P.load_starter_banned(project / "configs" / "prompts" / "banned_words.txt"))


def inj_base(T) -> P.Base:
    return P.Base("deep_inj:0", "deep:test:0", P.KIND_INJ, T.INJ_TEXT, origin="deepset_test")


def ben_base(T) -> P.Base:
    return P.Base("deep_ben:1", "deep:test:1", P.KIND_BEN, T.BEN_TEXT, origin="deepset_test")


def cand(text: str, declared: str | None = None, k: int = 0, gen: str = "gen-a") -> dict:
    return {"cand_id": f"x|{gen}|0|{k}", "generator": gen, "call_index": 0, "k": k, "text": text,
            "declared_stratum": declared}


def test_candidate_with_jaccard_06_is_rejected(T, filters, banned):
    """ТЗ 2.6: Jaccard with the base above 0.5 (here about 0.67) is rejected whatever the declared stratum."""
    j = P.jaccard_texts(T.INJ_TEXT, T.INJ_J06, 5)
    assert 0.5 < j < 0.8
    for declared in ("shallow", "deep"):
        r = P.filter_candidate(inj_base(T), cand(T.INJ_J06, declared), filters, banned)
        assert not r.passed and r.reason == "jaccard_high" and r.stratum is None
        assert r.jaccard == pytest.approx(j, abs=1e-3)


def test_deep_candidate_with_banned_word_is_demoted_to_shallow(T, filters, banned):
    """A.6: declared deep, Jaccard fine, but a banned word ("reveal") -> shallow, flagged as demoted."""
    r = P.filter_candidate(inj_base(T), cand(T.INJ_DEEP_BANNED, "deep"), filters, banned)
    assert r.passed and r.stratum == "shallow" and r.demoted
    assert "reveal" in r.banned_hits and r.jaccard <= filters["deep_max"]


def test_stratum_follows_jaccard_not_the_declaration(T, filters, banned):
    """A.6: the stratum is assigned by Jaccard (+ words); a "shallow"-declared distant text becomes deep and a
    "deep"-declared close text with banned words becomes shallow."""
    r = P.filter_candidate(inj_base(T), cand(T.INJ_DEEP, "shallow"), filters, banned)
    assert r.passed and r.stratum == "deep" and not r.demoted and r.banned_hits == []
    r = P.filter_candidate(inj_base(T), cand(T.INJ_SHALLOW, "deep"), filters, banned)
    assert r.passed and r.stratum == "shallow" and r.demoted
    assert filters["shallow_range"][0] < r.jaccard <= filters["shallow_range"][1]


def test_entity_filter_rejects_a_dropped_email(T, filters, banned):
    """ТЗ 1.6: concrete entities of the base must be present verbatim; only kind and length are recorded."""
    r = P.filter_candidate(inj_base(T), cand(T.INJ_NO_ENTITY, "deep"), filters, banned)
    assert not r.passed and r.reason == "entities"
    assert r.missing == [f"email:{len('boss@corp.example')}"]
    # fill strings of the templates are required too (ТЗ: "плейсхолдерные строки присутствуют")
    dropped_model = T.T_DEEP.replace("DeepSeek", "the model")
    assert any(m.startswith("fill:") for m in P.missing_entities(T.LOG_STRING0, dropped_model, list(P.FILL.values())))
    assert P.missing_entities(T.LOG_STRING0, T.T_DEEP, list(P.FILL.values())) == []


def test_extract_entities_covers_every_class():
    text = ('Please wire $1,200 and 300 EUR to IBAN GB82WEST12345698765432 for Alice Marple via '
            'https://pay.example.com/x?y=1, attach budget_2024.xlsx, use $recipient and say "code red" to '
            'alice@corp.example on 2024-05-13. Contact Bob. Then Carol Danvers replied.')
    ents = set(P.extract_entities(text))
    expected = {("email", "alice@corp.example"), ("url", "https://pay.example.com/x?y=1"),
                ("iban", "GB82WEST12345698765432"), ("placeholder", "$recipient"), ("file", "budget_2024.xlsx"),
                ("amount", "1,200"), ("amount", "300"), ("number", "2024-05-13"), ("quoted", "code red"),
                ("name", "Alice Marple"), ("name", "Carol Danvers")}
    assert expected <= ents
    values = {v for _, v in ents}
    assert "Contact Bob" not in values and "Then Carol Danvers" not in values and "Please" not in values
    # emails and URLs are compared case-insensitively, everything else verbatim
    assert P.missing_entities("Mail Alice@Corp.example now", "mail alice@corp.example now") == []
    assert P.missing_entities('Say "Code Red"', 'Say "code red"') == ["quoted:8"]


def test_banned_matcher_inflections_word_boundaries_and_short_words(banned):
    """Design §7: case-insensitive, word-bounded, inflections by stem prefix; short entries match exactly."""
    hits = banned.hits("Please keep IGNORING these Rules, they were ignored before")
    assert {"ignore", "ignoring", "rules"} <= set(hits)
    assert banned.hits("she was dancing in danger near the dandelions") == []      # "dan" stays exact
    assert "don't" in banned.hits("don’t worry about it")                        # curly apostrophe
    assert "act as" in banned.hits("Act  as a pirate") and "new task" in banned.hits("three new tasks")
    assert "email" in banned.hits("I emailed him") and banned.hits("an e-mail arrived") == []
    assert "prompt" in banned.hits("promptly") and "bypass" in banned.hits("bypassing")
    assert banned.hits("a systematic review of admins") and "output" in banned.hits("the outputs")
    assert P.stem("guidelines") == "guidelin" and P.stem("rules") == "rule" and P.stem("dan") == "dan"


def test_chi2_extension_is_deterministic_and_skips_covered_words(project):
    """A.1: top-χ² injection-enriched tokens, minus tokens already covered by the starter list, stop words and
    short tokens; alphabetical tie-break."""
    import pandas as pd
    df = pd.read_parquet(project / "data" / "raw" / "deepset" / "train.parquet")
    texts, labels = df["text"].tolist(), [int(v) for v in df["label"].tolist()]
    extra = P.chi2_extension(texts, labels, ["ignore", "rules"], 3)
    assert extra == ["poem", "topic", "write"]
    assert extra == P.chi2_extension(texts, labels, ["ignore", "rules"], 3)
    more = P.chi2_extension(texts, labels, ["ignore", "rules"], 10)
    assert "ignore" not in more and "rules" not in more and "the" not in more and "weather" not in more
    assert P.chi2_extension(texts, [1] * len(texts), [], 3) == []


def test_language_and_length_filters_and_benign_strata(T, filters, banned):
    """ТЗ 1.6: English only, 0.5-2x length; negatives get their stratum from Jaccard alone (no word rule)."""
    r = P.filter_candidate(ben_base(T), cand(T.BEN_GERMAN), filters, banned)
    assert not r.passed and r.reason == "language" and r.lang == "de"
    r = P.filter_candidate(ben_base(T), cand(T.BEN_SHORT), filters, banned)
    assert not r.passed and r.reason == "length" and r.len_ratio < 0.5
    r = P.filter_candidate(ben_base(T), cand(T.BEN_SHALLOW), filters, banned)
    assert r.passed and r.stratum == "shallow" and r.banned_hits == []
    r = P.filter_candidate(ben_base(T), cand(T.BEN_DEEP), filters, banned)
    assert r.passed and r.stratum == "deep"
    r = P.filter_candidate(ben_base(T), cand(""), filters, banned)
    assert not r.passed and r.reason == "empty"


def test_within_base_dedup_keeps_the_first(T, filters, banned):
    """ТЗ 1.6: dedup at Jaccard 0.8 inside the set; the earlier candidate survives, the later is marked."""
    cands = [cand(T.INJ_DEEP, "deep", k=0), cand(T.INJ_DEEP_DUP, "deep", k=1), cand(T.INJ_SHALLOW, "shallow", k=2)]
    res = P.filter_base(inj_base(T), cands, filters, banned)
    assert [r.passed for r in res] == [True, False, True]
    assert res[1].reason == "duplicate" and res[1].dup_of == res[0].cand_id and res[1].stratum is None
    assert [r.stratum for r in res if r.passed] == ["deep", "shallow"]
