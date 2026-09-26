"""ТЗ 1.4: BIPIA pair construction, insertion rules, spans, determinism, test-file journaling."""
import json

import pandas as pd

from flyguard.config import seeds_for
from flyguard.data import loaders
from flyguard.data.loaders import (bipia_context_key, insert_end, insert_middle, insert_start, locate_span,
                                   sentence_starts, slug)
from flyguard.data.normalize import normalize_text


def test_sentence_starts_rule():
    ctx = "Hello world. Second one!  Third? Last words"
    assert sentence_starts(ctx) == [0, 13, 26, 33]
    assert sentence_starts("no punctuation here\nsecond line") == [0]
    assert sentence_starts("Ends with a period.") == [0]                # end of text is not a middle boundary
    assert sentence_starts('He said "stop." Then left.') == [0, 16]     # closing quote after the period
    assert sentence_starts("x = a.b\ny = 2. z = 3") == [0, 15]           # code: only period + whitespace


def test_insertion_functions_match_bipia_formats():
    assert insert_start("ctx", "atk") == ("atk\nctx", "")
    assert insert_end("ctx", "atk") == ("ctx\natk", "ctx\n")
    text, prefix = insert_middle("First one. Second one.", "atk", 11)
    assert text == "First one. \natk\nSecond one." and prefix == "First one. \n"
    assert insert_middle("ctx", "atk", 0) == ("\natk\nctx", "\n")


def test_locate_span_in_normalised_coordinates():
    ctx, atk = "First  one.\tSecond one.", "Do  this\nnow"
    for pos, b in (("start", 0), ("middle", 12), ("end", 0)):
        text_orig, prefix = loaders.insert_attack(ctx, atk, pos, b)
        text = normalize_text(text_orig)
        s, e = locate_span(text, normalize_text(prefix), normalize_text(atk))
        assert text[s:e] == "Do this now"


def test_slug():
    assert slug("Space Removal & Grouping") == "space_removal_grouping"
    assert slug("Task Automation") == "task_automation"


def test_load_bipia_pairs_spans_and_variants(cfg, raw_root, recorder):
    seed = seeds_for(cfg, 0)["subsample"]
    docs, info = loaders.load_bipia(cfg, seed, raw_root, recorder)
    assert set(docs.source) == {"bipia"} and set(docs.label) == {0, 1}
    assert info["tasks"]["email"]["rows"] == 6 and info["tasks"]["email"]["distinct_contexts"] == 5
    assert info["tasks"]["email"]["clusters"] == 4                      # near-duplicate contexts share a cluster
    assert info["tasks"]["table"]["distinct_contexts"] == 4 and info["tasks"]["code"]["distinct_contexts"] == 4
    assert set(info["dropped_tasks"]) == {"qa", "abstract"}
    meta = pd.DataFrame(docs["meta"].tolist())
    docs = pd.concat([docs, meta], axis=1)
    docs["ctx"] = docs["doc_id"].map(bipia_context_key)
    assert docs.groupby("ctx")["cluster_id"].nunique().max() == 1
    merged = docs[docs.cluster_id == "bipia:email:1"]
    assert set(merged.ctx) == {"bipia:email:1", "bipia:email:4"}
    for cluster, g in docs.groupby("ctx"):
        task = g["task"].iloc[0]
        names = 3 if task != "code" else 2
        clean = g[g.label == 0]
        assert len(clean) == 1 and clean.iloc[0]["doc_id"] == f"{cluster}:clean" and clean.iloc[0]["spans"] == []
        attacked = g[g.label == 1]
        assert len(attacked) == names * 3                              # attack names x positions (E6 set)
        assert (attacked["variant"] == "main").sum() == 1              # exactly one main attacked doc per context
        assert sorted(set(attacked["position"])) == ["end", "middle", "start"]
        assert (attacked["pair_of"] == f"{cluster}:clean").all()
        ctx_text = clean.iloc[0]["text_orig"]
        for r in attacked.itertuples(index=False):
            (s, e), = r.spans
            assert r.text[s:e] == normalize_text(loaders.bipia_attacks(raw_root, task, "test", recorder)[r.attack][int(r.attack_index)])
            assert r.text == normalize_text(r.text_orig)
            assert r.doc_id == f"{cluster}:{r.attack_id}:{r.position}"
            if r.position == "start":
                assert s == 0
            if r.position == "end":
                assert e == len(r.text)
            if r.position == "middle":
                assert r.boundary in sentence_starts(ctx_text)
        if task == "code":
            assert "\n" in ctx_text                                   # lines joined with newline
    # journaled test reads: three context files and three attack files
    logged = [p for p, split, _ in recorder.calls if split == "test"]
    assert any(p.endswith("email/test.jsonl") for p in logged)
    assert any(p.endswith("text_attack_test.json") for p in logged)
    assert any(p.endswith("code_attack_test.json") for p in logged)


def test_load_bipia_is_deterministic_per_seed(cfg, raw_root, recorder):
    seed = seeds_for(cfg, 0)["subsample"]
    a, _ = loaders.load_bipia(cfg, seed, raw_root, recorder)
    b, _ = loaders.load_bipia(cfg, seed, raw_root, recorder)
    pd.testing.assert_frame_equal(a.drop(columns=["meta"]), b.drop(columns=["meta"]))
    assert [json.dumps(m, sort_keys=True) for m in a.meta] == [json.dumps(m, sort_keys=True) for m in b.meta]
    c, _ = loaders.load_bipia(cfg, seed + 1, raw_root, recorder)
    ma = a[[bool(m["variant"] == "main" and m["attack"]) for m in a.meta]]["doc_id"].tolist()
    mc = c[[bool(m["variant"] == "main" and m["attack"]) for m in c.meta]]["doc_id"].tolist()
    assert set(a.doc_id) >= set(ma) and len(ma) == len(mc)       # same contexts; draws may differ by seed


def test_other_loaders_and_journal(cfg, raw_root, recorder):
    tr = loaders.load_deepset(cfg, "train", raw_root, recorder)
    te = loaders.load_deepset(cfg, "test", raw_root, recorder)
    ni = loaders.load_notinject(cfg, raw_root, recorder)
    assert len(tr) == 40 and len(te) == 12 and len(ni) == 12
    assert tr.doc_id.iloc[0] == "deep:train:0" and (tr.cluster_id == tr.doc_id).all()
    assert (ni.label == 0).all() and ni.doc_id.iloc[0] == "notinject:one:0"
    assert set(ni["meta"].map(lambda m: m["subset"])) == {"one", "two", "three"}
    assert (tr.text == tr.text_orig.map(normalize_text)).all()
    splits = [s for _, s, _ in recorder.calls]
    assert splits and set(splits) == {"test"}
    assert not any(p.endswith("train.parquet") for p, _, _ in recorder.calls)
    assert loaders.load_paraphrases(cfg, raw_root, recorder) is None
    ep, docs, note = loaders.load_trace_documents(cfg, "agentdojo", raw_root)
    assert ep is None and docs is None and note.startswith("dojo:")
