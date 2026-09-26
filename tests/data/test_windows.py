"""ТЗ 1.3 / 2.6: windows cover the document, labels agree with spans."""
import random

import pandas as pd
import xxhash

from flyguard.data.windows import build_windows, make_windows, text_hash, window_label, windows_for_document

SIZE, STRIDE, MIN_SPAN = 256, 192, 64


def test_short_and_empty_documents_are_one_window():
    assert make_windows("", SIZE, STRIDE) == [(0, 0)]
    assert make_windows("a" * 255, SIZE, STRIDE) == [(0, 255)]
    assert make_windows(256, SIZE, STRIDE) == [(0, 256)]
    assert make_windows(257, SIZE, STRIDE) == [(0, 256), (192, 257)]


def test_windows_cover_document_with_fixed_overlap():
    for n in list(range(1, 1300, 7)) + [448, 449, 640, 4545]:
        w = make_windows(n, SIZE, STRIDE)
        assert w[0][0] == 0 and w[-1][1] == n
        covered = set()
        for s, e in w:
            assert 0 <= s < e <= n
            covered.update(range(s, e))
        assert covered == set(range(n))
        for (s1, e1), (s2, e2) in zip(w, w[1:]):
            assert s2 - s1 == STRIDE and e1 - s2 == SIZE - STRIDE
        assert all(e - s == SIZE for s, e in w[:-1])
        if len(w) > 1:
            assert w[-1][1] - w[-1][0] > SIZE - STRIDE


def test_window_label_span_rules():
    long_span = [(300, 400)]
    assert window_label(256, 512, long_span, MIN_SPAN, 0) == 1
    assert window_label(192, 448, long_span, MIN_SPAN, 0) == 1
    assert window_label(0, 256, long_span, MIN_SPAN, 0) == 0
    assert window_label(0, 256, [(200, 400)], MIN_SPAN, 0) == 0     # 56 chars < 64
    assert window_label(0, 256, [(192, 400)], MIN_SPAN, 0) == 1     # exactly 64
    assert window_label(0, 256, [(190, 400)], MIN_SPAN, 0) == 1
    short = [(250, 270)]
    assert window_label(0, 256, short, MIN_SPAN, 0) == 0            # only 6 of 20 chars
    assert window_label(192, 448, short, MIN_SPAN, 0) == 1          # whole shorter span
    assert window_label(0, 256, [{"start": 10, "end": 30}], MIN_SPAN, 0) == 1
    assert window_label(0, 256, [(0, 0)], MIN_SPAN, 1) == 0 or True  # degenerate span never raises


def test_window_label_spanless_sources_take_document_label():
    assert window_label(0, 256, [], MIN_SPAN, 1) == 1
    assert window_label(0, 256, None, MIN_SPAN, 0) == 0
    assert window_label(500, 700, [], MIN_SPAN, 1) == 1


def test_windows_for_document_ids_text_and_hash():
    text = "abcdefghij" * 60  # 600 chars
    rows = windows_for_document("d:1", text, [(100, 180)], 1, SIZE, STRIDE, MIN_SPAN)
    assert [r["window_id"] for r in rows] == ["d:1#w0", "d:1#w1", "d:1#w2"]
    for r in rows:
        assert r["text"] == text[r["start"]:r["end"]]
        assert r["text_hash"] == xxhash.xxh64(r["text"].encode()).hexdigest() == text_hash(r["text"])
        assert r["label"] == window_label(r["start"], r["end"], [(100, 180)], MIN_SPAN, 1)
    assert [r["label"] for r in rows] == [1, 0, 0]


def test_every_span_yields_a_positive_window_and_positives_touch_spans():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 3000)
        text = "".join(rng.choice("abcdefgh ") for _ in range(n))
        spans = []
        for _ in range(rng.randint(0, 3)):
            s = rng.randint(0, n - 1)
            e = min(n, s + rng.randint(1, 500))
            spans.append((s, e))
        rows = windows_for_document("x", text, spans, 1 if spans else 0, SIZE, STRIDE, MIN_SPAN)
        labels = [r["label"] for r in rows]
        if spans:
            assert any(labels), (n, spans)   # ТЗ 2.6: label agrees with the span
        for r in rows:
            if r["label"] == 1:
                assert any(min(r["end"], e) - max(r["start"], s) >= min(MIN_SPAN, e - s) for s, e in spans)
            else:
                assert not any(min(r["end"], e) - max(r["start"], s) >= min(MIN_SPAN, e - s) for s, e in spans)


def test_build_windows_uses_config_and_carries_document_fields(cfg):
    docs = pd.DataFrame([
        {"doc_id": "a", "source": "deep", "split": "train", "label": 1, "text": "z" * 700, "spans": [], "cluster_id": "a"},
        {"doc_id": "b", "source": "bipia", "split": "test", "label": 1, "text": "y" * 300, "spans": [(280, 300)], "cluster_id": "c"},
        {"doc_id": "c", "source": "notinject", "split": "test", "label": 0, "text": "short", "spans": [], "cluster_id": "c"},
    ])
    w = build_windows(docs, cfg)
    size, stride = cfg.default["windows"]["size"], cfg.default["windows"]["stride"]
    assert list(w.columns) == ["window_id", "doc_id", "source", "split", "start", "end", "text", "label", "cluster_id",
                               "text_hash", "dedup_excluded", "dup_of"]
    assert (w[w.doc_id == "a"]["label"] == 1).all() and len(w[w.doc_id == "a"]) == len(make_windows(700, size, stride))
    assert w[w.doc_id == "b"]["label"].tolist() == [0, 1]
    assert w[w.doc_id == "c"].iloc[0]["text"] == "short"
    assert not w["dedup_excluded"].any() and w["dup_of"].isna().all()
    assert (w["cluster_id"] == w["doc_id"].map({"a": "a", "b": "c", "c": "c"})).all()


def test_text_hash_reads_config_and_refuses_unknown_algo(cfg, monkeypatch):
    """windows.text_hash {algo, seed, encoding} is the single definition of the cache key (ASSUMPTIONS A17)."""
    from flyguard.data.windows import text_hash_from_cfg

    spec = cfg.default["windows"]["text_hash"]
    h = text_hash_from_cfg(cfg)
    assert h("héllo") == xxhash.xxh64("héllo".encode(spec["encoding"]), seed=int(spec["seed"])).hexdigest()
    assert h("héllo") == text_hash("héllo")                       # cfg-less callers get the same key
    assert text_hash("x", seed=1) != text_hash("x")
    w = build_windows(pd.DataFrame([{"doc_id": "a", "source": "deep", "split": "train", "label": 0,
                                     "text": "some text", "spans": [], "cluster_id": "a"}]), cfg)
    assert w.iloc[0]["text_hash"] == h("some text")
    monkeypatch.setitem(cfg.default, "windows", dict(cfg.default["windows"], text_hash={"algo": "md5", "seed": 0}))
    import pytest
    with pytest.raises(ValueError):
        text_hash_from_cfg(cfg)
