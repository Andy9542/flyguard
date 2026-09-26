"""End-to-end build on the synthetic raw tree: outputs, schema, idempotency, smoke, stage handling."""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from flyguard.config import ROOT
from flyguard.data import build
from flyguard.data.loaders import bipia_context_key
from flyguard.data.windows import make_windows


def _log_size():
    p = ROOT / "logs" / "data_access.log"
    return p.stat().st_size if p.exists() else -1


def test_build_without_traces_end_to_end(cfg, raw_root, recorder):
    before = _log_size()
    res = build.build_all(cfg, without_traces=True, smoke=False, root=raw_root, access_log=recorder, write=True)
    assert _log_size() == before                                   # tests never touch the real journal
    docs, win = res["documents"], res["windows"]
    assert set(docs.source) == {"deep", "bipia", "notinject"}
    assert set(docs.columns) >= {"doc_id", "source", "split", "label", "text", "text_orig", "lang", "lang_stratum",
                                 "cluster_id", "spans", "meta", "dedup_dropped"}
    assert docs.doc_id.is_unique and win.window_id.is_unique
    assert set(docs.lang_stratum) <= {"en", "non-en"} and "unk" in set(docs.lang)
    size, stride = cfg.default["windows"]["size"], cfg.default["windows"]["stride"]
    for doc_id, g in win.groupby("doc_id"):
        n = len(docs.set_index("doc_id").loc[doc_id, "text"])
        assert list(zip(g.start, g.end)) == make_windows(n, size, stride)
    # the duplicated deepset test document is excluded and dropped
    assert win[win.doc_id == "deep:test:1"].dedup_excluded.all()
    assert "deep:test:1" in res["splits"]["dropped_by_dedup"]
    assert "deep:test:1" not in res["splits"]["e1"]["test"]["deep"]
    # manifests
    processed, manifests = build.output_dirs(raw_root, False)
    for name in ("splits.json", "pools.json", "dedup.json", "audit.md"):
        assert (manifests / name).exists()
    assert (processed / "documents.parquet").exists() and (processed / "windows.parquet").exists()
    assert not (processed / "episodes.parquet").exists()
    audit = (manifests / "audit.md").read_text(encoding="utf-8")
    assert "deepset" in audit and "NotInject" in audit and "qa" in audit and "--without-traces" in audit
    assert "Ignore all previous" not in audit                      # counts only, never texts
    sp = json.load(open(manifests / "splits.json"))
    assert set(sp["e1"]["test"]) == {"deep", "bipia", "notinject"} and sp["e3"]["cross_template"] == []
    assert sp["counts"]["c_unl"] == len(sp["c_unl"]) > 0
    pools = json.load(open(manifests / "pools.json"))
    assert set(pools["p_val"]["by_source"]) <= {"deep", "bipia"} and pools["p_val"]["n"] > 0
    # parquet types follow design §2
    schema = pq.read_schema(processed / "documents.parquet")
    assert schema.field("label").type == pa.int8() and schema.field("spans").type == build.SPAN_TYPE
    wschema = pq.read_schema(processed / "windows.parquet")
    assert wschema.field("dup_of").type == pa.string() and wschema.field("dedup_excluded").type == pa.bool_()
    back = build.read_documents(processed / "documents.parquet")
    assert len(back) == len(docs) and isinstance(back.spans.iloc[0], list) and isinstance(back.meta.iloc[0], dict)
    main_bip = docs[(docs.source == "bipia") & ~docs.dedup_dropped & docs.meta.map(lambda m: m.get("variant", "main") == "main")]
    for _, g in main_bip.groupby(main_bip.doc_id.map(bipia_context_key)):
        assert set(g.label) == {0, 1} and len(g) == 2               # no half-pair in the main test
    bip = back[(back.source == "bipia") & (back.label == 1)].iloc[0]
    (s, e), = bip.spans
    assert bip.text[s:e].startswith("Ignore all previous")
    wb = build.read_windows(processed / "windows.parquet")
    assert len(wb) == len(win) and wb.dup_of.notna().sum() == win.dedup_excluded.sum()


def test_build_is_idempotent(cfg, raw_root, recorder):
    processed, manifests = build.output_dirs(raw_root, False)
    build.build_all(cfg, without_traces=True, root=raw_root, access_log=recorder)
    first = {n: (manifests / n).read_bytes() for n in ("splits.json", "pools.json", "dedup.json", "audit.md")}
    d1, w1 = build.read_documents(processed / "documents.parquet"), build.read_windows(processed / "windows.parquet")
    build.build_all(cfg, without_traces=True, root=raw_root, access_log=recorder)
    for n, data in first.items():
        assert (manifests / n).read_bytes() == data, n
    d2, w2 = build.read_documents(processed / "documents.parquet"), build.read_windows(processed / "windows.parquet")
    assert d1.drop(columns=["meta"]).equals(d2.drop(columns=["meta"])) and w1.equals(w2)


def test_full_stage_without_trace_module_or_paraphrases(cfg, raw_root, recorder):
    res = build.build_all(cfg, without_traces=False, root=raw_root, access_log=recorder, write=False)
    assert set(res["documents"].source) == {"deep", "bipia", "notinject"}
    notes = " ".join(res["notes"])
    assert "dojo" in notes and "dyn" in notes and "paraphrases.csv missing" in notes
    assert res["episodes"] is None and res["splits"]["e3"]["cross_suite"] == []


def test_smoke_subset_respects_caps_and_clusters(cfg, raw_root, recorder, monkeypatch):
    small = dict(cfg.default["smoke"], docs_per_source=8)
    monkeypatch.setitem(cfg.default, "smoke", small)
    res = build.build_all(cfg, without_traces=True, smoke=True, root=raw_root, access_log=recorder, write=True)
    docs = res["documents"]
    processed, manifests = build.output_dirs(raw_root, True)
    assert processed.name == "smoke" and manifests.name == "smoke" and (manifests / "splits.json").exists()
    for src, g in docs.groupby("source"):
        max_cluster = g.groupby("cluster_id").size().max()
        assert len(g) <= 8 + max_cluster
    bip = docs[docs.source == "bipia"]
    for _, g in bip.groupby(bip.doc_id.map(bipia_context_key)):
        assert set(g.label) == {0, 1}                              # whole contexts only
    assert set(bip.split) == {"val", "test"}                       # smoke keeps material of every split
    assert set(docs[docs.source == "deep"].split) == {"train", "val", "test"}
    full = build.build_all(cfg, without_traces=True, smoke=False, root=raw_root, access_log=recorder, write=False)
    sub = docs.set_index("doc_id")["split"]
    assert (full["documents"].set_index("doc_id").loc[sub.index, "split"] == sub).all()   # same split as the full build


def test_cli_parses_flags(monkeypatch):
    calls = {}

    def fake(cfg, without_traces=False, smoke=False):
        calls.update(without_traces=without_traces, smoke=smoke)
        import pandas as pd
        return {"documents": pd.DataFrame({"source": ["deep"]}), "windows": pd.DataFrame(),
                "dedup": {"documents_dropped_total": 0, "test_windows_excluded": 0}, "notes": ["n"]}

    monkeypatch.setattr(build, "build_all", fake)
    monkeypatch.setattr(build, "load_configs", lambda: object())
    assert build.main(["--without-traces", "--smoke"]) == 0
    assert calls == {"without_traces": True, "smoke": True}
