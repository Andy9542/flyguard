"""dojo/dyn loading: trace directory resolution (shared.traces_dir), smoke manifest routing, error propagation."""
import json
from pathlib import Path

import pandas as pd
import pytest

from flyguard.data import loaders


def test_trace_logdir_follows_operator_and_root(cfg, tmp_path, monkeypatch):
    assert loaders.trace_logdir(cfg, "agentdojo", tmp_path) == tmp_path / cfg.default["traces"]["logdir"] / "agentdojo"
    monkeypatch.setitem(cfg.operator, "shared", {"traces_dir": "handed/over"})
    assert loaders.trace_logdir(cfg, "agentdyn", tmp_path) == tmp_path / "handed" / "over" / "agentdyn"
    monkeypatch.setitem(cfg.operator, "shared", {"traces_dir": str(tmp_path / "abs")})
    assert loaders.trace_logdir(cfg, "agentdojo", tmp_path) == tmp_path / "abs" / "agentdojo"
    from flyguard.agentdojo_io import extract
    from flyguard.config import ROOT
    monkeypatch.setitem(cfg.operator, "shared", {"traces_dir": None})
    assert loaders.trace_logdir(cfg, "agentdojo", ROOT) == extract.traces_root(cfg, "agentdojo")   # one rule


def test_no_logs_is_a_note_not_an_error(cfg, tmp_path):
    ep, docs, note = loaders.load_trace_documents(cfg, "agentdojo", tmp_path)
    assert ep is None and docs is None and note.startswith("dojo: no trace logs under")


def _fake_logs(root: Path) -> Path:
    d = root / "data" / "traces" / "agentdojo" / "model" / "workspace" / "user_task_0" / "none"
    d.mkdir(parents=True)
    json.dump({"messages": []}, open(d / "none.json", "w"))
    return d


def test_extraction_kwargs_and_smoke_manifest(cfg, tmp_path, monkeypatch):
    _fake_logs(tmp_path)
    import flyguard.agentdojo_io.extract as extract
    seen = {}

    def fake(cfg_, benchmark, **kw):
        seen.update(kw)
        docs = pd.DataFrame([{"doc_id": "dojo:e#0", "source": "dojo", "split": "test", "label": 0, "text": "t",
                              "text_orig": "t", "cluster_id": "workspace/user_task_0", "spans": [],
                              "meta_json": json.dumps({"episode_id": "e"})}])
        return pd.DataFrame([{"episode_id": "e"}]), docs

    monkeypatch.setattr(extract, "build_episode_documents", fake)
    calls = []
    ep, docs, note = loaders.load_trace_documents(cfg, "agentdojo", tmp_path, smoke=True,
                                                  access_log=lambda p, s, u: calls.append((p, s)))
    assert seen["traces_dir"] == tmp_path / "data" / "traces" / "agentdojo"
    assert seen["manifest_path"] == tmp_path / "data" / "manifests" / "smoke" / "traces_extraction.json"
    assert seen["data_access_log"] == tmp_path / "logs" / "data_access.log"        # never the repository journal
    assert calls == [(tmp_path / "data" / "traces" / "agentdojo", "test")]
    assert len(docs) == 1 and docs.iloc[0]["meta"] == {"episode_id": "e"} and "1 documents" in note
    loaders.load_trace_documents(cfg, "agentdojo", tmp_path, smoke=False, access_log=lambda *a: None)
    assert seen["manifest_path"] == tmp_path / "data" / "manifests" / "traces_extraction.json"
    loaders.load_trace_documents(cfg, "agentdojo", tmp_path, manifest_dir=tmp_path / "m", access_log=lambda *a: None)
    assert seen["manifest_path"] == tmp_path / "m" / "traces_extraction.json"


def test_extraction_errors_propagate(cfg, tmp_path, monkeypatch):
    _fake_logs(tmp_path)
    import flyguard.agentdojo_io.extract as extract

    def boom(cfg_, benchmark, **kw):
        raise FileNotFoundError("meta file missing")

    monkeypatch.setattr(extract, "build_episode_documents", boom)
    with pytest.raises(FileNotFoundError):
        loaders.load_trace_documents(cfg, "agentdojo", tmp_path, access_log=lambda *a: None)

    def empty(cfg_, benchmark, **kw):
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(extract, "build_episode_documents", empty)
    ep, docs, note = loaders.load_trace_documents(cfg, "agentdojo", tmp_path, access_log=lambda *a: None)
    assert docs is None and "produced no documents" in note
