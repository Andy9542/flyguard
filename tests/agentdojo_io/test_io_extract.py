"""extract.py: episodes/documents frames with the exact design §2 columns, exclusions, manifest, data-access log."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from flyguard.agentdojo_io import extract as X
from flyguard.agentdojo_io.contract import validate_split_manifest, write_split_manifest

from .conftest import FIXTURES

EID = {
    "hijacked": "toy/user_task_1/injection_task_0/important_instructions/toy-model",
    "ignored": "toy/user_task_0/injection_task_1/tool_knowledge/toy-model",
    "clean": "toy/user_task_4/none/none/toy-model",
    "error": "toy/user_task_2/injection_task_2/important_instructions/toy-model",
    "single_quoted": "toy/user_task_3/injection_task_3/ignore_previous/toy-model",
    "no_surface": "toy/user_task_8/injection_task_4/injecagent/toy-model",
}


@pytest.fixture
def built(tmp_path, cfg, traces_tree):
    root, _ = traces_tree
    episodes, documents = X.build_episode_documents(
        cfg, "agentdojo", model="toy-model", traces_dir=root / "agentdojo", harm_refs_path=FIXTURES / "harm_references.yaml",
        meta_dir=FIXTURES / "meta", manifest_path=tmp_path / "traces_extraction.json",
        data_access_log=tmp_path / "data_access.log")
    return episodes, documents, tmp_path


def test_episode_frame_columns_and_values(built):
    episodes, _, _ = built
    assert list(episodes.columns) == X.EPISODE_COLUMNS
    assert len(episodes) == 6 and set(episodes["episode_id"]) == set(EID.values())
    by = episodes.set_index("episode_id")
    h = by.loc[EID["hijacked"]]
    assert (h["episode_class"], h["injection_step"], h["first_harmful_step"], h["match"]) == ("hijacked", 1, 2, "full")
    assert (h["contract_split"], bool(h["e1_val_task"]), h["n_steps"], h["model"], h["benchmark"]) == ("test", False, 3, "toy-model", "agentdojo")
    assert (h["suite"], h["user_task"], h["injection_task"], h["attack"]) == ("toy", "user_task_1", "injection_task_0", "important_instructions")
    assert h["security"] is True or h["security"] == True and h["utility"] == False  # noqa: E712 - pandas booleans
    assert len(h["sha256"]) == 64 and h["log_path"].endswith("injection_task_0.json")
    i = by.loc[EID["ignored"]]
    assert (i["injection_step"], i["first_harmful_step"], i["match"]) == (0, 2, "name_only") and i["contract_split"] in ("train", "val")
    c = by.loc[EID["clean"]]
    assert c["episode_class"] == "benign" and pd.isna(c["injection_step"]) and pd.isna(c["match"]) and pd.isna(c["first_harmful_step"])
    assert bool(c["e1_val_task"]) is True and pd.isna(c["injection_task"]) and pd.isna(c["attack"]) and pd.isna(c["security"])
    e = by.loc[EID["error"]]
    assert e["episode_class"] == "error" and e["contract_split"] == "test" and e["match"] == "unmatched"
    s = by.loc[EID["single_quoted"]]
    assert (s["episode_class"], s["injection_step"], s["first_harmful_step"], s["match"]) == ("hijacked", 0, 1, "full")
    n = by.loc[EID["no_surface"]]
    assert pd.isna(n["injection_step"]) and (n["first_harmful_step"], n["match"]) == (1, "full")
    assert str(episodes["injection_step"].dtype) == "Int64" and str(episodes["utility"].dtype) == "boolean"


def test_document_frame_columns_labels_and_spans(built, cfg):
    _, docs, _ = built
    assert list(docs.columns) == X.DOCUMENT_COLUMNS
    assert set(docs["source"]) == {"dojo"} and str(docs["label"].dtype) == "int8"
    eids = {json.loads(m)["episode_id"] for m in docs["meta_json"]}
    assert EID["error"] not in eids and EID["no_surface"] not in eids
    assert eids == {EID["hijacked"], EID["ignored"], EID["clean"], EID["single_quoted"]}
    h = docs[docs["doc_id"].str.startswith("dojo:" + EID["hijacked"] + "#")].set_index("doc_id")
    assert list(h.index) == [f"dojo:{EID['hijacked']}#{k}" for k in range(3)]
    assert h["label"].tolist() == [0, 1, 0]
    pos = h.loc[f"dojo:{EID['hijacked']}#1"]
    assert len(pos["spans"]) == 1 and set(pos["spans"][0]) == {"start", "end"}
    a, b = pos["spans"][0]["start"], pos["spans"][0]["end"]
    assert pos["text"][a:b].startswith("<INFORMATION> This is an important message")
    assert "  " not in pos["text"] and pos["text_orig"] != pos["text"] and pos["cluster_id"] == "toy/user_task_1"
    assert h.loc[f"dojo:{EID['hijacked']}#0", "spans"] == [] and h["split"].tolist() == ["test"] * 3
    meta = json.loads(pos["meta_json"])
    assert (meta["step"], meta["tool"], meta["span_mode"], meta["contract_split"]) == (1, "read_inbox", "exact", "test")
    clean = docs[docs["doc_id"].str.startswith("dojo:" + EID["clean"] + "#")]
    assert clean["split"].tolist() == ["val", "val"] and clean["label"].tolist() == [0, 0]
    sq = docs[docs["doc_id"] == f"dojo:{EID['single_quoted']}#0"].iloc[0]
    assert sq["label"] == 1 and json.loads(sq["meta_json"])["span_mode"] == "yaml_escaped" and sq["spans"]
    ig = docs[docs["doc_id"] == f"dojo:{EID['ignored']}#0"].iloc[0]
    assert ig["label"] == 1 and "\\n" in ig["text"]  # the document keeps the escaped rendering the agent saw
    assert set(docs["lang_stratum"]) <= {"en", "non-en"} and docs["lang"].map(lambda v: isinstance(v, str)).all()
    assert int(docs["label"].sum()) == 3


def test_manifest_and_data_access_log(built):
    episodes, docs, tmp_path = built
    m = json.loads((tmp_path / "traces_extraction.json").read_text())["agentdojo"]
    assert m["n_logs"] == 6 and m["model"] == "toy-model"
    assert m["episodes_by_class"] == {"hijacked": 2, "injection_ignored": 2, "benign": 1, "error": 1}
    assert m["attacked_without_span"] == {"count": 1, "episode_ids": [EID["no_surface"]]}
    assert m["errors"] == {"count": 1, "episode_ids": [EID["error"]]}
    assert m["steps_labelled_by_mode"] == {"exact": 2, "yaml_escaped": 2}
    assert m["steps_labelled"] == 4 and m["steps_total"] == int(episodes["n_steps"].sum())
    assert m["documents"] == len(docs) and m["documents_positive"] == 3
    assert m["fill_strings"] == {"user": "Emma Johnson", "model": "DeepSeek"}
    assert m["reference_sources"] == {"yaml:ground_truth": 1, "yaml:manual": 3, "meta": 1}
    assert m["match_counts"] == {"full": 3, "name_only": 1, "unmatched": 1}
    assert m["unmatched"]["episode_ids"] == [] and m["unfilled_placeholders"] == 0
    assert len(m["val_tasks"]["toy"]) == 2 and m["episodes_by_contract_split"]["test"] == 2
    lines = (tmp_path / "data_access.log").read_text().splitlines()
    assert len(lines) == 1 and "\ttest\t" in lines[0] and "6 trace logs" in lines[0]


def test_split_manifest_from_episodes(built, tmp_path):
    episodes, _, _ = built
    m = write_split_manifest(episodes.to_dict("records"), tmp_path / "split_manifest.json")
    assert validate_split_manifest(m) == []
    assert m["test"] == [EID["hijacked"]] and m["counts"]["error"] == 1
    assert set(m["observation"] + m["validation_clean"]) == {EID["clean"]}
    assert set(m["train_attacks"] + m["validation_attacks"]) == {EID["ignored"], EID["single_quoted"], EID["no_surface"]}


def test_empty_when_no_logs(tmp_path, cfg):
    episodes, docs = X.build_episode_documents(cfg, "agentdojo", traces_dir=tmp_path / "nothing", model="m",
                                               manifest_path=tmp_path / "m.json", data_access_log=tmp_path / "da.log")
    assert list(episodes.columns) == X.EPISODE_COLUMNS and list(docs.columns) == X.DOCUMENT_COLUMNS and len(episodes) == 0
    assert json.loads((tmp_path / "m.json").read_text())["agentdojo"]["note"] == "no trace logs found"
    assert not (tmp_path / "da.log").exists()
    with pytest.raises(ValueError):
        X.build_episode_documents(cfg, "unknown", traces_dir=tmp_path)


def test_choose_model_records_are_binding_then_single_directory(cfg, traces_tree, tmp_path):
    root, _ = traces_tree
    none = tmp_path / "missing.json"
    assert X.choose_model(cfg, "agentdojo", root / "agentdojo", pilot_path=none, manifest_path=none) == "toy-model"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"chosen_model": "chosen-model"}))
    # the pilot's choice binds both benchmarks even when its logs have not arrived: no other model's episodes
    assert X.choose_model(cfg, "agentdojo", root / "agentdojo", pilot_path=pilot, manifest_path=none) == "chosen-model"
    manifest = tmp_path / "traces_manifest.json"
    manifest.write_text(json.dumps({"agent_model": "frozen-model"}))
    assert X.choose_model(cfg, "agentdyn", root / "agentdyn", pilot_path=none, manifest_path=manifest) == "frozen-model"
    (root / "agentdojo" / "other-model").mkdir()
    assert X.choose_model(cfg, "agentdojo", root / "agentdojo", pilot_path=none, manifest_path=none) is None
    episodes, docs = X.build_episode_documents(cfg, "agentdyn", traces_dir=root / "agentdyn", model="chosen-model",
                                               manifest_path=tmp_path / "m.json", data_access_log=tmp_path / "da.log")
    assert len(episodes) == 0 and len(docs) == 0


def test_cli_writes_parquet_and_prints_counts(tmp_path, traces_tree, capsys, monkeypatch):
    root, _ = traces_tree
    monkeypatch.setattr(X, "MANIFEST_PATH", tmp_path / "m.json")
    monkeypatch.setattr(X, "DATA_ACCESS_LOG", tmp_path / "da.log")
    monkeypatch.setattr(X.L, "HARM_REFERENCES_PATH", FIXTURES / "harm_references.yaml")
    monkeypatch.setattr(X.L, "META_DIR", FIXTURES / "meta")
    monkeypatch.setattr(X.build_episode_documents, "__defaults__", (None, None, FIXTURES / "harm_references.yaml",
                                                                    FIXTURES / "meta", tmp_path / "m.json",
                                                                    tmp_path / "da.log", 0))
    rc = X.main(["--benchmark", "agentdojo", "--traces-dir", str(root / "agentdojo"), "--model", "toy-model",
                 "--episodes-out", str(tmp_path / "e.parquet"), "--documents-out", str(tmp_path / "d.parquet"),
                 "--split-manifest", str(tmp_path / "s.json")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["episodes"] == 6 and out["positive_documents"] == 3
    e = pd.read_parquet(tmp_path / "e.parquet")
    d = pd.read_parquet(tmp_path / "d.parquet")
    assert list(e.columns) == X.EPISODE_COLUMNS and list(d.columns) == X.DOCUMENT_COLUMNS and len(d) == out["documents"]
    assert (tmp_path / "s.json").exists()


def test_attacked_val_task_episodes_get_the_unused_e1_role(built):
    """A25: only clean outputs of AgentDojo validation tasks play a role in E1; attacked ones are 'unused'."""
    from flyguard.agentdojo_io import labels as L
    episodes, documents, _ = built
    val_eps = episodes[episodes["e1_val_task"]]
    attacked_val = set(val_eps[val_eps["injection_task"].notna()]["episode_id"])
    clean_val = set(val_eps[val_eps["injection_task"].isna()]["episode_id"])
    assert clean_val, "fixture must contain a clean validation-task episode"
    ep_of = documents["meta_json"].map(lambda m: json.loads(m)["episode_id"])
    assert set(documents.loc[ep_of.isin(clean_val), "split"]) == {L.SPLIT_VAL}
    assert set(documents.loc[ep_of.isin(attacked_val), "split"]) <= {L.SPLIT_UNUSED}
    others = documents.loc[~ep_of.isin(clean_val | attacked_val), "split"]
    assert set(others) <= {L.SPLIT_TEST}
