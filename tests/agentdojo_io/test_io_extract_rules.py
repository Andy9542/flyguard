"""extract.py rules frozen after the first pass: the D7 decode switch, contract §6 for AgentDyn (A19), the
config-sourced constants recorded in the manifest, journal-before-read, and the joint split manifest of
`build_all` / `--benchmark all`. Synthetic logs only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flyguard.agentdojo_io import extract as X
from flyguard.agentdojo_io.contract import validate_split_manifest
from flyguard.config import seeds_for

from .conftest import FIXTURES
from .fixtures import synthetic as S

DYN = "dynsuite"
DYN_NON_TEST = ("user_task_0", "user_task_3", "user_task_4")   # crc32 % 3 != 2 (see test_hash_split_rules)
DYN_TEST = ("user_task_1", "user_task_2")                      # crc32 % 3 == 2


def _dyn_id(user_task: str, attacked: bool = False) -> str:
    tail = "injection_task_0/important_instructions" if attacked else "none/none"
    return f"{DYN}/{user_task}/{tail}/{S.MODEL}"


def write_dyn_tree(root: Path) -> None:
    """AgentDyn-shaped tree (D6 volume: clean runs of every task, `important_instructions` on contract-test tasks,
    plus one attack on a non-test task that contract §6 excludes)."""
    clean, hijacked = S.log_clean(), S.log_hijacked()
    logs = [S.make_log(t, None, None, clean["messages"], {}, security=None, utility=True, suite=DYN)
            for t in DYN_NON_TEST + DYN_TEST]
    for t in ("user_task_1", "user_task_0"):
        logs.append(S.make_log(t, "injection_task_0", "important_instructions", hijacked["messages"], hijacked["injections"],
                               security=True, utility=False, suite=DYN))
    for log in logs:
        path = S.log_path(root, log, benchmark="agentdyn")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log), encoding="utf-8")


def _build(cfg, benchmark: str, root: Path, tmp_path: Path, **kw):
    return X.build_episode_documents(cfg, benchmark, model=S.MODEL, traces_dir=root / benchmark,
                                     harm_refs_path=FIXTURES / "harm_references.yaml", meta_dir=FIXTURES / "meta",
                                     manifest_path=tmp_path / "traces_extraction.json",
                                     data_access_log=tmp_path / "data_access.log", **kw)


def test_agentdyn_clean_non_test_runs_are_all_validation(tmp_path, cfg):
    write_dyn_tree(tmp_path / "traces")
    episodes, _ = _build(cfg, "agentdyn", tmp_path / "traces", tmp_path)
    by = episodes.set_index("episode_id")["contract_split"]
    assert all(by[_dyn_id(t)] == "val" for t in DYN_NON_TEST)
    assert all(by[_dyn_id(t)] == "test" for t in DYN_TEST)
    assert by[_dyn_id("user_task_1", attacked=True)] == "test" and by[_dyn_id("user_task_0", attacked=True)] == "excluded"
    assert "train" not in set(by)
    m = json.loads((tmp_path / "traces_extraction.json").read_text())["agentdyn"]
    assert m["val_tasks"][DYN] == sorted(DYN_NON_TEST) and "agentdyn_clean_non_test" in m["validation_rule"]
    assert m["episodes_by_contract_split"] == {"val": 3, "test": 3, "excluded": 1}


def test_agentdyn_falls_back_to_fraction_without_the_config_key(tmp_path, cfg):
    write_dyn_tree(tmp_path / "traces")
    cfg2 = copy.deepcopy(cfg)
    cfg2.default["splits"]["contract"].pop("agentdyn_clean_non_test")
    episodes, _ = _build(cfg2, "agentdyn", tmp_path / "traces", tmp_path)
    clean_non_test = episodes[episodes["user_task"].isin(DYN_NON_TEST) & episodes["attack"].isna()]
    assert sorted(clean_non_test["contract_split"]) == ["train", "train", "val"]   # round(0.2 * 3) -> 1 val task


def test_decode_switch_false_gives_only_exact_spans(tmp_path, cfg, traces_tree):
    root, _ = traces_tree
    cfg2 = copy.deepcopy(cfg)
    cfg2.default["extraction"]["decode_yaml_quoted_scalars"] = False
    episodes, docs = _build(cfg2, "agentdojo", root, tmp_path)
    m = json.loads((tmp_path / "traces_extraction.json").read_text())["agentdojo"]
    assert m["decode_yaml_quoted_scalars"] is False and m["steps_labelled_by_mode"] == {"exact": 2}
    assert m["attacked_without_span"]["count"] == 3 and m["documents_positive"] == 1 == int(docs["label"].sum())
    by = episodes.set_index("episode_id")["injection_step"]
    assert by["toy/user_task_1/injection_task_0/important_instructions/toy-model"] == 1   # exact, kept
    assert by["toy/user_task_2/injection_task_2/important_instructions/toy-model"] == 0   # exact in an `error` episode
    assert by.drop(["toy/user_task_1/injection_task_0/important_instructions/toy-model",
                    "toy/user_task_2/injection_task_2/important_instructions/toy-model"]).isna().all()
    assert {json.loads(x)["span_mode"] for x in docs["meta_json"]} <= {"exact", None}


def test_manifest_records_config_constants_and_target_args_matter(tmp_path, cfg, traces_tree):
    root, _ = traces_tree
    episodes, _ = _build(cfg, "agentdojo", root, tmp_path)
    m = json.loads((tmp_path / "traces_extraction.json").read_text())["agentdojo"]
    assert m["fill_strings"] == {str(k): str(v) for k, v in cfg.default["traces"]["fill_strings"].items()}
    assert m["target_args"] == list(cfg.default["extraction"]["harm_matching"]["target_args"])
    assert m["decode_yaml_quoted_scalars"] is True and m["escaped_pass_skipped_map_mismatch"] == 0
    assert m["val_seed"] == seeds_for(cfg, 0)["subsample"] and "0.2" in m["validation_rule"]
    assert m["match_counts_by_class"] == {"hijacked": {"full": 2}, "injection_ignored": {"name_only": 1, "full": 1},
                                          "error": {"unmatched": 1}}
    assert m["unmatched_hijacked"] == {"count": 0, "episode_ids": []} and m["unmatched"]["count"] == 0
    no_surface = "toy/user_task_8/injection_task_4/injecagent/toy-model"
    assert episodes.set_index("episode_id").loc[no_surface, "match"] == "full"   # meta fallback keyed on `recipients`
    cfg2 = copy.deepcopy(cfg)
    cfg2.default["extraction"]["harm_matching"]["target_args"] = ["url"]
    episodes2, _ = _build(cfg2, "agentdojo", root, tmp_path)
    assert episodes2.set_index("episode_id").loc[no_surface, "match"] == "unmatched"
    assert json.loads((tmp_path / "traces_extraction.json").read_text())["agentdojo"]["target_args"] == ["url"]


def test_test_read_is_journalled_before_parsing(tmp_path, cfg, traces_tree):
    root, _ = traces_tree
    corrupt = root / "agentdojo" / S.MODEL / S.SUITE / "user_task_9" / "none" / "none.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        _build(cfg, "agentdojo", root, tmp_path)
    lines = (tmp_path / "data_access.log").read_text().splitlines()
    assert len(lines) == 1 and "\ttest\t" in lines[0] and "7 trace logs" in lines[0]
    assert not (tmp_path / "traces_extraction.json").exists()


def test_build_all_writes_one_joint_split_manifest(tmp_path, cfg, traces_tree):
    root, _ = traces_tree
    write_dyn_tree(root)
    frames, m = X.build_all(cfg, split_manifest_path=tmp_path / "split_manifest.json",
                            traces_dirs={"agentdojo": root / "agentdojo", "agentdyn": root / "agentdyn"}, model=S.MODEL,
                            harm_refs_path=FIXTURES / "harm_references.yaml", meta_dir=FIXTURES / "meta",
                            manifest_path=tmp_path / "te.json", data_access_log=tmp_path / "da.log")
    assert set(frames) == {"agentdojo", "agentdyn"} and validate_split_manifest(m) == []
    assert json.loads((tmp_path / "split_manifest.json").read_text()) == m
    assert {e.split("/")[0] for e in m["test"]} == {"toy", DYN}
    assert set(m["validation_clean"]) >= {_dyn_id(t) for t in DYN_NON_TEST}
    assert m["counts"]["by_benchmark"]["agentdyn"] == {"test": 3, "validation_clean": 3}
    assert m["rule"]["agentdyn_clean_non_test"] == "validation" and "all non-test" in m["rule"]["validation"]["agentdyn"]
    assert "crc32('<benchmark>/<suite>')" in m["rule"]["val_seed_rule"] and m["rule"]["val_seed"] == seeds_for(cfg, 0)["subsample"]
    te = json.loads((tmp_path / "te.json").read_text())
    assert set(te) == {"agentdojo", "agentdyn"} and te["agentdyn"]["n_logs"] == 7


def test_cli_all_uses_config_roots_and_default_split_manifest(tmp_path, traces_tree, capsys, monkeypatch):
    root, _ = traces_tree
    write_dyn_tree(root)
    monkeypatch.setattr(X, "traces_root", lambda cfg, b: root / b)
    monkeypatch.setattr(X, "SPLIT_MANIFEST_PATH", tmp_path / "s.json")
    monkeypatch.setattr(X.build_episode_documents, "__defaults__", (None, None, FIXTURES / "harm_references.yaml",
                                                                    FIXTURES / "meta", tmp_path / "m.json",
                                                                    tmp_path / "da.log", 0))
    assert X.main(["--benchmark", "all", "--model", S.MODEL, "--episodes-out", str(tmp_path / "e.parquet")]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["episodes"] == 13 and set(out["per_benchmark"]) == {"agentdojo", "agentdyn"}
    assert out["split_manifest"]["validation_clean"] >= 3 and (tmp_path / "s.json").exists()
    assert (tmp_path / "e.parquet").exists()
    with pytest.raises(SystemExit):
        X.main(["--benchmark", "all", "--traces-dir", str(root)])
