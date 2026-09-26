"""parse.py: log fields, step numbering from 0 in call order (contract §3), episode ids, log discovery."""
from __future__ import annotations

import json

import pytest

from flyguard.agentdojo_io import parse as P

from .fixtures import synthetic as S


def test_read_log_fields(traces_tree):
    _, paths = traces_tree
    log = P.read_log(paths["hijacked"])
    assert (log.suite_name, log.pipeline_name, log.user_task_id) == ("toy", "toy-model", "user_task_1")
    assert log.injection_task_id == "injection_task_0" and log.attack_type == "important_instructions"
    assert list(log.injections) == ["email_body_3"] and log.is_attacked
    assert log.security is True and log.utility is False and log.error is None
    assert log.duration == 1.5 and log.benchmark_version == "v0.0-toy"
    assert "evaluation_timestamp" in log.extra and log.path == paths["hijacked"]


def test_parse_log_dict_tolerates_none_tokens_and_rejects_bad_shapes():
    data = S.log_clean()
    data["attack_type"], data["injection_task_id"], data["error"] = "none", "none", ""
    log = P.parse_log_dict(data)
    assert log.attack_type is None and log.injection_task_id is None and log.error is None and not log.is_attacked
    with pytest.raises(ValueError):
        P.parse_log_dict({"suite_name": "x"})
    with pytest.raises(ValueError):
        P.parse_log_dict({"suite_name": "x", "user_task_id": "u", "messages": [], "injections": []})


def test_tool_steps_are_numbered_from_zero_in_call_order():
    log = P.parse_log_dict(S.log_hijacked())
    steps = P.tool_steps(log)
    assert [s.index for s in steps] == [0, 1, 2]
    assert [s.tool for s in steps] == ["get_balance", "read_inbox", "send_email"]
    assert steps[0].output_text == "balance: 1000.0\ncurrency: EUR"
    assert steps[2].args["recipients"] == ["Attacker@Example.com"] and steps[2].call_id == "c2"
    assert steps[0].message_index < steps[1].message_index < steps[2].message_index
    assert all(s.error is None for s in steps)


def test_content_text_variants():
    assert P.content_text("plain") == "plain"
    assert P.content_text(None) == ""
    assert P.content_text([{"type": "text", "content": "a"}, {"type": "text", "content": "b"}]) == "a\nb"
    assert P.content_text([{"type": "text", "text": "legacy"}]) == "legacy"
    assert P.content_text([{"type": "thinking", "content": None}, {"type": "text", "content": "x"}]) == "x"
    assert json.loads(P.content_text({"k": 1})) == {"k": 1}


def test_string_content_and_fallback_to_assistant_tool_calls():
    data = S.log_clean()  # first tool message carries string content
    steps = P.tool_steps(P.parse_log_dict(data))
    assert steps[0].output_text == "balance: 42.0" and steps[0].tool == "get_balance"
    # a tool message without `tool_call` is matched to the assistant's call by id, then FIFO
    data["messages"][3].pop("tool_call")
    steps = P.tool_steps(P.parse_log_dict(data))
    assert steps[0].tool == "get_balance" and steps[0].call_id == "c0"
    data["messages"][3]["tool_call_id"] = None
    steps = P.tool_steps(P.parse_log_dict(data))
    assert steps[0].tool == "get_balance"


def test_args_given_as_json_string_are_parsed():
    data = S.log_hijacked()
    data["messages"][7]["tool_call"]["args"] = json.dumps({"recipients": ["a@b.c"]})
    steps = P.tool_steps(P.parse_log_dict(data))
    assert steps[2].args == {"recipients": ["a@b.c"]}
    data["messages"][7]["tool_call"]["args"] = "not json"
    assert P.tool_steps(P.parse_log_dict(data))[2].args == {"_raw": "not json"}


def test_unanswered_tool_calls_are_not_steps():
    data = S.log_clean()
    data["messages"].append(S.assistant([S.call("get_balance", {}, "c9"), S.call("read_inbox", {}, "c10")]))
    log = P.parse_log_dict(data)
    assert len(P.tool_steps(log)) == 2 and P.unanswered_tool_calls(log) == 2


def test_tool_error_is_kept():
    data = S.log_clean()
    data["messages"][5]["error"] = "ToolError: boom"
    steps = P.tool_steps(P.parse_log_dict(data))
    assert steps[1].error == "ToolError: boom"


def test_episode_id_format_and_inverse():
    attacked = P.parse_log_dict(S.log_hijacked())
    clean = P.parse_log_dict(S.log_clean())
    assert P.episode_id(attacked) == "toy/user_task_1/injection_task_0/important_instructions/toy-model"
    assert P.episode_id(clean) == "toy/user_task_4/none/none/toy-model"
    assert P.episode_id(clean, model="other") == "toy/user_task_4/none/none/other"
    parts = P.split_episode_id(P.episode_id(attacked))
    assert parts == {"suite": "toy", "user_task": "user_task_1", "injection_task": "injection_task_0",
                     "attack": "important_instructions", "model": "toy-model"}
    assert P.split_episode_id(P.episode_id(clean))["injection_task"] is None
    with pytest.raises(ValueError):
        P.split_episode_id("toy/user_task_4/none/none")


def test_iter_log_paths_layout_and_skips_utility_checks(traces_tree, tmp_path):
    root, paths = traces_tree
    found = list(P.iter_log_paths(root / "agentdojo", "toy-model"))
    assert len(found) == len(S.ALL_LOGS) - 1
    assert paths["utility_check"] not in found and paths["hijacked"] in found
    assert found == sorted(found)
    assert list(P.iter_log_paths(root / "agentdojo")) == found  # model=None: every model directory
    assert list(P.iter_log_paths(tmp_path / "missing")) == []
    (root / "agentdojo" / "toy-model" / "stray.json").write_text("{}")  # wrong depth: ignored
    assert len(list(P.iter_log_paths(root / "agentdojo", "toy-model"))) == len(found)
