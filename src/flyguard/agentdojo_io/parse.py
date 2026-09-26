"""Reading harness JSON logs (AgentDojo / AgentDyn `TraceLogger` files) into steps — ТЗ 1.5 "Извлечение", contract §3.

The logs are written unchanged by the harness (`agentdojo.logging.TraceLogger.save`): keys `suite_name`,
`pipeline_name`, `user_task_id`, `injection_task_id`, `attack_type`, `injections`, `messages`, `error` plus the
context arguments `utility`, `security`, `duration`, `benchmark_version` (and timestamps). Messages follow
`agentdojo.types`: `content` is a string (older harness versions) or a list of `{type, content}` blocks; assistant
messages carry `tool_calls: [{function, args, id, ...}]`; tool messages carry `tool_call`, `tool_call_id`, `error`.

Why a dedicated parser: every other module (windows, contract CSV, the second team's step numbering) depends on
one definition of "step i" — the i-th tool call of the episode, counted from 0 in call order (contract §3). This
file is the only place where that numbering is derived from the raw messages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

NONE_TOKEN = "none"
"""Value used by the harness for the attack / injection-task position of clean runs (directory and file names)."""


@dataclass
class Step:
    """One executed tool call of an episode (contract §3: step i = the i-th tool call, counted from 0)."""

    index: int
    tool: str
    args: dict[str, Any]
    output_text: str
    error: str | None = None
    call_id: str | None = None
    message_index: int = -1


@dataclass
class TraceLog:
    """The fields of one harness JSON log (see module docstring); `extra` keeps any other context keys."""

    suite_name: str
    pipeline_name: str
    user_task_id: str
    injection_task_id: str | None
    attack_type: str | None
    injections: dict[str, str]
    messages: list[dict[str, Any]]
    error: str | None = None
    utility: bool | None = None
    security: bool | None = None
    duration: float | None = None
    benchmark_version: str | None = None
    path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_attacked(self) -> bool:
        return self.injection_task_id is not None


_KNOWN_KEYS = ("suite_name", "pipeline_name", "user_task_id", "injection_task_id", "attack_type", "injections",
               "messages", "error", "utility", "security", "duration", "benchmark_version")


def content_text(content: Any) -> str:
    """Text of a message `content` field: a string, `None`, or a list of `{type, content|text}` blocks.

    Mirrors `agentdojo.types.get_text_content_as_str` (blocks joined with a newline) so that the parser sees the
    same text the agent model saw; a non-string block payload is JSON-dumped rather than dropped, because a tool
    output must never silently disappear from the documents (ТЗ 1.5 "Извлечение").
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                payload = block.get("content", block.get("text"))
                if payload is None:
                    continue
                parts.append(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))
            elif block is not None:
                parts.append(block if isinstance(block, str) else json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _none_if_token(value: Any) -> Any:
    """`attack_type`/`injection_task_id` of clean runs are `None` in the log; tolerate the literal 'none' too."""
    if isinstance(value, str) and value.strip().lower() in ("", NONE_TOKEN, "null"):
        return None
    return value


def parse_log_dict(data: dict[str, Any], path: Path | None = None) -> TraceLog:
    """Build a `TraceLog` from an already-loaded JSON object (used by `read_log` and by tests)."""
    if not isinstance(data, dict):
        raise ValueError(f"trace log is not a JSON object: {path}")
    missing = [k for k in ("suite_name", "user_task_id", "messages") if k not in data]
    if missing:
        raise ValueError(f"trace log {path} lacks keys {missing}")
    injections = data.get("injections")
    if injections is None:
        injections = {}
    if not isinstance(injections, dict):
        raise ValueError(f"trace log {path}: `injections` must be a dict")
    messages = data.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError(f"trace log {path}: `messages` must be a list")
    error = data.get("error")
    return TraceLog(
        suite_name=str(data["suite_name"]),
        pipeline_name=str(data.get("pipeline_name") or "unknown"),
        user_task_id=str(data["user_task_id"]),
        injection_task_id=_none_if_token(data.get("injection_task_id")),
        attack_type=_none_if_token(data.get("attack_type")),
        injections={str(k): str(v) for k, v in injections.items()},
        messages=messages,
        error=(str(error) if error not in (None, "") else None),
        utility=data.get("utility"),
        security=data.get("security"),
        duration=data.get("duration"),
        benchmark_version=data.get("benchmark_version"),
        path=Path(path) if path is not None else None,
        extra={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
    )


def read_log(path: str | Path) -> TraceLog:
    """Read one harness JSON log (ТЗ 1.5 "Извлечение"; design §3 `read_log`).

    The file is parsed as data only: nothing from it is printed here, and the tool outputs it contains are
    real prompt injections (CLAUDE.md data safety).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_log_dict(data, path)


def _call_parts(call: Any) -> tuple[str, dict[str, Any], str | None]:
    """(function, args, id) of a serialized `FunctionCall`; args may arrive as a JSON string from some providers."""
    if not isinstance(call, dict):
        return "", {}, None
    function = str(call.get("function") or call.get("name") or "")
    args = call.get("args")
    if args is None:
        args = call.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed if isinstance(parsed, dict) else {"_raw": args}
        except ValueError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_value": args}
    call_id = call.get("id")
    return function, dict(args), (str(call_id) if call_id is not None else None)


def tool_steps(log: TraceLog) -> list[Step]:
    """Tool steps of an episode, numbered from 0 in call order (contract §3; design §3 `tool_steps`).

    A step is one *executed* tool call, i.e. one `role == "tool"` message: FlyGuard scores the response of call
    i, so a call without a response has no document. The tool's name and arguments are taken from the tool
    message's own `tool_call` (what the harness actually executed); when a log lacks it, the matching entry of
    the preceding assistant `tool_calls` is used (by `tool_call_id`, else first-in-first-out), so that older log
    shapes still parse. Assistant tool calls that never received a response are not steps (they were never
    executed) and are counted by `unanswered_tool_calls`.
    """
    steps: list[Step] = []
    pending: list[tuple[str, dict[str, Any], str | None]] = []
    for m_idx, message in enumerate(log.messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                pending.append(_call_parts(call))
        elif role == "tool":
            function, args, call_id = _call_parts(message.get("tool_call"))
            tool_call_id = message.get("tool_call_id")
            matched = None
            if pending:
                if tool_call_id is not None:
                    for k, cand in enumerate(pending):
                        if cand[2] == tool_call_id:
                            matched = pending.pop(k)
                            break
                if matched is None:
                    matched = pending.pop(0)
            if not function and matched is not None:
                function, args = matched[0], matched[1]
            if call_id is None:
                call_id = tool_call_id if tool_call_id is not None else (matched[2] if matched else None)
            error = message.get("error")
            steps.append(Step(index=len(steps), tool=function, args=args, output_text=content_text(message.get("content")),
                              error=(str(error) if error not in (None, "") else None), call_id=call_id,
                              message_index=m_idx))
    return steps


def unanswered_tool_calls(log: TraceLog) -> int:
    """Number of assistant tool calls without a tool response (diagnostic for the extraction manifest)."""
    requested = sum(len(m.get("tool_calls") or []) for m in log.messages if isinstance(m, dict) and m.get("role") == "assistant")
    answered = sum(1 for m in log.messages if isinstance(m, dict) and m.get("role") == "tool")
    return max(requested - answered, 0)


def episode_id(log: TraceLog, model: str | None = None) -> str:
    """Contract §8 episode id `<suite>/<user_task>/<injection_task|none>/<attack|none>/<model>` (model = pipeline_name)."""
    return "/".join([
        log.suite_name,
        log.user_task_id,
        log.injection_task_id or NONE_TOKEN,
        log.attack_type or NONE_TOKEN,
        model or log.pipeline_name,
    ])


def split_episode_id(eid: str) -> dict[str, str | None]:
    """Inverse of `episode_id`: dict with suite, user_task, injection_task, attack, model (`none` -> None)."""
    parts = eid.split("/")
    if len(parts) != 5:
        raise ValueError(f"episode id must have 5 '/'-separated parts: {eid!r}")
    suite, user_task, injection_task, attack, model = parts
    return {"suite": suite, "user_task": user_task, "injection_task": _none_if_token(injection_task),
            "attack": _none_if_token(attack), "model": model}


def iter_log_paths(root: str | Path, model: str | None = None) -> Iterator[Path]:
    """Log files under `<root>/<model>/<suite>/<user_task>/<attack>/<injection_task>.json`, sorted.

    Directories named `injection_task_*` at the user-task level are skipped: the harness runs injection tasks
    as user tasks for its utility checks, and those runs are not episodes of the dataset (same rule as
    `flyguard.gen.traces.freeze`).
    """
    root = Path(root)
    if not root.exists():
        return
    models = [root / model] if model else sorted(p for p in root.iterdir() if p.is_dir())
    for model_dir in models:
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.rglob("*.json")):
            rel = path.relative_to(model_dir).parts
            if len(rel) != 4:  # suite / user_task / attack / file
                continue
            if rel[1].startswith("injection_task_"):
                continue
            yield path
