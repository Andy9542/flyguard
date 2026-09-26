"""OpenAITransport without the network (ТЗ "Сеть", 1.6): exponential backoff on 429 / 5xx / connection errors,
one logs/network.log line per attempt, a definitive 4xx as a non-retryable TransportError, reply extraction.
The client points at a .invalid host and its `create` is stubbed, so no request can leave the machine."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from flyguard.gen import paraphrases as P

REQ = httpx.Request("POST", "https://x.invalid/chat/completions")
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
OPTS = {"max_tokens": 5, "response_format": {"type": "json_object"}, "extra_body": {"thinking": {"type": "disabled"}},
        "purpose": "unit test"}


def status_error(cls, code: int):
    return cls("err", response=httpx.Response(code, request=REQ), body=None)


class Usage:
    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": 2}


def ok_response(text='{"paraphrases": []}'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                           usage=Usage())


@pytest.fixture
def transport(tmp_path):
    sleeps: list[float] = []
    t = P.OpenAITransport("https://x.invalid/", "not-a-real-key", tmp_path / "network.log", sleep=sleeps.append,
                          max_attempts=3)
    t.sleeps = sleeps  # type: ignore[attr-defined]
    return t


def netlog_lines(t) -> list[str]:
    return t.netlog_path.read_text(encoding="utf-8").splitlines() if t.netlog_path.exists() else []


def test_retries_on_429_with_backoff_then_returns_reply(transport):
    attempts = []

    def create(**kw):
        attempts.append(kw)
        if len(attempts) < 3:
            raise status_error(openai.RateLimitError, 429)
        return ok_response()

    transport.client.chat.completions.create = create
    reply = transport(MSGS, "m", 0.9, **OPTS)
    assert reply.text == '{"paraphrases": []}' and reply.finish_reason == "stop"
    assert reply.usage == {"prompt_tokens": 10, "completion_tokens": 2}
    assert len(attempts) == 3 and len(transport.sleeps) == 2
    assert 1.0 <= transport.sleeps[0] <= 3.0 and 2.0 <= transport.sleeps[1] <= 6.0      # 2*2^k * jitter(0.5..1.5)
    lines = netlog_lines(transport)
    assert len(lines) == 3                                                            # one line per attempt
    assert all("x.invalid" in ln and "POST" in ln and "unit test" in ln and "stream 2 of 2" in ln for ln in lines)
    kw = attempts[0]
    assert kw["model"] == "m" and kw["temperature"] == 0.9 and kw["messages"] == MSGS and "purpose" not in kw
    assert kw["max_tokens"] == 5 and kw["response_format"] == {"type": "json_object"}
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert getattr(transport.client, "max_retries", 0) == 0                          # no silent SDK retries


def test_definitive_4xx_is_a_non_retryable_error(transport):
    calls = []

    def create(**kw):
        calls.append(1)
        raise status_error(openai.BadRequestError, 400)

    transport.client.chat.completions.create = create
    with pytest.raises(P.TransportError) as ei:
        transport(MSGS, "m", 0.0, **OPTS)
    assert ei.value.retryable is False and "400" in str(ei.value)
    assert len(calls) == 1 and transport.sleeps == [] and len(netlog_lines(transport)) == 1


def test_exhausted_5xx_and_connection_errors_are_retryable(transport):
    errors = [status_error(openai.InternalServerError, 503), openai.APIConnectionError(request=REQ),
              status_error(openai.RateLimitError, 429)]

    def create(**kw):
        raise errors.pop(0)

    transport.client.chat.completions.create = create
    with pytest.raises(P.TransportError) as ei:
        transport(MSGS, "m", 0.0, **OPTS)
    assert ei.value.retryable is True and "RateLimitError" in str(ei.value)
    assert len(netlog_lines(transport)) == 3 and len(transport.sleeps) == 2 and errors == []


def test_empty_choices_and_missing_usage(transport):
    transport.client.chat.completions.create = lambda **kw: SimpleNamespace(choices=[], usage=None)
    reply = transport(MSGS, "m", 0.0, **OPTS)
    assert reply.text is None and reply.usage == {} and reply.finish_reason is None
    assert len(netlog_lines(transport)) == 1


def test_backoff_cap_from_config(tmp_path):
    """paraphrase.transport.backoff_max_s caps every back-off sleep (jitter 0.5..1.5 of the capped value)."""
    sleeps: list[float] = []
    t = P.OpenAITransport("https://x.invalid/", "not-a-real-key", tmp_path / "network.log", sleep=sleeps.append,
                          max_attempts=5, timeout=7.0, backoff_max=1.0)
    t.client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(status_error(openai.RateLimitError, 429))
    with pytest.raises(P.TransportError):
        t(MSGS, "m", 0.0, **OPTS)
    assert len(sleeps) == 4 and all(0.5 <= s <= 1.5 for s in sleeps) and len(netlog_lines(t)) == 5
    assert t.timeout == 7.0 and t.client.timeout == 7.0 and t.max_attempts == 5


def test_make_transport_reads_the_frozen_transport_config(make_rt, T, monkeypatch):
    """The three paraphrase.transport keys of configs/default.yaml govern the real transport (review: they were
    dead config); the key comes from the provider's key_env and is never stored in the config."""
    monkeypatch.setenv("FAKE_KEY_ENV", "not-a-real-key")
    t = P.make_transport(make_rt())
    assert (t.max_attempts, t.timeout, t.backoff_max) == (8, 600.0, 90.0) and t.client.timeout == 600.0
    assert t.client.max_retries == 0 and t.base_url == "https://fake.invalid"
    cfg = T.make_cfg(transport={"max_attempts": 2, "timeout_s": 5, "backoff_max_s": 1})
    t2 = P.make_transport(make_rt(cfg=cfg))
    assert (t2.max_attempts, t2.timeout, t2.backoff_max) == (2, 5.0, 1.0) and t2.client.timeout == 5.0
    monkeypatch.delenv("FAKE_KEY_ENV")
    with pytest.raises(SystemExit):
        P.make_transport(make_rt())
