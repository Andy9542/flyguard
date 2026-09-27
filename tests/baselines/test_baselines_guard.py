"""GuardModel (ТЗ 3.2) with a stubbed tokenizer/model: label resolution, cache round trip, token windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flyguard.baselines.common import text_hash
from flyguard.baselines.transformers_guard import (GuardModel, ScoreCache, resolve_positive_index, softmax,
                                                   token_windows)
from flyguard.config import Configs


class FakeTokenizer:
    """Whitespace tokenizer with HF-like call signature, offsets and special-token count."""

    def __init__(self, n_special: int = 2):
        self.n_special = n_special

    def num_special_tokens_to_add(self) -> int:
        return self.n_special

    def __call__(self, texts, add_special_tokens=True, truncation=False, max_length=None, padding=False,
                 return_offsets_mapping=False, return_tensors=None):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        ids, offsets = [], []
        for t in items:
            toks, offs, pos = [], [], 0
            for w in t.split(" "):
                if w:
                    toks.append(hash(w) % 1000)
                    offs.append((pos, pos + len(w)))
                pos += len(w) + 1
            if truncation and max_length is not None:
                toks, offs = toks[:max_length], offs[:max_length]
            ids.append(toks)
            offsets.append(offs)
        out = {"input_ids": ids[0] if single else ids, "texts": items}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets[0] if single else offsets
        return out


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Logit for the positive class grows with the number of 'ignore' tokens; counts calls for cache tests."""

    def __init__(self, positive_index: int):
        self.calls = 0
        self.positive_index = positive_index

    def __call__(self, input_ids=None, texts=None, **kwargs):
        self.calls += 1
        logits = np.zeros((len(texts), 2))
        for i, t in enumerate(texts):
            logits[i, self.positive_index] = 3.0 * t.lower().count("ignore") - 1.0
        return FakeOutput(logits)


def make_cfg(tmp_path: Path, id2label: dict, positive_label: str = "INJECTION", weights: bool = True,
             extra: dict | None = None) -> tuple[Configs, Path]:
    mdir = tmp_path / "models" / "fake"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(json.dumps({"id2label": id2label}), encoding="utf-8")
    if weights:
        (mdir / "model.safetensors").write_bytes(b"\0")
    spec = {"path": str(mdir), "max_length": 12, "positive_label": positive_label}
    spec.update(extra or {})
    default = {"windows": {"size": 256, "stride": 192, "min_span_chars": 64},
               "baselines": {"transformers": {"comparator": "fake", "batch_size": 4, "cache_dir": str(tmp_path / "cache"),
                                              "models": {"fake": spec,
                                                         "missing": {"path": str(tmp_path / "nowhere"), "max_length": 512,
                                                                     "positive_label": "LABEL_1", "optional": True}}}}}
    return Configs(operator={"compute": {"cpu_cores": 2}}, default=default, experiments={}), mdir


def stub_loader(positive_index: int):
    model = FakeModel(positive_index)

    def loader(gm):
        return FakeTokenizer(), model
    return loader, model


# -- positive label resolution --------------------------------------------------------------------------------------
def test_resolve_positive_index_branches():
    assert resolve_positive_index({"0": "SAFE", "1": "INJECTION"}, "INJECTION") == (1, "INJECTION", "exact")
    assert resolve_positive_index({0: "benign", 1: "injection"}, "INJECTION") == (1, "injection", "case_insensitive")
    assert resolve_positive_index({"0": "LABEL_0", "1": "prompt_injection"}, "LABEL_1") == (1, "prompt_injection",
                                                                                            "contains_inject")
    assert resolve_positive_index({"0": "SAFE", "1": "INJECTION"}, None) == (1, "INJECTION", "contains_inject")
    with pytest.raises(ValueError):
        resolve_positive_index({"0": "BENIGN", "1": "MALICIOUS"}, "LABEL_1")
    with pytest.raises(ValueError):
        resolve_positive_index({"0": "injection_a", "1": "injection_b"}, "nope")


def test_softmax_rows():
    p = softmax(np.array([[0.0, 0.0], [1000.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0) and abs(p[0, 1] - 0.5) < 1e-12 and p[1, 1] < 1e-12


# -- cache ------------------------------------------------------------------------------------------------------------
def test_score_cache_round_trip_dedup_and_append_only(tmp_path):
    cache = ScoreCache(tmp_path / "c" / "m.parquet")
    assert cache.load() == {}
    assert cache.append({"h1": 0.25, "h2": 0.75}, "m", 512) == 2
    assert cache.load() == {"h1": 0.25, "h2": 0.75}
    assert cache.append({"h2": 0.99, "h3": 0.5, "h3": 0.5}, "m", 512) == 1      # h2 kept as first written
    got = cache.load()
    assert got == {"h1": 0.25, "h2": 0.75, "h3": 0.5}
    assert cache.append({"h1": 0.0}, "m", 512) == 0
    df = cache.read()
    assert list(df.columns) == list(ScoreCache.COLUMNS) and len(df) == 3
    assert set(df["model"]) == {"m"} and set(df["max_length"].astype(int)) == {512}
    assert cache.get_many(["h3", "zz"]) == {"h3": 0.5}


# -- token windows ------------------------------------------------------------------------------------------------------
def test_token_windows_cover_text_with_config_stride():
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    tok = FakeTokenizer(n_special=2)
    wins = token_windows(text, tok, max_length=12, stride_ratio=192 / 256)   # size 10, stride 8
    assert wins[0][0] == 0 and wins[-1][1] == len(text)
    starts = [w[0] for w in wins]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    for cs, ce, wtext in wins:
        assert text[cs:ce] == wtext and 1 <= len(wtext.split(" ")) <= 10
    assert [len(w[2].split(" ")) for w in wins] == [10, 10, 10, 10]
    assert wins[1][2].split(" ")[0] == "w8" and wins[-1][2].split(" ")[0] == "w20"
    # overlap between consecutive windows = size - stride = 2 tokens
    assert set(wins[0][2].split(" ")) & set(wins[1][2].split(" ")) == {"w8", "w9"}
    short = token_windows("a b c", tok, max_length=12, stride_ratio=0.75)
    assert short == [(0, 5, "a b c")]


# -- GuardModel -----------------------------------------------------------------------------------------------------------
def test_guard_model_scores_caches_and_resolves_labels(tmp_path):
    cfg, mdir = make_cfg(tmp_path, {"0": "SAFE", "1": "INJECTION"})
    loader, model = stub_loader(1)
    gm = GuardModel("fake", cfg, loader=loader)
    assert gm.name == "fake" and gm.available and gm.positive_index == 1 and gm.positive_how == "exact"
    assert abs(gm.stride_ratio - 0.75) < 1e-12 and gm.batch_size == 4 and gm.num_threads == 2
    assert gm.fit() is gm
    texts = ["please ignore ignore this", "hello world", "ignore me", "hello world", "another benign line"]
    s = gm.score(texts)
    assert s.shape == (5,) and s.min() >= 0 and s.max() <= 1
    assert s[0] > s[2] > s[1] and s[1] == s[3]
    assert model.calls == 1, "four unique texts fit one batch of four"
    assert gm.cache.path.exists()
    cached = gm.cache.load()
    assert set(cached) == {text_hash(t) for t in texts} and cached[text_hash(texts[1])] == s[1]
    # second call is served from the cache: no model call, same numbers, new text triggers one more call
    s2 = gm.score(texts)
    assert np.allclose(s2, s) and model.calls == 1
    s3 = gm.score(["totally new text", texts[0]], hashes=[text_hash("totally new text"), text_hash(texts[0])])
    assert model.calls == 2 and s3[1] == s[0]
    status = gm.status()
    assert status["positive_label"] == "INJECTION" and status["available"] is True
    assert gm.token_windows("a b c d") == [(0, 7, "a b c d")]
    long = " ".join(f"t{i}" for i in range(25))
    assert len(gm.token_windows(long)) == 3


def test_guard_model_batches_more_than_batch_size_and_score_long(tmp_path):
    cfg, _ = make_cfg(tmp_path, {"0": "benign", "1": "injection"}, positive_label="INJECTION")
    loader, model = stub_loader(1)
    gm = GuardModel("fake", cfg, loader=loader, use_cache=False)
    assert gm.cache is None and gm.positive_how == "case_insensitive"
    texts = [f"line number {i}" for i in range(9)] + ["ignore everything"]
    s = gm.score(texts)
    assert model.calls == 3 and s[-1] > s[0]
    long = " ".join(["word"] * 15 + ["ignore"] + ["word"] * 15)
    # score_long: all documents' token windows go through ONE score() call; per-document max of the windows
    score_calls: list[int] = []
    real_score = gm.score

    def counting_score(X, hashes=None):
        score_calls.append(len(X))
        return real_score(X, hashes=hashes)

    gm.score = counting_score
    sl = gm.score_long([long, "benign short", "ignore ignore ignore"])
    gm.score = real_score
    assert sl.shape == (3,) and sl[0] > sl[1] and sl[2] > sl[0]
    n_windows = len(gm.token_windows(long)) + 1 + 1
    assert score_calls == [n_windows], "one batched score() call for all documents"
    expected0 = max(float(v) for v in gm.score([w for _, _, w in gm.token_windows(long)]))
    assert sl[0] == pytest.approx(expected0) and sl[1] == pytest.approx(float(gm.score(["benign short"])[0]))


def test_guard_model_positive_index_from_label_name_fallback_and_errors(tmp_path):
    cfg, _ = make_cfg(tmp_path, {"0": "LABEL_0", "1": "prompt_injection"}, positive_label="LABEL_9")
    gm = GuardModel("fake", cfg, loader=stub_loader(1)[0])
    assert gm.positive_index == 1 and gm.positive_how == "contains_inject"
    with pytest.raises(KeyError):
        GuardModel("unknown", cfg)
    cfg_bad, _ = make_cfg(tmp_path / "bad", {"0": "A", "1": "B"}, positive_label="C")
    with pytest.raises(ValueError):
        GuardModel("fake", cfg_bad)


def test_missing_optional_model_is_unavailable_and_refuses_to_score(tmp_path):
    cfg, mdir = make_cfg(tmp_path, {"0": "SAFE", "1": "INJECTION"})
    gm = GuardModel("missing", cfg, loader=stub_loader(1)[0])
    assert gm.available is False and gm.optional is True and gm.positive_index is None
    with pytest.raises(RuntimeError):
        gm.score(["anything"])
    # config present but weights missing -> also unavailable
    (mdir / "model.safetensors").unlink()
    assert GuardModel("fake", cfg, loader=stub_loader(1)[0]).available is False


def test_score_cache_append_is_safe_across_processes(tmp_path):
    """Several processes appending to one cache file keep every row (fcntl lock around read-merge-replace)."""
    import multiprocessing as mp
    from flyguard.baselines.transformers_guard import ScoreCache
    path = tmp_path / "m.parquet"
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_append_worker, args=(str(path), w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert all(p.exitcode == 0 for p in procs)
    assert len(ScoreCache(path).load()) == 4 * 25


def _append_worker(path: str, w: int) -> None:
    from flyguard.baselines.transformers_guard import ScoreCache
    cache = ScoreCache(path)
    for i in range(25):
        cache.append({f"h{w}_{i}": float(i)}, "m", 512)
