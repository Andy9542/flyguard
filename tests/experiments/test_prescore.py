"""Guard prescoring (``flyguard.experiments.prescore``) on the E6 toy tree (conftest helpers + BIPIA E6 variants and
both trace benchmarks, built by ``test_e6_contract.make_e6_root``): the selected texts are exactly the ones E1 and E6
ask a guard for (real and smoke profiles), and an interrupted run resumes by scoring only the missing hashes.
Stub models, no network, no real data."""
from __future__ import annotations

import copy
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from flyguard.baselines.transformers_guard import GuardModel, text_hash
from flyguard.experiments import Context, FeatureContext, e1, e6, prescore, read_result

_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("prescore_conftest_helpers", _here / "conftest.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
_spec6 = importlib.util.spec_from_file_location("prescore_e6_helpers", _here / "test_e6_contract.py")
E6T = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(E6T)

MODEL = "protectai_v2"


class CountingModel(H.FakeModel):
    """The stub guard; records every text it scores and raises on forward call number ``fail_on`` (1-based)."""

    def __init__(self, positive_index: int = 1, fail_on: int | None = None):
        super().__init__(positive_index)
        self.fail_on, self.seen = fail_on, []

    def __call__(self, input_ids=None, texts=None, **kwargs):
        if self.fail_on is not None and self.calls + 1 == self.fail_on:
            self.calls += 1
            raise RuntimeError("simulated interruption inside a chunk")
        self.seen.extend(texts)
        return super().__call__(input_ids=input_ids, texts=texts, **kwargs)


class Guard(GuardModel):
    """GuardModel with a stub model; ``requested`` collects every key a caller asks the cache for."""

    requested: set[str]

    def score(self, X, hashes=None):
        keys = list(hashes) if hashes is not None else [text_hash(str(t)) for t in X]
        self.requested.update(keys)
        return super().score(X, hashes=hashes)


def make_factory(cfg, root: Path, cache_dir: Path, fail_on: int | None = None, forbid_model: bool = False):
    """A guard factory sharing one model and one ``requested`` set; ``forbid_model`` makes any forward pass fail
    (consumers must be served by the cache alone)."""
    state = {"model": CountingModel(fail_on=1 if forbid_model else fail_on), "requested": set()}

    def loader(gm):
        return H.FakeTokenizer(), state["model"]

    def factory(name, **kw):
        g = Guard(name, cfg, root=root, loader=loader, cache_dir=cache_dir, **kw)
        g.requested = state["requested"]
        return g

    factory.state = state
    return factory


@pytest.fixture(scope="module")
def pcfg():
    cfg = copy.deepcopy(H.make_toy_cfg())
    cfg.default["stats"]["bootstrap"]["n"] = 20
    cfg.default["smoke"]["bootstrap"] = 20
    cfg.experiments["E6"]["gammas"] = [0.0, 0.99]
    return cfg


@pytest.fixture(scope="module")
def proot(tmp_path_factory, pcfg):
    """The E6 toy tree, also copied into the smoke directories (the smoke profile reads data/*/smoke)."""
    root = tmp_path_factory.mktemp("flyguard_prescore")
    E6T.make_e6_root(root, pcfg)
    processed, manifests = root / "data" / "processed", root / "data" / "manifests"
    (processed / "smoke").mkdir()
    for name in ("documents.parquet", "windows.parquet", "episodes.parquet"):
        shutil.copy(processed / name, processed / "smoke" / name)
    shutil.copytree(manifests, manifests / "smoke_tmp")
    shutil.move(str(manifests / "smoke_tmp"), str(manifests / "smoke"))
    return root


def _cache_keys(cache_dir: Path) -> set[str]:
    path = cache_dir / f"{MODEL}.parquet"
    return set(pd.read_parquet(path)["text_hash"].astype(str)) if path.exists() else set()


def _e6_only_hashes(root: Path, smoke: bool = False) -> set[str]:
    """Hashes of E6-variant BIPIA windows whose text no other window shares (only ``bipia_all`` asks for them)."""
    from flyguard.data.build import output_dirs
    from flyguard.io import read_json

    processed, manifests = output_dirs(root, smoke)
    w = pd.read_parquet(processed / "windows.parquet")
    ids = set(read_json(manifests / "splits.json")["bipia"]["e6_docs"]["test"])
    e6w = w[w["doc_id"].isin(ids) & ~w["dedup_excluded"].fillna(False)]
    return set(e6w["text_hash"]) - set(w.loc[~w["doc_id"].isin(ids), "text_hash"])


def _consume(cfg, root: Path, factory, smoke: bool, recorder_class, latency_guards=False) -> dict:
    """Run E1 and E6 (default parts of the profile) with the given guard factory. Guard latency is an uncached
    forward pass, so it is off unless the test checks that the smoke profile switches it off."""
    ctx = Context(cfg, smoke=smoke, root=root, access_log=recorder_class())
    e1.run_e1([0], smoke=smoke, root=root, cfg=cfg, force=True, latency_guards=latency_guards, ctx=ctx,
              guard_factory=factory, cache=False)
    path = e6.run(ctx, 0, smoke, root=root, cfg=cfg, force=True, guard_factory=factory, cache=False)
    return read_result(path)


# ---------------------------------------------------------------------------------------------- tests
def test_selection_equals_what_e1_and_e6_score(pcfg, proot, tmp_path, recorder_class):
    cache = tmp_path / "cache"
    pre = make_factory(pcfg, proot, cache)
    rec = recorder_class()
    out = prescore.prescore(MODEL, pcfg, smoke=False, root=proot, chunk=7, guard_factory=pre, access_log=rec)
    assert out["tok512"] == "scored" and {"bipia_all", "tok512"} <= set(out["e6_parts"])
    assert out["windows_cached_before"] == 0 and out["windows_scored"] == out["windows_unique"] > 0
    assert set(out["windows_by_role"]) >= {"val_all", "p_val", "test:deep", "test:bipia", "test:notinject", "bipia_all"}
    assert all(split == "test" and "guard prescoring" in why for _, split, why in rec.calls)
    prescored = _cache_keys(cache)
    assert _e6_only_hashes(proot) <= prescored                       # the E6 variants are scored in the real run
    train = Context(pcfg, root=proot, access_log=recorder_class()).train_windows
    only_train = set(train["text_hash"]) - set(pd.read_parquet(proot / "data/processed/windows.parquet")
                                                 .query("split != 'train'")["text_hash"])
    assert only_train and not (only_train & prescored)               # train windows are never scored

    consumer = make_factory(pcfg, proot, cache, forbid_model=True)   # every guard score must come from the cache
    res = _consume(pcfg, proot, consumer, False, recorder_class)
    assert consumer.state["model"].seen == []
    assert consumer.state["requested"] == prescored                  # nothing missing, nothing scored in vain
    assert "auc/bipia_all/protectai_v2" in res["numbers"] and "auc/deep/protectai_v2_tok512" in res["numbers"]


def test_smoke_selection_follows_smoke_e6_parts(pcfg, proot, tmp_path, recorder_class):
    assert set(e6.resolve_parts(pcfg, None, smoke=True)) == set(pcfg.default["smoke"]["e6_parts"]) & set(e6.PART_FLAGS)
    cache = tmp_path / "cache_smoke"
    pre = make_factory(pcfg, proot, cache)
    out = prescore.prescore(MODEL, pcfg, smoke=True, root=proot, chunk=50, guard_factory=pre,
                            access_log=recorder_class())
    assert out["tok512"].startswith("not needed") and "bipia_all" not in out["windows_by_role"]
    assert "tok512" not in out["e6_parts"] and "bipia_all" not in out["e6_parts"]
    prescored = _cache_keys(cache)
    e6_only = _e6_only_hashes(proot, smoke=True)
    assert e6_only and not (e6_only & prescored)

    consumer = make_factory(pcfg, proot, cache, forbid_model=True)
    res = _consume(pcfg, proot, consumer, True, recorder_class, latency_guards="always")   # smoke never times guards
    assert consumer.state["model"].seen == [] and consumer.state["requested"] == prescored
    status = {r["part"]: r["status"] for r in res["tables"]["parts"]}
    for part in ("bipia_all", "tok512", "flyhash40"):
        assert status[part] == e6.STATUS_SMOKE
    assert status["k"] == status["tau80"] == e6.STATUS_RUN and status["k_flyhash"] == e6.STATUS_NOT_REQUESTED
    assert res["e6_parts"]["status"] == status and res["smoke"] is True
    assert not any("bipia_all" in k or "tok512" in k or k.startswith(("auc/deep/flyhash8", "macro_auc/flyhash8"))
                   for k in res["numbers"])
    assert any("не выполнено в смоуке" in n for n in res["notes"])
    e1res = read_result(proot / "results" / "smoke" / "E1" / "0.json")
    assert not any(k.startswith("latency_ms/protectai_v2") for k in e1res["numbers"])
    assert any("transformer latency not measured" in n for n in e1res["notes"])


def test_interrupted_chunk_resumes_with_only_the_missing_hashes(pcfg, proot, tmp_path, recorder_class):
    cache = tmp_path / "cache_resume"
    batch = int(pcfg.default["baselines"]["transformers"]["batch_size"])
    chunk = 2 * batch - 3                                             # two forward calls per chunk
    first = make_factory(pcfg, proot, cache, fail_on=4)               # the 2nd call of chunk 2 fails
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prescore.prescore(MODEL, pcfg, root=proot, chunk=chunk, tok512=False, guard_factory=first,
                          access_log=recorder_class())
    ctx = Context(pcfg, root=proot, access_log=recorder_class())
    fc = FeatureContext(ctx, 0, cache=False)
    selected, _ = prescore.guard_windows(fc, e6.resolve_parts(pcfg))
    order = selected["text_hash"].tolist()
    assert len(order) > 3 * chunk
    after_crash = _cache_keys(cache)
    assert after_crash == set(order[:chunk])                          # chunk 1 kept, the interrupted chunk 2 lost

    second = make_factory(pcfg, proot, cache)
    out = prescore.prescore(MODEL, pcfg, root=proot, chunk=chunk, tok512=False, guard_factory=second,
                            access_log=recorder_class())
    rescored = [text_hash(t) for t in second.state["model"].seen]
    assert sorted(rescored) == sorted(set(order) - after_crash)       # each missing hash exactly once, nothing else
    assert out["windows_cached_before"] == chunk and out["windows_scored"] == len(order) - chunk
    assert _cache_keys(cache) == set(order) and out["tok512"] == "skipped (--no-tok512)"

    third = make_factory(pcfg, proot, cache, forbid_model=True)       # a complete cache: nothing to do
    again = prescore.prescore(MODEL, pcfg, root=proot, chunk=chunk, tok512=False, guard_factory=third,
                              access_log=recorder_class())
    assert again["windows_scored"] == 0 and third.state["model"].seen == []


def test_unavailable_model_and_cli(pcfg, proot, tmp_path, recorder_class, capsys):
    missing = prescore.prescore("piguard", pcfg, root=proot, access_log=recorder_class(),
                                guard_factory=lambda n, **kw: GuardModel(n, pcfg, root=proot))
    assert missing == {"model": "piguard", "available": False}
    cache = tmp_path / "cache_cli"
    assert prescore.main(["--model", MODEL, "--no-tok512", "--chunk", "40"], root=proot,
                         guard_factory=make_factory(pcfg, proot, cache), access_log=recorder_class()) == 0
    out = capsys.readouterr().out
    assert '"windows_scored"' in out and '"tok512": "skipped (--no-tok512)"' in out and _cache_keys(cache)
