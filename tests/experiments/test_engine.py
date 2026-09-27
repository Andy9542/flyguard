"""Engine, context and results on the toy fixture (tests/experiments/conftest.py): roles and the journaled test
door, per-seed caches, noses on C_unl, matrices, codes, detector fitting with validation choices, scoring,
document-level evaluation, Runner and summary. Deterministic, synthetic, no network."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from threadpoolctl import threadpool_info

from flyguard import nose
from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import config_hash
from flyguard.experiments import (Context, FeatureContext, Runner, detector_spec, engine, fly_spec, read_result,
                                  standard_evaluation, summarize)
from flyguard.experiments.results import number, result_path, summarize_numbers, write_result
from flyguard.readout import LinearReadout

NAMES = ["regex", "tfidf_lr", "knn1", "knn5", "centroid", "lr_svd", "real_fly_bloom", "real_fly_linear",
         "flyhash_bloom", "flyhash_linear", "protectai_v2", "piguard"]


@pytest.fixture(scope="module")
def mctx(toy_cfg, toy_root, recorder_class):
    rec = recorder_class()
    ctx = Context(toy_cfg, root=toy_root, access_log=rec)
    ctx.recorder = rec
    return ctx


@pytest.fixture(scope="module")
def gf(toy_cfg, toy_root, fake_loader):
    return lambda name, **kw: GuardModel(name, toy_cfg, root=toy_root, loader=fake_loader, **kw)


@pytest.fixture(scope="module")
def fc(mctx, gf):
    return FeatureContext(mctx, 0, purpose="tests", guard_factory=gf)


@pytest.fixture(scope="module")
def fitted(fc):
    return fc.fit_many(NAMES)


@pytest.fixture(scope="module")
def evaluation(fc, fitted):
    return standard_evaluation(fc, fitted)


# ---------------------------------------------------------------------------------------------- context
def test_context_roles_never_contain_test_rows(ctx, recorder):
    assert not (ctx.windows["split"] == "test").any() and not (ctx.documents["split"] == "test").any()
    tr = ctx.train_windows
    assert (tr["source"] == "deep").all() and set(tr["label"]) == {0, 1}
    assert (ctx.val_windows_deep["source"] == "deep").all()
    assert {"bipia", "dojo", "deep"} <= set(ctx.val_windows("all")["source"])
    assert (ctx.c_unl_windows["split"] != "test").all() and len(ctx.c_unl_windows) > len(tr)
    pv = ctx.p_val_windows
    assert (pv["label"] == 0).all() and set(pv["doc_id"]) == ctx.p_val_doc_ids
    assert ctx.test_sources == ["deep", "bipia", "dojo", "dyn", "para", "notinject"]
    assert ctx.p_test_size == len(ctx.p_test_doc_ids) and recorder.calls == []    # episodes: lazy, journaled once
    assert len(ctx.episodes) and ctx.episodes is ctx.episodes
    assert [c[:2] for c in recorder.calls] == [(str(ctx.episodes_path), "test")]
    assert ctx.connectome()[0].shape == (12, 6) and ctx.available_guards() == ["protectai_v2"]


def test_test_door_is_the_only_accessor_and_is_journaled(ctx, recorder):
    assert recorder.calls == []
    with pytest.raises(KeyError):
        ctx.documents_for(["deep:test:0"])          # unloaded test source: no silent access
    w1 = ctx.load_test_windows("deep", "unit test")
    w2 = ctx.load_test_windows("deep", "again")
    assert w1 is w2 and (w1["split"] == "test").all() and (w1["source"] == "deep").all()
    assert len(recorder.calls) == 2 and all(c[1] == "test" and "deep" in c[2] for c in recorder.calls)
    assert {c[0].split("/")[-1] for c in recorder.calls} == {"windows.parquet", "documents.parquet"}
    docs = ctx.test_documents("deep")
    assert set(docs["doc_id"]) == set(ctx.test_doc_ids("deep")) >= set(w1["doc_id"])
    assert ctx.documents_for(w1["doc_id"].head(3))["label"].isin([0, 1]).all()
    with pytest.raises(KeyError):
        ctx.load_test_windows("nowhere")
    assert ctx.test_reads == [("deep", recorder.calls[0][2])]
    one = ctx.test_doc_ids("bipia")[:1]
    sub = ctx.load_test_windows("bipia", "e6-like list", doc_ids=one, name="sub")
    assert set(sub["doc_id"]) == set(one) and len(recorder.calls) == 4 and "bipia#sub" in recorder.calls[2][2]
    assert len(ctx.test_documents("bipia", name="sub")) == 1
    with pytest.raises(ValueError):
        ctx.load_test_windows("bipia", doc_ids=one)


# ---------------------------------------------------------------------------------------------- features
def test_counts_cache_budget_and_cleanup(mctx, tmp_path):
    a = FeatureContext(mctx, 0, cache_dir=tmp_path / "f")
    X = a.counts("train")
    files = list(a.cache_dir.glob("*.npz"))
    assert len(files) == 1 and files[0].name == "counts512_train.npz" and a.cache_bytes() > 0
    assert a.counts("train") is X
    b = FeatureContext(mctx, 0, cache_dir=tmp_path / "f")
    assert (b.counts("train") != X).nnz == 0                       # served from the file, identical
    c = FeatureContext(mctx, 0, cache_dir=tmp_path / "g", cache_budget_bytes=10)
    c.counts("train")
    assert not c.cache_dir.exists()                                 # over budget -> computed, not written
    assert FeatureContext(mctx, 0, cache_dir=tmp_path / "h", cache=False).counts("train").shape == X.shape
    assert a.cleanup() > 0 and not a.cache_dir.exists()
    assert a.counts("train").shape == X.shape                       # recomputable after cleanup


def test_noses_are_fitted_on_c_unl_and_shapes_follow_the_connectome(fc):
    assert fc.d_glom == 6 and fc.bins == 512
    assert fc.n16k.mean_.shape == (512,) and fc.n51_svd.rank == 6 and fc.n51_hash.bins == 6
    U = fc.features("n16k", "train")
    assert U.shape == (fc.window_set("train").n, 512)
    assert np.allclose(np.asarray(U.sum(axis=1)).ravel(), 1.0)      # N16k rows sum to one
    assert fc.features("n51_svd", "val").shape == (12, 6) and fc.features("n51_hash", "val").shape == (12, 6)
    assert np.allclose(fc.n51_svd.transform(fc.features("n16k", "c_unl")).mean(axis=0), 0.0, atol=1e-4)
    assert fc.features("text", "val") == fc.window_set("val").texts


def test_matrices(fc):
    M = fc.matrix("measured")
    R = fc.matrix("random")
    assert M.shape == (12, 6) and set(np.unique(M.data)) == {1.0}
    assert np.array_equal(np.diff(R.indptr), np.diff(M.indptr))     # same in-degree per cell
    F = fc.matrix("flyhash", "n16k")
    assert F.shape == (4 * 512, 512) and set(np.diff(F.indptr)) == {6}
    assert fc.matrix("flyhash8", "n16k").shape == (8 * 512, 512)
    assert fc.matrix("random:100", "n51_svd").shape == (100, 6) and fc.matrix("random:1024", "n16k").shape == (1024, 512)
    nulls = fc.curveball_set()
    assert len(nulls) == 3 and all(np.array_equal(np.diff(n.indptr), np.diff(M.indptr)) for n in nulls)
    assert fc.matrix("curveball:2") is nulls[2] and fc.matrix("measured") is M
    assert fc.matrix("dense_sign").shape == (round(M.nnz / 6), 6) and fc.matrix("weighted").max() > 1
    with pytest.raises(ValueError):
        fc.matrix("measured", "n16k")
    with pytest.raises(ValueError):
        fly_spec("bad", readout="bloom", matrix="dense_sign")


def test_codes_have_exactly_k_winners_and_subsets_are_slices(fc):
    Z = fc.codes(detector_spec("real_fly_bloom"), "train")
    assert Z.shape == (fc.window_set("train").n, 12) and set(np.diff(Z.indptr)) == {fc.k_for(12)}
    Zf = fc.codes(detector_spec("flyhash_bloom"), "train")
    assert set(np.diff(Zf.indptr)) == {fc.k_for(2048)} == {102}
    few = fc.fewshot_set(3, 0)
    assert few.frame["doc_id"].nunique() == 6 and few.parent == "train"
    assert (fc.codes(detector_spec("real_fly_bloom"), few) != Z[few.rows]).nnz == 0
    assert set(fc.fewshot_set(3, 1).frame["doc_id"]) != set(few.frame["doc_id"])
    streamed = [z for _, z in fc.iter_codes(("n51_svd", "measured", None), "val", batch_rows=5)]
    assert len(streamed) == 3 and (fc.codes(("n51_svd", "measured", None), "val") != sp.vstack(streamed)).nnz == 0
    with pytest.raises(ValueError):
        fc.fewshot_set(1000, 0)


def test_streamed_bloom_fit_is_bit_identical(mctx, fc, fitted, gf, monkeypatch):
    """Training sets too large to memoise (E3 folds) are fitted from streamed class counts: same model, same γ."""
    monkeypatch.setattr(engine, "CODE_BATCH_BYTES", 1)                    # 32-row batches: two per training set
    st = FeatureContext(mctx, 0, cache=False, guard_factory=gf, code_memo_bytes=1)
    key = detector_spec("flyhash_bloom").code_key
    assert (FeatureContext.codes(st, key, "train") != fc.codes(key, "train")).nnz == 0   # preallocated, 2 batches
    monkeypatch.setattr(st, "codes", lambda *a, **k: pytest.fail("the streamed fit materialised a code matrix"))
    for spec in (detector_spec("real_fly_bloom"), detector_spec("flyhash_bloom"), fly_spec("g5", gamma=0.5)):
        a, b = fitted.get(spec.name) or fc.fit(spec), st.fit(spec)
        assert a.choices == b.choices and len(set(b.choices["balanced_counts"])) == 1, spec.name
        for attr in ("F_", "n_seen_", "class_counts_", "balanced_counts_", "balanced_index_"):
            assert np.array_equal(getattr(a.model, attr), getattr(b.model, attr)), (spec.name, attr)
        assert st.score_windows(b, "test:deep").equals(fc.score_windows(a, "test:deep"))


def test_svd_and_fits_are_pinned_to_one_thread(mctx, monkeypatch):
    """ТЗ 2.6: the randomized SVD and the lbfgs fits change bits with the BLAS thread count (``--jobs``)."""
    seen = []
    for cls in (nose.N51Svd, LinearReadout):
        monkeypatch.setattr(cls, "fit", lambda self, *a, _o=cls.fit: seen.append(
            {p["num_threads"] for p in threadpool_info()}) or _o(self, *a))
    one = FeatureContext(mctx, 0, cache=False)
    one.n51_svd
    one.fit("real_fly_linear")
    assert len(seen) >= 2 and all(s == {1} for s in seen)          # the SVD, then every C of the grid


# ---------------------------------------------------------------------------------------------- detectors
def test_fit_records_validation_choices(fc, fitted, toy_cfg):
    grid = toy_cfg.default["readout"]["linear"]["C_grid"]
    assert fitted["tfidf_lr"].choices["idf_source"] == "c_unl" and fitted["tfidf_lr"].choices["C_source"] == "val"
    assert fitted["real_fly_bloom"].choices["gamma"] in toy_cfg.default["readout"]["bloom"]["gammas"]
    assert set(fitted["real_fly_bloom"].choices["gamma_table"]) == {"0.0", "0.5", "0.9", "0.99"}
    assert fitted["real_fly_linear"].choices["C"] in grid and fitted["flyhash_linear"].choices["C"] in grid
    assert fitted["regex"].choices["n_patterns"] >= 15
    assert fitted["protectai_v2"].available and not fitted["piguard"].available
    assert all(isinstance(f.state_size_bytes(), int) for n, f in fitted.items() if f.available)
    few = fc.fit("real_fly_bloom", train=fc.fewshot_set(1, 0))
    assert few.n_train == fc.fewshot_set(1, 0).n and "gamma_table" in few.choices
    fixed = fc.fit(fly_spec("g99", gamma=0.99, k_frac=0.25))
    assert fixed.choices == {"gamma": 0.99, "gamma_source": "fixed", "balanced_counts": fixed.choices["balanced_counts"]}
    assert fixed.model.k == fc.k_for(12, 0.25)
    no_kc = fc.fit(fly_spec("nose_only", nose="n51_svd", matrix=None, readout="linear"))
    assert no_kc.model.model_.coef_.shape == (1, 6)


def test_scoring_is_grouped_cached_and_deterministic(mctx, fc, fitted, gf):
    ws = fc.window_set("test:deep")
    raw = mctx.load_test_windows("deep")
    assert ws.n == len(raw) - int(raw["dedup_excluded"].sum()) and not ws.frame["dedup_excluded"].any()
    scores = fc.score_many(fitted, ws)
    assert set(scores) == set(NAMES) - {"piguard"}
    for s in scores.values():
        assert list(s.index) == list(ws.window_ids) and float(s.min()) >= 0.0 and float(s.max()) <= 1.0
    assert fc.score_windows(fitted["flyhash_bloom"], ws).equals(scores["flyhash_bloom"])
    assert fc.score_windows(fitted["piguard"], ws) is None
    calls = fc.guard("protectai_v2")._model.calls
    fc.score_guard("protectai_v2", ws)
    assert fc.guard("protectai_v2")._model.calls == calls          # in-memory and hash cache: no new forward pass
    same = FeatureContext(mctx, 0, cache=False, guard_factory=gf)
    other = FeatureContext(mctx, 1, cache=False, guard_factory=gf)
    s_same = same.score_windows(same.fit("flyhash_linear"), ws)
    s_other = other.score_windows(other.fit("flyhash_linear"), ws)
    assert s_same.equals(scores["flyhash_linear"]) and not s_other.equals(scores["flyhash_linear"])


# ---------------------------------------------------------------------------------------------- evaluation
def test_doc_frame_joins_labels_and_strata(fc, fitted):
    ws = fc.window_set("test:notinject")
    df = fc.doc_frame(fc.score_many([fitted["tfidf_lr"], fitted["regex"]], ws), ws)
    assert {"doc_id", "label", "cluster_id", "source", "lang_stratum", "subset", "tfidf_lr", "regex"} <= set(df.columns)
    assert (df["label"] == 0).all() and set(df["subset"]) == {"one", "two", "three"} and set(df["lang_stratum"]) == {"en", "non-en"}
    single = fc.doc_frame(fc.score_windows(fitted["knn1"], "val"), "val")
    assert "score" in single.columns and set(single["label"]) == {0, 1} and single["doc_id"].is_unique


def test_standard_evaluation_numbers_thresholds_and_keys(evaluation):
    nums, th = evaluation["numbers"], evaluation["thresholds"]
    for key in ("auc/deep/tfidf_lr", "auc/para_deep/regex", "macro_auc/real_fly_bloom", "tpr_at_fpr/deep/tfidf_lr",
                "fpr_ptest/tfidf_lr", "fpr_notinject/tau_fpr/tfidf_lr", "fpr_notinject/tau90_deep/real_fly_bloom/one",
                "fpr_notinject/tau90_deep/real_fly_bloom/non-en", "val_auc/deep/real_fly_bloom", "val_macro_auc/tfidf_lr",
                "hyper/real_fly_bloom/gamma", "hyper/tfidf_lr/C", "latency_ms/flyhash_bloom", "state_bytes/regex",
                "diff/macro_auc/real_fly_linear-lr_svd", "diff90/macro_auc/real_fly_linear-lr_svd",
                "diff/auc/deep/tfidf_lr-protectai_v2", "diff/fpr_notinject/tau90_deep/real_fly_bloom-protectai_v2"):
        assert key in nums, key
    for key, rec in nums.items():
        assert set(rec) >= {"value", "ci_low", "ci_high", "n", "note"}
        if rec["ci_low"] is not None:
            assert rec["ci_low"] - 1e-12 <= rec["value"] <= rec["ci_high"] + 1e-12, key
    d95, d90 = nums["diff/macro_auc/real_fly_linear-lr_svd"], nums["diff90/macro_auc/real_fly_linear-lr_svd"]
    assert 0.0 <= d95["p"] <= 1.0 and d95["ci_low"] <= d90["ci_low"] and d90["ci_high"] <= d95["ci_high"]
    assert nums["macro_auc/tfidf_lr"]["note"] == "sources=bipia,deep,dojo,dyn,para"
    assert set(th["tau_fpr/tfidf_lr"]) >= {"value", "source", "target", "n"} and th["tau_fpr/tfidf_lr"]["source"] == "P_val"
    assert th["tau90_deep/regex"]["target"] == "tpr>=0.9" and "tau90_dojo/regex" in th
    assert not any(k.startswith("auc/notinject/") for k in nums) and "macro_auc/piguard" not in nums
    assert len(evaluation["tables"]["detectors"]) == len(NAMES) and len(evaluation["tables"]["sources"]) == 6
    assert any("piguard" in n and "unavailable" in n for n in evaluation["notes"])
    assert set(evaluation["doc_tables"]) == {"deep", "bipia", "dojo", "dyn", "para", "notinject", "para_deep"}


# ---------------------------------------------------------------------------------------------- results
def _body(fc, rb):
    fitted = fc.fit_many(["tfidf_lr", "real_fly_bloom"])
    rb.merge(standard_evaluation(fc, fitted, pairs=[("macro_auc", "real_fly_bloom", "tfidf_lr")], latency=False,
                                 validation=False, notinject=False))
    rb.add_number("custom/x", 1.5, n=3)
    rb.note("n_docs=5")


def test_runner_writes_skips_and_summarizes(toy_cfg, toy_root, mctx, gf):
    paths = [Runner("E1", s, root=toy_root, ctx=mctx, guard_factory=gf).run(_body) for s in (0, 1)]
    assert paths[0] == result_path("E1", 0, root=toy_root) and paths[0].exists()
    r = read_result(paths[0])
    assert r["config_hash"] == config_hash(toy_root) and r["smoke"] is False and r["seed"] == 0
    assert set(r["seeds"]) == set(toy_cfg.default["seeds"]["children"])
    assert r["timing"]["test_reads"] == [] == read_result(paths[1])["timing"]["test_reads"]  # shared ctx: no new reads
    assert set(r["timing"]["threads"]) == {"env", "cpu_count", "pools"}      # thread counts in force (A39)
    assert r["numbers"]["custom/x"] == {"value": 1.5, "ci_low": None, "ci_high": None, "n": 3, "note": None}
    assert "n_docs=5" in r["notes"] and "macro_auc/real_fly_bloom" in r["numbers"]
    runner = Runner("E1", 0, root=toy_root, ctx=mctx, guard_factory=gf)
    mtime = paths[0].stat().st_mtime_ns
    assert runner.should_skip() and runner.run(_body) == paths[0] and paths[0].stat().st_mtime_ns == mtime
    assert not (toy_root / "data" / "processed" / "features" / "seed1").exists()   # cache cleaned after the run
    summ = summarize("E1", root=toy_root)
    m = summ["numbers"]["macro_auc/real_fly_bloom"]
    per = [m["per_seed"][s]["value"] for s in ("0", "1")]
    assert summ["n_seeds"] == 2 and m["n_seeds"] == 2 and m["mean"] == pytest.approx(np.mean(per))
    assert m["seed_ci_low"] is not None and m["seed_ci_low"] <= m["mean"] <= m["seed_ci_high"]
    assert m["per_seed"]["0"]["ci_low"] is not None and summ["warnings"] == [] and summ["config_hash"] == r["config_hash"]
    assert {row["seed"] for row in summ["tables"]["detectors"]} == {0, 1}
    assert (toy_root / "results" / "E1" / "summary.json").exists()
    smoke = Runner("E1", 0, smoke=True, root=toy_root, cfg=toy_cfg, guard_factory=gf)
    smoke._ctx = Context(toy_cfg, smoke=False, root=toy_root, access_log=lambda *a: None)
    assert smoke.run(_body) == toy_root / "results" / "smoke" / "E1" / "0.json"
    sr = read_result(smoke.result_path())
    assert sr["smoke"] is True and set(sr["timing"]["test_reads"]) == set(mctx.test_sources)


def test_results_helpers(tmp_path):
    one = summarize_numbers({0: {"a/b": number(0.5, {"low": 0.4, "high": 0.6, "level": 0.95, "n": 7})}})
    assert one["a/b"]["sd"] is None and one["a/b"]["seed_ci_low"] is None and one["a/b"]["n"] == 7
    assert number(float("nan"))["value"] is None and number(1.0, p=0.02)["p"] == 0.02
    with pytest.raises(ValueError):
        write_result("E9", 0, {"bad key/": 1.0}, {}, {}, [], root=tmp_path)
    p = write_result("E9", 3, {"m/x": 0.25}, {"t": [{"k": 1}]}, {"tau": {"value": 0.5, "source": "s", "target": "t", "n": 1}},
                     ["ok"], smoke=True, root=tmp_path)
    assert p == tmp_path / "results" / "smoke" / "E9" / "3.json" and read_result(p)["numbers"]["m/x"]["value"] == 0.25
