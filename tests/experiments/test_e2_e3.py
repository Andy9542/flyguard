"""E2 learning curves and E3 transfer folds on synthetic tables (tests/experiments/conftest.py fixtures plus a richer
AgentDojo tree with two attack templates and three suites). Deterministic, no network, no real data. The Runner
entry points write into a private copy of the toy tree: the session-scoped ``toy_root`` is shared with the other
test modules (``test_report`` copies it), so no results file may be left in it."""
from __future__ import annotations

import importlib.util
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import Configs, config_hash
from flyguard.data.build import DOCUMENTS_SCHEMA, WINDOWS_SCHEMA, documents_to_frame, write_parquet
from flyguard.data.pools import build_pools
from flyguard.data.splits import build_splits
from flyguard.data.windows import build_windows
from flyguard.experiments import Context, FeatureContext, ResultBuilder, read_result, summarize
from flyguard.experiments import e2, e3
from flyguard.experiments.results import check_key
from flyguard.io import atomic_write_json

_spec = importlib.util.spec_from_file_location("exp_conftest", Path(__file__).with_name("conftest.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

E2_DETS = ["real_fly_bloom", "flyhash_bloom", "real_fly_linear", "tfidf_lr", "knn1", "centroid"]
E3_DETS = ("tfidf_lr", "knn1", "real_fly_bloom", "flyhash_bloom")


def cfg_with_e2(cfg: Configs, **over) -> Configs:
    exps = dict(cfg.experiments)
    exps["E2"] = {**exps["E2"], **over}
    return Configs(cfg.operator, cfg.default, exps)


def make_transfer_root(root: Path, cfg: Configs) -> None:
    """The toy tree plus AgentDojo episodes of three suites x five tasks x two templates (ТЗ 1.10 folds)."""
    documents, episodes = C.make_documents()
    rng = random.Random(11)
    rows, eps = [], []
    for suite in ("workspace", "travel", "banking"):
        for t in range(5):
            task = f"user_task_{10 + t}"
            split = "val" if zlib.crc32(task.encode()) % 5 == 0 else "test"
            for attack in (None, "important_instructions", "tool_knowledge"):
                itask = f"injection_task_{t}" if attack else None
                eid = f"{suite}/{task}/{itask or 'none'}/{attack or 'none'}/{C.MODEL}"
                cls = "benign" if attack is None else ("hijacked" if t % 2 else "injection_ignored")
                for step in range(2):
                    body = "tool output: " + C.paragraph(rng, rng.randint(2, 4))
                    spans, label = [], 0
                    if attack and step == 1:
                        body, spans = C.with_injection(rng, body)
                        label = 1
                    rows.append(C._doc(f"dojo:{eid}#{step}", "dojo", split, label, body, f"{suite}/{task}", spans,
                                       {"suite": suite, "user_task": task, "injection_task": itask, "attack": attack,
                                        "episode_id": eid, "step": step, "episode_class": cls, "variant": "main",
                                        "model": C.MODEL}))
                eps.append({"episode_id": eid, "benchmark": "agentdojo", "suite": suite, "user_task": task,
                            "injection_task": itask, "attack": attack, "model": C.MODEL, "episode_class": cls,
                            "utility": True, "security": cls == "hijacked", "n_steps": 2,
                            "injection_step": 1 if attack else None, "first_harmful_step": None, "match": None,
                            "contract_split": "train", "e1_val_task": split == "val", "log_path": "x", "sha256": "0" * 64})
    documents = pd.concat([documents, pd.DataFrame(rows)], ignore_index=True).sort_values("doc_id", kind="stable")
    documents = documents.reset_index(drop=True)
    episodes = pd.concat([episodes, pd.DataFrame(eps)], ignore_index=True)
    windows = build_windows(documents, cfg)
    documents["dedup_dropped"] = False
    splits = build_splits(documents, cfg, set(), episodes[episodes["benchmark"] == "agentdojo"])
    processed, manifests = root / "data" / "processed", root / "data" / "manifests"
    write_parquet(documents_to_frame(documents), processed / "documents.parquet", DOCUMENTS_SCHEMA)
    write_parquet(windows, processed / "windows.parquet", WINDOWS_SCHEMA)
    write_parquet(episodes, processed / "episodes.parquet", None)
    atomic_write_json(manifests / "splits.json", splits)
    atomic_write_json(manifests / "pools.json", build_pools(documents, cfg, set()))
    C.make_toy_connectome(processed / "connectome" / "malecns_R.npz")
    C.make_fake_guard(root, cfg)


@pytest.fixture(scope="module")
def gf(toy_cfg, toy_root, fake_loader):
    return lambda name, **kw: GuardModel(name, toy_cfg, root=toy_root, loader=fake_loader, **kw)


@pytest.fixture(scope="module")
def toy(toy_cfg, toy_root, recorder_class, gf):
    rec = recorder_class()
    ctx = Context(toy_cfg, root=toy_root, access_log=rec)
    return ctx, rec, FeatureContext(ctx, 0, purpose="e2e3 tests", guard_factory=gf, cache=False)


@pytest.fixture(scope="module")
def transfer(toy_cfg, tmp_path_factory, recorder_class, fake_loader):
    root = tmp_path_factory.mktemp("flyguard_transfer")
    make_transfer_root(root, toy_cfg)
    rec = recorder_class()
    ctx = Context(toy_cfg, root=root, access_log=rec)
    gfr = lambda name, **kw: GuardModel(name, toy_cfg, root=root, loader=fake_loader, **kw)  # noqa: E731
    return ctx, rec, FeatureContext(ctx, 0, purpose="e3 tests", guard_factory=gfr, cache=False), root


def _check_numbers(numbers: dict) -> None:
    for key, rec in numbers.items():
        check_key(key)
        assert set(rec) >= {"value", "ci_low", "ci_high", "n", "note"}, key
        if rec["ci_low"] is not None:
            assert rec["ci_low"] - 1e-12 <= rec["value"] <= rec["ci_high"] + 1e-12, key


# ---------------------------------------------------------------------------------------------- E2
@pytest.fixture(scope="module")
def e2_run(toy):
    _, _, fc = toy
    rb = ResultBuilder()
    out = e2.run_e2(fc, rb, detectors=E2_DETS, shots=[1, 10, 100, "full"], n_reps=2)
    return rb, out


def test_e2_levels_fits_and_skipped_level(toy, e2_run, toy_cfg):
    _, _, fc = toy
    rb, out = e2_run
    levels = {r["level"]: r for r in rb.tables["fewshot_levels"]}
    assert set(levels) == {"1", "10", "100", "full"}
    assert levels["100"]["status"] == e2.STATUS_NO_DATA and "positive" in levels["100"]["reason"]
    assert levels["1"]["n_reps"] == 2 and levels["1"]["n_train_docs"] == 2 and levels["full"]["n_reps"] == 1
    assert any("100" in n and e2.STATUS_NO_DATA in n for n in rb.notes)
    assert len(rb.tables["fewshot_fits"]) == len(out["index"]) == 2 * 2 * 6 + 6
    names = [it.name for it in out["index"]]
    assert len(set(names)) == len(names) and "real_fly_bloom@1:0" in names and "knn1@full:0" in names
    full = out["fitted"]["real_fly_bloom@full:0"]
    assert full.choices["gamma"] == fc.fit("real_fly_bloom").choices["gamma"] and full.spec.code_key == ("n51_svd", "measured", None)
    assert out["fitted"]["real_fly_bloom@1:0"].n_train < full.n_train
    gam = rb.numbers["fewshot/hyper/real_fly_bloom/gamma/shots1"]
    assert gam["value"] in toy_cfg.default["readout"]["bloom"]["gammas"] and gam["n"] == 2 and gam["note"].startswith("reps")
    assert rb.numbers["fewshot/n_train_docs/shots1"]["value"] == 2 and "fewshot/n_train_docs/shots100" not in rb.numbers


def test_e2_curve_numbers_bands_and_h1b_pairs(toy, e2_run):
    rb, out = e2_run
    nums = rb.numbers
    _check_numbers(nums)
    m = nums["fewshot/macro_auc/shots1/real_fly_bloom"]
    assert m["n_reps"] == 2 and m["band_low"] <= m["value"] <= m["band_high"] and m["rep_sd"] is not None
    assert m["note"] == "sources=bipia,deep,dojo,dyn,para" and m["ci_low"] is not None
    assert nums["fewshot/macro_auc/full/tfidf_lr"]["n_reps"] == 1 and nums["fewshot/macro_auc/full/tfidf_lr"].get("rep_sd") is None
    for level in ("1", "10"):
        for a in ("real_fly_bloom", "flyhash_bloom"):
            d95, d90 = nums[f"diff/macro_auc/shots{level}/{a}-knn1"], nums[f"diff90/macro_auc/shots{level}/{a}-knn1"]
            assert 0.0 <= d95["p"] <= 1.0 and d95["ci_low"] <= d90["ci_low"] <= d90["ci_high"] <= d95["ci_high"]
            assert d95["n_reps"] == 2
    assert "diff/macro_auc/full/real_fly_bloom-tfidf_lr" in nums and "diff/macro_auc/shots100/real_fly_bloom-knn1" not in nums
    assert "fewshot/auc/deep/shots1/knn1" in nums and "fewshot/auc/para_deep/shots10/centroid" in nums
    from flyguard.experiments.verdicts_run import SHOT_TOKENS, find_pair_key   # the H1b(ii) consumer finds the keys
    for k in ("1", "10"):
        assert find_pair_key(nums, "diff", "macro_auc", "flyhash_bloom", "knn1",
                             ["|".join(t.format(k=k) for t in SHOT_TOKENS)]) == f"diff/macro_auc/shots{k}/flyhash_bloom-knn1"
    assert not any(k.startswith("fewshot/auc/notinject") for k in nums)
    rows = pd.DataFrame(rb.tables["fewshot_curve"])
    assert len(rows) == len(out["index"]) and {"macro_auc", "auc_deep", "auc_para_deep"} <= set(rows.columns)
    one = rows[(rows["detector"] == "real_fly_bloom") & (rows["level"] == "1")]
    assert one["macro_auc"].mean() == pytest.approx(m["value"]) and one["macro_auc"].min() == pytest.approx(m["band_low"])
    assert set(out["doc_tables"]) == {"deep", "bipia", "dojo", "dyn", "para", "para_deep"}
    assert {r["source"] for r in rb.tables["fewshot_sources"] if r["in_macro"]} == {"deep", "bipia", "dojo", "dyn", "para"}


def test_e2_shot_levels_parse():
    assert e2.shot_levels([1, "10", "Full"]) == [1, 10, "full"] and [e2.level_key(x) for x in (1, "full")] == ["shots1", "full"]
    with pytest.raises(ValueError):
        e2.shot_levels([1, 1])
    with pytest.raises(ValueError):
        e2.shot_levels([0])


# ---------------------------------------------------------------------------------------------- E3
def test_e3_degenerate_folds_report_no_data(toy):
    ctx, rec, fc = toy
    before = len(rec.calls)
    rb = ResultBuilder()
    res = e3.run_e3(fc, rb, detectors=E3_DETS)
    fam = {r["family"]: r for r in rb.tables["transfer_families"]}
    assert fam["cross_template"]["status"] == e3.STATUS_NO_DATA and "D6" in fam["cross_template"]["reason"]
    assert fam["double_holdout"]["status"] == e3.STATUS_NO_DATA and fam["double_holdout"]["n_usable"] == 0
    assert fam["cross_suite"]["status"] == e3.STATUS_PARTIAL and fam["cross_suite"]["n_run"] == 2   # banking, slack absent
    assert "important_instructions" in fam["cross_suite"]["templates_present"]
    folds = {(r["family"], r["fold"]): r for r in rb.tables["transfer_folds"]}
    assert len(folds) == 4 + 4 + 16 and folds[("cross_template", "important_instructions")]["reason"].startswith("нет n_train_pos")
    assert folds[("cross_suite", "workspace")]["status"] == e3.STATUS_OK and folds[("cross_suite", "workspace")]["n_val_clusters"] == 1
    assert all(r["status"] == e3.STATUS_NO_DATA for r in res["double_holdout"])
    new = rec.calls[before:]
    assert len(new) == 2 and all("dojo#e3" in c[2] and c[1] == "test" for c in new)
    assert [k for k, _ in ctx.test_reads if k == "dojo#e3"] == ["dojo#e3"]
    _check_numbers(rb.numbers)
    for det in E3_DETS + ("regex",):
        assert f"transfer/cross_suite/workspace/fpr_at_tau/{det}" in rb.numbers
    assert not any("/tpr_at_tau/" in k or "/auc/" in k for k in rb.numbers)    # cross-suite: FPR only (ТЗ 1.10)
    th = rb.thresholds["tau_fold/cross_suite/travel/tfidf_lr"]
    assert set(th) >= {"value", "source", "target", "n", "achieved_fpr", "below_pool_min"}
    assert th["source"] == "fold_val_negatives:cross_suite/travel" and th["target"] == "fpr<=0.05" and th["below_pool_min"]
    mean = rb.numbers["transfer/cross_suite/fpr_at_tau/knn1"]
    assert mean["n"] == 2 and mean["fold_min"] <= mean["value"] <= mean["fold_max"]
    assert any("static" in n for n in rb.notes) and any(e3.STATUS_NO_DATA in n and "cross_template" in n for n in rb.notes)


def test_e3_two_templates_three_suites(transfer):
    ctx, rec, fc, _ = transfer
    e3_manifest = ctx.splits["e3"]
    assert sorted(e3_manifest["templates_present"]) == ["important_instructions", "tool_knowledge"]
    rb = ResultBuilder()
    res = e3.run_e3(fc, rb, detectors=E3_DETS)
    fam = {r["family"]: r for r in rb.tables["transfer_families"]}
    assert (fam["cross_template"]["n_run"], fam["cross_suite"]["n_run"], fam["double_holdout"]["n_run"]) == (2, 3, 6)
    assert fam["double_holdout"]["status"] == e3.STATUS_PARTIAL
    assert len(rec.calls) == 2 and "dojo#e3" in rec.calls[0][2]
    nums = rb.numbers
    _check_numbers(nums)
    for det in E3_DETS:
        for t in ("important_instructions", "tool_knowledge"):
            tpr, auc = nums[f"transfer/cross_template/{t}/tpr_at_tau/{det}"], nums[f"transfer/cross_template/{t}/auc/{det}"]
            assert tpr["n_pos"] > 0 and tpr["n"] == tpr["n_pos"] and 0.0 <= tpr["value"] <= 1.0 and 0.0 <= auc["value"] <= 1.0
            assert f"transfer/cross_template/{t}/fpr_at_tau/{det}" not in nums
        for s in ("workspace", "travel", "banking"):
            fpr = nums[f"transfer/cross_suite/{s}/fpr_at_tau/{det}"]
            assert fpr["n"] == fpr["n_neg"] > 0
            for t in ("important_instructions", "tool_knowledge"):
                dh = f"transfer/double_holdout/{s}x{t}"
                assert {f"{dh}/tpr_at_tau/{det}", f"{dh}/fpr_at_tau/{det}", f"{dh}/auc/{det}"} <= set(nums)
        assert nums[f"transfer/double_holdout/auc/{det}"]["n"] == 6
    fold = {r["fold"]: r for r in rb.tables["transfer_folds"] if r["family"] == "cross_template"}["tool_knowledge"]
    assert fold["n_val_clusters"] == max(1, round(0.2 * fold["n_train_clusters"])) and fold["fpr_level"] == 0.05
    fits = pd.DataFrame(rb.tables["transfer_fits"])
    assert set(fits["detector"]) == set(E3_DETS) | {"regex"} and (fits["gamma_source"].dropna() == "val").all()
    assert set(fits.loc[fits["detector"] == "regex", "trained"]) == {False}
    r = res["cross_suite"][0]
    assert r["status"] == e3.STATUS_OK and r["fold_row"]["usable"]


def test_e3_holdout_is_deterministic_and_content_based():
    frame = pd.DataFrame({"cluster_id": [f"s/t{i}" for i in range(10) for _ in range(3)]})
    a = e3.holdout_clusters(frame, 0.2, 5, "cross_suite/x")
    assert a == e3.holdout_clusters(frame.iloc[::-1], 0.2, 5, "cross_suite/x") and len(a) == 2
    assert a != e3.holdout_clusters(frame, 0.2, 5, "cross_suite/y") or a != e3.holdout_clusters(frame, 0.2, 6, "cross_suite/x")
    assert e3.holdout_clusters(frame.iloc[:3], 0.2, 5, "z") == set() and len(e3.holdout_clusters(frame.iloc[:6], 0.9, 5, "z")) == 1
    assert e3.fold_reason({"n_train_pos": 0, "n_train_neg": 3, "n_test_pos": 1, "n_test_neg": 0}) == "нет n_train_pos, n_test_neg"
    # class-aware redraw: two positive clusters out of ten, three held out -> both sides must keep both classes
    lab = frame.assign(label=frame["cluster_id"].isin(["s/t3", "s/t7"]).astype(int))
    for seed in range(6):
        chosen, attempts = e3.draw_holdout(lab, 0.3, seed, "cross_suite/x", labels=True)
        held = lab["cluster_id"].isin(chosen)
        assert len(chosen) == 3 and 1 <= attempts <= e3.HOLDOUT_ATTEMPTS
        assert lab.loc[held, "label"].nunique() == 2 and lab.loc[~held, "label"].nunique() == 2
        assert (attempts == 1) == (chosen == e3.holdout_clusters(lab, 0.3, seed, "cross_suite/x"))
    one, n = e3.draw_holdout(lab.assign(label=0), 0.3, 5, "cross_suite/x", labels=True)   # impossible -> last draw
    assert n == e3.HOLDOUT_ATTEMPTS and len(one) == 3


# ---------------------------------------------------------------------------------------------- Runner entry points
def test_run_entry_points_write_results(toy_cfg, toy_root, tmp_path_factory, recorder_class, fake_loader):
    root = tmp_path_factory.mktemp("flyguard_e2e3") / "repo"     # private copy: results never land in the shared tree
    shutil.copytree(toy_root, root)
    cfg = cfg_with_e2(toy_cfg, subsamples_per_point=2)
    rec = recorder_class()
    ctx = Context(cfg, root=root, access_log=rec)
    gf = lambda name, **kw: GuardModel(name, toy_cfg, root=root, loader=fake_loader, **kw)  # noqa: E731
    p2 = e2.run(ctx, 0, False, root=root, cfg=cfg, guard_factory=gf)
    p3 = e3.run(ctx, 0, False, root=root, cfg=cfg, guard_factory=gf)
    assert p2 == root / "results" / "E2" / "0.json" and p3 == root / "results" / "E3" / "0.json"
    assert not (toy_root / "results" / "E2").exists() and not (toy_root / "results" / "E3").exists()
    r2, r3 = read_result(p2), read_result(p3)
    assert r2["config_hash"] == config_hash(root) == r3["config_hash"] and r2["experiment"] == "E2"
    assert set(r2["timing"]["test_reads"]) == {"deep", "bipia", "dojo", "dyn", "para"} and r3["timing"]["test_reads"] == ["dojo#e3"]
    assert "diff/macro_auc/shots10/flyhash_bloom-knn1" in r2["numbers"] and r2["numbers"]["fewshot/macro_auc/shots1/knn1"]["n_reps"] == 2
    assert any(r["level"] == "100" and r["status"] == e3.STATUS_NO_DATA for r in r2["tables"]["fewshot_levels"])
    assert "transfer/cross_suite/workspace/fpr_at_tau/real_fly_bloom" in r3["numbers"] and r3["thresholds"]
    assert {r["family"] for r in r3["tables"]["transfer_families"]} == set(e3.FAMILIES)
    assert not (root / "data" / "processed" / "features" / "seed0").exists()
    assert e2.run(ctx, 0, False, root=root, cfg=cfg, guard_factory=gf) == p2      # current -> skipped
    s2 = summarize("E2", root=root)
    assert s2["numbers"]["fewshot/macro_auc/full/real_fly_bloom"]["n_seeds"] == 1 and s2["warnings"] == []
    assert len(s2["tables"]["fewshot_curve"]) == len(r2["tables"]["fewshot_curve"])
