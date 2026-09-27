"""E6 and the contract run on a synthetic root (conftest helpers + BIPIA E6 variants, contract train/val/test
episodes of both benchmarks, a split manifest). Deterministic, no network, no real data."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyguard.agentdojo_io.contract import validate_csv, write_split_manifest
from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import config_hash, seeds_for
from flyguard.data.build import DOCUMENTS_SCHEMA, WINDOWS_SCHEMA, documents_to_frame, write_parquet
from flyguard.data.pools import build_pools
from flyguard.data.splits import build_splits
from flyguard.data.windows import build_windows
from flyguard.experiments import Context, read_result, summarize
from flyguard.experiments import contract_run, e6
from flyguard.io import atomic_write_json, read_json

_spec = importlib.util.spec_from_file_location("e6_conftest_helpers", Path(__file__).with_name("conftest.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

ATTACK = "important_instructions"


def _tasks() -> dict[str, str]:
    """Task ids by role from the crc32 rules: contract test (mod 3 == 2), one of them also E1-val (mod 5 == 0)."""
    ids = [f"user_task_{i}" for i in range(60)]
    test = [t for t in ids if zlib.crc32(t.encode()) % 3 == 2]
    both = [t for t in test if zlib.crc32(t.encode()) % 5 == 0]
    plain = [t for t in test if t not in both]
    non_test = [t for t in ids if zlib.crc32(t.encode()) % 3 != 2 and zlib.crc32(t.encode()) % 5 != 0]
    return {"test_a": plain[0], "test_b": plain[1], "test_e1val": both[0], "test_nodocs": plain[2],
            "train": non_test[0], "val": non_test[1]}


def make_e6_root(root: Path, cfg) -> dict:
    docs, _ = H.make_documents()
    docs = docs[~docs["source"].isin(["dojo", "dyn"])].reset_index(drop=True)
    rng = random.Random(11)
    rows: list[dict] = []
    for c in range(6):  # BIPIA E6 variants: 2 attack names x 3 positions per context, minus the sampled main pair
        split, cid = ("val" if c < 2 else "test"), f"bipia:email:{c}"
        context = docs.loc[docs["doc_id"] == f"{cid}:clean", "text"].iloc[0]
        for a_i, attack in enumerate(("attack one", "attack two")):
            inj = H.INJECTIONS[a_i]
            for pos in ("start", "middle", "end"):
                if attack == "attack one" and pos == "middle":
                    continue
                if pos == "start":
                    text, spans = inj + " " + context, [(0, len(inj))]
                elif pos == "end":
                    text, spans = context + " " + inj, [(len(context) + 1, len(context) + 1 + len(inj))]
                else:
                    text, spans = H.with_injection(rng, context)
                rows.append(H._doc(f"{cid}:attack-{a_i}-0:{pos}", "bipia", split, 1, text, cid, spans,
                                   {"task": "email", "context": c, "variant": "e6", "attack": attack,
                                    "attack_id": f"attack-{a_i}-0", "position": pos, "pair_of": f"{cid}:clean"}))
    T = _tasks()
    episodes: list[dict] = []

    def add_episode(src: str, bench: str, suite: str, task: str, attacked: bool, cls: str, n_steps: int, e1: str,
                    csplit: str, with_docs: bool = True, inj_step=1, harmful=None, match=None) -> None:
        itask, attack = (f"injection_task_{task[-1]}", ATTACK) if attacked else (None, None)
        eid = f"{suite}/{task}/{itask or 'none'}/{attack or 'none'}/{H.MODEL}"
        for step in range(n_steps if with_docs else 0):
            body = "tool output: " + H.paragraph(rng, rng.randint(2, 4))
            spans, label = [], 0
            if attacked and step == inj_step:
                body, spans = H.with_injection(rng, body)
                label = 1
            rows.append(H._doc(f"{src}:{eid}#{step}", src, e1, label, body, f"{suite}/{task}", spans,
                               {"suite": suite, "user_task": task, "injection_task": itask, "attack": attack,
                                "episode_id": eid, "step": step, "episode_class": cls, "variant": "main",
                                "contract_split": csplit, "model": H.MODEL}))
        episodes.append({"episode_id": eid, "benchmark": bench, "suite": suite, "user_task": task, "injection_task": itask,
                         "attack": attack, "model": H.MODEL, "episode_class": cls, "utility": True,
                         "security": cls == "hijacked", "n_steps": n_steps,
                         "injection_step": (inj_step if attacked and with_docs else None),
                         "first_harmful_step": harmful, "match": match, "contract_split": csplit,
                         "e1_val_task": zlib.crc32(task.encode()) % 5 == 0, "log_path": f"data/traces/{bench}/{eid}.json",
                         "sha256": "0" * 64})

    for role, suite in (("test_a", "workspace"), ("test_b", "travel"), ("test_e1val", "workspace")):
        t = T[role]
        e1_clean, e1_atk = ("val", "unused") if role == "test_e1val" else ("test", "test")
        add_episode("dojo", "agentdojo", suite, t, False, "benign", 3, e1_clean, "test")
        hij = role != "test_b"
        add_episode("dojo", "agentdojo", suite, t, True, "hijacked" if hij else "injection_ignored", 3, e1_atk, "test",
                    harmful=2 if hij else None, match="full" if hij else "unmatched")
    add_episode("dojo", "agentdojo", "travel", T["test_nodocs"], False, "benign", 0, "test", "test")
    add_episode("dojo", "agentdojo", "travel", T["test_nodocs"], True, "injection_ignored", 2, "test", "test",
                with_docs=False, inj_step=None, match="unmatched")
    for role, csplit in (("train", "train"), ("val", "val")):
        add_episode("dojo", "agentdojo", "workspace", T[role], False, "benign", 3, "test", csplit)
        add_episode("dojo", "agentdojo", "workspace", T[role], True, "injection_ignored", 3, "test", "excluded",
                    match="unmatched")
    add_episode("dyn", "agentdyn", "shopping", T["test_a"], False, "benign", 2, "test", "test")
    add_episode("dyn", "agentdyn", "shopping", T["test_a"], True, "hijacked", 3, "test", "test", harmful=2, match="full")
    add_episode("dyn", "agentdyn", "shopping", T["train"], False, "benign", 2, "test", "val")
    add_episode("dyn", "agentdyn", "shopping", T["val"], False, "benign", 2, "test", "val")

    documents = pd.concat([docs, pd.DataFrame(rows)], ignore_index=True).sort_values("doc_id", kind="stable").reset_index(drop=True)
    documents["dedup_dropped"] = False
    episodes_df = pd.DataFrame(episodes)
    windows = build_windows(documents, cfg)
    e6_ids = {d["doc_id"] for d in rows if d["source"] == "bipia" and d["split"] == "test"}
    multi = windows[windows["doc_id"].isin(e6_ids)].groupby("doc_id").size()
    victim = sorted(multi[multi >= 2].index)[0]
    windows.loc[windows.index[windows["doc_id"] == victim][-1], "dedup_excluded"] = True   # one excluded E6 window (ТЗ 1.7)
    splits = build_splits(documents, cfg, set(), episodes_df[episodes_df["benchmark"] == "agentdojo"])
    pools = build_pools(documents, cfg, set())
    processed, manifests = root / "data" / "processed", root / "data" / "manifests"
    write_parquet(documents_to_frame(documents), processed / "documents.parquet", DOCUMENTS_SCHEMA)
    write_parquet(windows, processed / "windows.parquet", WINDOWS_SCHEMA)
    write_parquet(episodes_df, processed / "episodes.parquet", None)
    atomic_write_json(manifests / "splits.json", splits)
    atomic_write_json(manifests / "pools.json", pools)
    atomic_write_json(manifests / "dedup.json", {"test_windows_excluded": 1, "documents_dropped_total": 0})
    H.make_toy_connectome(processed / "connectome" / "malecns_R.npz")
    H.make_fake_guard(root, cfg)
    (root / "logs").mkdir(exist_ok=True)
    write_split_manifest(episodes_df.to_dict("records"), root / "results" / "shared" / "split_manifest.json",
                         val_seed=seeds_for(cfg, 0)["subsample"])
    return {"episodes": episodes_df, "tasks": T, "splits": splits}


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def e6_cfg(toy_cfg):
    """The toy config with 20 bootstrap draws and two γ values: every E6 code path at a third of the runtime."""
    cfg = copy.deepcopy(toy_cfg)
    cfg.default["stats"]["bootstrap"]["n"] = 20
    cfg.experiments["E6"]["gammas"] = [0.0, 0.99]
    return cfg


@pytest.fixture(scope="module")
def e6_root(tmp_path_factory, e6_cfg):
    root = tmp_path_factory.mktemp("flyguard_e6")
    make_e6_root(root, e6_cfg)
    return root


@pytest.fixture(scope="module")
def gf(e6_cfg, e6_root, fake_loader):
    return lambda name, **kw: GuardModel(name, e6_cfg, root=e6_root, loader=fake_loader, **kw)


@pytest.fixture(scope="module")
def e6_ctx(e6_cfg, e6_root, recorder_class):
    rec = recorder_class()
    ctx = Context(e6_cfg, root=e6_root, access_log=rec)
    ctx.recorder = rec
    return ctx


@pytest.fixture(scope="module")
def e6_result(e6_ctx, e6_root, gf):
    return read_result(e6.run(e6_ctx, 0, root=e6_root, guard_factory=gf))


# ---------------------------------------------------------------------------------------------- E6
def test_e6_parts_and_detector_set(toy_cfg):
    assert e6.resolve_parts(toy_cfg) == tuple(e6.PART_FLAGS)
    specs, base_of, table = e6.e6_detectors(toy_cfg, e6.resolve_parts(toy_cfg), ["protectai_v2", "piguard"])
    names = set(specs)
    assert {"real_fly_bloom_k2.5", "real_fly_linear_k10", "real_fly_bloom_gamma0.99", "flyhash_bloom_gamma0",
            "real_fly_linear_weighted", "flyhash_bloom_normalized", "flyhash8_bloom", "flyhash8_linear",
            "protectai_v2", "lr_svd"} <= names
    assert not any("-" in n for n in names) and "flyhash_linear_k2.5" not in names
    assert specs["real_fly_bloom_gamma0.5"].gamma == 0.5 and specs["real_fly_bloom_k10"].k_frac == 0.1
    assert specs["flyhash8_bloom"].matrix == "flyhash8" and specs["real_fly_bloom_weighted"].matrix == "weighted"
    assert base_of["flyhash_bloom_normalized"] == "flyhash_bloom" and len(table) == len(base_of)
    opt = e6.e6_detectors(toy_cfg, ("k", "k_flyhash"), [])[0]
    assert "flyhash_linear_k2.5" in opt and e6.resolve_parts(toy_cfg, ["gamma", "gamma"]) == ("gamma",)
    with pytest.raises(KeyError):
        e6.resolve_parts(toy_cfg, ["nope"])
    smoke = e6.resolve_parts(toy_cfg, smoke=True)                   # A54: E6.yaml flags ∩ smoke.e6_parts
    assert set(smoke) == set(toy_cfg.default["smoke"]["e6_parts"]) and not {"bipia_all", "tok512", "flyhash40"} & set(smoke)
    st = {r["part"]: r["status"] for r in e6.part_status(toy_cfg, smoke, True, False, {"para_shallow": "no data"})}
    assert st["tok512"] == e6.STATUS_SMOKE and st["k"] == e6.STATUS_RUN and st["para_shallow"] == e6.STATUS_NO_DATA


def test_e6_result_covers_every_part(e6_result, e6_ctx):
    nums, th, tables = e6_result["numbers"], e6_result["thresholds"], e6_result["tables"]
    for key in ("auc/deep/real_fly_bloom_k2.5", "diff/macro_auc/real_fly_bloom_k10-real_fly_bloom",
                "hyper/real_fly_bloom_gamma0.99/gamma", "auc/bipia/real_fly_linear_weighted",
                "macro_auc/flyhash_bloom_normalized", "macro_auc/flyhash8_linear", "latency_ms/flyhash8_bloom",
                "fpr_notinject/tau80_deep/real_fly_bloom", "fpr_notinject/tau80_deep/real_fly_bloom/one",
                "diff/fpr_notinject/tau80_deep/real_fly_bloom-protectai_v2",
                "auc/para_shallow/tfidf_lr", "macro_auc_strata/real_fly_bloom", "diff/macro_auc_strata/real_fly_linear-lr_svd",
                "auc/bipia_all/real_fly_bloom", "diff/auc/bipia_all/protectai_v2-tfidf_lr",
                "auc/deep/protectai_v2_tok512", "macro_auc/protectai_v2_tok512", "diff/macro_auc/protectai_v2_tok512-protectai_v2",
                "tpr_at_fpr/deep/protectai_v2_tok512", "fpr_notinject/tau90_deep/protectai_v2_tok512"):
        assert key in nums, key
    for key, rec in nums.items():
        if rec["ci_low"] is not None and rec["value"] is not None:
            assert rec["ci_low"] - 1e-12 <= rec["value"] <= rec["ci_high"] + 1e-12, key
    assert nums["hyper/real_fly_bloom_gamma0.99/gamma"] == {"value": 0.99, "ci_low": None, "ci_high": None, "n": None, "note": "fixed"}
    assert th["tau80_deep/real_fly_bloom"]["target"] == "tpr>=0.8" and "tau80_dojo/tfidf_lr" in th
    assert th["tau80_deep/real_fly_bloom"]["value"] >= th["tau90_deep/real_fly_bloom"]["value"]   # fewer positives to catch
    assert "para_shallow" in nums["macro_auc_strata/real_fly_bloom"]["note"] and "tau_fpr/protectai_v2_tok512" in th
    rows = tables["bipia_attacks"]
    dets = {r["detector"] for r in rows}
    assert {"real_fly_bloom", "protectai_v2", "flyhash8_bloom"} <= dets
    mine = [r for r in rows if r["detector"] == "real_fly_bloom"]
    assert {r["position"] for r in mine if r["attack"] is None} == {"start", "middle", "end"}
    assert {r["attack"] for r in mine if r["position"] is None} == {"attack one", "attack two"}
    assert all(r["ci_low"] is not None for r in mine if r["attack"] is None or r["position"] is None)
    assert sum(1 for r in mine if r["attack"] and r["position"]) == 6
    src = {r["source"]: r for r in tables["sources"]}
    assert src["bipia_all"]["n_e6_variant_docs"] == 4 * 5 and src["bipia_all"]["n_pos"] == 4 * 6 and src["bipia_all"]["n_neg"] == 4
    assert {r["part"] for r in tables["variants"]} == {"k", "gamma", "weighted", "normalized", "flyhash40"}
    assert "bipia#e6" in e6_result["timing"]["test_reads"] and any("bipia#e6" in c[2] for c in e6_ctx.recorder.calls)
    assert e6_result["config_hash"] == config_hash(e6_ctx.root) and any("tok512" in n for n in e6_result["notes"])
    status = {r["part"]: r["status"] for r in tables["parts"]}
    assert status == {**{p: e6.STATUS_RUN for p in e6.PART_FLAGS}, "k_flyhash": e6.STATUS_NOT_REQUESTED}
    assert e6_result["e6_parts"]["status"] == status and "git_dirty" in e6_result


def test_e6_parts_subset_skip_and_summary(e6_ctx, e6_root, gf, e6_result):
    path = e6.run(e6_ctx, 1, root=e6_root, guard_factory=gf, parts=("k",))
    r = read_result(path)
    keys = set(r["numbers"])
    assert "auc/deep/real_fly_bloom_k2.5" in keys and "auc/deep/real_fly_linear_k10" in keys
    assert not any("_gamma" in k or "tok512" in k or "bipia_all" in k or "weighted" in k for k in keys)
    assert not any(k.startswith("tau80") for k in r["thresholds"]) and "auc/para_shallow/tfidf_lr" not in keys
    assert {x["part"]: x["status"] for x in r["tables"]["parts"]}["gamma"] == e6.STATUS_NOT_REQUESTED
    mtime = path.stat().st_mtime_ns
    assert e6.run(e6_ctx, 1, root=e6_root, guard_factory=gf, parts=("k",)) == path and path.stat().st_mtime_ns == mtime
    summ = summarize("E6", root=e6_root)
    assert summ["n_seeds"] == 2 and summ["numbers"]["macro_auc/real_fly_bloom"]["n_seeds"] == 2
    assert e6.parse_parts("k, gamma") == ("k", "gamma") and e6.parse_parts("all") is None
    assert not (e6_root / "data" / "processed" / "features" / "seed1").exists()


# ---------------------------------------------------------------------------------------------- contract
@pytest.fixture(scope="module")
def contract(e6_cfg, e6_root, recorder_class, gf):
    rec = recorder_class()
    ctx = Context(e6_cfg, root=e6_root, access_log=rec)
    path = contract_run.run(ctx, 0, root=e6_root, guard_factory=gf)
    return {"ctx": ctx, "rec": rec, "path": path, "res": read_json(path), "episodes": ctx.episodes}


def test_contract_csv_rows_thresholds_and_metrics(contract, e6_root):
    res, ep = contract["res"], contract["episodes"]
    csv_path = e6_root / "results" / "shared" / "flyguard.csv"
    assert res["csv"] == "results/shared/flyguard.csv" and validate_csv(csv_path) == []
    rows = pd.read_csv(csv_path, keep_default_na=False)
    test_eps = ep[ep["contract_split"] == "test"]
    assert len(rows) == 4 * len(test_eps) == res["n_rows"] and set(rows["variant"]) == set(contract_run.VARIANTS)
    n_val_benign = int(((ep["contract_split"] == "val") & (ep["episode_class"] == "benign")).sum())
    assert n_val_benign == 3 and set(rows["threshold_n_benign"]) == {n_val_benign}
    for det in contract_run.VARIANTS.values():
        rec = res["thresholds"][f"contract/{det}"]
        assert rec["n"] == n_val_benign and rec["target"] == "fa_per_100<=5" and "fewer than 200" in rec["note"]
        assert rec["n_by_benchmark"] == {"agentdojo": 1, "agentdyn": 2}
    steps = pd.read_csv(e6_root / "results" / "contract_steps.csv")
    for r in rows.itertuples(index=False):
        s = steps[steps["episode_id"] == r.episode_id].sort_values("step")
        if len(s) == 0:
            assert r.max_score == 0.0 and r.alarm_step == ""
            continue
        assert r.max_score == pytest.approx(float(s[r.variant].max()))
        hits = s[s[r.variant] >= r.threshold]["step"]
        assert (r.alarm_step == "" and len(hits) == 0) or (r.alarm_step != "" and int(r.alarm_step) == int(hits.min()))
    nums = res["numbers"]
    for key in ("contract/stopped_before_harm/real_fly_bloom", "contract/false_alarms_per_100_benign/tfidf_lr",
                "contract/agentdyn/false_alarms_per_100_benign/flyhash_linear", "contract/n_hijacked/real_fly_linear",
                "contract/threshold/real_fly_bloom", "hyper/real_fly_bloom/gamma", "contract/n_test_episodes"):
        assert key in nums, key
    assert nums["contract/n_test_episodes"]["value"] == len(test_eps) and nums["contract/n_hijacked/real_fly_bloom"]["value"] == 3
    assert nums["contract/n_unmatched/real_fly_bloom"]["value"] == 0 and nums["contract/n_benign/tfidf_lr"]["value"] == 5
    assert set(res["metrics"]) == set(contract_run.VARIANTS) and set(res["metrics_by_benchmark"]) == {"agentdojo", "agentdyn"}
    assert "git_dirty" in res and "pools" in res["timing"]["threads"]              # A41/A52 provenance
    assert res["config_hash"] == config_hash(e6_root) and res["smoke"] is False and res["seed"] == 0


def test_contract_training_follows_d11_and_doors_are_ordered(contract, e6_root):
    res, rec = contract["res"], contract["rec"]
    comp = {r["set"]: r for r in res["tables"]["training"]}
    assert comp["train"]["n_windows"] == comp["train"]["n_deep"] and comp["val"]["n_windows"] == comp["val"]["n_deep"]
    assert comp["c_unl"]["n_dojo"] > 0 and comp["dojo_clean_train_episodes"]["n_pos"] == 0 and comp["dojo_clean_val_episodes"]["n_windows"] > 0
    assert res["training_mode"] == "D11" and any("D11" in n for n in res["notes"])
    purposes = [c[2] for c in rec.calls]
    first = lambda key: next(i for i, p in enumerate(purposes) if key in p)
    assert first("dojo#contract_trainval") < first("dyn#contract_val") < first("dojo#contract_test")
    assert first("dyn#contract_val") < first("dyn#contract_test")
    assert set(res["timing"]["test_reads"]) == {"dojo#contract_trainval", "dyn#contract_val", "dojo#contract_test", "dyn#contract_test"}
    without = res["episodes_without_documents"]
    assert len(without["nothing_to_scan"]) == 1 and len(without["not_extracted"]) == 1 and without["no_step_document"] == []
    bench = res["numbers"]
    assert bench["contract/agentdyn/false_alarms_per_100_benign/real_fly_bloom"]["ci_low"] is not None   # headline rate: interval
    assert bench["contract/agentdyn/n_benign/real_fly_bloom"]["value"] == 1 and bench["contract/agentdojo/n_benign/tfidf_lr"]["value"] == 4
    assert bench["contract/agentdyn/alarms_on_injection_ignored_share/tfidf_lr"]["ci_low"] is None   # point only per benchmark
    assert any("per-benchmark" in n for n in res["notes"])
    assert res["split_manifest"]["ok"] and res["split_manifest"]["n_test"] == res["numbers"]["contract/n_test_episodes"]["value"]
    ep_rows = {(r["benchmark"], r["contract_split"], r["episode_class"]): r for r in res["tables"]["episodes"]}
    assert ep_rows[("agentdojo", "excluded", "injection_ignored")]["n"] == 2 and ep_rows[("agentdyn", "val", "benign")]["n"] == 2


def test_contract_skip_manifest_mismatch_and_dojo_negatives(contract, e6_cfg, e6_root, gf):
    path = contract["path"]
    mtime = path.stat().st_mtime_ns
    assert contract_run.is_current(e6_root) and contract_run.run(None, 0, root=e6_root, cfg=e6_cfg, guard_factory=gf) == path
    ns = argparse.Namespace(seeds=[0, 1], smoke=False, root=e6_root, force=False, keep_cache=False, dojo_negatives=False)
    assert contract_run.run_cli(ns) == path and path.stat().st_mtime_ns == mtime     # current -> skipped, first seed only
    assert "validation_episodes" not in contract["res"]["tables"] and len(contract["res"]["validation_episodes"]) == 3 * 4
    mpath = e6_root / "results" / "shared" / "split_manifest.json"
    original = mpath.read_text(encoding="utf-8")
    try:
        bad = json.loads(original)
        bad["test"] = bad["test"][1:]
        atomic_write_json(mpath, bad)
        with pytest.raises(ValueError, match="split_manifest"):
            contract_run.run(Context(e6_cfg, root=e6_root, access_log=lambda *a: None), 0, root=e6_root, guard_factory=gf, force=True)
    finally:
        mpath.write_text(original, encoding="utf-8")
    ctx = Context(e6_cfg, root=e6_root, access_log=lambda *a: None)
    res = read_json(contract_run.run(ctx, 0, root=e6_root, guard_factory=gf, force=True, dojo_negatives=True))
    comp = {r["set"]: r for r in res["tables"]["training"]}
    d11 = {r["set"]: r for r in contract["res"]["tables"]["training"]}
    assert comp["train"]["n_dojo"] > 0 and comp["train"]["n_pos"] == d11["train"]["n_pos"]   # dojo windows add negatives only
    assert comp["train"]["n_neg"] == d11["train"]["n_neg"] + comp["train"]["n_dojo"] and comp["val"]["n_dojo"] > 0
    assert res["training_mode"] == "D11 + dojo_negatives" and validate_csv(e6_root / "results" / "shared" / "flyguard.csv") == []
