"""ТЗ 1.9 / 1.10: E1 split rules, C_unl, disjointness, AgentDojo val rule, E3 folds."""
import zlib

import pandas as pd

from flyguard.data import splits


def _doc(doc_id, source, label, cluster=None, meta=None, split=None):
    return {"doc_id": doc_id, "source": source, "label": label, "text": doc_id, "text_orig": doc_id, "spans": [],
            "cluster_id": cluster or doc_id, "meta": meta or {}, "split": split}


def _dojo_docs():
    rows, eps = [], []
    suites, templates = ["workspace", "travel", "banking", "slack"], ["important_instructions", "tool_knowledge",
                                                                       "injecagent", "ignore_previous"]
    for s in suites:
        for t in range(6):
            task = f"user_task_{t}"
            for atk in [None] + templates:
                ep = f"{s}/{task}/{'none' if atk is None else 'injection_task_1'}/{atk or 'none'}/M"
                cls = "benign" if atk is None else ("hijacked" if t % 2 else "injection_ignored")
                if s == "slack" and t == 5 and atk == "injecagent":
                    cls = "error"
                eps.append({"episode_id": ep, "suite": s, "user_task": task, "attack": atk, "episode_class": cls})
                for step in range(2):
                    label = 1 if (atk is not None and step == 1) else 0
                    rows.append(_doc(f"dojo:{ep}#{step}", "dojo", label, f"{s}/{task}",
                                     {"episode_id": ep, "suite": s, "user_task": task, "attack": atk,
                                      "episode_class": cls, "step": step}))
    return pd.DataFrame(rows), pd.DataFrame(eps)


def test_crc32_rule_matches_zlib(cfg):
    r = cfg.default["splits"]["agentdojo_val_rule"]
    for task in ["user_task_0", "user_task_18", "user_task_3", "abc"]:
        assert splits.is_agentdojo_val_task(task, cfg) == (zlib.crc32(task.encode()) % r["mod"] == r["rem"])
    assert splits.crc32_rule("user_task_2", 3, 2) == (zlib.crc32(b"user_task_2") % 3 == 2)


def test_stratified_val_ids():
    ids = [f"d{i}" for i in range(100)]
    labels = [1 if i < 30 else 0 for i in range(100)]
    val = splits.stratified_val_ids(ids, labels, 0.2, 42)
    assert len(val) == 20 and sum(1 for i in val if int(i[1:]) < 30) == 6
    assert val == splits.stratified_val_ids(ids, labels, 0.2, 42)
    assert val != splits.stratified_val_ids(ids, labels, 0.2, 43)
    assert val == splits.stratified_val_ids(list(reversed(ids)), list(reversed(labels)), 0.2, 42)   # order-free


def test_bipia_context_split_per_task():
    clusters = [f"bipia:email:{i}" for i in range(10)] + [f"bipia:table:{i}" for i in range(20)]
    m = splits.bipia_context_split(clusters, 0.2, 5)
    assert sum(v == "val" for k, v in m.items() if ":email:" in k) == 2
    assert sum(v == "val" for k, v in m.items() if ":table:" in k) == 4
    assert m == splits.bipia_context_split(clusters, 0.2, 5)


def _corpus():
    docs = []
    for i in range(50):
        docs.append(_doc(f"deep:train:{i}", "deep", 1 if i % 2 else 0, meta={"deepset_split": "train"}))
    for i in range(10):
        docs.append(_doc(f"deep:test:{i}", "deep", 1 if i % 2 else 0, meta={"deepset_split": "test"}))
    for i in range(10):
        c = f"bipia:email:{i}"
        docs.append(_doc(f"{c}:clean", "bipia", 0, c, {"variant": "main", "task": "email"}))
        docs.append(_doc(f"{c}:a-0:end", "bipia", 1, c, {"variant": "main", "task": "email"}))
        docs.append(_doc(f"{c}:a-0:start", "bipia", 1, c, {"variant": "e6", "task": "email"}))
    dojo, eps = _dojo_docs()
    docs += dojo.to_dict("records")
    docs.append(_doc("dyn:x#0", "dyn", 0, "shopping/user_task_1", {"episode_id": "x", "user_task": "user_task_1"}))
    docs.append(_doc("para:b1:0", "para", 1, "b1", {"stratum": "deep"}))
    docs.append(_doc("notinject:one:0", "notinject", 0))
    return pd.DataFrame(docs), eps


def test_assign_e1_split_rules(cfg):
    docs, _ = _corpus()
    docs["split"] = splits.assign_e1_split(docs, cfg, 123).to_numpy()
    deep_train = docs[(docs.source == "deep") & docs.doc_id.str.startswith("deep:train")]
    assert set(deep_train.split) == {"train", "val"}
    assert (deep_train.split == "val").sum() == 10                                  # 20 % stratified: 5 + 5
    assert (deep_train[deep_train.label == 1].split == "val").sum() == 5
    assert set(docs[docs.doc_id.str.startswith("deep:test")].split) == {"test"}
    bip = docs[docs.source == "bipia"]
    assert bip.groupby("cluster_id")["split"].nunique().max() == 1                   # by cluster
    assert (bip.drop_duplicates("cluster_id").split == "val").sum() == 2
    dojo = docs[docs.source == "dojo"]
    for r in dojo.itertuples(index=False):
        assert r.split == ("val" if splits.is_agentdojo_val_task(r.meta["user_task"], cfg) else "test")
    assert set(docs[docs.source.isin(["dyn", "para", "notinject"])].split) == {"test"}
    # a val/test role already assigned by agentdojo_io is kept; a "train" role is ignored (labels: deepset only)
    docs2 = docs.copy()
    docs2.loc[docs2.source == "dojo", "split"] = "val"
    assert set(splits.assign_e1_split(docs2, cfg, 123)[docs2.source == "dojo"]) == {"val"}
    docs2.loc[docs2.source == "dojo", "split"] = "train"
    got = splits.assign_e1_split(docs2, cfg, 123)[docs2.source == "dojo"]
    assert set(got) <= {"val", "test"} and (got.to_numpy() == docs.loc[docs.source == "dojo", "split"].to_numpy()).all()
    docs3 = docs.copy()
    docs3.loc[docs3.source == "dojo", "split"] = "train"
    import pytest
    with pytest.raises(ValueError):
        splits.build_splits(docs3, cfg, set(), None)


def test_build_splits_invariants(cfg):
    docs, eps = _corpus()
    docs["split"] = splits.assign_e1_split(docs, cfg, 123).to_numpy()
    dropped = {"deep:test:3"}
    s = splits.build_splits(docs, cfg, dropped, eps)
    train, val = set(s["e1"]["train"]), set(s["e1"]["val"])
    test = {d for ids in s["e1"]["test"].values() for d in ids}
    assert not (train & val) and not (train & test) and not (val & test)
    cluster = dict(zip(docs.doc_id, docs.cluster_id))
    assert not ({cluster[d] for d in train | val} & {cluster[d] for d in test})
    assert set(s["c_unl"]) == train | val
    assert not ({cluster[d] for d in s["c_unl"]} & {cluster[d] for d in test})
    assert "deep:test:3" not in test and "deep:test:3" in s["dropped_by_dedup"]
    assert set(s["e1"]["test"]) == {"deep", "bipia", "dojo", "dyn", "para", "notinject"}
    assert all(d.startswith("deep:train") for d in train)
    e6 = set(s["bipia"]["e6_docs"]["val"]) | set(s["bipia"]["e6_docs"]["test"])
    assert e6 and not (e6 & (train | val | test))
    assert not (set(s["bipia"]["val_contexts"]) & set(s["bipia"]["test_contexts"]))
    assert s["bipia"]["e6_role"]["in_c_unl"] is False and s["bipia"]["e6_role"]["dedup_reference_when_val"] is True
    err = docs[docs.meta.map(lambda m: m.get("episode_class") == "error")]["doc_id"]
    assert not (set(err) & test)
    assert s["rules"] == cfg.default["splits"]
    assert set(s["template_names"]) == set(cfg.default["traces"]["agentdojo"]["attacks"])
    for d in s["e1"]["test"]["dojo"]:
        assert not splits.is_agentdojo_val_task(docs.set_index("doc_id").loc[d, "meta"]["user_task"], cfg)


def test_e3_folds(cfg):
    dojo, eps = _dojo_docs()
    dojo["split"] = "test"
    f = splits.build_e3_folds(dojo, eps, cfg)
    assert len(f["cross_template"]) == 4 and len(f["cross_suite"]) == 4 and len(f["double_holdout"]) == 16
    meta = {r.doc_id: r.meta for r in dojo.itertuples(index=False)}
    assert f["templates_present"] == cfg.default["traces"]["agentdojo"]["attacks"] and f["templates_missing"] == []
    assert f["usable"] == {"cross_template": 4, "cross_suite": 4, "double_holdout": 16}
    for kind in splits.E3_FOLD_KINDS:
        for fold in f[kind]:
            assert fold["usable"] is True
            tr, te = set(fold["train"]), set(fold["test"])
            assert tr and te and not (tr & te)
            assert not any(meta[d]["episode_class"] == "error" for d in tr | te)
    for fold in f["cross_template"]:
        t = fold["fold"]
        assert {meta[d]["attack"] for d in fold["test"] if meta[d]["attack"]} == {t}
        assert t not in {meta[d]["attack"] for d in fold["train"]}
        assert fold["n_test_pos"] > 0 and fold["n_test_neg"] > 0
    for fold in f["cross_suite"]:
        assert {meta[d]["suite"] for d in fold["test"]} == {fold["fold"]}
        assert fold["fold"] not in {meta[d]["suite"] for d in fold["train"]}
    for fold in f["double_holdout"]:
        s, t = fold["fold"].split("x")
        assert {meta[d]["suite"] for d in fold["test"]} == {s}
        assert {meta[d]["attack"] for d in fold["test"] if meta[d]["attack"]} == {t}
        assert s not in {meta[d]["suite"] for d in fold["train"]} and t not in {meta[d]["attack"] for d in fold["train"]}
    empty = splits.build_e3_folds(dojo, None, cfg)
    assert all(empty[k] == [] for k in splits.E3_FOLD_KINDS) and empty["templates_missing"] == f["templates_present"]


def test_e3_folds_flag_missing_templates(cfg):
    """DEVIATIONS D6: with only important_instructions traces, the folds exist but most are not usable."""
    dojo, eps = _dojo_docs()
    keep = dojo.meta.map(lambda m: m["attack"] in (None, "important_instructions"))
    dojo = dojo[keep].reset_index(drop=True)
    dojo["split"] = "test"
    eps = eps[eps.attack.isna() | (eps.attack == "important_instructions")]
    f = splits.build_e3_folds(dojo, eps, cfg)
    assert f["templates_present"] == ["important_instructions"]
    assert f["templates_missing"] == ["tool_knowledge", "injecagent", "ignore_previous"]
    assert len(f["cross_template"]) == 4 and len(f["double_holdout"]) == 16       # the ТЗ shape is kept
    by_name = {x["fold"]: x for x in f["cross_template"]}
    assert by_name["important_instructions"]["n_train_pos"] == 0 and not by_name["important_instructions"]["usable"]
    assert all(by_name[t]["n_test_pos"] == 0 and not by_name[t]["usable"] for t in f["templates_missing"])
    assert f["usable"]["cross_template"] == 0 and f["usable"]["cross_suite"] == 4 and f["usable"]["double_holdout"] == 0
    assert f["usable_rule"] == splits.E3_USABLE_RULE
