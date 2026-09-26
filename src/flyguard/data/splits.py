"""Splits (ТЗ 1.9, 1.10; docs/design.md §2): the E1 split column, C_unl, BIPIA contexts, E3 folds, splits.json.

The E1 split is data, not an experiment: it is drawn once from seed ``subsample`` of global seed 0 and written to
``documents.parquet``. Everything that must never see test clusters (C_unl, dedup references, P_val) is derived
from that column.
"""
from __future__ import annotations

import json
import zlib
from typing import Any, Iterable

import numpy as np
import pandas as pd

SOURCE_ORDER = ("deep", "bipia", "dojo", "dyn", "para", "notinject")
BENIGN_ATTACKS = (None, "", "none", "None")


def crc32_rule(key: str, mod: int, rem: int) -> bool:
    """``crc32(key) % mod == rem`` on the id as given (``user_task_18``), the rule shared with the second team."""
    return zlib.crc32(str(key).encode("utf-8")) % int(mod) == int(rem)


def is_agentdojo_val_task(user_task: str, cfg: Any) -> bool:
    """ТЗ 1.8/1.10: AgentDojo tasks with crc32(user_task_id) mod 5 == 0 are validation (clean outputs feed P_val)."""
    r = cfg.default["splits"]["agentdojo_val_rule"]
    return crc32_rule(user_task, r["mod"], r["rem"])


def stratified_val_ids(ids: Iterable[str], labels: Iterable[int], val_fraction: float, seed: int) -> set[str]:
    """ТЗ 1.10: stratified ``val_fraction`` of deepset train, by label, permuted by a Generator seeded with
    ``subsample`` of global seed 0. Ids are sorted before permutation so the draw depends on content, not on
    load order."""
    rng = np.random.default_rng(int(seed))
    df = pd.DataFrame({"id": list(ids), "label": list(labels)})
    val: set[str] = set()
    for label in sorted(df["label"].unique()):
        group = sorted(df[df["label"] == label]["id"].tolist())
        n_val = int(round(val_fraction * len(group)))
        perm = rng.permutation(len(group))
        val.update(group[i] for i in perm[:n_val])
    return val


def bipia_context_split(cluster_ids: Iterable[str], val_fraction: float, seed: int) -> dict[str, str]:
    """ТЗ 1.4: contexts 20/80 val/test by cluster, per task (each task keeps its share of validation contexts)."""
    rng = np.random.default_rng(int(seed))
    clusters = sorted(set(cluster_ids))
    by_task: dict[str, list[str]] = {}
    for c in clusters:
        by_task.setdefault(c.split(":")[1] if c.count(":") >= 2 else "", []).append(c)
    out: dict[str, str] = {}
    for task in sorted(by_task):
        group = by_task[task]
        n_val = int(round(val_fraction * len(group)))
        perm = rng.permutation(len(group))
        chosen = {group[i] for i in perm[:n_val]}
        for c in group:
            out[c] = "val" if c in chosen else "test"
    return out


def _meta(documents: pd.DataFrame) -> list[dict[str, Any]]:
    if "meta" in documents.columns:
        return [m if isinstance(m, dict) else (json.loads(m) if m else {}) for m in documents["meta"]]
    return [json.loads(m) if m else {} for m in documents["meta_json"]]


def assign_e1_split(documents: pd.DataFrame, cfg: Any, seed_subsample: int) -> pd.Series:
    """E1 role of every document (ТЗ 1.10): deepset train -> train/val (stratified 20 %), deepset test -> test;
    BIPIA contexts -> val/test by cluster; dojo -> val when crc32(user_task) % 5 == 0 (or the val/test role
    agentdojo_io already assigned; a ``train`` role is ignored because E1 trains on deepset labels only, the
    contract split of design §3 is a different column), else test; dyn, paraphrases and NotInject -> test only."""
    sp = cfg.default["splits"]
    metas = _meta(documents)
    split = pd.Series([None] * len(documents), index=documents.index, dtype=object)
    existing = documents["split"] if "split" in documents.columns else pd.Series([None] * len(documents), index=documents.index)
    src = documents["source"].to_numpy()
    ids = documents["doc_id"].to_numpy()
    deep_train = [(i, ids[i]) for i in range(len(documents)) if src[i] == "deep" and metas[i].get("deepset_split") == "train"]
    val_ids = stratified_val_ids([d for _, d in deep_train], [int(documents["label"].iloc[i]) for i, _ in deep_train],
                                 float(sp["val_fraction"]), seed_subsample)
    bip_map = bipia_context_split(documents.loc[documents["source"] == "bipia", "cluster_id"],
                                  float(sp["bipia"]["val_fraction"]), seed_subsample)
    for i in range(len(documents)):
        s = src[i]
        if s == "deep":
            if metas[i].get("deepset_split") == "train":
                split.iloc[i] = "val" if ids[i] in val_ids else "train"
            else:
                split.iloc[i] = "test"
        elif s == "bipia":
            split.iloc[i] = bip_map[documents["cluster_id"].iloc[i]]
        elif s == "dojo":
            given = existing.iloc[i]
            if isinstance(given, str) and given in ("val", "test"):
                split.iloc[i] = given          # agentdojo_io's E1 role; "train" is never accepted (labels: deepset only)
            else:
                split.iloc[i] = "val" if is_agentdojo_val_task(str(metas[i].get("user_task", "")), cfg) else "test"
        else:
            split.iloc[i] = "test"
    return split


def _e3_frame(documents: pd.DataFrame, episodes: pd.DataFrame | None) -> pd.DataFrame:
    """dojo documents with episode fields (from meta, completed from episodes.parquet when present)."""
    dojo = documents[documents["source"] == "dojo"].copy()
    metas = _meta(dojo)
    for col in ("episode_id", "suite", "user_task", "attack", "episode_class"):
        dojo[col] = [m.get(col) for m in metas]
    if episodes is not None and len(episodes) and "episode_id" in episodes.columns:
        ep = episodes.drop_duplicates("episode_id").set_index("episode_id")
        for col in ("suite", "user_task", "attack", "episode_class"):
            if col in ep.columns:
                filled = dojo["episode_id"].map(ep[col])
                dojo[col] = dojo[col].where(dojo[col].notna(), filled)
    dojo["benign"] = dojo["attack"].isin(BENIGN_ATTACKS) | dojo["attack"].isna()
    return dojo[dojo["episode_class"].fillna("") != "error"]


def build_e3_folds(documents: pd.DataFrame, episodes: pd.DataFrame | None, cfg: Any) -> dict[str, list[dict]]:
    """ТЗ 1.10 E3 folds over AgentDojo documents (ТЗ: 4 cross-template, 4 cross-suite, 16 double-holdout).

    cross_template: test = episodes of the held-out template plus benign episodes of E1-validation tasks (the
    fold's negatives for the threshold); train = the other templates plus benign episodes of the remaining tasks.
    cross_suite: test = every document of the held-out suite; train = the other suites. double_holdout: test =
    held-out template on the held-out suite plus that suite's benign episodes; train = other suites x other
    templates plus their benign episodes. Environments are static (contract §10), so the benign side is a stated
    choice, not a leakage-free guarantee. ``error`` episodes are excluded.
    """
    out: dict[str, list[dict]] = {"cross_template": [], "cross_suite": [], "double_holdout": []}
    if episodes is None or documents.empty or not (documents["source"] == "dojo").any():
        return out
    dojo = _e3_frame(documents, episodes)
    if dojo.empty:
        return out
    templates = list(cfg.default["traces"]["agentdojo"]["attacks"])
    suites = list(cfg.default["traces"]["agentdojo"]["suites"])
    val_task = dojo["user_task"].map(lambda t: is_agentdojo_val_task(str(t), cfg)).astype(bool)

    def fold(name: str, train_mask: pd.Series, test_mask: pd.Series) -> dict:
        tr, te = dojo[train_mask], dojo[test_mask]
        return {"fold": name, "train": sorted(tr["doc_id"]), "test": sorted(te["doc_id"]),
                "n_train_pos": int((tr["label"] == 1).sum()), "n_train_neg": int((tr["label"] == 0).sum()),
                "n_test_pos": int((te["label"] == 1).sum()), "n_test_neg": int((te["label"] == 0).sum())}

    for t in templates:
        is_t = dojo["attack"] == t
        out["cross_template"].append(fold(t, (~dojo["benign"] & ~is_t) | (dojo["benign"] & ~val_task),
                                          is_t | (dojo["benign"] & val_task)))
    for s in suites:
        is_s = dojo["suite"] == s
        out["cross_suite"].append(fold(s, ~is_s, is_s))
    for s in suites:
        is_s = dojo["suite"] == s
        for t in templates:
            is_t = dojo["attack"] == t
            out["double_holdout"].append(fold(f"{s}x{t}", ~is_s & ((~dojo["benign"] & ~is_t) | dojo["benign"]),
                                              is_s & (is_t | dojo["benign"])))
    return out


def build_splits(documents: pd.DataFrame, cfg: Any, dropped: set[str] | None = None,
                 episodes: pd.DataFrame | None = None) -> dict[str, Any]:
    """``splits.json`` of design §2. Dropped documents (dedup, ТЗ 1.7) leave every list; BIPIA E6 variants are listed
    under ``bipia.e6_docs`` and not in the E1 lists or C_unl (ТЗ 1.9: C_unl is the train+val *texts*, and 45
    near-copies of each validation context would swamp it); ``error`` episodes leave the dojo/dyn test lists."""
    dropped = dropped or set()
    metas = _meta(documents)
    df = documents[["doc_id", "source", "split", "label", "cluster_id"]].copy()
    df["variant"] = [m.get("variant", "main") for m in metas]
    df["episode_class"] = [m.get("episode_class") for m in metas]
    usable = df[~df["doc_id"].isin(dropped)]
    main = usable[usable["variant"] != "e6"]
    e1_train = sorted(main[main["split"] == "train"]["doc_id"])
    if not (main[main["split"] == "train"]["source"] == "deep").all():
        raise ValueError("E1 train may only contain deepset documents (ТЗ 1.10)")
    e1_val = sorted(main[main["split"] == "val"]["doc_id"])
    test_main = main[(main["split"] == "test") & (main["episode_class"].fillna("") != "error")]
    e1_test = {s: sorted(test_main[test_main["source"] == s]["doc_id"]) for s in SOURCE_ORDER
               if (df["source"] == s).any()}
    c_unl = sorted(set(e1_train) | set(e1_val))
    bip = df[df["source"] == "bipia"]
    e6 = usable[(usable["source"] == "bipia") & (usable["variant"] == "e6")]
    templates = list(cfg.default["traces"]["agentdojo"]["attacks"])
    return {
        "e1": {"train": e1_train, "val": e1_val, "test": e1_test},
        "c_unl": c_unl,
        "e3": build_e3_folds(documents[~documents["doc_id"].isin(dropped)], episodes, cfg),
        "bipia": {"val_contexts": sorted(set(bip[bip["split"] == "val"]["cluster_id"])),
                  "test_contexts": sorted(set(bip[bip["split"] == "test"]["cluster_id"])),
                  "e6_docs": {"val": sorted(e6[e6["split"] == "val"]["doc_id"]),
                              "test": sorted(e6[e6["split"] == "test"]["doc_id"])}},
        "rules": cfg.default["splits"],
        "template_names": {t: t for t in templates},
        "dropped_by_dedup": sorted(dropped),
        "counts": {"train": len(e1_train), "val": len(e1_val), "test": {k: len(v) for k, v in e1_test.items()},
                   "c_unl": len(c_unl), "val_by_source": {s: int((main[(main["split"] == "val") & (main["source"] == s)]).shape[0])
                                                          for s in SOURCE_ORDER if (df["source"] == s).any()}},
    }
