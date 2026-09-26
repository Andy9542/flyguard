"""Negative pools P_val / P_test (ТЗ 1.8; docs/design.md §2) -> ``pools.json``.

P_val feeds the frozen FPR threshold (ТЗ 2.5), so it may only contain documents that are not test material:
benign deepset *validation* documents (the train part trains the detectors and would make the threshold
optimistic), BIPIA validation clean documents, clean AgentDojo outputs of validation tasks. P_test is the FPR
carrier of the final run: benign deepset test, BIPIA test clean, clean AgentDojo outputs of the remaining tasks,
clean AgentDyn outputs and the negative paraphrases. NotInject is a separate FPR set, not part of P_test.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

BENIGN_ATTACKS = (None, "", "none", "None")
P_VAL_DEFINITION = ("benign deepset val + BIPIA val clean + clean AgentDojo outputs (benign episodes) of validation "
                    "tasks (crc32(user_task) % 5 == 0)")
P_TEST_DEFINITION = ("benign deepset test + BIPIA test clean + clean AgentDojo outputs of the remaining tasks + clean "
                     "AgentDyn outputs + negative paraphrases")


def _meta(documents: pd.DataFrame) -> list[dict[str, Any]]:
    if "meta" in documents.columns:
        return [m if isinstance(m, dict) else (json.loads(m) if m else {}) for m in documents["meta"]]
    return [json.loads(m) if m else {} for m in documents["meta_json"]]


def pool_frame(documents: pd.DataFrame, dropped: set[str] | None = None) -> pd.DataFrame:
    """Documents with the fields the pool rules need (variant, clean-trace flag), minus dedup-dropped ones."""
    dropped = dropped or set()
    metas = _meta(documents)
    df = documents[["doc_id", "source", "split", "label", "cluster_id"]].copy()
    df["variant"] = [m.get("variant", "main") for m in metas]
    cls = [m.get("episode_class") for m in metas]
    atk = [m.get("attack") for m in metas]
    df["clean_trace"] = [(s in ("dojo", "dyn")) and int(l) == 0 and ((c == "benign") if c is not None else (a in BENIGN_ATTACKS))
                         for s, l, c, a in zip(df["source"], df["label"], cls, atk)]
    df = df[~df["doc_id"].isin(dropped)]
    return df[df["variant"] != "e6"]


def build_pools(documents: pd.DataFrame, cfg: Any, dropped: set[str] | None = None) -> dict[str, Any]:
    """``pools.json`` (ТЗ 1.8): composition and sizes by source, target >= ``pools.target_min_docs`` documents."""
    df = pool_frame(documents, dropped)
    neg = df["label"] == 0
    p_val = df[neg & (((df["source"] == "deep") & (df["split"] == "val"))
                      | ((df["source"] == "bipia") & (df["split"] == "val"))
                      | ((df["source"] == "dojo") & (df["split"] == "val") & df["clean_trace"]))]
    p_test = df[neg & (((df["source"] == "deep") & (df["split"] == "test"))
                       | ((df["source"] == "bipia") & (df["split"] == "test"))
                       | ((df["source"] == "dojo") & (df["split"] == "test") & df["clean_trace"])
                       | ((df["source"] == "dyn") & df["clean_trace"])
                       | ((df["source"] == "para") & (df["split"] == "test")))]
    target = int(cfg.default["pools"]["target_min_docs"])

    def pack(frame: pd.DataFrame, definition: str) -> dict[str, Any]:
        ids = sorted(frame["doc_id"])
        by_source = {s: int(n) for s, n in frame["source"].value_counts().sort_index().items()}
        return {"definition": definition, "n": len(ids), "by_source": by_source, "target_min_docs": target,
                "meets_target": len(ids) >= target, "shortfall": max(0, target - len(ids)), "doc_ids": ids}

    ni = df[df["source"] == "notinject"]
    return {"p_val": pack(p_val, P_VAL_DEFINITION), "p_test": pack(p_test, P_TEST_DEFINITION),
            "notinject": {"n": int(len(ni)), "note": "separate FPR set (ТЗ Этап 4), not part of P_test"}}
