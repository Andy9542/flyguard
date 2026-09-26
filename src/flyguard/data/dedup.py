"""Window-level deduplication (ТЗ 1.7, docs/design.md §2).

Rule: MinHash over character 5-gram shingles, within class, across all sources; a *test* window whose Jaccard
similarity with any train/validation window of the same class is >= the threshold is excluded from the test
(``dedup_excluded=True``, ``dup_of`` = the most similar reference window). Train/val windows are never touched, so
the training set is not changed by dedup and the test cannot leak memorised text. LSH candidates are verified
with the exact Jaccard so the result is deterministic and does not depend on LSH banding luck.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
from datasketch import MinHash, MinHashLSH

REF_SPLITS = ("train", "val")


def char_shingles(text: str, n: int) -> set[str]:
    """Character n-gram shingle set (ТЗ 1.7; reused by the paraphrase filters, ТЗ 1.6).

    A text shorter than ``n`` yields the text itself as its single shingle (so two identical short texts still have
    Jaccard 1); the empty text yields the empty set. Case is kept: the normalised text is compared as the detectors
    see it.
    """
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Exact Jaccard similarity of two shingle sets; 0.0 when both are empty."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def minhash_of(shingles: Iterable[str], num_perm: int) -> MinHash:
    """datasketch MinHash of a shingle set with ``dedup.minhash_perm`` permutations (ТЗ 1.7)."""
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m


def find_test_duplicates(windows: pd.DataFrame, shingle: int, num_perm: int, threshold: float) -> dict[str, str]:
    """Map ``test window_id -> reference window_id`` for every test window duplicating a train/val window.

    Within class: label-1 test windows are only compared with label-1 reference windows, label-0 with label-0. The
    LSH is built over the (small) reference set and queried with each test window; every candidate is verified with
    the exact Jaccard, ties broken by the lexicographically smallest reference id.
    """
    dup_of: dict[str, str] = {}
    ref = windows[windows["split"].isin(REF_SPLITS)]
    test = windows[windows["split"] == "test"]
    if ref.empty or test.empty:
        return dup_of
    for label in sorted(windows["label"].unique()):
        ref_l = ref[ref["label"] == label]
        test_l = test[test["label"] == label]
        if ref_l.empty or test_l.empty:
            continue
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        ref_text: dict[str, str] = {}
        for wid, text in ref_l[["window_id", "text"]].itertuples(index=False, name=None):
            sh = char_shingles(text, shingle)
            if not sh:
                continue
            ref_text[wid] = text
            lsh.insert(wid, minhash_of(sh, num_perm))
        if not ref_text:
            continue
        for wid, text in test_l[["window_id", "text"]].itertuples(index=False, name=None):
            sh = char_shingles(text, shingle)
            if not sh:
                continue
            cands = lsh.query(minhash_of(sh, num_perm))
            best_id, best_j = None, -1.0
            for cid in sorted(cands):
                j = jaccard(sh, char_shingles(ref_text[cid], shingle))
                if j >= threshold and j > best_j:
                    best_id, best_j = cid, j
            if best_id is not None:
                dup_of[wid] = best_id
    return dup_of


def apply_window_exclusions(windows: pd.DataFrame, dup_of: dict[str, str]) -> pd.DataFrame:
    """Set ``dedup_excluded``/``dup_of`` on the window table (only test windows can appear in ``dup_of``)."""
    out = windows.copy()
    ids = out["window_id"]
    out["dup_of"] = ids.map(dup_of).astype(object).where(ids.isin(dup_of.keys()), None)
    out["dedup_excluded"] = ids.isin(dup_of.keys()).astype(bool)
    return out


def _variants(documents: pd.DataFrame) -> pd.Series:
    if "meta" in documents.columns:
        metas = [m if isinstance(m, dict) else (json.loads(m) if m else {}) for m in documents["meta"]]
    elif "meta_json" in documents.columns:
        metas = [json.loads(m) if m else {} for m in documents["meta_json"]]
    else:
        metas = [{} for _ in range(len(documents))]
    return pd.Series([m.get("variant", "main") for m in metas], index=documents.index)


def document_drops(documents: pd.DataFrame, windows: pd.DataFrame) -> dict[str, list[str]]:
    """Document-level consequences of the window exclusions (ТЗ 1.7).

    * ``positives_all_excluded``: a positive test document whose positive windows are all excluded leaves the
      positives (its remaining windows are clean text, so scoring it as a positive would be meaningless).
    * ``no_windows_left``: any test document with no window left cannot be scored and leaves the test.
    * ``bipia_pairs``: BIPIA pairs are never broken (ТЗ 1.7). The pair is the clean document plus the *main*
      attacked document of a context; when either of them is dropped by the rules above, every document of that
      context (both members and its E6 variants) is dropped, so no half-pair remains. An E6 variant dropped on
      its own (the benchmark reuses long attack strings across contexts, so windows inside such a string duplicate
      a validation window) does not touch the pair.
    Documents stay in ``documents.parquet`` (flag ``dedup_dropped``); the split lists and pools omit them.
    """
    test_docs = documents[documents["split"] == "test"]
    w = windows[windows["doc_id"].isin(test_docs["doc_id"])]
    kept = w[~w["dedup_excluded"]]
    kept_any = set(kept["doc_id"])
    kept_pos = set(kept[kept["label"] == 1]["doc_id"])
    had_pos = set(w[w["label"] == 1]["doc_id"])
    positives_all_excluded = sorted(d for d in had_pos if d not in kept_pos)
    no_windows_left = sorted(d for d in set(w["doc_id"]) if d not in kept_any and d not in positives_all_excluded)
    dropped = set(positives_all_excluded) | set(no_windows_left)
    bip = test_docs[test_docs["source"] == "bipia"].copy()
    bipia_pairs: list[str] = []
    broken: set[str] = set()
    if not bip.empty:
        bip["variant"] = _variants(bip)
        bip["ctx"] = [":".join(d.split(":")[:3]) for d in bip["doc_id"]]
        essential = (bip["label"] == 0) | (bip["variant"] == "main")
        broken = set(bip[essential & bip["doc_id"].isin(dropped)]["ctx"])
        bipia_pairs = sorted(set(bip[bip["ctx"].isin(broken)]["doc_id"]) - dropped)
    return {"positives_all_excluded": positives_all_excluded, "no_windows_left": no_windows_left,
            "bipia_pairs": bipia_pairs, "bipia_contexts": sorted(broken)}


def _drops_by_variant(documents: pd.DataFrame, dropped: set[str]) -> dict[str, int]:
    """Counts of dropped documents by ``<source>/<variant>/<label>`` so the audit separates main-test losses from
    E6 extras."""
    d = documents[documents["doc_id"].isin(dropped)]
    if d.empty:
        return {}
    keys = [f"{s}/{v}/{int(l)}" for s, v, l in zip(d["source"], _variants(d), d["label"])]
    out: dict[str, int] = {}
    for k in keys:
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def dedup_windows(documents: pd.DataFrame, windows: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, set[str], dict]:
    """Run ТЗ 1.7 end to end: returns (windows with flags, dropped doc ids, dedup.json payload)."""
    d = cfg.default["dedup"]
    shingle, num_perm, thr = int(d["shingle"]), int(d["minhash_perm"]), float(d["jaccard"])
    dup_of = find_test_duplicates(windows, shingle, num_perm, thr)
    out = apply_window_exclusions(windows, dup_of)
    drops = document_drops(documents, out)
    dropped = set(drops["positives_all_excluded"]) | set(drops["no_windows_left"]) | set(drops["bipia_pairs"])
    src_of = dict(zip(out["window_id"], out["source"]))
    ref_src = dict(zip(out["window_id"], out["source"]))
    by_source: dict[str, int] = {}
    by_pair: dict[str, int] = {}
    for wid, rid in dup_of.items():
        by_source[src_of[wid]] = by_source.get(src_of[wid], 0) + 1
        key = f"{src_of[wid]}->{ref_src[rid]}"
        by_pair[key] = by_pair.get(key, 0) + 1
    report = {
        "rule": {"shingle": shingle, "minhash_perm": num_perm, "jaccard": thr, "scope": "test vs train+val, within class"},
        "windows_total": int(len(out)),
        "windows_test": int((out["split"] == "test").sum()),
        "windows_reference": int(out["split"].isin(REF_SPLITS).sum()),
        "test_windows_excluded": int(len(dup_of)),
        "test_windows_excluded_by_label": {str(int(l)): int(((out["dedup_excluded"]) & (out["label"] == l)).sum())
                                           for l in sorted(out["label"].unique())} if len(out) else {},
        "test_windows_excluded_by_source": dict(sorted(by_source.items())),
        "test_windows_excluded_by_source_pair": dict(sorted(by_pair.items())),
        "documents_dropped": {k: len(v) for k, v in drops.items()},
        "documents_dropped_by_source_variant": _drops_by_variant(documents, dropped),
        "documents_dropped_ids": drops,
        "documents_dropped_total": int(len(dropped)),
    }
    return out, dropped, report
