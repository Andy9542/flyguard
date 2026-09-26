"""Window-level deduplication (ТЗ 1.7, docs/design.md §2).

Rule: MinHash over character 5-gram shingles, within class, across all sources; a *test* window whose Jaccard
similarity with any train/validation window of the same class is >= the threshold is excluded from the test
(``dedup_excluded=True``, ``dup_of`` = the most similar reference window). Train/val windows are never touched, so
the training set is not changed by dedup and the test cannot leak memorised text.

Candidates come from a MinHash LSH and every candidate is verified with the exact Jaccard, so the decision itself
never depends on the sketch: a false candidate is discarded, and a pair the LSH misses is the only error mode.
The banding is therefore chosen for *recall at the threshold*, not for datasketch's balanced false-positive /
false-negative optimum: with the config's 128 permutations, ``threshold=0.8`` would give b=9, r=13 and only a 40 %
chance of surfacing a pair at J=0.80 (review finding); :func:`lsh_params` picks the largest band size whose
candidate probability at J=threshold is at least ``LSH_RECALL`` (b=25, r=5: 0.99995 at J=0.80, 0.059 at J=0.30),
and the numbers are recorded in ``dedup.json['rule']``.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
from datasketch import MinHash, MinHashLSH

REF_SPLITS = ("train", "val")
LSH_RECALL = 0.9999
"""Minimum probability that a pair with Jaccard exactly at the threshold becomes an LSH candidate (engineering
constant of the candidate stage, not a ТЗ number: the ТЗ fixes the exact rule, Jaccard >= 0.8)."""


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


def containment(a: Iterable[str], b: Iterable[str]) -> float:
    """Share of ``a``'s shingles present in ``b`` (|a ∩ b| / |a|); 0.0 for an empty ``a``. Used by the
    contamination audit (ТЗ 3.2) for copies embedded in a longer training text, where Jaccard is diluted."""
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def minhash_of(shingles: Iterable[str], num_perm: int) -> MinHash:
    """datasketch MinHash of a shingle set with ``dedup.minhash_perm`` permutations (ТЗ 1.7)."""
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m


def candidate_probability(j: float, b: int, r: int) -> float:
    """Probability that a pair of Jaccard ``j`` shares at least one of ``b`` bands of ``r`` rows."""
    return 1.0 - (1.0 - float(j) ** r) ** b


def lsh_params(num_perm: int, threshold: float, recall: float = LSH_RECALL) -> tuple[int, int]:
    """Banding ``(b, r)`` with ``b * r <= num_perm``: the largest ``r`` (fewest spurious candidates) whose candidate
    probability at ``J = threshold`` is still >= ``recall``. Falls back to ``r = 1`` (every row its own band, the
    highest recall possible) when no banding reaches the floor."""
    best: tuple[int, int] | None = None
    for r in range(1, int(num_perm) + 1):
        b = int(num_perm) // r
        if b < 1:
            break
        if candidate_probability(threshold, b, r) >= recall:
            best = (b, r)
    return best or (int(num_perm), 1)


class NearDuplicateIndex:
    """MinHash-LSH candidates + exact Jaccard verification over character shingles (ТЗ 1.7 machinery).

    Shared by :func:`find_test_duplicates` (test windows against train/val windows) and by the contamination audit
    (ТЗ 3.2, :mod:`flyguard.data.audit`). Texts are kept to recompute shingles on demand for verification, so the
    index costs one string per key instead of one shingle set.
    """

    def __init__(self, shingle: int, num_perm: int, threshold: float, recall: float = LSH_RECALL):
        self.shingle, self.num_perm, self.threshold, self.recall = int(shingle), int(num_perm), float(threshold), float(recall)
        self.b, self.r = lsh_params(self.num_perm, self.threshold, self.recall)
        self.lsh = MinHashLSH(num_perm=self.num_perm, params=(self.b, self.r))
        self.texts: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def shingles(self, text: str) -> set[str]:
        return char_shingles(text, self.shingle)

    def minhash(self, text: str) -> MinHash | None:
        sh = self.shingles(text)
        return minhash_of(sh, self.num_perm) if sh else None

    def add(self, key: str, text: str, minhash: MinHash | None = None) -> bool:
        """Insert ``key``; returns False for a text without shingles (empty), which can never match."""
        if key in self.texts:
            raise KeyError(f"duplicate key {key!r}")
        m = minhash if minhash is not None else self.minhash(text)
        if m is None:
            return False
        self.texts[key] = text
        self.lsh.insert(key, m)
        return True

    def query(self, text: str, minhash: MinHash | None = None, shingles: set[str] | None = None) -> list[tuple[str, float]]:
        """Every indexed key whose exact Jaccard with ``text`` is >= the threshold, best first (ties by key).
        ``shingles``/``minhash`` of ``text`` may be passed when the caller already computed them."""
        sh = shingles if shingles is not None else self.shingles(text)
        if not sh:
            return []
        m = minhash if minhash is not None else minhash_of(sh, self.num_perm)
        hits = []
        for key in self.lsh.query(m):
            j = jaccard(sh, self.shingles(self.texts[key]))
            if j >= self.threshold:
                hits.append((key, j))
        hits.sort(key=lambda kv: (-kv[1], kv[0]))
        return hits

    def best(self, text: str, minhash: MinHash | None = None, shingles: set[str] | None = None) -> tuple[str, float] | None:
        hits = self.query(text, minhash, shingles)
        return hits[0] if hits else None

    def rule(self) -> dict[str, Any]:
        """The numbers of the candidate stage for manifests."""
        return {"minhash_perm": self.num_perm, "lsh_bands": self.b, "lsh_rows": self.r,
                "candidate_recall_at_threshold": round(candidate_probability(self.threshold, self.b, self.r), 6),
                "candidate_recall_floor": self.recall,
                "lsh_equivalent_threshold": round((1.0 / self.b) ** (1.0 / self.r), 4),
                "verification": "exact Jaccard >= threshold over the same shingles"}


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
        index = NearDuplicateIndex(shingle, num_perm, threshold)
        for wid, text in ref_l[["window_id", "text"]].itertuples(index=False, name=None):
            index.add(wid, text)
        if not len(index):
            continue
        for wid, text in test_l[["window_id", "text"]].itertuples(index=False, name=None):
            hit = index.best(text)
            if hit is not None:
                dup_of[wid] = hit[0]
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


def dedup_rule(cfg: Any) -> dict[str, Any]:
    """The ТЗ 1.7 numbers plus the candidate-stage banding, as written to ``dedup.json['rule']``."""
    d = cfg.default["dedup"]
    shingle, num_perm, thr = int(d["shingle"]), int(d["minhash_perm"]), float(d["jaccard"])
    rule = {"shingle": shingle, "jaccard": thr, "scope": "test vs train+val, within class",
            "reference_splits": list(REF_SPLITS)}
    rule.update(NearDuplicateIndex(shingle, num_perm, thr).rule())
    return rule


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
        "rule": dedup_rule(cfg),
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
