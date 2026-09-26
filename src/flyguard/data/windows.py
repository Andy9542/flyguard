"""Windows over documents (ТЗ 1.3, docs/design.md §2).

One windowing scheme for every detector: size 256, stride 192 (overlap 64); a document shorter than the size is a
single window. A window is positive when it contains at least ``min_span_chars`` characters of a known injection
span, or the whole span when the span is shorter; span-less sources pass the document label to every window.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import pandas as pd
import xxhash

Span = tuple[int, int]


def make_windows(text: str | int, size: int, stride: int) -> list[tuple[int, int]]:
    """ТЗ 1.3: half-open character windows ``[start, end)`` covering the whole text.

    Windows start at 0, stride, 2*stride, ... and the last one is clipped to the text length, so every character
    belongs to at least one window and consecutive windows overlap by ``size - stride`` characters. A text no
    longer than ``size`` (including the empty text) is exactly one window. ``text`` may be a string or its length.
    """
    n = len(text) if isinstance(text, str) else int(text)
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    if n <= size:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + size, n)
        out.append((start, end))
        if end >= n:
            break
        start += stride
    return out


def _span_pairs(spans: Any) -> list[Span]:
    """Accept ``[(s, e)]``, ``[{'start': s, 'end': e}]`` or None and return a list of int pairs."""
    if spans is None:
        return []
    out: list[Span] = []
    for sp in spans:
        if isinstance(sp, dict):
            out.append((int(sp["start"]), int(sp["end"])))
        else:
            out.append((int(sp[0]), int(sp[1])))
    return out


def window_label(start: int, end: int, spans: Sequence[Any] | None, min_span_chars: int, doc_label: int) -> int:
    """ТЗ 1.3 window label.

    With spans: 1 iff some span overlaps the window by >= min(min_span_chars, span length) characters (the whole
    span when it is shorter than ``min_span_chars``), else 0. Without spans (deepset, NotInject, paraphrases): the
    document label, because those sources have no localisation of the injection.
    """
    pairs = _span_pairs(spans)
    if not pairs:
        return int(doc_label)
    for s, e in pairs:
        span_len = e - s
        if span_len <= 0:
            continue
        overlap = min(end, e) - max(start, s)
        if overlap > 0 and overlap >= min(min_span_chars, span_len):
            return 1
    return 0


TEXT_HASH_ALGO = "xxhash64"
TEXT_HASH_DEFAULTS = {"algo": TEXT_HASH_ALGO, "seed": 0, "encoding": "utf-8"}


def text_hash(text: str, seed: int = 0, encoding: str = "utf-8") -> str:
    """xxhash64 hex of the window text: THE key of the score caches (design §2, §6; ASSUMPTIONS A17).

    The single definition of ``configs/default.yaml`` ``windows.text_hash`` {algo xxhash64, seed 0, utf-8};
    baselines import this function. The defaults equal the config so cfg-less callers get the same key;
    :func:`text_hash_from_cfg` binds the config values and refuses an unknown algorithm.
    """
    return xxhash.xxh64(text.encode(encoding), seed=int(seed)).hexdigest()


def text_hash_from_cfg(cfg: Any) -> Callable[[str], str]:
    """``text_hash`` bound to ``cfg.default['windows']['text_hash']`` (design §2: one cache key everywhere)."""
    spec = dict(TEXT_HASH_DEFAULTS, **((cfg.default.get("windows") or {}).get("text_hash") or {}))
    if str(spec["algo"]) != TEXT_HASH_ALGO:
        raise ValueError(f"windows.text_hash.algo={spec['algo']!r} is not supported; only {TEXT_HASH_ALGO}")
    seed, encoding = int(spec["seed"]), str(spec["encoding"])
    return lambda text: text_hash(text, seed, encoding)


def windows_for_document(doc_id: str, text: str, spans: Sequence[Any] | None, doc_label: int, size: int,
                         stride: int, min_span_chars: int, hasher: Callable[[str], str] = text_hash) -> list[dict[str, Any]]:
    """Rows of ``windows.parquet`` for one document (window_id = ``<doc_id>#w<k>``)."""
    rows = []
    for k, (s, e) in enumerate(make_windows(text, size, stride)):
        wtext = text[s:e]
        rows.append({"window_id": f"{doc_id}#w{k}", "doc_id": doc_id, "start": s, "end": e, "text": wtext,
                     "label": window_label(s, e, spans, min_span_chars, doc_label), "text_hash": hasher(wtext)})
    return rows


def build_windows(documents: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Window table for every document (ТЗ 1.3) with the config's size/stride/min_span_chars.

    Carries ``source``, ``split`` and ``cluster_id`` from the document so dedup and the experiments never need a
    join; ``dedup_excluded``/``dup_of`` start empty and are filled by :mod:`flyguard.data.dedup`.
    """
    w = cfg.default["windows"]
    size, stride, min_span = int(w["size"]), int(w["stride"]), int(w["min_span_chars"])
    hasher = text_hash_from_cfg(cfg)
    rows: list[dict[str, Any]] = []
    cols = documents[["doc_id", "source", "split", "label", "text", "spans", "cluster_id"]]
    for doc_id, source, split, label, text, spans, cluster_id in cols.itertuples(index=False, name=None):
        for r in windows_for_document(doc_id, text, spans, int(label), size, stride, min_span, hasher):
            r.update({"source": source, "split": split, "cluster_id": cluster_id, "dedup_excluded": False,
                      "dup_of": None})
            rows.append(r)
    order = ["window_id", "doc_id", "source", "split", "start", "end", "text", "label", "cluster_id", "text_hash",
             "dedup_excluded", "dup_of"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in order})
    return pd.DataFrame(rows)[order]


def spans_to_json(spans: Sequence[Any] | None) -> str:
    """Stable JSON of spans for manifests/tests."""
    return json.dumps([{"start": s, "end": e} for s, e in _span_pairs(spans)])
