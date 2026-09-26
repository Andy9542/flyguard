"""Source loaders -> document rows (docs/design.md §4; ТЗ 1.1–1.4, 1.6).

Every loader returns a DataFrame with the columns ``doc_id, source, label, text, text_orig, cluster_id, spans,
meta`` (``meta`` is a dict; ``spans`` a list of ``(start, end)`` in normalised-text coordinates). The E1 ``split``
column is assigned afterwards by :mod:`flyguard.data.splits`. Raw texts are data: nothing here prints them.
Reads of test files go through ``access_log`` (default: ``flyguard.netlog.log_data_access``), ТЗ "Честность".
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from flyguard.config import ROOT
from flyguard.data.dedup import char_shingles, jaccard
from flyguard.data.normalize import normalizer_from_cfg

AccessLog = Callable[[Path, str, str], None]
DOC_COLUMNS = ["doc_id", "source", "label", "text", "text_orig", "cluster_id", "spans", "meta"]
BIPIA_TASKS = ("email", "table", "code")
BIPIA_DROPPED_TASKS = ("qa", "abstract")   # not shipped in the BIPIA repo (BLOCKERS B4)
BIPIA_POSITIONS = ("start", "middle", "end")
NOTINJECT_SUBSETS = ("one", "two", "three")


def _log(access_log: AccessLog | None, path: Path, split: str, purpose: str) -> None:
    if access_log is None:
        from flyguard.netlog import log_data_access
        log_data_access(path, split=split, purpose=purpose)
    else:
        access_log(path, split, purpose)


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in DOC_COLUMNS})
    return pd.DataFrame(rows)[DOC_COLUMNS]


# ------------------------------------------------------------------------------------------ deepset

def load_deepset(cfg: Any, split: str, root: Path = ROOT, access_log: AccessLog | None = None) -> pd.DataFrame:
    """deepset/prompt-injections parquet (ТЗ 1.1; design §4): ids by row order, ``cluster_id`` = own id (ТЗ 1.10).

    The test parquet is a test file: its read is journaled. Labels of train are the only training labels in E1.
    """
    norm = normalizer_from_cfg(cfg)
    path = root / "data" / "raw" / "deepset" / f"{split}.parquet"
    if split == "test":
        _log(access_log, path, "test", "flyguard.data.loaders.load_deepset: build documents")
    df = pd.read_parquet(path, columns=["text", "label"])
    rows = []
    for i, (text, label) in enumerate(zip(df["text"].tolist(), df["label"].tolist())):
        doc_id = f"deep:{split}:{i}"
        text_orig = "" if text is None else str(text)
        rows.append({"doc_id": doc_id, "source": "deep", "label": int(label), "text": norm(text_orig),
                     "text_orig": text_orig, "cluster_id": doc_id, "spans": [],
                     "meta": {"deepset_split": split, "row": i}})
    return _frame(rows)


# ---------------------------------------------------------------------------------------- NotInject

def load_notinject(cfg: Any, root: Path = ROOT, access_log: AccessLog | None = None) -> pd.DataFrame:
    """NotInject (ТЗ 1.1, 1.2): three JSON subsets, label 0, test only, FPR only; ``cluster_id`` = own id."""
    norm = normalizer_from_cfg(cfg)
    rows = []
    for subset in NOTINJECT_SUBSETS:
        path = root / "data" / "raw" / "notinject" / f"NotInject_{subset}.json"
        _log(access_log, path, "test", "flyguard.data.loaders.load_notinject: build documents")
        with open(path, encoding="utf-8") as fh:
            items = json.load(fh)
        for i, item in enumerate(items):
            doc_id = f"notinject:{subset}:{i}"
            text_orig = str(item.get("prompt", ""))
            rows.append({"doc_id": doc_id, "source": "notinject", "label": 0, "text": norm(text_orig),
                         "text_orig": text_orig, "cluster_id": doc_id, "spans": [],
                         "meta": {"subset": subset, "category": item.get("category"),
                                  "word_list": list(item.get("word_list") or [])}})
    return _frame(rows)


# -------------------------------------------------------------------------------------- paraphrases

PARAPHRASE_COLUMNS = ["para_id", "base_id", "kind", "label", "stratum", "text", "jaccard_to_base", "generator",
                      "judge_confidence"]


def load_paraphrases(cfg: Any, root: Path = ROOT, access_log: AccessLog | None = None,
                     path: Path | None = None) -> pd.DataFrame | None:
    """Paraphrases CSV of design §7 (ТЗ 1.6): test only, ``cluster_id`` = base id, stratum kept in meta.

    Returns None when the file does not exist yet (the two-stage build of design §2).
    """
    path = path or root / "data" / "paraphrases" / "paraphrases.csv"
    if not path.exists():
        return None
    _log(access_log, path, "test", "flyguard.data.loaders.load_paraphrases: build documents")
    norm = normalizer_from_cfg(cfg)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in PARAPHRASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"paraphrases.csv lacks columns {missing}")
    rows = []
    counter: dict[str, int] = {}
    for rec in df.to_dict("records"):
        base_id = str(rec["base_id"])
        k = counter.get(base_id, 0)
        counter[base_id] = k + 1
        text_orig = str(rec["text"])
        rows.append({"doc_id": f"para:{base_id}:{k}", "source": "para", "label": int(float(rec["label"])),
                     "text": norm(text_orig), "text_orig": text_orig, "cluster_id": base_id, "spans": [],
                     "meta": {"para_id": rec["para_id"], "base_id": base_id, "kind": rec["kind"],
                              "stratum": rec["stratum"], "generator": rec["generator"],
                              "jaccard_to_base": rec["jaccard_to_base"],
                              "judge_confidence": rec["judge_confidence"]}})
    return _frame(rows)


# -------------------------------------------------------------------------------------------- BIPIA

_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*\s+")


def sentence_starts(context: str) -> list[int]:
    """Candidate insertion offsets for the ``middle`` position (own re-implementation of BIPIA's rule).

    BIPIA's ``bipia/data/utils.py::insert_middle`` samples one sentence *start* from
    ``PunktSentenceTokenizer().span_tokenize(context)`` (so offset 0 is a candidate) and joins
    ``context[:start], attack, context[start:]`` with newlines. nltk is not a project dependency, so the boundary
    set is: offset 0 plus every offset that follows a run of ``.!?`` (optionally closing quotes/brackets) and at
    least one whitespace character, excluding the end of the text. Line breaks without terminal punctuation are
    not boundaries, as in Punkt; the same rule applies to the ``code`` task (lines joined with ``\\n``, as BIPIA does).
    """
    n = len(context)
    starts = {0}
    for m in _SENTENCE_END.finditer(context):
        if m.end() < n:
            starts.add(m.end())
    return sorted(starts)


def insert_start(context: str, attack: str) -> tuple[str, str]:
    """BIPIA ``insert_start``: ``attack + "\\n" + context``. Returns (text, prefix before the attack)."""
    return "\n".join([attack, context]), ""


def insert_end(context: str, attack: str) -> tuple[str, str]:
    """BIPIA ``insert_end``: ``context + "\\n" + attack``."""
    return "\n".join([context, attack]), context + "\n"


def insert_middle(context: str, attack: str, boundary: int) -> tuple[str, str]:
    """BIPIA ``insert_middle`` at a sentence start chosen by the caller's seeded RNG (see :func:`sentence_starts`)."""
    return "\n".join([context[:boundary], attack, context[boundary:]]), context[:boundary] + "\n"


def insert_attack(context: str, attack: str, position: str, boundary: int = 0) -> tuple[str, str]:
    """Dispatch to the BIPIA insertion of ``position`` (ТЗ 1.4); returns ``(text, prefix before the attack)``."""
    if position == "start":
        return insert_start(context, attack)
    if position == "end":
        return insert_end(context, attack)
    if position == "middle":
        return insert_middle(context, attack, boundary)
    raise ValueError(f"unknown position {position!r}")


def locate_span(norm_text: str, norm_prefix: str, norm_attack: str) -> tuple[int, int]:
    """Span of the inserted attack in normalised coordinates (design §2: spans live in ``text`` coordinates).

    The attack is separated from the context by newlines, so after NFKC + whitespace collapse it appears verbatim,
    preceded by ``normalize(prefix)`` and one space (none when the prefix is empty). The expected offset is
    verified; the first occurrence is the fallback; a miss is an error (never silently mislabelled).
    """
    expected = len(norm_prefix) + (1 if norm_prefix else 0)
    L = len(norm_attack)
    if norm_text[expected:expected + L] == norm_attack:
        return expected, expected + L
    pos = norm_text.find(norm_attack)
    if pos < 0 or L == 0:
        raise ValueError("inserted attack not found in the normalised document")
    return pos, pos + L


def bipia_context_key(doc_id: str) -> str:
    """``bipia:<task>:<ctx>`` of a BIPIA document id: the pair/context key (its cluster may be a wider group)."""
    return ":".join(doc_id.split(":")[:3])


def slug(name: str) -> str:
    """Attack names contain spaces and ``&``; ids use ``[a-z0-9_]``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def bipia_attacks(root: Path, task: str, split: str = "test", access_log: AccessLog | None = None) -> dict[str, list[str]]:
    """Attack strings of the BIPIA protocol: ``text_attack_<split>.json`` for email/table, ``code_attack_<split>.json``
    for code (design §4). Test contexts are paired with the *test* attacks, as in the benchmark itself."""
    kind = "code" if task == "code" else "text"
    path = root / "data" / "raw" / "bipia" / f"{kind}_attack_{split}.json"
    if split == "test":
        _log(access_log, path, "test", f"flyguard.data.loaders.bipia_attacks: {task}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): [str(s) for s in v] for k, v in data.items()}


def bipia_contexts(root: Path, task: str, split: str, norm: Callable[[str], str],
                   access_log: AccessLog | None = None) -> tuple[list[tuple[int, str, int]], int]:
    """Distinct contexts of a BIPIA task file as ``(first_row_index, context_text, n_rows)``, plus the row count.

    BIPIA test files repeat a context under several questions; one pair per distinct normalised context keeps the
    cluster = base context (ТЗ 1.4). Code contexts are lists of lines, joined with ``\\n`` as BIPIA's code task does.
    """
    path = root / "data" / "raw" / "bipia" / task / f"{split}.jsonl"
    if split == "test":
        _log(access_log, path, "test", f"flyguard.data.loaders.bipia_contexts: {task}")
    seen: dict[str, int] = {}
    out: list[tuple[int, str, int]] = []
    n_rows = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            ctx = json.loads(line)["context"]
            if isinstance(ctx, list):
                ctx = "\n".join(str(x) for x in ctx)
            ctx = str(ctx)
            key = norm(ctx)
            if key in seen:
                idx = seen[key]
                out[idx] = (out[idx][0], out[idx][1], out[idx][2] + 1)
            else:
                seen[key] = len(out)
                out.append((i, ctx, 1))
    return out, n_rows


def context_clusters(contexts: list[tuple[int, str, int]], norm: Callable[[str], str], shingle: int,
                     threshold: float) -> dict[int, int]:
    """Map ``context index -> representative index`` grouping near-identical contexts (ТЗ 1.4 "cluster = base
    context", ТЗ 1.10). BIPIA files contain contexts that differ by a few characters; if two such contexts fell on
    different sides of the val/test split, dedup (ТЗ 1.7) would exclude the test one and its pair. Connected
    components of document-level Jaccard >= ``threshold`` over character shingles (the dedup rule's own numbers)
    share a cluster, so the split and the cluster bootstrap treat them as one base context."""
    idx = [c[0] for c in contexts]
    sh = [char_shingles(norm(c[1]), shingle) for c in contexts]
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if jaccard(sh[a], sh[b]) >= threshold:
                ra, rb = find(idx[a]), find(idx[b])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in idx}


def context_rng(seed_subsample: int, task: str, ctx_index: int) -> np.random.Generator:
    """Per-context RNG from seed ``subsample`` (global seed 0): adding or removing a context never shifts the
    draws of another one (spawn key = task number, context index)."""
    task_no = BIPIA_TASKS.index(task) if task in BIPIA_TASKS else 10 + zlib.crc32(task.encode()) % 1000
    return np.random.default_rng(np.random.SeedSequence(int(seed_subsample), spawn_key=(task_no, int(ctx_index))))


def load_bipia(cfg: Any, seed_subsample: int, root: Path = ROOT, access_log: AccessLog | None = None,
               tasks: tuple[str, ...] = BIPIA_TASKS, split: str = "test") -> tuple[pd.DataFrame, dict[str, Any]]:
    """BIPIA pairs (ТЗ 1.4, design §4) for every *test* context of ``email``, ``table``, ``code``.

    Per distinct context: the clean document (label 0) and, for each attack name and each position
    (start/middle/end), one attacked document (label 1) whose span is the inserted attack. One string per attack
    name (index ``k`` of the five) is drawn by the context's RNG; the *main* test document is one attack name and
    one position drawn by the same RNG (meta ``variant='main'``); the rest are the E6 "all attacks x positions" set
    (``variant='e6'``; ТЗ says 15 x 3, i.e. attack names x positions, so code gets 10 x 3). All documents of a context
    share ``cluster_id = bipia:<task>:<rep>`` where ``rep`` is the representative of its near-duplicate group
    (:func:`context_clusters`; usually the context itself); document ids keep the context's own index. The middle
    boundary is drawn once per context, as BIPIA reuses one seed for every insertion. Train contexts are not used in the build (labels come from deepset train only);
    ``split="train"`` exists for sanity checks on train material without opening a test file.
    """
    norm = normalizer_from_cfg(cfg)
    dd = cfg.default["dedup"]
    rows: list[dict[str, Any]] = []
    info: dict[str, Any] = {"tasks": {}, "dropped_tasks": {}, "positions": list(BIPIA_POSITIONS)}
    for task in BIPIA_DROPPED_TASKS:
        info["dropped_tasks"][task] = "context corpus not in the BIPIA repository (BLOCKERS B4)"
    for task in tasks:
        ctx_path = root / "data" / "raw" / "bipia" / task / f"{split}.jsonl"
        if not ctx_path.exists():
            info["dropped_tasks"][task] = f"{split}.jsonl missing"
            continue
        attacks = bipia_attacks(root, task, split, access_log)
        names = sorted(attacks)
        contexts, n_rows = bipia_contexts(root, task, split, norm, access_log)
        rep_of = context_clusters(contexts, norm, int(dd["shingle"]), float(dd["jaccard"]))
        n_main = n_e6 = 0
        for ctx_index, context, n_occ in contexts:
            rng = context_rng(seed_subsample, task, ctx_index)
            k_by_name = {name: int(rng.integers(0, len(attacks[name]))) for name in names}
            main_name = names[int(rng.integers(0, len(names)))]
            main_pos = BIPIA_POSITIONS[int(rng.integers(0, len(BIPIA_POSITIONS)))]
            cands = sentence_starts(context)
            boundary = int(cands[int(rng.integers(0, len(cands)))])
            ctx_key = f"bipia:{task}:{ctx_index}"
            cluster = f"bipia:{task}:{rep_of[ctx_index]}"
            base_meta = {"task": task, "context_index": ctx_index, "context_rows": n_occ,
                         "cluster_rep": rep_of[ctx_index], "n_sentence_boundaries": len(cands)}
            rows.append({"doc_id": f"{ctx_key}:clean", "source": "bipia", "label": 0, "text": norm(context),
                         "text_orig": context, "cluster_id": cluster, "spans": [],
                         "meta": {**base_meta, "variant": "main", "attack": None, "attack_id": None,
                                  "position": None, "pair_of": None}})
            for name in names:
                k = k_by_name[name]
                attack = attacks[name][k]
                attack_id = f"{slug(name)}-{k}"
                norm_attack = norm(attack)
                for pos in BIPIA_POSITIONS:
                    text_orig, prefix = insert_attack(context, attack, pos, boundary)
                    text = norm(text_orig)
                    span = locate_span(text, norm(prefix), norm_attack)
                    is_main = name == main_name and pos == main_pos
                    n_main += int(is_main)
                    n_e6 += int(not is_main)
                    rows.append({"doc_id": f"{ctx_key}:{attack_id}:{pos}", "source": "bipia", "label": 1,
                                 "text": text, "text_orig": text_orig, "cluster_id": cluster, "spans": [span],
                                 "meta": {**base_meta, "variant": "main" if is_main else "e6", "attack": name,
                                          "attack_id": attack_id, "attack_index": k, "position": pos,
                                          "boundary": boundary if pos == "middle" else None,
                                          "pair_of": f"{ctx_key}:clean"}})
        info["tasks"][task] = {"rows": n_rows, "distinct_contexts": len(contexts),
                               "clusters": len(set(rep_of.values())), "attack_names": len(names),
                               "attack_strings": sum(len(v) for v in attacks.values()),
                               "docs_main": len(contexts) + n_main, "docs_e6_extra": n_e6}
    return _frame(rows), info


# --------------------------------------------------------------------------------- dojo / dyn traces

def load_trace_documents(cfg: Any, benchmark: str, root: Path = ROOT) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """dojo/dyn documents through ``flyguard.agentdojo_io.extract.build_episode_documents`` (design §3), imported
    lazily; returns ``(episodes, documents, note)`` with ``None`` frames and the reason when the module or the trace
    logs are missing (design §2: the sources are skipped and the audit records it)."""
    src = {"agentdojo": "dojo", "agentdyn": "dyn"}[benchmark]
    logdir = root / cfg.default["traces"]["logdir"] / benchmark
    if not logdir.exists() or not any(logdir.rglob("*.json")):
        return None, None, f"{src}: no trace logs under {logdir.relative_to(root) if logdir.is_relative_to(root) else logdir}"
    try:
        from flyguard.agentdojo_io.extract import build_episode_documents
    except ImportError as exc:
        return None, None, f"{src}: flyguard.agentdojo_io.extract unavailable ({exc})"
    try:
        episodes, docs = build_episode_documents(cfg, benchmark)
    except FileNotFoundError as exc:
        return None, None, f"{src}: extraction failed ({exc})"
    if docs is None or len(docs) == 0:
        return episodes, None, f"{src}: extraction produced no documents"
    return episodes, adapt_trace_documents(docs, src, cfg), f"{src}: {len(docs)} documents from {0 if episodes is None else len(episodes)} episodes"


def adapt_trace_documents(docs: pd.DataFrame, source: str, cfg: Any) -> pd.DataFrame:
    """Coerce the §2 frame produced by agentdojo_io into the loader schema (meta dict, spans as pairs, source)."""
    norm = normalizer_from_cfg(cfg)
    df = docs.copy()
    if "meta" not in df.columns:
        df["meta"] = [json.loads(m) if isinstance(m, str) and m else {} for m in df.get("meta_json", [""] * len(df))]
    else:
        df["meta"] = [json.loads(m) if isinstance(m, str) else dict(m or {}) for m in df["meta"]]
    if "text_orig" not in df.columns:
        df["text_orig"] = df["text"]
    if "text" not in df.columns:
        df["text"] = [norm(t) for t in df["text_orig"]]
    if "spans" not in df.columns:
        df["spans"] = [[] for _ in range(len(df))]
    df["spans"] = [[(int(sp["start"]), int(sp["end"])) if isinstance(sp, dict) else (int(sp[0]), int(sp[1]))
                    for sp in (s if s is not None else [])] for s in df["spans"]]
    df["source"] = source
    df["label"] = df["label"].astype(int)
    keep = DOC_COLUMNS + (["split"] if "split" in df.columns else [])
    return df[keep].reset_index(drop=True)
