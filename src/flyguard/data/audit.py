"""``data/manifests/audit.md`` (ТЗ 1.2) and the contamination record of ТЗ 3.2 (``contamination.json``).

audit.md: documents, classes, languages, lengths, windows per source, share of German in deepset, NotInject
composition, BIPIA tasks dropped, skipped sources, dedup, pools, contamination. Counts only: the audit never quotes
a document (data-safety rule), and it carries no timestamp so idempotent rebuilds produce identical files.

Contamination (ТЗ 3.2: "по карточкам моделей и открытому обучающему набору PIGuard записывается пересечение с
источниками; направление сдвига не предполагается"): the PIGuard open training set is third-party training data,
read once, journaled through ``log_data_access`` and never printed. For every source of ours the audit counts the
documents and windows that have a near-duplicate in it under the dedup rule (ТЗ 1.7 machinery: MinHash candidates,
exact Jaccard >= ``dedup.jaccard`` over the same character 5-gram shingles of normalised text), adds an exact
containment check against the PIGuard records tagged with the counterpart dataset (copies embedded in a longer
prompt dilute Jaccard), and repeats the static facts from the model cards. The result feeds the threats-to-validity
section of the report; no direction of the effect is assumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from flyguard.config import ROOT
from flyguard.data.dedup import NearDuplicateIndex, char_shingles, containment, minhash_of
from flyguard.data.normalize import normalizer_from_cfg
from flyguard.data.windows import make_windows

PIGUARD_TRAIN = Path("data") / "raw" / "piguard_train" / "train.json"
EXTERNAL_SPLIT = "external"
"""Journal label of the PIGuard training-set read: not our train/val/test, third-party training data."""
TEXT_FIELDS = ("text", "prompt")
TARGETED_TAGS = {"deep": ("prompt-injections",), "bipia": ("BIPIA",), "dojo": ("InjecAgent",), "dyn": ("InjecAgent",)}
"""PIGuard ``source`` tags that name the dataset behind one of our sources; used by the containment check."""

MODEL_CARDS = {
    "protectai_v2": {
        "model": "protectai/deberta-v3-base-prompt-injection-v2",
        "training_datasets_on_card": ["natolambert/xstest-v2-copy", "VMware/open-instruct",
                                      "alespalla/chatbot_instruction_prompts", "HuggingFaceH4/grok-conversation-harmless",
                                      "Harelix/Prompt-Injection-Mixed-Techniques-2024", "OpenSafetyLab/Salad-Data",
                                      "jackhhao/jailbreak-classification"],
        "named_overlap_with_our_sources": "none of the listed datasets is deepset/prompt-injections, BIPIA, NotInject, "
                                          "AgentDojo or AgentDyn; the card's list is not exhaustive, the open training "
                                          "data is not published, so no measured overlap is possible",
    },
    "piguard": {
        "model": "leolee99/PIGuard",
        "training_datasets_on_card": ["the authors' own train.json (data/raw/piguard_train/train.json; composition "
                                      "measured below by its `source` tag)"],
        "named_overlap_with_our_sources": "train.json tags 546 records `prompt-injections` (deepset train: 343 benign / "
                                          "203 injections) and 1116 records `BIPIA`; NotInject is the authors' own "
                                          "over-defense benchmark",
    },
    "prompt_guard_2": {
        "model": "meta-llama/Llama-Prompt-Guard-2-86M",
        "training_datasets_on_card": ["not disclosed on the card; optional model, skipped without HF access (BLOCKERS B2)"],
        "named_overlap_with_our_sources": "unknown",
    },
}
THREATS_TO_VALIDITY = [
    "BIPIA is part of the PIGuard authors' test set and 1116 BIPIA-tagged records are in the PIGuard open training set",
    "NotInject, PIGuard and AgentDyn share a first author: the over-defense set and the agent benchmark were built by "
    "the team that trained one of the comparators",
    "deepset/prompt-injections train (546 records, 343/203) is in the PIGuard open training set under the tag "
    "`prompt-injections`; E1 trains FlyGuard on the same 546 documents, so on deepset both sides saw the train split",
    "ProtectAI v2's card lists datasets by name only; an unmeasured overlap with deepset or BIPIA cannot be excluded",
    "No direction of the shift is assumed (ТЗ 3.2); the counts above are reported, not corrected for",
]


# ----------------------------------------------------------------------------------------------- helpers

def _meta(documents: pd.DataFrame) -> list[dict[str, Any]]:
    if "meta" in documents.columns:
        return [m if isinstance(m, dict) else (json.loads(m) if m else {}) for m in documents["meta"]]
    return [json.loads(m) if m else {} for m in documents["meta_json"]]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return head + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def _len_stats(s: pd.Series) -> list[Any]:
    if s.empty:
        return [0, 0, 0, 0]
    return [int(s.min()), int(s.median()), round(float(s.mean()), 1), int(s.max())]


def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


# ------------------------------------------------------------------------------------- contamination

def read_piguard_train(path: Path, norm: Callable[[str], str]) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    """PIGuard ``train.json`` -> (distinct normalised texts, their ``source`` tags, counts).

    The file is a JSON list of records ``{prompt|text, label, source}`` (inspected with code, never printed);
    a dict with one list value is accepted too. Empty texts are counted and dropped; identical normalised texts
    are merged and keep every tag. Texts are returned sorted so the audit is deterministic.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else []
    field = next((f for f in TEXT_FIELDS if data and isinstance(data[0], dict) and f in data[0]), None)
    if field is None:
        raise ValueError("PIGuard train.json: no record with a text/prompt field")
    by_text: dict[str, set[str]] = {}
    by_source_label: dict[str, dict[str, int]] = {}
    n_empty = 0
    for rec in data:
        if not isinstance(rec, dict):
            continue
        tag = str(rec.get("source", "?"))
        label = str(rec.get("label", "?"))
        by_source_label.setdefault(tag, {}).setdefault(label, 0)
        by_source_label[tag][label] += 1
        text = norm(rec.get(field) or "")
        if not text:
            n_empty += 1
            continue
        by_text.setdefault(text, set()).add(tag)
    texts = sorted(by_text)
    tags = [sorted(by_text[t]) for t in texts]
    lengths = pd.Series([len(t) for t in texts]) if texts else pd.Series(dtype=int)
    stats = {"records": len(data), "text_field": field, "empty_texts": n_empty, "distinct_texts": len(texts),
             "by_source_label": {k: dict(sorted(v.items())) for k, v in sorted(by_source_label.items())},
             "text_length": dict(zip(("min", "median", "mean", "max"), _len_stats(lengths)))}
    return texts, tags, stats


def _source_key(source: str, variant: str) -> str:
    return f"{source}_e6" if variant == "e6" else source


def _nested_counts(frame: pd.DataFrame, matched: set[str], id_col: str, name: str) -> dict[str, Any]:
    """``{source_key: {split: {label: {n, matched, share}}}}`` plus totals per source_key."""
    out: dict[str, Any] = {}
    for (sk, split, label), g in frame.groupby(["source_key", "split", "label"], sort=True):
        n = int(len(g))
        m = int(g[id_col].isin(matched).sum())
        out.setdefault(sk, {}).setdefault(str(split), {})[str(int(label))] = {name: n, f"{name}_matched": m, "share": _share(m, n)}
    for sk, g in frame.groupby("source_key", sort=True):
        n, m = int(len(g)), int(g[id_col].isin(matched).sum())
        out.setdefault(sk, {})["total"] = {name: n, f"{name}_matched": m, "share": _share(m, n)}
    return out


def _by_tag(frame: pd.DataFrame, tags_of: dict[str, set[str]], id_col: str) -> dict[str, dict[str, int]]:
    """Matched items per source_key and PIGuard tag (an item counts once per tag it matches)."""
    out: dict[str, dict[str, int]] = {}
    for sk, g in frame.groupby("source_key", sort=True):
        counts: dict[str, int] = {}
        for i in g[id_col]:
            for tag in tags_of.get(i, ()):
                counts[tag] = counts.get(tag, 0) + 1
        if counts:
            out[sk] = dict(sorted(counts.items()))
    return out


def targeted_containment(docs: pd.DataFrame, texts: list[str], tags: list[list[str]], shingle: int,
                         threshold: float) -> dict[str, Any]:
    """Exact containment between our main documents of a source and the PIGuard records tagged with the counterpart
    dataset (``TARGETED_TAGS``): ``ours_in_piguard`` = share of the document's shingles inside one PIGuard text
    (a copy wrapped in a longer prompt), ``piguard_in_ours`` = share of a PIGuard text inside our document (an
    attack string embedded in a tool output). For dojo/dyn only positive documents are checked (the tag is an
    attack-string collection). Counts of documents whose best value is >= ``threshold``."""
    out: dict[str, Any] = {}
    for source, wanted in TARGETED_TAGS.items():
        ours = docs[(docs["source"] == source) & (docs["variant"] != "e6")]
        if source in ("dojo", "dyn"):
            ours = ours[ours["label"] == 1]
        pig_idx = [i for i, t in enumerate(tags) if any(w in t for w in wanted)]
        if ours.empty or not pig_idx:
            out[source] = {"tags": list(wanted), "piguard_texts": len(pig_idx), "documents": int(len(ours)), "note": "nothing to compare"}
            continue
        pig_sh = [char_shingles(texts[i], shingle) for i in pig_idx]
        n_in_pig = n_pig_in = 0
        per_split: dict[str, dict[str, int]] = {}
        for split, text in ours[["split", "text"]].itertuples(index=False, name=None):
            sh = char_shingles(text, shingle)
            best_a = best_b = 0.0
            for ps in pig_sh:
                inter = len(sh & ps)
                if inter:
                    best_a = max(best_a, inter / len(sh))
                    best_b = max(best_b, inter / len(ps))
            a, b = best_a >= threshold, best_b >= threshold
            n_in_pig += int(a)
            n_pig_in += int(b)
            ps_ = per_split.setdefault(str(split), {"documents": 0, "ours_in_piguard": 0, "piguard_in_ours": 0})
            ps_["documents"] += 1
            ps_["ours_in_piguard"] += int(a)
            ps_["piguard_in_ours"] += int(b)
        out[source] = {"tags": list(wanted), "piguard_texts": len(pig_idx), "documents": int(len(ours)),
                       "ours_in_piguard": n_in_pig, "ours_in_piguard_share": _share(n_in_pig, len(ours)),
                       "piguard_in_ours": n_pig_in, "piguard_in_ours_share": _share(n_pig_in, len(ours)),
                       "by_split": per_split}
    return out


def contamination_audit(cfg: Any, documents: pd.DataFrame, windows: pd.DataFrame, root: Path = ROOT,
                        access_log: Callable[[Path, str, str], None] | None = None,
                        train_path: Path | None = None) -> dict[str, Any]:
    """ТЗ 3.2 contamination record -> ``data/manifests/contamination.json`` (counts only).

    Three measurements against ``data/raw/piguard_train/train.json`` (skipped with a note when the file is absent):

    * ``document_level``: our documents whose whole normalised text has a near-duplicate PIGuard text (Jaccard >=
      ``dedup.jaccard``): the unit the review asked for, exact for deepset (short, unwrapped prompts).
    * ``window_level``: our windows (ТЗ 1.3 grid) with a near-duplicate among the PIGuard texts cut on the same
      grid; alignment-sensitive (a copy embedded after a different prefix shifts the grid), hence a lower bound.
    * ``targeted_containment``: exact containment against the PIGuard records tagged with the counterpart dataset.

    Our side is indexed (the smaller side) and PIGuard is streamed as queries, so memory stays with our data; the
    LSH banding is the dedup one (recorded in ``rule``). BIPIA E6 variants are reported as ``bipia_e6``.
    """
    path = Path(train_path) if train_path is not None else root / PIGUARD_TRAIN
    static = {"model_cards": MODEL_CARDS, "threats_to_validity": THREATS_TO_VALIDITY}
    if not path.exists():
        return {"skipped": True, "train_file": str(PIGUARD_TRAIN),
                "note": f"{PIGUARD_TRAIN} missing: PIGuard overlap not measured (scripts/fetch.sh downloads it)", **static}
    purpose = ("flyguard.data.audit.contamination_audit: third-party training set (PIGuard) read for the ТЗ 3.2 "
               "overlap counts; texts never printed")
    if access_log is None:
        from flyguard.netlog import log_data_access
        log_data_access(path, split=EXTERNAL_SPLIT, purpose=purpose)
    else:
        access_log(path, EXTERNAL_SPLIT, purpose)
    norm = normalizer_from_cfg(cfg)
    dd, w = cfg.default["dedup"], cfg.default["windows"]
    shingle, num_perm, thr = int(dd["shingle"]), int(dd["minhash_perm"]), float(dd["jaccard"])
    size, stride = int(w["size"]), int(w["stride"])
    texts, tags, pig_stats = read_piguard_train(path, norm)

    docs = documents[["doc_id", "source", "split", "label", "text"]].copy()
    docs["variant"] = [m.get("variant", "main") for m in _meta(documents)]
    docs["source_key"] = [_source_key(s, v) for s, v in zip(docs["source"], docs["variant"])]
    win = windows[["window_id", "doc_id", "source", "split", "label", "text"]].copy()
    win["source_key"] = win["doc_id"].map(dict(zip(docs["doc_id"], docs["source_key"])))

    doc_index = NearDuplicateIndex(shingle, num_perm, thr)
    for doc_id, text in docs[["doc_id", "text"]].itertuples(index=False, name=None):
        doc_index.add(doc_id, text)
    win_index = NearDuplicateIndex(shingle, num_perm, thr)
    for wid, text in win[["window_id", "text"]].itertuples(index=False, name=None):
        win_index.add(wid, text)

    doc_tags: dict[str, set[str]] = {}
    win_tags: dict[str, set[str]] = {}
    n_pig_windows = 0
    for text, tg in zip(texts, tags):
        sh = char_shingles(text, shingle)
        m = minhash_of(sh, num_perm)
        for doc_id, _ in doc_index.query(text, m, sh):
            doc_tags.setdefault(doc_id, set()).update(tg)
        for s, e in make_windows(text, size, stride):
            n_pig_windows += 1
            if e - s == len(text):
                hits = win_index.query(text, m, sh)
            else:
                hits = win_index.query(text[s:e])
            for wid, _ in hits:
                win_tags.setdefault(wid, set()).update(tg)

    matched_docs = set(doc_tags)
    matched_wins = set(win_tags)
    win["matched"] = win["window_id"].isin(matched_wins)
    per_doc = win.groupby("doc_id")["matched"].agg(["any", "all"])
    docs_any = set(per_doc.index[per_doc["any"]])
    docs_all = set(per_doc.index[per_doc["all"]])
    window_level = _nested_counts(win, matched_wins, "window_id", "windows")
    docs_with_window = _nested_counts(docs, docs_any, "doc_id", "documents")
    docs_all_windows = _nested_counts(docs, docs_all, "doc_id", "documents")
    for sk in window_level:
        for split, labels in docs_with_window.get(sk, {}).items():
            if split == "total":
                window_level[sk]["total"]["documents_with_matched_window"] = labels["documents_matched"]
                window_level[sk]["total"]["documents_all_windows_matched"] = docs_all_windows[sk]["total"]["documents_matched"]
                continue
            for label, c in labels.items():
                window_level[sk][split][label]["documents_with_matched_window"] = c["documents_matched"]
                window_level[sk][split][label]["documents_all_windows_matched"] = docs_all_windows[sk][split][label]["documents_matched"]

    rule = {"unit_document": "whole normalised document text vs whole PIGuard prompt",
            "unit_window": f"ТЗ 1.3 windows (size {size}, stride {stride}) of both sides; alignment-sensitive, a lower bound",
            "decision": f"exact Jaccard >= {thr} over character {shingle}-gram shingles (dedup.jaccard, ТЗ 1.7)",
            "candidates": doc_index.rule(), "containment": f"exact |A ∩ B| / |A| >= {thr} against the tagged records",
            "index_side": "our documents / windows; PIGuard streamed as queries"}
    return {"skipped": False, "train_file": str(PIGUARD_TRAIN), "journal_split": EXTERNAL_SPLIT, "rule": rule,
            "piguard_train": {**pig_stats, "windows_scanned": n_pig_windows},
            "document_level": _nested_counts(docs, matched_docs, "doc_id", "documents"),
            "document_level_by_piguard_source": _by_tag(docs, doc_tags, "doc_id"),
            "window_level": window_level,
            "window_level_by_piguard_source": _by_tag(win, win_tags, "window_id"),
            "targeted_containment": targeted_containment(docs, texts, tags, shingle, thr),
            **static}


# ------------------------------------------------------------------------------------------- audit.md

def _contamination_section(c: dict[str, Any]) -> list[str]:
    out = ["\n## Пересечение с обучающими данными промышленных детекторов (ТЗ 3.2)\n"]
    if not c:
        out.append("- аудит пересечения не запускался\n")
        return out
    if c.get("skipped"):
        out.append(f"- пропущено: {c.get('note')}\n")
    else:
        pig = c["piguard_train"]
        out.append(f"Открытый обучающий набор PIGuard (`{c['train_file']}`, чтение журналировано как `{c['journal_split']}`): "
                   f"записей {pig['records']}, пустых {pig['empty_texts']}, уникальных текстов {pig['distinct_texts']}, "
                   f"окон при сканировании {pig['windows_scanned']}. Правило: {c['rule']['decision']}; "
                   f"кандидаты — MinHash LSH (b={c['rule']['candidates']['lsh_bands']}, r={c['rule']['candidates']['lsh_rows']}).\n")
        rows = [[tag, v.get("0", 0), v.get("1", 0)] for tag, v in pig["by_source_label"].items()]
        out.append("\nСостав train.json по полю `source` (метка 0 / 1):\n\n" + _table(["тег PIGuard", "0", "1"], rows))
        keys = set()
        for level in ("document_level", "window_level"):
            for sk, splits_ in c[level].items():
                keys.update((sk, split, label) for split, labels in splits_.items() if split != "total" for label in labels)
        rows = []
        for sk, split, label in sorted(keys):
            d = c["document_level"].get(sk, {}).get(split, {}).get(label)
            wl = c["window_level"].get(sk, {}).get(split, {}).get(label, {})
            rows.append([sk, split, label,
                         d["documents"] if d else "-", d["documents_matched"] if d else "-", d["share"] if d else "-",
                         wl.get("windows", 0), wl.get("windows_matched", 0), wl.get("share", "-"),
                         wl.get("documents_with_matched_window", "-")])
        out.append("\nНаши документы и окна с почти-дубликатом в train.json (по источнику, разбиению E1 и метке; "
                   "метка документа для столбцов документов, метка окна (ТЗ 1.3) для столбцов окон — у документов E6 "
                   "есть чистые окна контекста; `bipia_e6` — варианты E6):\n\n"
                   + _table(["источник", "split", "метка", "документов", "совпало документов", "доля", "окон",
                             "совпало окон", "доля окон", "документов с совпавшим окном"], rows))
        if c["document_level_by_piguard_source"]:
            out.append("\nСовпавшие документы по тегу PIGuard: " + "; ".join(
                f"{sk}: {v}" for sk, v in c["document_level_by_piguard_source"].items()) + "\n")
        out.append("\nТочное вхождение (containment) против записей с тегом источника-двойника:\n")
        for src, t in c["targeted_containment"].items():
            if "note" in t:
                out.append(f"- {src} vs {t['tags']}: {t['note']} (записей PIGuard {t['piguard_texts']}, наших документов {t['documents']})\n")
            else:
                out.append(f"- {src} vs {t['tags']} ({t['piguard_texts']} записей, {t['documents']} наших документов): "
                           f"наш документ внутри записи PIGuard — {t['ours_in_piguard']} ({t['ours_in_piguard_share']}), "
                           f"запись PIGuard внутри нашего документа — {t['piguard_in_ours']} ({t['piguard_in_ours_share']}); "
                           f"по split: {t['by_split']}\n")
        out.append(f"\nОграничение: {c['rule']['unit_window']}.\n")
    out.append("\nКарточки моделей (статично):\n")
    for name, card in c.get("model_cards", {}).items():
        out.append(f"- `{name}` ({card['model']}): обучающие наборы по карточке — {', '.join(card['training_datasets_on_card'])}; "
                   f"пересечение по именам: {card['named_overlap_with_our_sources']}\n")
    out.append("\nУгрозы валидности (ТЗ 3.2):\n")
    out.extend(f"- {t}\n" for t in c.get("threats_to_validity", []))
    return out


def build_audit(documents: pd.DataFrame, windows: pd.DataFrame, cfg: Any, info: dict[str, Any]) -> str:
    """Render the audit as Markdown. ``info`` carries ``stage``, ``smoke``, ``notes`` (skipped sources), ``bipia``
    (loader info), ``dedup`` (dedup.json payload), ``pools`` sizes, ``contamination`` (contamination.json payload)
    and, for a smoke build, ``smoke_rule``."""
    docs = documents.copy()
    metas = _meta(docs)
    docs["variant"] = [m.get("variant", "main") for m in metas]
    docs["length"] = docs["text"].str.len()
    main = cfg.default.get("language", {}).get("main", "en")
    out: list[str] = []
    out.append("# Аудит данных (ТЗ 1.2)\n")
    out.append(f"Стадия: `{info.get('stage', '?')}`; smoke: `{bool(info.get('smoke', False))}`. "
               "Только счётчики: тексты документов в аудит не попадают.\n")
    if info.get("smoke_rule"):
        out.append(f"\nПравило smoke-подвыборки: {info['smoke_rule']}\n")

    # documents by source
    out.append("\n## Документы по источникам\n")
    rows = []
    for src, g in docs.groupby("source", sort=True):
        rows.append([src, len(g), int((g["label"] == 1).sum()), int((g["label"] == 0).sum()),
                     int((g["variant"] == "e6").sum()),
                     int((g["split"] == "train").sum()), int((g["split"] == "val").sum()), int((g["split"] == "test").sum()),
                     int(g.get("dedup_dropped", pd.Series(False, index=g.index)).sum())])
    out.append(_table(["источник", "документов", "инъекции", "чистые", "из них E6-варианты", "train", "val", "test",
                       "выпало по дедупу"], rows))

    # languages
    out.append("\n## Языки (langdetect, сид из конфига)\n")
    rows = []
    for src, g in docs.groupby("source", sort=True):
        top = g["lang"].value_counts().head(6)
        rows.append([src, int((g["lang_stratum"] == main).sum()), int((g["lang_stratum"] != main).sum()),
                     int((g["lang"] == "unk").sum()), ", ".join(f"{k}:{v}" for k, v in top.items())])
    out.append(_table(["источник", main, f"non-{main}", "unk", "топ кодов"], rows))
    deep = docs[docs["source"] == "deep"]
    if not deep.empty:
        ds = [m.get("deepset_split") for m in metas]
        deep = deep.assign(deepset_split=[d for d, s in zip(ds, docs["source"]) if s == "deep"])
        rows = []
        for sp, g in deep.groupby("deepset_split", sort=True):
            rows.append([sp, len(g), int((g["lang"] == "de").sum()), f"{100.0 * (g['lang'] == 'de').mean():.1f} %",
                         int((g["lang"] == "en").sum())])
        rows.append(["all", len(deep), int((deep["lang"] == "de").sum()), f"{100.0 * (deep['lang'] == 'de').mean():.1f} %",
                     int((deep["lang"] == "en").sum())])
        out.append("\nДоля немецкого в deepset:\n\n" + _table(["часть", "документов", "de", "доля de", "en"], rows))

    # lengths
    out.append("\n## Длины нормализованного текста (символы)\n")
    rows = [[src] + _len_stats(g["length"]) for src, g in docs.groupby("source", sort=True)]
    out.append(_table(["источник", "min", "медиана", "среднее", "max"], rows))

    # windows
    out.append("\n## Окна (ТЗ 1.3)\n")
    rows = []
    if len(windows):
        for src, g in windows.groupby("source", sort=True):
            n_docs = g["doc_id"].nunique()
            rows.append([src, len(g), round(len(g) / max(n_docs, 1), 2), int((g["label"] == 1).sum()),
                         int((g["label"] == 0).sum()), int(g["dedup_excluded"].sum())])
    out.append(_table(["источник", "окон", "окон на документ", "окна-инъекции", "окна-чистые", "исключено дедупом"], rows))

    # NotInject
    ni = docs[docs["source"] == "notinject"]
    out.append("\n## Состав NotInject\n")
    if ni.empty:
        out.append("NotInject не загружен.\n")
    else:
        ni = ni.assign(subset=[m.get("subset") for m, s in zip(metas, docs["source"]) if s == "notinject"],
                       category=[m.get("category") for m, s in zip(metas, docs["source"]) if s == "notinject"])
        piv = ni.groupby(["subset", "category"]).size().unstack(fill_value=0)
        rows = [[sub] + [int(piv.loc[sub, c]) for c in piv.columns] + [int((ni[ni["subset"] == sub]["lang_stratum"] != main).sum())]
                for sub in piv.index]
        out.append(_table(["подмножество"] + list(piv.columns) + [f"non-{main}"], rows))

    # BIPIA
    bip = info.get("bipia") or {}
    out.append("\n## BIPIA (ТЗ 1.4)\n")
    if bip.get("tasks"):
        rows = [[t, v["rows"], v["distinct_contexts"], v.get("clusters", v["distinct_contexts"]), v["attack_names"],
                 v["attack_strings"], v["docs_main"], v["docs_e6_extra"]] for t, v in sorted(bip["tasks"].items())]
        out.append(_table(["задача", "строк в test.jsonl", "уникальных контекстов", "кластеров (почти-дубли слиты)",
                           "имён атак", "строк атак", "документов (основной тест)", "документов E6 (доп.)"], rows))
        bd = docs[docs["source"] == "bipia"]
        out.append(f"\nКонтексты: val {bd[bd['split'] == 'val']['cluster_id'].nunique()}, "
                   f"test {bd[bd['split'] == 'test']['cluster_id'].nunique()}; позиции {bip.get('positions')}; "
                   "правило `middle`: начало предложения по regex `[.!?]+[кавычки]*\\s+` (плюс смещение 0), выбранное RNG контекста.\n")
    for task, why in sorted((bip.get("dropped_tasks") or {}).items()):
        out.append(f"- задача `{task}` выпала: {why}\n")

    # dedup
    dd = info.get("dedup") or {}
    out.append("\n## Дедупликация (ТЗ 1.7)\n")
    if dd:
        rule = dd.get("rule") or {}
        out.append(f"- правило: Жаккар >= {rule.get('jaccard')} по символьным {rule.get('shingle')}-граммам, кандидаты MinHash LSH "
                   f"({rule.get('minhash_perm')} перестановок, b={rule.get('lsh_bands')}, r={rule.get('lsh_rows')}, "
                   f"вероятность кандидата на пороге {rule.get('candidate_recall_at_threshold')}), проверка точным Жаккаром\n")
        out.append(f"- окон в тесте: {dd.get('windows_test')}, эталонных (train+val): {dd.get('windows_reference')}, "
                   f"исключено тестовых окон: {dd.get('test_windows_excluded')} "
                   f"(по источникам: {dd.get('test_windows_excluded_by_source')}; пары источников: "
                   f"{dd.get('test_windows_excluded_by_source_pair')})\n")
        out.append(f"- документов выпало: {dd.get('documents_dropped')}; по источнику/варианту/метке: "
                   f"{dd.get('documents_dropped_by_source_variant')}\n")

    # pools
    pools = info.get("pools") or {}
    if pools:
        out.append("\n## Пулы негативов (ТЗ 1.8)\n")
        for name in ("p_val", "p_test"):
            p = pools.get(name, {})
            out.append(f"- {name}: {p.get('n')} документов {p.get('by_source')}; цель {p.get('target_min_docs')}, "
                       f"{'достигнута' if p.get('meets_target') else 'не достигнута'}"
                       f"{'; deepset-часть: ' + str(p.get('p_val_deepset')) if name == 'p_val' else ''}\n")

    # contamination (ТЗ 3.2)
    out.extend(_contamination_section(info.get("contamination") or {}))

    notes = info.get("notes") or []
    out.append("\n## Пропущенные источники и замечания\n")
    if notes:
        out.extend(f"- {n}\n" for n in notes)
    else:
        out.append("- нет\n")
    return "".join(out)
