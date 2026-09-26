"""``data/manifests/audit.md`` (ТЗ 1.2): documents, classes, languages, lengths, windows per source, share of
German in deepset, NotInject composition, BIPIA tasks dropped, skipped sources. Counts only: the audit never
quotes a document (data-safety rule), and it carries no timestamp so idempotent rebuilds produce identical files.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd


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


def build_audit(documents: pd.DataFrame, windows: pd.DataFrame, cfg: Any, info: dict[str, Any]) -> str:
    """Render the audit as Markdown. ``info`` carries ``stage``, ``smoke``, ``notes`` (skipped sources), ``bipia``
    (loader info), ``dedup`` (dedup.json payload) and ``pools`` sizes."""
    docs = documents.copy()
    metas = _meta(docs)
    docs["variant"] = [m.get("variant", "main") for m in metas]
    docs["length"] = docs["text"].str.len()
    main = cfg.default.get("language", {}).get("main", "en")
    out: list[str] = []
    out.append("# Аудит данных (ТЗ 1.2)\n")
    out.append(f"Стадия: `{info.get('stage', '?')}`; smoke: `{bool(info.get('smoke', False))}`. "
               "Только счётчики: тексты документов в аудит не попадают.\n")

    # documents by source
    out.append("## Документы по источникам\n")
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
        out.append(f"- окон в тесте: {dd.get('windows_test')}, эталонных (train+val): {dd.get('windows_reference')}, "
                   f"исключено тестовых окон: {dd.get('test_windows_excluded')} "
                   f"(по источникам: {dd.get('test_windows_excluded_by_source')})\n")
        out.append(f"- документов выпало: {dd.get('documents_dropped')}; по источнику/варианту/метке: "
                   f"{dd.get('documents_dropped_by_source_variant')}\n")

    # pools
    pools = info.get("pools") or {}
    if pools:
        out.append("\n## Пулы негативов (ТЗ 1.8)\n")
        for name in ("p_val", "p_test"):
            p = pools.get(name, {})
            out.append(f"- {name}: {p.get('n')} документов {p.get('by_source')}; цель {p.get('target_min_docs')}, "
                       f"{'достигнута' if p.get('meets_target') else 'не достигнута'}\n")

    notes = info.get("notes") or []
    out.append("\n## Пропущенные источники и замечания\n")
    if notes:
        out.extend(f"- {n}\n" for n in notes)
    else:
        out.append("- нет\n")
    return "".join(out)
