"""Regex baseline (ТЗ 3.1 "Регулярки": 15–30 patterns, committed before the first test read, one ROC point).

``configs/regex_patterns.txt`` holds one Python regular expression per line (``#`` lines are comments), compiled
with ``re.IGNORECASE``; a window scores 1 when any pattern matches and 0 otherwise, so the regex detector is a
single point on the ROC rather than a curve. The file was authored from deepset **train** only: ``rank_ngrams``
ranks lower-cased word n-grams by precision and recall on train, the frequent high-precision n-grams were then
generalised by hand into the patterns, and ``train_stats`` reports the resulting train TPR/FPR (per language, since
German is about 40 % of deepset). ``configs/regex_patterns.txt`` is part of ``flyguard.config.CONFIG_FILES`` and
therefore of the frozen config hash; the acceptance criterion "коммит с regex_patterns.txt старше первого чтения
тестовых файлов" is checked by ``scripts/check_acceptance.py`` from git history and ``logs/data_access.log``.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from flyguard.config import ROOT, Configs, load_configs

_WORD = re.compile(r"\w+", re.UNICODE)


def patterns_path(cfg: Configs | None = None, root: Path = ROOT) -> Path:
    """Path of the ТЗ 3.1 pattern file from ``cfg.default['baselines']['regex_file']`` (never hard-coded)."""
    cfg = cfg or load_configs(root)
    rel = Path(cfg.default["baselines"]["regex_file"])
    return rel if rel.is_absolute() else root / rel


def read_pattern_lines(path: str | Path) -> list[str]:
    """Raw pattern strings of the ТЗ 3.1 file (design §6: one regex per line, ``#`` comments): blank lines and
    lines starting with ``#`` are dropped, nothing else is, so a pattern may contain ``#`` itself."""
    lines: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return lines


def compile_patterns(lines: Sequence[str]) -> list[re.Pattern[str]]:
    """Compile every pattern case-insensitively (ТЗ 3.1); a malformed line raises ``re.error`` with its index."""
    compiled = []
    for i, line in enumerate(lines):
        try:
            compiled.append(re.compile(line, re.IGNORECASE))
        except re.error as exc:  # pragma: no cover - exercised only by a broken file
            raise re.error(f"pattern {i + 1} does not compile: {exc}", line, exc.pos) from exc
    return compiled


def load_patterns(path: str | Path | None = None, cfg: Configs | None = None) -> list[re.Pattern[str]]:
    """Load and compile ``configs/regex_patterns.txt`` (or ``path``)."""
    return compile_patterns(read_pattern_lines(path or patterns_path(cfg)))


class RegexScorer:
    """Window scorer over raw strings: 1.0 if any pattern matches, else 0.0 (ТЗ 3.1 "точка на ROC").

    ``fit`` is a no-op because the patterns are frozen in the config file; the class exists so that the regex
    baseline follows the same ``WindowScorer`` protocol as every other detector in ``experiments.engine``.
    """

    name = "regex"

    def __init__(self, path: str | Path | None = None, patterns: Sequence[str] | None = None,
                 cfg: Configs | None = None):
        self.lines = list(patterns) if patterns is not None else read_pattern_lines(path or patterns_path(cfg))
        self.patterns = compile_patterns(self.lines)

    def fit(self, X_train: Any = None, y_train: Any = None, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "RegexScorer":
        """No-op: the patterns are frozen in the config file (``groups`` accepted, unused)."""
        return self

    def matches(self, text: str) -> list[int]:
        """Indices of the patterns matching ``text`` (for audits; never print the text itself)."""
        return [i for i, p in enumerate(self.patterns) if p.search(text)]

    def score(self, X: Sequence[str]) -> np.ndarray:
        return np.array([1.0 if any(p.search(t) for p in self.patterns) else 0.0 for t in X], dtype=np.float64)

    def hit_matrix(self, X: Sequence[str]) -> np.ndarray:
        """Boolean ``[n_texts, n_patterns]`` matrix of matches, used by ``train_stats``."""
        return np.array([[bool(p.search(t)) for p in self.patterns] for t in X], dtype=bool).reshape(
            len(X), len(self.patterns))


def rank_ngrams(texts: Sequence[str], labels: Sequence[int], n: int = 1, min_pos: int = 4,
                min_precision: float = 0.0) -> pd.DataFrame:
    """Rank lower-cased word n-grams of a labelled corpus by precision and recall (document frequency).

    This is the code step of ТЗ 3.1's pattern authoring: run on deepset train only, it lists n-grams frequent in
    injections and rare in benign texts, which were then hand-generalised into regexes (e.g. ``forget everything``
    -> ``\\b(forget|ignore|disregard) (the |all |about )?(everything|all|above|previous...)\\b``). Columns:
    ``ngram, pos, neg, precision, recall``; sorted by precision then pos count, descending.
    """
    labels = np.asarray(labels).astype(int).ravel()
    pos_c: collections.Counter[tuple[str, ...]] = collections.Counter()
    neg_c: collections.Counter[tuple[str, ...]] = collections.Counter()
    for text, y in zip(texts, labels):
        words = _WORD.findall(text.lower())
        grams = set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()
        (pos_c if y == 1 else neg_c).update(grams)
    n_pos = max(int((labels == 1).sum()), 1)
    rows = []
    for gram, k in pos_c.items():
        if k < min_pos:
            continue
        fp = neg_c.get(gram, 0)
        prec = k / (k + fp)
        if prec >= min_precision:
            rows.append({"ngram": " ".join(gram), "pos": int(k), "neg": int(fp), "precision": float(prec),
                         "recall": float(k / n_pos)})
    df = pd.DataFrame(rows, columns=["ngram", "pos", "neg", "precision", "recall"])
    return df.sort_values(["precision", "pos", "ngram"], ascending=[False, False, True]).reset_index(drop=True)


def train_stats(scorer: RegexScorer, texts: Sequence[str], labels: Sequence[int],
                langs: Sequence[str] | None = None) -> dict[str, Any]:
    """TPR/FPR of the pattern set on a labelled corpus, overall, per language and per pattern.

    Reported in the module report for deepset train (ТЗ 3.1). ``per_pattern[i] = {pos, neg, unique}`` where
    ``unique`` counts positives covered by that pattern alone, which is what decides whether a pattern earns its
    line. Texts are never included in the output.
    """
    labels = np.asarray(labels).astype(int).ravel()
    H = scorer.hit_matrix(list(texts))
    hit = H.any(axis=1) if H.size else np.zeros(len(labels), dtype=bool)
    pos, neg = labels == 1, labels == 0
    out: dict[str, Any] = {
        "n_patterns": len(scorer.patterns),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "tpr": float(hit[pos].mean()) if pos.any() else None,
        "fpr": float(hit[neg].mean()) if neg.any() else None,
        "per_pattern": [{"pos": int(H[pos, i].sum()), "neg": int(H[neg, i].sum()),
                         "unique": int((H[pos, i] & (H[pos].sum(axis=1) == 1)).sum())}
                        for i in range(H.shape[1])] if H.size else [],
    }
    if langs is not None:
        langs_arr = np.asarray(list(langs))
        by_lang = {}
        for lang in sorted(set(langs_arr.tolist())):
            m = langs_arr == lang
            by_lang[lang] = {
                "n_pos": int((m & pos).sum()), "n_neg": int((m & neg).sum()),
                "tpr": float(hit[m & pos].mean()) if (m & pos).any() else None,
                "fpr": float(hit[m & neg].mean()) if (m & neg).any() else None,
            }
        out["by_lang"] = by_lang
    return out
