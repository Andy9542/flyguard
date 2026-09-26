"""Baselines of ТЗ 3.1–3.2 (docs/design.md §6): regex, lexical models on the nose's features, transformer guards.

Importing this package does not import torch or transformers; ``GuardModel`` loads them lazily.
"""
from flyguard.baselines.common import WindowScorer, select_c, text_hash
from flyguard.baselines.lexical import KNN, LRSvd, NearestCentroid, TfidfLR, make_lexical_scorers
from flyguard.baselines.regex import RegexScorer, load_patterns, rank_ngrams, train_stats
from flyguard.baselines.transformers_guard import GuardModel, ScoreCache, resolve_positive_index, token_windows

__all__ = ["WindowScorer", "select_c", "text_hash", "KNN", "LRSvd", "NearestCentroid", "TfidfLR",
           "make_lexical_scorers", "RegexScorer", "load_patterns", "rank_ngrams", "train_stats", "GuardModel",
           "ScoreCache", "resolve_positive_index", "token_windows"]
