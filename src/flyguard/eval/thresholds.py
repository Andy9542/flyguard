"""Threshold records of ТЗ 2.5: every threshold is stored with its value, the data it was set on (``source``), the
target operating point (``target``) and the number of examples (``n``), as the acceptance criteria demand ("У каждого
порога записаны источник, целевая точка и число примеров"). Results files keep them under ``"thresholds"``
(docs/design.md §9).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from flyguard.config import Configs, load_configs
from flyguard.eval.metrics import fpr_at_threshold, tau_for_tpr, threshold_for_fpr, tpr_at_threshold


def threshold_record(value: float, source: str, target: str, n: int, **extra: Any) -> dict[str, Any]:
    """The record shape ``{value, source, target, n}`` of docs/design.md §9; ``extra`` adds e.g. the achieved rate."""
    rec: dict[str, Any] = {"value": float(value), "source": str(source), "target": str(target), "n": int(n)}
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec


def fpr_target_for_pool(n_pool: int, cfg: Configs | None = None) -> float | None:
    """Carrier rule of ТЗ Этап 0 / 2.5 for the frozen τ_FPR: pool >= ``pool_min_docs.fpr_1pct`` (2000) -> FPR 1 %,
    >= ``pool_min_docs.fpr_5pct`` (500) -> 5 %, smaller -> ``None`` (only ROC-AUC is reported). Below 500 negatives a
    1 % point rests on a handful of documents, so the ТЗ withdraws the TPR@FPR metric altogether."""
    cfg = cfg or load_configs()
    th = cfg.default["thresholds"]
    if n_pool >= int(th["pool_min_docs"]["fpr_1pct"]):
        return float(th["fpr_targets"]["primary"])
    if n_pool >= int(th["pool_min_docs"]["fpr_5pct"]):
        return float(th["fpr_targets"]["fallback"])
    return None


def tau_fpr_record(neg_pool_scores: Sequence[float] | np.ndarray, fpr: float | None = None,
                   source: str = "P_val", cfg: Configs | None = None, *,
                   n_test_pool: int | None = None) -> dict[str, Any] | None:
    """Frozen τ_FPR (ТЗ 2.5): set on the validation negative pool P_val at the FPR target chosen by E0.

    The 1 % / 5 % / AUC-only choice is the rule of ТЗ Этап 0 on the size of the *test* pool |P_test| ("TPR при FPR 1%
    определён при |P_test| ≥ 2 000; 500–1 999 → 5%"), never on the pool the threshold is set on: pass ``fpr``
    (the ``fpr_target`` of ``results/power.json``) or ``n_test_pool`` (|P_test|, from which the same rule
    :func:`fpr_target_for_pool` is applied here, as ``power.carrier_rule`` does); giving neither is an error.
    Returns ``None`` when the rule withdraws the metric (AUC only). The record keeps the target as ``"fpr<=0.01"``,
    the FPR actually achieved on the pool, ``n`` = |P_val| and, when known, ``n_test_pool`` as the provenance of the
    target."""
    neg = np.asarray(neg_pool_scores, dtype=float)
    if fpr is None:
        if n_test_pool is None:
            raise ValueError("tau_fpr_record needs the FPR target: pass fpr (results/power.json fpr_target) or "
                             "n_test_pool (|P_test|, the pool the ТЗ Этап 0 rule is defined on)")
        fpr = fpr_target_for_pool(int(n_test_pool), cfg)
        if fpr is None:
            return None
    tau = threshold_for_fpr(neg, float(fpr))
    return threshold_record(tau, source, f"fpr<={float(fpr):g}", int(neg.size),
                            achieved_fpr=fpr_at_threshold(neg, tau),
                            n_test_pool=None if n_test_pool is None else int(n_test_pool))


def tau_tpr_record(pos_scores: Sequence[float] | np.ndarray, source: str, tpr: float | None = None,
                   cfg: Configs | None = None) -> dict[str, Any]:
    """τ_90(S) (ТЗ 2.5): threshold reaching TPR >= ``tpr`` on the *test* injections of source S (primary S = deep,
    secondary dojo, τ_80 in E6 -- the rates live in ``thresholds.tau_tpr``). It is an equal-sensitivity point for FPR
    comparisons (H2), not a frozen operating threshold, so it is legitimately set on test positives."""
    cfg = cfg or load_configs()
    tpr = float(cfg.default["thresholds"]["tau_tpr"]["tpr"]) if tpr is None else float(tpr)
    pos = np.asarray(pos_scores, dtype=float)
    tau = tau_for_tpr(pos, tpr)
    return threshold_record(tau, source, f"tpr>={tpr:g}", int(pos.size), achieved_tpr=tpr_at_threshold(pos, tau))


def default_threshold_record(value: float, detector: str) -> dict[str, Any]:
    """The industrial detectors' own default threshold (ТЗ 2.5 "справочно"): recorded, never used for verdicts."""
    return threshold_record(value, f"default:{detector}", "default", 0)


def contract_threshold(benign_val_episode_scores: Sequence[float] | np.ndarray, cfg: Configs | None = None,
                       source: str = "validation_benign") -> dict[str, Any]:
    """Contract §7 threshold: on validation ``benign`` episodes (their max scores), at most one false alarm per 100
    episodes; fewer than ``stats.contract.threshold_min_val_episodes`` (200) validation episodes -> one per 20,
    flagged (``note``). ``n`` is ``threshold_n_benign`` of the contract CSV. Rates come from
    ``thresholds.fpr_targets`` (0.01 / 0.05); the cut-off is the contract's, frozen in ``configs/default.yaml``."""
    cfg = cfg or load_configs()
    rates = cfg.default["thresholds"]["fpr_targets"]
    min_n = int(cfg.default["stats"]["contract"]["threshold_min_val_episodes"])
    scores = np.asarray(benign_val_episode_scores, dtype=float)
    n = int(scores.size)
    enough = n >= min_n
    fpr = float(rates["primary"]) if enough else float(rates["fallback"])
    tau = threshold_for_fpr(scores, fpr)
    note = None if enough else (f"fewer than {min_n} validation benign episodes: one false alarm per "
                                f"{round(1 / fpr)} (contract §7)")
    return threshold_record(tau, source, f"fa_per_100<={100 * fpr:g}", n, achieved_fpr=fpr_at_threshold(scores, tau),
                            note=note, min_val_episodes=min_n)
