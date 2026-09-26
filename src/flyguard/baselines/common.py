"""Shared protocol and helpers of the baseline window scorers (ТЗ 3.1, 3.2; docs/design.md §6).

Every baseline follows the same three-method protocol as the fly's readouts so that ``experiments.engine`` can
treat them uniformly: ``name`` (the detector key used in results files, e.g. ``tfidf_lr``), ``fit(X_train,
y_train, X_val=None, y_val=None, groups=None)`` and ``score(X)`` returning one monotone score per *window* in
[0, 1]. Document scores (max over windows, ТЗ 1.3) are computed by ``eval.metrics.doc_scores``, never here. The
baselines do not compute features: the nose (``flyguard.nose``) hands them N16k or N51-svd matrices; the only
statistic a baseline fits on its own is TF-IDF's idf, and that one on C_unl (ТЗ 1.9). This module is below
``eval`` in the layer order, so validation AUC is taken from scikit-learn directly instead of ``eval.metrics``.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from flyguard.config import Configs
from flyguard.data.windows import text_hash  # one definition of the cache key (design §2), re-exported here
from flyguard.readout import make_logistic  # one logistic regression for MBON, LRSvd and TfidfLR (ТЗ 2.4, 3.1)

__all__ = ["WindowScorer", "text_hash", "as_matrix", "same_kind", "l2_normalize_rows", "cosine_similarity_blocks",
           "clip01", "safe_auc", "make_logreg", "grid_middle", "cv_feasible", "select_c", "warn_once"]


@runtime_checkable
class WindowScorer(Protocol):
    """Window-level detector protocol shared by baselines and readouts (docs/design.md §6).

    ``X`` is whatever lives in the detector's input space: sparse N16k rows for the lexical baselines, dense
    N51-svd rows for ``LRSvd``, raw window strings for ``RegexScorer`` and ``GuardModel``. ``groups`` is the
    per-train-window group key (``cluster_id``, else ``doc_id``): overlapping windows of one document must never
    be split across the folds that choose C (ТЗ 1.10 clusters), so the scorers that cross-validate require it
    when no validation set is given; the others accept and ignore it so the engine can pass it uniformly.
    """

    name: str

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "WindowScorer": ...

    def score(self, X: Any) -> np.ndarray: ...


def as_matrix(X: Any) -> sp.csr_matrix | np.ndarray:
    """Coerce detector input to float64 CSR (if sparse) or a 2-D float64 ndarray (if dense) (design §6 inputs).

    The baselines accept both because the nose emits CSR for N16k and ndarray for N51-svd (design §5), and the
    engine may hand either one to any scorer during E5/E6 controls.
    """
    if sp.issparse(X):
        return sp.csr_matrix(X, dtype=np.float64)
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def same_kind(Xq: Any, Xt: Any) -> tuple[Any, Any]:
    """Make a query and a reference matrix the same kind (both CSR if either is sparse) so products are defined.

    Needed because kNN/centroid (ТЗ 3.1) may see sparse N16k train rows and dense query rows in E5/E6 controls.
    """
    Xq, Xt = as_matrix(Xq), as_matrix(Xt)
    if sp.issparse(Xq) != sp.issparse(Xt):
        Xq = sp.csr_matrix(Xq) if not sp.issparse(Xq) else Xq
        Xt = sp.csr_matrix(Xt) if not sp.issparse(Xt) else Xt
    return Xq, Xt


def l2_normalize_rows(X: Any) -> sp.csr_matrix | np.ndarray:
    """Unit-L2 rows (zero rows stay zero), sparse or dense; cosine similarity is then a plain dot product."""
    X = as_matrix(X)
    if sp.issparse(X):
        X = X.copy()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms) @ X)
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1.0
    return X / norms[:, None]


def cosine_similarity_blocks(Xq_unit: Any, Xt_unit: Any, block_rows: int | None = None,
                             target_cells: float = 2e7) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, S)`` dense blocks of cosine similarities between unit-normalised rows.

    Blocks keep the dense ``n_query x n_train`` product at about ``target_cells`` floats (160 MB) so that KNN over
    tens of thousands of train windows fits in the operator's RAM; the inputs must already be unit-normalised.
    """
    Xq_unit, Xt_unit = same_kind(Xq_unit, Xt_unit)
    n_q, n_t = Xq_unit.shape[0], Xt_unit.shape[0]
    if block_rows is None:
        block_rows = max(1, int(target_cells // max(n_t, 1)))
    XtT = Xt_unit.T
    for start in range(0, n_q, block_rows):
        S = Xq_unit[start:start + block_rows] @ XtT
        yield start, (S.toarray() if sp.issparse(S) else np.asarray(S))


def clip01(s: Any) -> np.ndarray:
    """Clip a score vector to [0, 1], the window-score range of the protocol (design §6), and return float64."""
    return np.clip(np.asarray(s, dtype=np.float64).ravel(), 0.0, 1.0)


def safe_auc(y: Any, s: Any) -> float | None:
    """ROC AUC or ``None`` when only one class is present (E2 shots=1 validation folds can be single-class)."""
    y = np.asarray(y).astype(int).ravel()
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=np.float64).ravel()))


def make_logreg(C: float, seed: int, cfg: Configs | None = None) -> LogisticRegression:
    """The project's one logistic regression, for TF-IDF + LR and LR on N51-svd (ТЗ 3.1).

    H1b pairs ``LRSvd`` with the linear MBON readout to compare *input spaces*, so both sides must fit the same
    estimator (solver, iterations, tolerance, class weights, penalty). That estimator is built only by
    ``flyguard.readout.make_logistic`` from ``readout.linear`` of ``configs/default.yaml``; this wrapper exists so
    the baselines have one call site and so a partial ``Configs`` without a ``readout`` block (unit tests of the
    baselines' own config keys) falls back to the repository defaults instead of failing.
    """
    lin = cfg.default.get("readout", {}).get("linear") if cfg is not None else None
    return make_logistic(float(C), int(seed), cfg if lin else None)


def grid_middle(grid: Sequence[float]) -> float:
    """The default C when nothing can be validated (E2 shots=1, ТЗ 3.1 "C по валидации" impossible): the middle
    of the configured grid, a documented fallback rather than a hidden constant."""
    grid = list(grid)
    return float(grid[len(grid) // 2])


def cv_feasible(y_train: np.ndarray, groups: np.ndarray, cv_folds: int) -> bool:
    """True when a ``cv_folds``-fold grouped split can score AUC: both classes present and each class spread over
    at least ``cv_folds`` distinct groups (otherwise some test folds would miss a class, or ``StratifiedGroupKFold``
    would have fewer groups than splits — E2 with one or ten documents per class)."""
    y_train = np.asarray(y_train).astype(int).ravel()
    if len(np.unique(y_train)) < 2:
        return False
    groups = np.asarray(groups)
    return all(len(np.unique(groups[y_train == c])) >= cv_folds for c in (0, 1))


def select_c(make_model: Callable[[float], Any], X_train: Any, y_train: Any, X_val: Any = None,
             y_val: Any = None, grid: Sequence[float] = (1.0,), seed: int = 0, cv_folds: int = 3,
             groups: Any = None) -> tuple[float, str, dict[str, float | None]]:
    """Choose C on validation AUC only (ТЗ 3.1 "C по валидации"); returns ``(C, source, auc_by_C)``.

    Order of preference, each recorded in ``source`` so that results files can say where C came from:
    ``val`` — AUC on the given validation set (both classes present); ``cv`` — mean AUC over a seeded
    ``StratifiedGroupKFold(cv_folds)`` of the train set when no usable validation set is given; ``default`` — the
    middle of the grid when neither is possible (E2 shots=1 trains on two rows). The cross-validation folds are
    split by ``groups`` (the engine passes ``cluster_id``, else ``doc_id``, per train window), never by window:
    consecutive 256-character windows of one document overlap by 64 characters (ТЗ 1.3) and paraphrases, BIPIA
    pairs and AgentDojo steps share clusters (ТЗ 1.10), so a window-level split would put near-copies of a test
    fold's rows in its training part and inflate every C's AUC. For that reason the ``cv`` route *refuses* to run
    without ``groups`` (``ValueError``) instead of silently degrading; ``groups`` is not needed when a validation
    set decides or when the default applies. Ties are broken towards the smallest C (strongest regularisation),
    the conservative choice on a 546-document train set.
    """
    grid = [float(c) for c in grid]
    y_train = np.asarray(y_train).astype(int).ravel()
    if groups is not None:
        groups = np.asarray(groups).ravel()
        if groups.shape[0] != y_train.shape[0]:
            raise ValueError(f"{groups.shape[0]} groups for {y_train.shape[0]} train rows")
    table: dict[str, float | None] = {}
    if X_val is not None and y_val is not None and safe_auc(y_val, np.zeros(len(np.asarray(y_val)))) is None:
        y_val = None  # single-class validation set is useless for AUC
    if X_val is not None and y_val is not None and len(np.unique(y_train)) == 2:
        for C in grid:
            model = make_model(C).fit(X_train, y_train)
            table[str(C)] = safe_auc(y_val, model.predict_proba(X_val)[:, 1])
        source = "val"
    else:
        counts = np.bincount(y_train, minlength=2)
        if len(np.unique(y_train)) < 2 or counts.min() < cv_folds:
            return grid_middle(grid), "default", {str(c): None for c in grid}
        if groups is None:
            raise ValueError("select_c: cross-validated C needs per-window groups (cluster_id, else doc_id); "
                             "a window-level split leaks overlapping windows across folds (ТЗ 1.3, 1.10)")
        if not cv_feasible(y_train, groups, cv_folds):
            return grid_middle(grid), "default", {str(c): None for c in grid}
        sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=int(seed))
        for C in grid:
            aucs = []
            for tr, te in sgkf.split(np.zeros(len(y_train)), y_train, groups):
                if len(np.unique(y_train[tr])) < 2:
                    continue  # a fold whose training part lost a class cannot be fitted; its AUC is skipped
                model = make_model(C).fit(_rows(X_train, tr), y_train[tr])
                a = safe_auc(y_train[te], model.predict_proba(_rows(X_train, te))[:, 1])
                if a is not None:
                    aucs.append(a)
            table[str(C)] = float(np.mean(aucs)) if aucs else None
        source = "cv"
    valid = [(table[str(C)], C) for C in grid if table[str(C)] is not None]
    if not valid:
        return grid_middle(grid), "default", table
    best_auc = max(a for a, _ in valid)
    best_C = min(C for a, C in valid if a == best_auc)
    return best_C, source, table


def _rows(X: Any, idx: np.ndarray) -> Any:
    return X[idx]


def warn_once(message: str) -> None:
    """A loud, non-fatal warning for a baseline fallback (e.g. idf without C_unl, ТЗ 1.9), so that a deviation is
    never silent (CLAUDE.md journals rule)."""
    warnings.warn(message, RuntimeWarning, stacklevel=3)
