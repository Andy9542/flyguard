"""Shared protocol and helpers of the baseline window scorers (ТЗ 3.1, 3.2; docs/design.md §6).

Every baseline follows the same three-method protocol as the fly's readouts so that ``experiments.engine`` can
treat them uniformly: ``name`` (the detector key used in results files, e.g. ``tfidf_lr``), ``fit(X_train,
y_train, X_val=None, y_val=None)`` and ``score(X)`` returning one monotone score per *window* in [0, 1]. Document
scores (max over windows, ТЗ 1.3) are computed by ``eval.metrics.doc_scores``, never here. The baselines do not
compute features: the nose (``flyguard.nose``) hands them N16k or N51-svd matrices; the only statistic a baseline
fits on its own is TF-IDF's idf, and that one on C_unl (ТЗ 1.9). This module is below ``eval`` in the layer
order, so validation AUC is taken from scikit-learn directly instead of ``eval.metrics``.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp
import xxhash
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@runtime_checkable
class WindowScorer(Protocol):
    """Window-level detector protocol shared by baselines and readouts (docs/design.md §6).

    ``X`` is whatever lives in the detector's input space: sparse N16k rows for the lexical baselines, dense
    N51-svd rows for ``LRSvd``, raw window strings for ``RegexScorer`` and ``GuardModel``.
    """

    name: str

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> "WindowScorer": ...

    def score(self, X: Any) -> np.ndarray: ...


def text_hash(text: str) -> str:
    """Key of the transformer score caches: xxhash64 (seed 0) hex digest of the UTF-8 window text.

    docs/design.md §2 defines ``windows.parquet.text_hash`` as "xxhash64 hex of ``text``"; ``flyguard.data.windows``
    must produce exactly this value so that cached scores (ТЗ 3.2 "оценки кешируются по хешу окна") can be joined
    back to windows without re-hashing. Seed 0 is xxhash's default and is written down here so that it is not an
    implicit choice.
    """
    return xxhash.xxh64(text.encode("utf-8"), seed=0).hexdigest()


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


def make_logreg(C: float, seed: int, max_iter: int = 2000) -> LogisticRegression:
    """L2 logistic regression with balanced class weights (ТЗ 2.4 MBON / 3.1 TF-IDF + LR).

    L2 is scikit-learn's default (``penalty`` is deprecated in 1.9, so it is not passed); lbfgs is deterministic
    for these sizes and handles sparse input; ``class_weight='balanced'`` follows ``configs/default.yaml``
    ``readout.linear.class_weight`` because deepset train is 343/203 and E2 subsamples are skewed further.
    """
    return LogisticRegression(C=float(C), class_weight="balanced", solver="lbfgs", max_iter=max_iter,
                              random_state=int(seed))


def grid_middle(grid: Sequence[float]) -> float:
    """The default C when nothing can be validated (E2 shots=1, ТЗ 3.1 "C по валидации" impossible): the middle
    of the configured grid, a documented fallback rather than a hidden constant."""
    grid = list(grid)
    return float(grid[len(grid) // 2])


def select_c(make_model: Callable[[float], Any], X_train: Any, y_train: Any, X_val: Any = None,
             y_val: Any = None, grid: Sequence[float] = (1.0,), seed: int = 0,
             cv_folds: int = 3) -> tuple[float, str, dict[str, float | None]]:
    """Choose C on validation AUC only (ТЗ 3.1 "C по валидации"); returns ``(C, source, auc_by_C)``.

    Order of preference, each recorded in ``source`` so that results files can say where C came from:
    ``val`` — AUC on the given validation set (both classes present); ``cv`` — mean AUC of a seeded stratified
    ``cv_folds``-fold split of the train set when no usable validation set is given but every class has at least
    ``cv_folds`` examples; ``default`` — the middle of the grid otherwise (E2 shots=1 trains on two rows). Ties are
    broken towards the smallest C (strongest regularisation), the conservative choice on a 546-document train set.
    """
    grid = [float(c) for c in grid]
    y_train = np.asarray(y_train).astype(int).ravel()
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
        if len(np.unique(y_train)) == 2 and counts.min() >= cv_folds:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(seed))
            for C in grid:
                aucs = []
                for tr, te in skf.split(np.zeros(len(y_train)), y_train):
                    model = make_model(C).fit(_rows(X_train, tr), y_train[tr])
                    a = safe_auc(y_train[te], model.predict_proba(_rows(X_train, te))[:, 1])
                    if a is not None:
                        aucs.append(a)
                table[str(C)] = float(np.mean(aucs)) if aucs else None
            source = "cv"
        else:
            return grid_middle(grid), "default", {str(c): None for c in grid}
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
