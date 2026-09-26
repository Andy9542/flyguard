"""Readouts on the KC code (ТЗ 2.4, preregistration §3.4): the Bloom filter of FlyNN and the linear MBON.

Both consume a binary csr code ``Z [n, m]`` (from :func:`flyguard.fly.fly_code` or ``sign_code``) and return a
score in [0, 1] per row. Hyperparameters (gamma, C) are chosen by validation AUC only (ТЗ "Честность
эксперимента"); the grids live in ``configs/default.yaml`` (``readout.bloom.gammas``, ``readout.linear.C_grid``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from flyguard.config import load_configs


@lru_cache(maxsize=1)
def _readout_cfg() -> dict[str, Any]:
    return load_configs().default["readout"]


def _as_code(Z: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(Z, dtype=np.float32)


def _labels(y: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y).astype(np.int64).ravel()
    if y.shape[0] != n:
        raise ValueError(f"{y.shape[0]} labels for {n} rows")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y


def balance_indices(y: Sequence[int] | np.ndarray, seed: int) -> np.ndarray:
    """Indices of a class-balanced subset: the majority class is subsampled without replacement to the size of
    the minority class with seed ``subsample`` (ТЗ 2.4 "Классы подвыбираются до меньшего"). Sorted, so the
    training order is data order; the Bloom update commutes anyway.
    """
    y = np.asarray(y).astype(np.int64).ravel()
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    present = [g for g in groups if g.size]
    if not present:
        return np.zeros(0, dtype=np.int64)
    n_min = min(g.size for g in present)
    keep = [np.sort(rng.choice(g, n_min, replace=False)) if g.size > n_min else g for g in groups]
    return np.sort(np.concatenate(keep))


class BloomReadout:
    """Bloom-filter readout of FlyNN (ТЗ 2.4 "Bloom (FlyNN)").

    Per class c a filter F_c in (0, 1]^m initialised to ones; an example of class c with code z applies
    F_c[i] <- gamma · F_c[i] for the active i, so after training F_c[i] = gamma^{n_c(i)} with n_c(i) the number
    of class-c training codes in which cell i fired (the updates commute, so the closed form equals the
    sequential rule and is computed with one sparse column sum). Familiarity phi_c(z) = 1 - mean_{i in supp z}
    F_c[i] and the score s = (phi_1 - phi_0 + 1)/2 in [0, 1]. ``fit`` balances the classes by subsampling the
    majority to the minority with ``seed_subsample``; ``normalized=True`` (E6) instead uses every example with
    the exponent n_c(i) · N_min / N_c, N_c the full class counts, so both classes weigh equally without
    discarding data (our reading of the ТЗ formula). gamma = 0 saturates after ~100 examples per class (a cell
    is zeroed by its first activation); gamma -> 1 turns the filter into a log counter, i.e. naive Bayes on the
    code — they are different models and are reported under different names.
    """

    def __init__(self, m: int, k: int, gamma: float, seed_subsample: int = 0, normalized: bool = False) -> None:
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        self.m = int(m)
        self.k = int(k)
        self.gamma = float(gamma)
        self.seed_subsample = int(seed_subsample)
        self.normalized = bool(normalized)
        self.F_ = np.ones((2, self.m), dtype=np.float32)
        self.n_seen_ = np.zeros(2, dtype=np.int64)
        self.class_counts_: np.ndarray | None = None
        self.balanced_counts_: np.ndarray | None = None
        self.balanced_index_: np.ndarray | None = None

    def _check(self, Z: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
        Zc = _as_code(Z)
        if Zc.shape[1] != self.m:
            raise ValueError(f"code has {Zc.shape[1]} cells, readout has {self.m}")
        return Zc

    def _apply(self, c: int, exponent: np.ndarray) -> None:
        # float64 powers: 0.5**200 underflows float32, and 0.0**0 == 1.0 keeps untouched cells at one
        self.F_[c] = (self.F_[c].astype(np.float64) * np.power(self.gamma, exponent)).astype(np.float32)

    def partial_fit(self, Z: sp.spmatrix | np.ndarray, y: Sequence[int] | np.ndarray) -> "BloomReadout":
        """The raw sequential rule F_c[i] <- gamma · F_c[i] for every active cell of every example, no balancing
        (online / few-shot use; ``fit`` calls it on the balanced subset)."""
        Zc = self._check(Z)
        yc = _labels(y, Zc.shape[0])
        for c in (0, 1):
            rows = np.flatnonzero(yc == c)
            if rows.size:
                n_c = np.asarray(Zc[rows].sum(axis=0)).ravel().astype(np.float64)
                self._apply(c, n_c)
                self.n_seen_[c] += rows.size
        return self

    def fit(self, Z: sp.spmatrix | np.ndarray, y: Sequence[int] | np.ndarray) -> "BloomReadout":
        """Reset the filters and train on the class-balanced subset (or, ``normalized``, on everything with
        class-normalised exponents)."""
        Zc = self._check(Z)
        yc = _labels(y, Zc.shape[0])
        self.F_.fill(1.0)
        self.n_seen_[:] = 0
        self.class_counts_ = np.bincount(yc, minlength=2)
        if self.normalized:
            present = self.class_counts_[self.class_counts_ > 0]
            n_min = int(present.min()) if present.size else 0
            for c in (0, 1):
                rows = np.flatnonzero(yc == c)
                if rows.size:
                    n_c = np.asarray(Zc[rows].sum(axis=0)).ravel().astype(np.float64)
                    self._apply(c, n_c * (n_min / rows.size))
                    self.n_seen_[c] += rows.size
            self.balanced_index_ = np.arange(yc.size)
            self.balanced_counts_ = np.full(2, n_min, dtype=np.int64) * (self.class_counts_ > 0)
        else:
            idx = balance_indices(yc, self.seed_subsample)
            self.balanced_index_ = idx
            self.balanced_counts_ = np.bincount(yc[idx], minlength=2)
            self.partial_fit(Zc[idx], yc[idx])
        return self

    def familiarity(self, Z: sp.spmatrix | np.ndarray) -> np.ndarray:
        """phi_c(z) = 1 - mean_{i in supp z} F_c[i] for c = 0, 1, as ``[n, 2]``; a row with empty support (which a
        k-WTA code never has) gets phi = 0 for both classes, i.e. score 0.5."""
        Zc = self._check(Z)
        support = np.asarray(Zc.sum(axis=1)).ravel().astype(np.float64)
        sums = np.asarray(Zc @ self.F_.T.astype(np.float64))  # [n, 2]
        mean = np.divide(sums, support[:, None], out=np.ones_like(sums), where=support[:, None] > 0)
        return (1.0 - mean).astype(np.float32)

    def score(self, Z: sp.spmatrix | np.ndarray) -> np.ndarray:
        """s(z) = (phi_1 - phi_0 + 1)/2 in [0, 1], one value per row."""
        phi = self.familiarity(Z)
        return np.clip((phi[:, 1] - phi[:, 0] + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

    def state_size_bytes(self) -> int:
        """Size of the trained state (ТЗ Этап 4 "размер состояния"): the two filters."""
        return int(self.F_.nbytes)


class LinearReadout:
    """Linear MBON readout (ТЗ 2.4 "Линейный (MBON)"): logistic regression on z with L2, balanced class weights,
    C chosen on validation; the score is the sigmoid (``predict_proba[:, 1]``).

    liblinear handles the sparse binary codes at FlyHash width directly and is deterministic for the primal L2
    problem. sklearn >= 1.8 deprecates ``penalty``; ``l1_ratio=0.0`` is the L2 penalty. Balanced class weights
    (``readout.linear.class_weight``) replace subsampling here because the linear model can reweight.
    """

    def __init__(self, C: float = 1.0, seed: int = 0, max_iter: int = 1000, tol: float = 1e-4,
                 solver: str = "liblinear", class_weight: str | dict | None = None) -> None:
        cfg = _readout_cfg()["linear"]
        self.C = float(C)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.solver = solver
        self.class_weight = cfg.get("class_weight", "balanced") if class_weight is None else class_weight
        self.model_: LogisticRegression | None = None

    def fit(self, Z: sp.spmatrix | np.ndarray, y: Sequence[int] | np.ndarray) -> "LinearReadout":
        """Fit the L2 logistic regression; needs both classes."""
        Zc = _as_code(Z)
        yc = _labels(y, Zc.shape[0])
        if np.unique(yc).size < 2:
            raise ValueError("LinearReadout.fit needs examples of both classes")
        self.model_ = LogisticRegression(C=self.C, l1_ratio=0.0, class_weight=self.class_weight, solver=self.solver,
                                         random_state=self.seed, max_iter=self.max_iter, tol=self.tol)
        self.model_.fit(Zc, yc)
        return self

    def score(self, Z: sp.spmatrix | np.ndarray) -> np.ndarray:
        """P(injection | z) = sigmoid(w · z + b)."""
        if self.model_ is None:
            raise RuntimeError("LinearReadout.fit must run before score")
        return self.model_.predict_proba(_as_code(Z))[:, 1].astype(np.float32)

    def state_size_bytes(self) -> int:
        """Size of the trained state: weights and intercept."""
        if self.model_ is None:
            return 0
        return int(self.model_.coef_.nbytes + self.model_.intercept_.nbytes)


def _first_argmax(table: dict[Any, float]) -> Any:
    keys = list(table)
    vals = np.array([table[k] for k in keys], dtype=np.float64)
    return keys[int(np.argmax(vals))]  # first maximum in grid order -> deterministic ties


def select_gamma(Z_train: sp.spmatrix, y_train: np.ndarray, Z_val: sp.spmatrix, y_val: np.ndarray, m: int, k: int,
                 seed_subsample: int = 0, gammas: Sequence[float] | None = None,
                 normalized: bool = False) -> tuple[float, dict[float, float]]:
    """Choose gamma for the Bloom readout by validation AUC only (ТЗ 2.4 "gamma по валидации"; grid
    ``readout.bloom.gammas``). Returns ``(best_gamma, {gamma: val_auc})``; ties go to the first grid entry. Called
    separately for few-shot and full training, as the ТЗ requires."""
    grid = [float(g) for g in (_readout_cfg()["bloom"]["gammas"] if gammas is None else gammas)]
    y_val = np.asarray(y_val)
    table: dict[float, float] = {}
    for g in grid:
        model = BloomReadout(m, k, g, seed_subsample=seed_subsample, normalized=normalized).fit(Z_train, y_train)
        table[g] = float(roc_auc_score(y_val, model.score(Z_val)))
    return _first_argmax(table), table


def select_C(Z_train: sp.spmatrix, y_train: np.ndarray, Z_val: sp.spmatrix, y_val: np.ndarray,
             C_grid: Sequence[float] | None = None, seed: int = 0, **linear_kwargs: Any) -> tuple[float, dict[float, float]]:
    """Choose C for the linear readout by validation AUC only (ТЗ 2.4 "C по валидации"; grid
    ``readout.linear.C_grid``). Returns ``(best_C, {C: val_auc})``; ties go to the first grid entry."""
    grid = [float(c) for c in (_readout_cfg()["linear"]["C_grid"] if C_grid is None else C_grid)]
    y_val = np.asarray(y_val)
    table: dict[float, float] = {}
    for c in grid:
        model = LinearReadout(C=c, seed=seed, **linear_kwargs).fit(Z_train, y_train)
        table[c] = float(roc_auc_score(y_val, model.score(Z_val)))
    return _first_argmax(table), table
