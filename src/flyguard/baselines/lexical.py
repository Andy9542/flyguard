"""Lexical baselines of ТЗ 3.1 on the nose's features: TF-IDF + LR, kNN, nearest centroid, LR on N51-svd.

All four follow ``common.WindowScorer``: ``fit(X_train, y_train, X_val=None, y_val=None, groups=None)`` and
``score(X)`` in [0, 1]. They never compute features: ``TfidfLR``, ``KNN`` and ``NearestCentroid`` take the nose's
N16k rows (log1p counts, rows summing to one — ТЗ 2.1 "общий вход TF-IDF, kNN и FlyHash"), ``LRSvd`` takes the
51-dim standardised SVD features (ТЗ 3.1 "LR на N51-svd без KC-слоя — потолок носа, пара для H1b"). The one
statistic fitted here is TF-IDF's idf, and it is fitted on C_unl (ТЗ 1.9), passed in explicitly. The two logistic
regressions (``TfidfLR``, ``LRSvd``) are built by ``common.make_logreg`` — the same estimator as the MBON readout
(``readout.linear`` in the config) — and choose C through ``common.select_c``, which needs ``groups``
(``cluster_id``, else ``doc_id``, per train window) whenever no validation set is given.

Detector names match ``configs/experiments/E1.yaml``: ``tfidf_lr``, ``knn1``/``knn5``, ``centroid``, ``lr_svd``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from flyguard.config import Configs, load_configs
from flyguard.baselines.common import (as_matrix, clip01, cosine_similarity_blocks, l2_normalize_rows,
                                       make_logreg, same_kind, select_c, warn_once)


def _grid(cfg: Configs | None, key: str) -> list[float]:
    cfg = cfg or load_configs()
    return [float(c) for c in cfg.default["baselines"][key]["C_grid"]]


class TfidfLR:
    """TF-IDF + logistic regression on N16k (ТЗ 3.1: "TF-IDF + LR на N16k (idf по C_unl, C по валидации)").

    Input contract (``tf``): ``"n16k"`` (default) means ``X`` is the nose's N16k output, already log1p-scaled, so
    only idf weighting and L2 row normalisation are applied; ``"counts"`` means ``X`` holds raw 16 384-bin counts
    and ``log1p`` is applied here first. Both give the same tf-idf up to the per-row constant that L2
    normalisation removes; the flag exists so the engine cannot double-log by accident. idf is the smooth
    ``log((1 + N) / (1 + df)) + 1`` (scikit-learn's convention) over the document frequencies of ``X_unl`` =
    C_unl (ТЗ 1.9), given in the constructor or via ``fit_idf``. ТЗ 1.9 names C_unl as *the* corpus of the idf,
    so ``fit`` without a fitted idf raises ``ValueError``; only an explicit ``allow_train_idf=True`` (a documented
    control, never the main runs) lets the idf fall back to the train rows, with a ``RuntimeWarning`` and
    ``idf_source_ == "train"`` so the deviation reaches the results notes. C is chosen by validation AUC over
    ``cfg.default['baselines']['tfidf']['C_grid']`` (``common.select_c``; ``c_source_``/``c_table_`` record how).
    """

    name = "tfidf_lr"

    def __init__(self, C_grid: Sequence[float] | None = None, seed: int = 0, X_unl: Any = None,
                 tf: str = "n16k", cfg: Configs | None = None, allow_train_idf: bool = False):
        if tf not in ("n16k", "counts"):
            raise ValueError("tf must be 'n16k' or 'counts'")
        self.C_grid = [float(c) for c in C_grid] if C_grid is not None else _grid(cfg, "tfidf")
        self.seed = int(seed)
        self.tf = tf
        self.cfg = cfg
        self.allow_train_idf = bool(allow_train_idf)
        self.idf_: np.ndarray | None = None
        self.idf_source_: str | None = None
        self.C_: float | None = None
        self.c_source_: str | None = None
        self.c_table_: dict[str, float | None] = {}
        self.model_ = None
        if X_unl is not None:
            self.fit_idf(X_unl)

    def fit_idf(self, X_unl: Any) -> "TfidfLR":
        """idf from the document frequencies of C_unl rows (nonzero pattern; identical for counts and N16k).

        The caller's matrix is left untouched (``as_matrix`` shares the buffers of a float64 CSR input): explicit
        zeros are dropped from a *copy* before counting, so that C_unl rows used by other detectors keep their
        stored pattern and so that duplicate entries of a non-canonical CSR are merged before the count.
        """
        X = as_matrix(X_unl)
        if sp.issparse(X):
            X = X.copy()
            X.sum_duplicates()
            X.eliminate_zeros()
            df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float64)
        else:
            df = (X != 0).sum(axis=0).astype(np.float64)
        n = X.shape[0]
        self.idf_ = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.idf_source_ = "c_unl"
        return self

    def transform(self, X: Any) -> sp.csr_matrix | np.ndarray:
        """tf-idf rows: (log1p if counts) * idf, then unit L2 norm."""
        if self.idf_ is None:
            raise RuntimeError("idf not fitted: pass X_unl or call fit_idf before transform")
        X = as_matrix(X)
        if X.shape[1] != self.idf_.shape[0]:
            raise ValueError(f"feature dimension {X.shape[1]} != idf dimension {self.idf_.shape[0]}")
        if self.tf == "counts":
            if sp.issparse(X):
                X = X.copy()
                X.data = np.log1p(X.data)
            else:
                X = np.log1p(X)
        X = X @ sp.diags(self.idf_) if sp.issparse(X) else X * self.idf_[None, :]
        return l2_normalize_rows(X)

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "TfidfLR":
        if self.idf_ is None:
            if not self.allow_train_idf:
                raise ValueError("TfidfLR: idf must be fitted on C_unl (ТЗ 1.9): pass X_unl or call fit_idf; "
                                 "allow_train_idf=True opts into the train-row fallback explicitly")
            warn_once("TfidfLR: no C_unl given, idf fitted on the train rows (ТЗ 1.9 asks for C_unl)")
            self.fit_idf(X_train)
            self.idf_source_ = "train"
        y = np.asarray(y_train).astype(int).ravel()
        Xt = self.transform(X_train)
        Xv = self.transform(X_val) if X_val is not None else None
        self.C_, self.c_source_, self.c_table_ = select_c(self._make_model, Xt, y, Xv, y_val, self.C_grid,
                                                          self.seed, groups=groups)
        self.model_ = self._make_model(self.C_).fit(Xt, y)
        return self

    def _make_model(self, C: float):
        return make_logreg(C, self.seed, self.cfg)

    def score(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit before score")
        return clip01(self.model_.predict_proba(self.transform(X))[:, 1])


class KNN:
    """k nearest neighbours by cosine similarity on N16k (ТЗ 3.1 "kNN (k = 1, 5, косинус)").

    Score for k >= 2: the mean label of the k nearest train windows (k + 1 distinct levels). For k = 1 the mean
    label is the neighbour's label, a 0/1 score whose ROC has a single point; to keep a ranking (design §6 asks
    for a monotone score) the k = 1 score is the label weighted by the similarity: ``0.5 + 0.5 * sim`` when the
    nearest neighbour is an injection and ``0.5 - 0.5 * sim`` when it is benign (ASSUMPTIONS A15). Positives
    whose nearest positive neighbour is close rank highest, texts near a benign neighbour rank lowest, and
    undecided texts (low similarity to anything) sit near 0.5; the hard 1-NN decision is recovered at the 0.5
    threshold. Similarities are computed in dense blocks (``common.cosine_similarity_blocks``); candidates come
    from ``argpartition`` and are then ordered stably by (-similarity, index), so exact ties resolve to the lower
    train index. ``fit`` ignores ``X_val``/``y_val``/``groups``: kNN has nothing to validate.
    """

    def __init__(self, k: int = 5, metric: str = "cosine", block_rows: int | None = None):
        if metric != "cosine":
            raise ValueError("only cosine is implemented (ТЗ 3.1)")
        self.k = int(k)
        self.metric = metric
        self.block_rows = block_rows
        self.name = f"knn{self.k}"
        self.Xt_: Any = None
        self.y_: np.ndarray | None = None

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None, groups: Any = None) -> "KNN":
        self.Xt_ = l2_normalize_rows(X_train)
        self.y_ = np.asarray(y_train).astype(np.float64).ravel()
        if self.Xt_.shape[0] != self.y_.shape[0]:
            raise ValueError("X_train and y_train disagree in length")
        return self

    def score(self, X: Any) -> np.ndarray:
        if self.Xt_ is None or self.y_ is None:
            raise RuntimeError("fit before score")
        Xq = l2_normalize_rows(X)
        Xq, Xt = same_kind(Xq, self.Xt_)
        n_t = Xt.shape[0]
        k = min(self.k, n_t)
        out = np.empty(Xq.shape[0], dtype=np.float64)
        for start, S in cosine_similarity_blocks(Xq, Xt, self.block_rows):
            if k < n_t:
                cand = np.argpartition(-S, k - 1, axis=1)[:, :k]
            else:
                cand = np.tile(np.arange(n_t), (S.shape[0], 1))
            rows = np.arange(S.shape[0])[:, None]
            sims = S[rows, cand]
            order = np.lexsort((cand, -sims), axis=1)
            cand = np.take_along_axis(cand, order, axis=1)
            sims = np.take_along_axis(sims, order, axis=1)
            labels = self.y_[cand]
            if self.k == 1:
                y1, s1 = labels[:, 0], np.clip(sims[:, 0], -1.0, 1.0)
                block = np.where(y1 >= 0.5, 0.5 + 0.5 * s1, 0.5 - 0.5 * s1)
            else:
                block = labels.mean(axis=1)
            out[start:start + S.shape[0]] = block
        return clip01(out)


class NearestCentroid:
    """Nearest centroid by cosine (ТЗ 3.1 "ближайший центроид").

    Centroids are the means of the unit-normalised train rows of each class, re-normalised to unit length; the
    score is the difference of cosines to the injection and benign centroids mapped affinely to [0, 1]:
    ``s = (cos_1 - cos_0 + 1) / 2`` — the same map the Bloom readout uses for ``(phi_1 - phi_0 + 1) / 2`` (ТЗ 2.4),
    monotone in the two-class softmax argument ``cos_1 - cos_0`` and exactly inside [0, 1] for non-negative
    features (N16k); centred inputs are clipped.
    """

    name = "centroid"

    def __init__(self) -> None:
        self.c0_: np.ndarray | None = None
        self.c1_: np.ndarray | None = None

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "NearestCentroid":
        Xn = l2_normalize_rows(X_train)
        y = np.asarray(y_train).astype(int).ravel()
        if not ((y == 0).any() and (y == 1).any()):
            raise ValueError("NearestCentroid needs both classes in y_train")
        self.c0_ = self._centroid(Xn[y == 0])
        self.c1_ = self._centroid(Xn[y == 1])
        return self

    @staticmethod
    def _centroid(Xn: Any) -> np.ndarray:
        c = np.asarray(Xn.mean(axis=0)).ravel()
        norm = np.linalg.norm(c)
        return c / norm if norm > 0 else c

    def score(self, X: Any) -> np.ndarray:
        if self.c0_ is None or self.c1_ is None:
            raise RuntimeError("fit before score")
        Xn = l2_normalize_rows(X)
        cos1 = np.asarray(Xn @ self.c1_).ravel()
        cos0 = np.asarray(Xn @ self.c0_).ravel()
        return clip01((cos1 - cos0 + 1.0) / 2.0)


class LRSvd:
    """Logistic regression on the 51-dim standardised SVD features, no KC layer (ТЗ 3.1, the nose's ceiling).

    The features arrive standardised and permuted from ``nose.N51Svd`` (ТЗ 2.1), so nothing is rescaled here.
    L2, balanced class weights, C by validation AUC from ``cfg.default['baselines']['lr_svd']['C_grid']``; the
    estimator comes from ``common.make_logreg`` — the MBON readout's logistic regression (ТЗ 2.4, config
    ``readout.linear``) — so that H1b compares the input spaces, not the classifiers.
    """

    name = "lr_svd"

    def __init__(self, C_grid: Sequence[float] | None = None, seed: int = 0, cfg: Configs | None = None):
        self.C_grid = [float(c) for c in C_grid] if C_grid is not None else _grid(cfg, "lr_svd")
        self.seed = int(seed)
        self.cfg = cfg
        self.C_: float | None = None
        self.c_source_: str | None = None
        self.c_table_: dict[str, float | None] = {}
        self.model_ = None

    @staticmethod
    def _dense(X: Any) -> np.ndarray:
        X = as_matrix(X)
        return X.toarray() if sp.issparse(X) else X

    def fit(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "LRSvd":
        Xt = self._dense(X_train)
        y = np.asarray(y_train).astype(int).ravel()
        Xv = self._dense(X_val) if X_val is not None else None
        self.C_, self.c_source_, self.c_table_ = select_c(self._make_model, Xt, y, Xv, y_val, self.C_grid,
                                                          self.seed, groups=groups)
        self.model_ = self._make_model(self.C_).fit(Xt, y)
        return self

    def _make_model(self, C: float):
        return make_logreg(C, self.seed, self.cfg)

    def score(self, X: Any) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit before score")
        return clip01(self.model_.predict_proba(self._dense(X))[:, 1])


def make_lexical_scorers(cfg: Configs, seed: int, X_unl: Any = None) -> dict[str, Any]:
    """The ТЗ 3.1 lexical set keyed by detector name, from ``cfg.default['baselines']`` (ks, grids, centroid flag).

    ``X_unl`` are the C_unl rows for TF-IDF's idf; ``seed`` is the global seed's child that the engine passes for
    the classifier (the baselines themselves have no other randomness).
    """
    b = cfg.default["baselines"]
    scorers: dict[str, Any] = {"tfidf_lr": TfidfLR(b["tfidf"]["C_grid"], seed=seed, X_unl=X_unl, cfg=cfg)}
    for k in b["knn"]["ks"]:
        knn = KNN(int(k), metric=b["knn"].get("metric", "cosine"))
        scorers[knn.name] = knn
    if b.get("centroid", True):
        scorers["centroid"] = NearestCentroid()
    scorers["lr_svd"] = LRSvd(b["lr_svd"]["C_grid"], seed=seed, cfg=cfg)
    return scorers
