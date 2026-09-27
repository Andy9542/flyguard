"""Per-seed experiment engine (docs/design.md §9, docs/design_experiments.md §2): features, detectors, scores,
document-level evaluation and the ``Runner`` that turns an experiment body into ``results/<E>/<seed>.json``.

Pipeline for one global seed ``s`` (children from :func:`flyguard.config.seeds_for`):

1. *Window sets* (:class:`WindowSet`) by role -- ``train``, ``val`` (deepset val), ``val:<source>``, ``c_unl``,
   ``p_val``, ``test:<source>`` (through the context's single journaled accessor, dedup-excluded windows dropped),
   ``p_test:<source>``, few-shot subsets and any frame an experiment registers. A subset remembers its parent and
   row positions, so its features and codes are slices of the parent's, never recomputed.
2. *Counts*: :func:`flyguard.nose.char_ngram_hash_counts` with seed ``nose`` per set, cached on disk per set under
   ``data/processed/features/seed<s>/`` through :func:`flyguard.nose.cached_char_ngram_hash_counts` (its identity
   check keys the file by texts, seed, sizes and bins); the disk cache respects a per-seed byte budget and
   :meth:`FeatureContext.cleanup` deletes it after the seed's experiments.
3. *Noses* fitted on C_unl only (ТЗ 1.9): ``N16k``, ``N51Svd(seed_svd, seed_perm)`` with rank = the number of
   glomeruli of the measured matrix, ``N51Hash`` with as many bins.
4. *Matrices*: measured (binary; weighted for E6), random of the same density, Curveball nulls, FlyHash at any
   expansion, random matrices of an arbitrary cell count (E5) and the dense Gaussian sign matrix, all from seeds
   ``projection`` / ``curveball``.
5. *Codes*: ``fly_code(U, M, k)`` with ``k = k_for(m, k_frac)``; the N16k -> FlyHash path passes ``mean = N16k.mean_``
   (centring inside the product). Codes of small sets are memoised per (nose, matrix, k); large sets are streamed
   in row batches, and every readout sharing a code key is scored from the same batch (FlyHash-20 coding costs
   ~16 ms and ~130 KB per window, so BIPIA test is never materialised). A Bloom readout whose training set is too
   large to memoise (E3 folds) is fitted from per-class code counts accumulated over streamed batches, with the
   same numbers as the materialised fit (:meth:`FeatureContext._fit_bloom_streamed`).
6. *Detectors* (:data:`DETECTORS`, ТЗ Этап 2–3 / design_experiments table 5) are fitted on the train set with every
   hyperparameter chosen on validation AUC only: γ through ``readout.select_gamma``, C through ``readout.select_C``,
   the lexical baselines through their own ``fit(X_val, y_val, groups)``; guard models are inference only and
   scored once per window set (cached by ``text_hash`` inside ``GuardModel``). Fits and the N51-svd are computed
   with the BLAS/OpenMP pools pinned to one thread, so a seed's numbers do not depend on ``run_all.sh --jobs``.
7. *Evaluation* (:class:`Evaluator`, :func:`standard_evaluation`): document scores = max over non-excluded windows
   (``eval.metrics.doc_scores``), per-source AUC and macroAUC with cluster-bootstrap CIs, paired differences at
   95 % and 90 % from the same draws (with the bootstrap p), τ_FPR on P_val and TPR@FPR on the test positives,
   FPR on NotInject at τ_FPR and τ_90(deep) by subset and language stratum, latency and state size -- keyed as in
   :mod:`flyguard.experiments.results`.
"""
from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from flyguard import fly, nose
from flyguard.baselines.lexical import make_lexical_scorers
from flyguard.baselines.regex import RegexScorer
from flyguard.config import ROOT, Configs, seeds_for
from flyguard.connectome import (curveball_nulls, dense_gaussian_sign_matrix, flyhash_matrix, random_same_density)
from flyguard.eval.bootstrap import CI, cluster_bootstrap, macro_auc_bootstrap, paired_cluster_bootstrap, percentile_ci
from flyguard.eval.metrics import auc as auc_point, doc_scores, fpr_at_threshold, macro_auc, tpr_at_threshold
from flyguard.eval.thresholds import fpr_target_for_pool, tau_fpr_record, tau_tpr_record
from flyguard.eval.tost import bootstrap_p
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.io import read_json
from flyguard.readout import BloomReadout, LinearReadout, balance_indices, select_C, select_gamma

CACHE_BUDGET_BYTES = 200 << 20   # per seed (task constraint: feature caches per seed under 200 MB)
CODE_MEMO_BYTES = 256 << 20      # codes of a set are kept in memory below this size, streamed above it
CODE_BATCH_BYTES = 192 << 20     # streamed code batch size (csr bytes)
LATENCY_DOCS = 200
POSITIVE_SOURCES = ("deep", "bipia", "dojo", "dyn", "para")


# ----------------------------------------------------------------------------------------------------------------
# Window sets
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class WindowSet:
    """A named frame of ``windows.parquet`` rows; a subset keeps ``parent`` and its row positions there."""

    name: str
    frame: pd.DataFrame
    parent: str | None = None
    rows: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(len(self.frame))

    @property
    def texts(self) -> list[str]:
        return self.frame["text"].astype(str).tolist()

    @property
    def window_ids(self) -> np.ndarray:
        return self.frame["window_id"].to_numpy()

    @property
    def doc_ids(self) -> np.ndarray:
        return self.frame["doc_id"].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy().astype(int)

    @property
    def groups(self) -> np.ndarray:
        """Per-window group for cross-validated choices: ``cluster_id``, else ``doc_id`` (ТЗ 1.10)."""
        col = "cluster_id" if "cluster_id" in self.frame.columns else "doc_id"
        return self.frame[col].astype(str).to_numpy()


# ----------------------------------------------------------------------------------------------------------------
# Detector specifications
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectorSpec:
    """What to build: ``kind`` in ``fly`` (nose -> matrix -> k-WTA -> readout, or nose -> linear when ``matrix`` is
    None), ``lexical`` (``readout`` names the ``baselines.lexical`` scorer), ``regex``, ``guard``. ``matrix`` kinds:
    ``measured``, ``weighted``, ``random`` (same density as measured), ``curveball:<j>``, ``flyhash`` /
    ``flyhash<E>`` (expansion E, default the config primary), ``random:<m>`` (m cells, fan-in from config),
    ``dense_sign`` (linear readout only). ``gamma`` / ``C`` fix the hyperparameter (E6 "all γ"); ``None`` means
    choose on validation."""

    name: str
    kind: str
    nose: str | None = None
    matrix: str | None = None
    readout: str | None = None
    k_frac: float | None = None
    gamma: float | None = None
    C: float | None = None
    normalized: bool = False
    guard: str | None = None
    params: tuple = ()

    @property
    def code_key(self) -> tuple | None:
        """Detectors with the same code key share one coding pass."""
        if self.kind != "fly" or self.matrix is None:
            return None
        return (self.nose, self.matrix, self.k_frac)


def fly_spec(name: str, nose: str = "n51_svd", matrix: str | None = "measured", readout: str = "bloom",
             **kwargs: Any) -> DetectorSpec:
    if readout not in ("bloom", "linear"):
        raise ValueError("fly readout must be 'bloom' or 'linear'")
    if readout == "bloom" and (matrix is None or matrix == "dense_sign"):
        raise ValueError("the Bloom readout needs a sparse KC code (ТЗ 3.3: the sign code is linear only)")
    return DetectorSpec(name, "fly", nose=nose, matrix=matrix, readout=readout, **kwargs)


DETECTORS: dict[str, DetectorSpec] = {
    "regex": DetectorSpec("regex", "regex"),
    "tfidf_lr": DetectorSpec("tfidf_lr", "lexical", nose="n16k", readout="tfidf_lr"),
    "knn1": DetectorSpec("knn1", "lexical", nose="n16k", readout="knn1"),
    "knn5": DetectorSpec("knn5", "lexical", nose="n16k", readout="knn5"),
    "centroid": DetectorSpec("centroid", "lexical", nose="n16k", readout="centroid"),
    "lr_svd": DetectorSpec("lr_svd", "lexical", nose="n51_svd", readout="lr_svd"),
    "real_fly_bloom": fly_spec("real_fly_bloom", "n51_svd", "measured", "bloom"),
    "real_fly_linear": fly_spec("real_fly_linear", "n51_svd", "measured", "linear"),
    "flyhash_bloom": fly_spec("flyhash_bloom", "n16k", "flyhash", "bloom"),
    "flyhash_linear": fly_spec("flyhash_linear", "n16k", "flyhash", "linear"),
    "protectai_v2": DetectorSpec("protectai_v2", "guard", guard="protectai_v2"),
    "piguard": DetectorSpec("piguard", "guard", guard="piguard"),
    "prompt_guard_2": DetectorSpec("prompt_guard_2", "guard", guard="prompt_guard_2"),
}


def detector_spec(name: str) -> DetectorSpec:
    """Registry lookup by the ``configs/experiments/E1.yaml`` name."""
    try:
        return DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(DETECTORS)}") from None


@dataclass
class FittedDetector:
    """A fitted detector: the model, the validation choices (γ / C tables), timing and availability."""

    spec: DetectorSpec
    model: Any
    choices: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0
    train_set: str = "train"
    n_train: int = 0
    available: bool = True
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def state_size_bytes(self) -> int | None:
        """Trained state in bytes (ТЗ Этап 4 "размер состояния"): readout filters / weights, idf and coefficients,
        the kNN reference rows, the centroids, the pattern text or the guard's weight files."""
        m = self.model
        if m is None:
            return None
        if hasattr(m, "state_size_bytes"):
            return int(m.state_size_bytes())
        total = 0
        for attr in ("coef_", "intercept_"):
            lr = getattr(m, "model_", None)
            if lr is not None and hasattr(lr, attr):
                total += int(getattr(lr, attr).nbytes)
        for attr in ("idf_", "c0_", "c1_", "y_"):
            arr = getattr(m, attr, None)
            if arr is not None:
                total += int(np.asarray(arr).nbytes)
        xt = getattr(m, "Xt_", None)
        if xt is not None:
            total += int(xt.data.nbytes + xt.indices.nbytes + xt.indptr.nbytes) if sp.issparse(xt) else int(xt.nbytes)
        if hasattr(m, "lines"):
            total += sum(len(line.encode("utf-8")) for line in m.lines)
        if hasattr(m, "path") and hasattr(m, "available"):
            total += sum(p.stat().st_size for p in Path(m.path).glob("*") if p.suffix in (".safetensors", ".bin"))
        return total


def bloom_balance_choices(model: Any) -> dict[str, Any]:
    """The class counts a Bloom readout was fitted on (``balanced_counts_``) as the list and as two scalars.

    Why the scalars: the ``detectors`` table of a result keeps only scalar choices (:func:`standard_evaluation`), and
    the acceptance criterion "Bloom обучен на сбалансированных классах" (``check_acceptance.c_bloom``) needs the
    actual counts of every fit, not only ``readout.bloom.balance`` of the config."""
    counts = [int(c) for c in model.balanced_counts_]
    return {"balanced_counts": counts, "balanced_n0": counts[0], "balanced_n1": counts[1]}


def _parse_int_suffix(kind: str, prefix: str) -> int | None:
    rest = kind[len(prefix):]
    return int(rest.lstrip(":")) if rest else None


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# ----------------------------------------------------------------------------------------------------------------
# Feature context
# ----------------------------------------------------------------------------------------------------------------
class FeatureContext:
    """Everything that depends on the global seed: counts, noses, matrices, codes, fitted detectors and scores."""

    def __init__(self, ctx: Context, seed: int, purpose: str = "", cache: bool = True,
                 cache_dir: Path | None = None, cache_budget_bytes: int = CACHE_BUDGET_BYTES,
                 code_memo_bytes: int = CODE_MEMO_BYTES,
                 guard_factory: Callable[..., Any] | None = None) -> None:
        self.ctx = ctx
        self.cfg: Configs = ctx.cfg
        self.seed = int(seed)
        self.seeds = seeds_for(self.cfg, self.seed)
        self.purpose = purpose or f"seed={self.seed}"
        self.cache_enabled = bool(cache)
        base = Path(cache_dir) if cache_dir is not None else ctx.root / "data" / "processed" / "features"
        self.cache_dir = base / ("smoke" if ctx.smoke else "") / f"seed{self.seed}"
        self.cache_budget_bytes = int(cache_budget_bytes)
        self.code_memo_bytes = int(code_memo_bytes)
        self._guard_factory = guard_factory or (lambda name, **kw: ctx.guard(name, **kw))
        self._sets: dict[str, WindowSet] = {}
        self._counts: dict[tuple[str, int], sp.csr_matrix] = {}
        self._features: dict[tuple[str, str], Any] = {}
        self._codes: dict[tuple[tuple, str], sp.csr_matrix] = {}
        self._matrices: dict[tuple[str, int], Any] = {}
        self._nulls: list[sp.csr_matrix] | None = None
        self._noses: dict[str, Any] = {}
        self._guards: dict[str, Any] = {}
        self._guard_scores: dict[tuple[str, str], pd.Series] = {}
        self.cache_files: dict[str, int] = {}
        n = self.cfg.default["nose"]
        self.ngram_sizes = tuple(int(x) for x in n["ngram_sizes"])
        self.bins = int(n["n16k"]["bins"])

    # -- window sets ------------------------------------------------------------------------------------------------
    def register_set(self, name: str, frame: pd.DataFrame, parent: str | None = None,
                     rows: np.ndarray | None = None) -> WindowSet:
        ws = WindowSet(name, frame.reset_index(drop=True), parent, rows)
        self._sets[name] = ws
        return ws

    def subset(self, name: str, parent: str | WindowSet, mask: np.ndarray | None = None,
               doc_ids: Sequence[str] | None = None, window_ids: Sequence[str] | None = None) -> WindowSet:
        """A subset of ``parent`` (by boolean mask, document ids or window ids) whose features are slices."""
        p = self.window_set(parent)
        if mask is None:
            if doc_ids is not None:
                mask = p.frame["doc_id"].isin(set(doc_ids)).to_numpy()
            elif window_ids is not None:
                mask = p.frame["window_id"].isin(set(window_ids)).to_numpy()
            else:
                raise ValueError("subset needs mask, doc_ids or window_ids")
        rows = np.flatnonzero(np.asarray(mask, dtype=bool))
        root_name, root_rows = p.name, rows
        if p.parent is not None:  # flatten: positions relative to the root frame
            root_name, root_rows = p.parent, p.rows[rows]
        return self.register_set(name, p.frame.iloc[rows], root_name, root_rows)

    def window_set(self, name: str | WindowSet) -> WindowSet:
        """Built-in roles by name (module docstring) or an already registered set."""
        if isinstance(name, WindowSet):
            return name
        if name in self._sets:
            return self._sets[name]
        ctx = self.ctx
        if name == "train":
            return self.register_set(name, ctx.train_windows)
        if name == "val":
            return self.register_set(name, ctx.val_windows_deep)
        if name == "val_all":
            return self.register_set(name, ctx.val_windows("all"))
        if name.startswith("val:"):
            return self.register_set(name, ctx.val_windows((name[4:],)))
        if name == "c_unl":
            return self.register_set(name, ctx.c_unl_windows)
        if name == "p_val":
            return self.register_set(name, ctx.p_val_windows)
        if name == "notinject":
            return self.window_set("test:notinject")
        if name.startswith("test:"):
            frame = ctx.load_test_windows(name[5:], self.purpose)
            if "dedup_excluded" in frame.columns:
                frame = frame[~frame["dedup_excluded"].fillna(False).astype(bool)]
            return self.register_set(name, frame)
        if name.startswith("p_test:"):
            return self.subset(name, f"test:{name[7:]}", doc_ids=ctx.p_test_doc_ids)
        raise KeyError(f"unknown window set {name!r}")

    def fewshot_set(self, shots: int, rep: int = 0, parent: str = "train") -> WindowSet:
        """``shots`` training *documents* per class (E2, H1b-ii), drawn without replacement from
        ``SeedSequence([subsample, shots, rep])``; windows of the drawn documents form the set."""
        p = self.window_set(parent)
        name = f"{parent}:shots{int(shots)}:rep{int(rep)}"
        if name in self._sets:
            return self._sets[name]
        docs = p.frame.groupby("doc_id")["label"].max()
        rng = np.random.default_rng(np.random.SeedSequence([self.seeds["subsample"], int(shots), int(rep)]))
        chosen: list[str] = []
        for c in (0, 1):
            ids = sorted(docs.index[docs == c].tolist())
            if len(ids) < shots:
                raise ValueError(f"only {len(ids)} documents of class {c} for {shots} shots")
            chosen.extend(sorted(rng.choice(ids, int(shots), replace=False).tolist()))
        return self.subset(name, p, doc_ids=chosen)

    # -- counts and noses -------------------------------------------------------------------------------------------
    @property
    def d_glom(self) -> int:
        """Number of glomeruli = inputs of the measured matrix (51 on MaleCNS; the rank of N51-svd)."""
        return int(self.ctx.connectome()[0].shape[1])

    def _cache_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.cache_dir.glob("*.npz")) if self.cache_dir.is_dir() else 0

    def _disk_ok(self, n_texts: int, bins: int) -> bool:
        if not self.cache_enabled:
            return False
        est = n_texts * (2000 if bins >= 1024 else 200)
        return self._cache_bytes() + est <= self.cache_budget_bytes

    def counts(self, ws: str | WindowSet, bins: int | None = None) -> sp.csr_matrix:
        """Hashed n-gram counts of a set (``bins`` = N16k bins by default, ``d_glom`` for the N51-hash nose)."""
        ws = self.window_set(ws)
        bins = int(self.bins if bins is None else bins)
        if ws.parent is not None:
            return self.counts(ws.parent, bins)[ws.rows]
        key = (ws.name, bins)
        if key not in self._counts:
            seed = self.seeds["nose"]
            if self._disk_ok(ws.n, bins):
                path = self.cache_dir / f"counts{bins}_{_safe(ws.name)}.npz"
                X = nose.cached_char_ngram_hash_counts(ws.texts, path, sizes=self.ngram_sizes, bins=bins, seed=seed)
                self.cache_files[str(path)] = path.stat().st_size if path.exists() else 0
            else:
                X = nose.char_ngram_hash_counts(ws.texts, sizes=self.ngram_sizes, bins=bins, seed=seed)
            self._counts[key] = X
        return self._counts[key]

    @property
    def n16k(self) -> nose.N16k:
        if "n16k" not in self._noses:
            self._noses["n16k"] = nose.N16k().fit(self.counts("c_unl"))
        return self._noses["n16k"]

    @property
    def n51_svd(self) -> nose.N51Svd:
        """N51-svd fitted on C_unl with the BLAS pool pinned to one thread: the randomized TruncatedSVD gives
        different components at different OpenMP/BLAS thread counts (bit-level, ~1e-3 on the features), and
        ``run_all.sh --jobs N`` changes that count, so without the pin a seed would not reproduce across runs
        (ТЗ 2.6 "детерминизм при фиксированном seed", ASSUMPTIONS A14/A39). The thread counts in force are
        recorded in every result file (``timing.threads``, :func:`flyguard.experiments.results.thread_info`)."""
        if "n51_svd" not in self._noses:
            X = self.features("n16k", "c_unl")
            with threadpool_limits(limits=1):
                self._noses["n51_svd"] = nose.N51Svd(rank=self.d_glom, seed_svd=self.seeds["svd"],
                                                     seed_perm=self.seeds["perm"]).fit(X)
        return self._noses["n51_svd"]

    @property
    def n51_hash(self) -> nose.N51Hash:
        if "n51_hash" not in self._noses:
            self._noses["n51_hash"] = nose.N51Hash(bins=self.d_glom).fit(self.counts("c_unl", bins=self.d_glom))
        return self._noses["n51_hash"]

    def input_dim(self, space: str) -> int:
        return self.bins if space == "n16k" else self.d_glom

    def nose_mean(self, space: str) -> np.ndarray | None:
        """The centring vector to pass to ``fly.expand`` (N16k stays uncentred and sparse; the 51-d noses centre
        themselves in ``transform``)."""
        return self.n16k.mean_ if space == "n16k" else None

    def features(self, space: str, ws: str | WindowSet) -> Any:
        """``n16k`` (sparse, uncentred), ``n51_svd`` / ``n51_hash`` (dense, centred on C_unl) or ``text``."""
        ws = self.window_set(ws)
        if space == "text":
            return ws.texts
        if ws.parent is not None:
            return self.features(space, ws.parent)[ws.rows]
        key = (space, ws.name)
        if key not in self._features:
            if space == "n16k":
                self._features[key] = self.n16k.transform(self.counts(ws))
            elif space == "n51_svd":
                self._features[key] = self.n51_svd.transform(self.features("n16k", ws), centered=True)
            elif space == "n51_hash":
                self._features[key] = self.n51_hash.transform(self.counts(ws, bins=self.d_glom), centered=True)
            else:
                raise KeyError(f"unknown feature space {space!r}")
        return self._features[key]

    # -- matrices ---------------------------------------------------------------------------------------------------
    def curveball_set(self) -> list[sp.csr_matrix]:
        """The Curveball nulls of the measured matrix (``n_null`` from config / smoke, seed ``curveball``)."""
        if self._nulls is None:
            M = self.ctx.connectome()[0]
            spe = float(self.cfg.default["expansion"]["curveball"]["swaps_per_edge"])
            self._nulls = curveball_nulls(M, n_null=self.ctx.n_null(), seed=self.seeds["curveball"], swaps_per_edge=spe)
        return self._nulls

    def matrix(self, kind: str, space: str = "n51_svd") -> Any:
        """An expansion matrix ``[m, d]`` for inputs of ``space`` (see :class:`DetectorSpec` for the kinds)."""
        d = self.input_dim(space)
        key = (kind, d)
        if key in self._matrices:
            return self._matrices[key]
        M, meta = self.ctx.connectome()
        fh = self.cfg.default["expansion"]["flyhash"]
        fan_in = int(fh["fan_in"])
        seed = self.seeds["projection"]
        if kind in ("measured", "weighted", "random") or kind.startswith("curveball"):
            if d != M.shape[1]:
                raise ValueError(f"matrix {kind!r} has {M.shape[1]} inputs; space {space!r} has {d}")
            if kind == "measured":
                out = M
            elif kind == "weighted":
                out = meta["weighted"]
            elif kind == "random":
                out = random_same_density(M, seed)
            else:
                j = _parse_int_suffix(kind, "curveball")
                out = self.curveball_set()[0 if j is None else j]
        elif kind.startswith("flyhash"):
            exp = _parse_int_suffix(kind, "flyhash") or int(fh["primary_expansion"])
            out = flyhash_matrix(d, exp, min(fan_in, d), seed)
        elif kind.startswith("random:"):
            m = _parse_int_suffix(kind, "random")
            if m is None or m <= 0:
                raise ValueError("random:<m> needs a positive cell count")
            if m % d == 0:
                out = flyhash_matrix(d, m // d, min(fan_in, d), seed)
            else:  # constant fan-in template, inputs redrawn uniformly without replacement per cell
                k_in = min(fan_in, d)
                template = sp.csr_matrix((np.ones(m * k_in, dtype=np.float32),
                                          np.tile(np.arange(k_in, dtype=np.int32), m),
                                          np.arange(0, m * k_in + 1, k_in, dtype=np.int64)), shape=(m, d))
                out = random_same_density(template, seed)
        elif kind == "dense_sign":
            out = dense_gaussian_sign_matrix(d, int(M.nnz), seed)
        else:
            raise KeyError(f"unknown matrix kind {kind!r}")
        self._matrices[key] = out
        return out

    def k_for(self, m: int, k_frac: float | None = None) -> int:
        return fly.k_for(int(m), float(self.cfg.default["inhibition"]["k_frac"] if k_frac is None else k_frac))

    # -- codes ------------------------------------------------------------------------------------------------------
    def _code_batch(self, key: tuple, U: Any) -> sp.csr_matrix:
        space, kind, k_frac = key
        M = self.matrix(kind, space)
        mean = self.nose_mean(space)
        if kind == "dense_sign":
            return fly.sign_code(U, M, mean=mean)
        return fly.fly_code(U, M, self.k_for(M.shape[0], k_frac), mean=mean)

    def code_width(self, key: tuple) -> tuple[int, int]:
        """``(m, k)`` of a code key (``k`` = active bits per row; ~m/2 for the sign code)."""
        space, kind, k_frac = key
        M = self.matrix(kind, space)
        m = int(M.shape[0])
        return m, (m // 2 if kind == "dense_sign" else self.k_for(m, k_frac))

    def codes(self, spec_or_key: DetectorSpec | tuple, ws: str | WindowSet) -> sp.csr_matrix:
        """The full code matrix of a set (memoised when it fits ``code_memo_bytes``; subsets are slices)."""
        key = spec_or_key.code_key if isinstance(spec_or_key, DetectorSpec) else tuple(spec_or_key)
        ws = self.window_set(ws)
        if ws.parent is not None:
            return self.codes(key, ws.parent)[ws.rows]
        memo = (key, ws.name)
        if memo in self._codes:
            return self._codes[memo]
        m, k = self.code_width(key)
        if not ws.n:
            Z = sp.csr_matrix((0, m), dtype=np.float32)
        elif key[1] == "dense_sign":  # variable row weight; the sign code is only |M|/d cells wide
            Z = sp.vstack([z for _, z in self.iter_codes(key, ws)], format="csr")
        else:  # k-WTA: exactly k winners per row, so the csr arrays are preallocated and filled batch by batch
            # (a list of batches plus ``vstack`` held twice the code size, ~8 GB for an E3 fold at FlyHash-20)
            indices = np.empty(ws.n * k, dtype=np.int32)
            for sl, z in self.iter_codes(key, ws):
                if z.nnz != (sl.stop - sl.start) * k:
                    raise RuntimeError(f"code batch of {key} does not have exactly k={k} winners per row")
                indices[sl.start * k: sl.stop * k] = z.indices
            Z = sp.csr_matrix((np.ones(ws.n * k, dtype=np.float32), indices,
                               np.arange(0, ws.n * k + 1, k, dtype=np.int64)), shape=(ws.n, m))
            Z.has_sorted_indices = True
        if ws.n * k * 8 <= self.code_memo_bytes:
            self._codes[memo] = Z
        return Z

    def _codes_in_memory(self, key: tuple, ws: WindowSet) -> bool:
        """Whether :meth:`codes` of this set is cheap to hold: memoised (itself or its root set) or small enough to
        be memoised. Otherwise a fit streams the set (:meth:`_fit_bloom_streamed`) instead of materialising it."""
        root = ws.parent if ws.parent is not None else ws.name
        if (key, ws.name) in self._codes or (key, root) in self._codes:
            return True
        return ws.n * self.code_width(key)[1] * 8 <= self.code_memo_bytes

    def iter_codes(self, key: tuple, ws: str | WindowSet, batch_rows: int | None = None) -> Iterator[tuple[slice, sp.csr_matrix]]:
        """Stream ``(row slice, code batch)`` over a set; memoised codes are yielded whole."""
        key = tuple(key)
        ws = self.window_set(ws)
        memo = (key, ws.name)
        if memo in self._codes or (ws.parent is not None and (key, ws.parent) in self._codes):
            yield slice(0, ws.n), self.codes(key, ws)
            return
        space = key[0]
        U = self.features(space, ws)
        m, k = self.code_width(key)
        size = batch_rows or max(32, CODE_BATCH_BYTES // max(k * 8, 1))
        for start in range(0, ws.n, size):
            sl = slice(start, min(ws.n, start + size))
            yield sl, self._code_batch(key, U[sl])

    # -- detectors --------------------------------------------------------------------------------------------------
    def guard(self, name: str, **kwargs: Any) -> Any:
        if kwargs:
            return self._guard_factory(name, **kwargs)
        if name not in self._guards:
            self._guards[name] = self._guard_factory(name)
        return self._guards[name]

    def fit(self, spec: DetectorSpec | str, train: str | WindowSet = "train", val: str | WindowSet = "val") -> FittedDetector:
        """Fit one detector on ``train`` with validation-based choices on ``val`` (deepset val by default).

        The fit runs with the BLAS/OpenMP pools pinned to one thread (ТЗ 2.6 determinism, ASSUMPTIONS A14/A39):
        the multithreaded dot products inside lbfgs and the centroid means change the last bits of the fitted
        coefficients with the thread count (measured on the smoke tables: tfidf_lr and centroid window scores differ
        by 1-2e-16 between 1, 4 and 16 threads, enough to break or make a score tie), and ``run_all.sh --jobs N``
        changes that count. Nothing in a fit is BLAS-bound (sparse products, 51-d dense rows), so the pin costs
        nothing measurable. Scoring is not pinned: with the pinned fits, the fitted choices and the window scores of
        every non-guard E1 detector and of the E5-style controls (sign code, N51-hash, random, Curveball, random:m)
        were identical at 1, 4 and 16 threads on the smoke tables."""
        with threadpool_limits(limits=1):
            return self._fit(spec, train, val)

    def _fit(self, spec: DetectorSpec | str, train: str | WindowSet, val: str | WindowSet) -> FittedDetector:
        spec = detector_spec(spec) if isinstance(spec, str) else spec
        tr, va = self.window_set(train), self.window_set(val)
        t0 = time.perf_counter()
        choices: dict[str, Any] = {}
        y, y_val = tr.labels, va.labels
        seed = self.seeds["subsample"]
        if spec.kind == "fly":
            if spec.matrix is None:
                if spec.readout != "bloom":
                    X, Xv = self.features(spec.nose, tr), self.features(spec.nose, va)
                    model, choices = self._fit_linear(X, y, Xv, y_val, spec, seed)
                else:
                    raise ValueError("Bloom needs an expansion (E5: Bloom only with a KC layer)")
            elif spec.readout == "bloom" and not spec.normalized and not self._codes_in_memory(spec.code_key, tr):
                model, choices = self._fit_bloom_streamed(spec, tr, va, seed)
            else:
                Z, Zv = self.codes(spec, tr), self.codes(spec, va)
                m, k = self.code_width(spec.code_key)
                if spec.readout == "bloom":
                    gammas = [float(g) for g in self.cfg.default["readout"]["bloom"]["gammas"]]
                    if spec.gamma is None:
                        gamma, table = select_gamma(Z, y, Zv, y_val, m, k, seed_subsample=seed, gammas=gammas,
                                                    normalized=spec.normalized, cfg=self.cfg)
                        choices.update({"gamma": gamma, "gamma_source": "val", "gamma_table": {str(g): float(a) for g, a in table.items()},
                                        "val_auc_window": float(table[gamma])})
                    else:
                        gamma = float(spec.gamma)
                        choices.update({"gamma": gamma, "gamma_source": "fixed"})
                    model = BloomReadout(m, k, gamma, seed_subsample=seed, normalized=spec.normalized).fit(Z, y)
                    choices.update(bloom_balance_choices(model))
                else:
                    model, choices = self._fit_linear(Z, y, Zv, y_val, spec, seed)
        elif spec.kind == "lexical":
            scorers = make_lexical_scorers(self.cfg, seed, X_unl=self.features("n16k", "c_unl"))
            model = scorers[spec.readout]
            model.fit(self.features(spec.nose, tr), y, self.features(spec.nose, va), y_val, groups=tr.groups)
            for attr, key in (("C_", "C"), ("c_source_", "C_source"), ("c_table_", "C_table"), ("idf_source_", "idf_source")):
                if getattr(model, attr, None) is not None:
                    choices[key] = getattr(model, attr)
        elif spec.kind == "regex":
            model = RegexScorer(cfg=self.cfg)
            choices["n_patterns"] = len(model.patterns)
        elif spec.kind == "guard":
            gm = self.guard(spec.guard)
            status = gm.status()
            fitted = FittedDetector(spec, gm if gm.available else None, {"status": status},
                                    time.perf_counter() - t0, tr.name, 0, bool(gm.available), status)
            return fitted
        else:
            raise KeyError(f"unknown detector kind {spec.kind!r}")
        return FittedDetector(spec, model, choices, time.perf_counter() - t0, tr.name, tr.n)

    def _fit_bloom_streamed(self, spec: DetectorSpec, tr: WindowSet, va: WindowSet,
                            seed: int) -> tuple[BloomReadout, dict[str, Any]]:
        """The Bloom fit and its γ search for a training set whose codes are too large to hold (E3 folds: 19k-32k
        windows at FlyHash-20 are 2.4-4.1 GB of codes, and ``BloomReadout.fit`` copies the balanced rows once per
        γ), with results bit-identical to ``select_gamma`` + ``BloomReadout.fit`` on the materialised codes.

        Why it is exact: the trained filter is F_c = γ^{n_c} (readout docstring), where n_c counts per cell the
        class-c codes among the balanced rows ``readout.balance_indices(y, seed)``. Those counts are accumulated
        over streamed code batches (integers, exact in float64; scipy's float32 column sum of the materialised code
        is exact too below 2^24 rows), the filter is set through ``partial_fit`` on the 2 x m count matrix (one
        row per class, so its column sum *is* n_c), and the bookkeeping attributes are set as ``fit`` sets them.
        Every γ of the grid is scored on the validation set in one streamed pass; ties go to the first grid entry
        as in ``readout._first_argmax``. The normalised variant (E6) is not streamed, because E6 trains on deepset
        train; a normalised Bloom on a set too large to memoise still materialises its codes (the old path)."""
        key = spec.code_key
        m, k = self.code_width(key)
        y, y_val = tr.labels, va.labels
        idx = balance_indices(y, seed)
        in_bal = np.zeros(tr.n, dtype=bool)
        in_bal[idx] = True
        masks = [in_bal & (y == c) for c in (0, 1)]
        counts = np.zeros((2, m), dtype=np.float64)
        for sl, Z in self.iter_codes(key, tr):
            for c in (0, 1):
                rows = np.flatnonzero(masks[c][sl])
                if rows.size:
                    counts[c] += np.asarray(Z[rows].sum(axis=0), dtype=np.float64).ravel()
        count_code = sp.csr_matrix(counts.astype(np.float32))

        def bloom(gamma: float) -> BloomReadout:
            model = BloomReadout(m, k, gamma, seed_subsample=seed, normalized=False)
            model.partial_fit(count_code, np.array([0, 1]))
            model.class_counts_ = np.bincount(y, minlength=2)
            model.balanced_index_ = idx
            model.balanced_counts_ = np.bincount(y[idx], minlength=2)
            model.n_seen_[:] = model.balanced_counts_
            return model

        if spec.gamma is not None:
            gamma = float(spec.gamma)
            model = bloom(gamma)
            choices: dict[str, Any] = {"gamma": gamma, "gamma_source": "fixed"}
        else:
            grid = [float(g) for g in self.cfg.default["readout"]["bloom"]["gammas"]]
            models = {g: bloom(g) for g in grid}
            buf = {g: np.empty(va.n, dtype=np.float32) for g in grid}
            batches = ([(slice(0, va.n), self.codes(key, va))] if self._codes_in_memory(key, va)
                       else self.iter_codes(key, va))
            for sl, Zv in batches:
                for g in grid:
                    buf[g][sl] = models[g].score(Zv)
            table = {g: float(roc_auc_score(y_val, buf[g])) for g in grid}
            gamma = grid[int(np.argmax(np.array([table[g] for g in grid], dtype=np.float64)))]
            model = models[gamma]
            choices = {"gamma": gamma, "gamma_source": "val", "gamma_table": {str(g): float(a) for g, a in table.items()},
                       "val_auc_window": float(table[gamma])}
        choices.update(bloom_balance_choices(model))
        return model, choices

    def _fit_linear(self, Z: Any, y: np.ndarray, Zv: Any, y_val: np.ndarray, spec: DetectorSpec,
                    seed: int) -> tuple[LinearReadout, dict[str, Any]]:
        grid = [float(c) for c in self.cfg.default["readout"]["linear"]["C_grid"]]
        if spec.C is None:
            C, table = select_C(Z, y, Zv, y_val, C_grid=grid, seed=seed, cfg=self.cfg)
            choices = {"C": C, "C_source": "val", "C_table": {str(c): float(a) for c, a in table.items()},
                       "val_auc_window": float(table[C])}
        else:
            C, choices = float(spec.C), {"C": float(spec.C), "C_source": "fixed"}
        return LinearReadout(C=C, seed=seed, cfg=self.cfg).fit(Z, y), choices

    def fit_many(self, specs: Sequence[DetectorSpec | str], train: str | WindowSet = "train",
                 val: str | WindowSet = "val") -> dict[str, FittedDetector]:
        out = {}
        for s in specs:
            f = self.fit(s, train, val)
            out[f.name] = f
        return out

    # -- scoring ----------------------------------------------------------------------------------------------------
    def score_guard(self, name: str, ws: str | WindowSet) -> pd.Series | None:
        """Guard scores of a set (``None`` when the snapshot is unavailable); cached by ``text_hash`` in the
        model's parquet cache and per set in memory, so ten seeds never rescore a window."""
        ws = self.window_set(ws)
        key = (name, ws.name)
        if key not in self._guard_scores:
            gm = self.guard(name)
            if not gm.available:
                return None
            hashes = ws.frame["text_hash"].astype(str).tolist() if "text_hash" in ws.frame.columns else None
            vals = gm.score(ws.texts, hashes=hashes)
            self._guard_scores[key] = pd.Series(np.asarray(vals, dtype=float), index=ws.window_ids, name=name)
        return self._guard_scores[key]

    def score_many(self, fitted: Sequence[FittedDetector] | Mapping[str, FittedDetector],
                   ws: str | WindowSet) -> dict[str, pd.Series]:
        """Window scores (Series indexed by ``window_id``) of several detectors; fly detectors sharing a code key
        are scored from one coding pass. Unavailable guards are skipped."""
        ws = self.window_set(ws)
        items = list(fitted.values()) if isinstance(fitted, Mapping) else list(fitted)
        out: dict[str, pd.Series] = {}
        groups: dict[tuple, list[FittedDetector]] = {}
        for f in items:
            if f.spec.kind == "fly" and f.spec.matrix is not None:
                groups.setdefault(f.spec.code_key, []).append(f)
        for key, members in groups.items():
            buf = {f.name: np.empty(ws.n, dtype=np.float64) for f in members}
            for sl, Z in self.iter_codes(key, ws):
                for f in members:
                    buf[f.name][sl] = np.asarray(f.model.score(Z), dtype=np.float64)
            for f in members:
                out[f.name] = pd.Series(buf[f.name], index=ws.window_ids, name=f.name)
        for f in items:
            if f.name in out:
                continue
            if f.spec.kind == "guard":
                s = self.score_guard(f.spec.guard, ws) if f.available else None
                if s is not None:
                    out[f.name] = s.rename(f.name)
                continue
            if f.spec.kind == "regex":
                vals = f.model.score(ws.texts)
            else:  # lexical or fly without expansion
                vals = f.model.score(self.features(f.spec.nose, ws))
            out[f.name] = pd.Series(np.asarray(vals, dtype=np.float64), index=ws.window_ids, name=f.name)
        return out

    def score_windows(self, fitted: FittedDetector, ws: str | WindowSet) -> pd.Series | None:
        """Window scores of one detector as a Series indexed by ``window_id`` (``None`` for an unavailable guard)."""
        return self.score_many([fitted], ws).get(fitted.name)

    # -- document level ---------------------------------------------------------------------------------------------
    def doc_frame(self, scores: pd.Series | Mapping[str, pd.Series], ws: str | WindowSet) -> pd.DataFrame:
        """Document scores (max over non-excluded windows, ``eval.metrics.doc_scores``) joined with the document
        labels, clusters and strata: columns ``doc_id, label, cluster_id, source, lang_stratum, subset, stratum``
        plus one score column per detector (a single Series gives the column ``score``)."""
        ws = self.window_set(ws)
        frame = ws.frame
        table: pd.DataFrame | None = None
        scores = {"score": scores} if isinstance(scores, pd.Series) else dict(scores)
        for name, s in scores.items():
            ds = doc_scores(pd.DataFrame({"window_id": s.index.to_numpy(), "score": s.to_numpy(dtype=float)}), frame)
            ds = ds[["doc_id", "score"]].rename(columns={"score": name})
            table = ds if table is None else table.merge(ds, on="doc_id", how="outer")
        if table is None:
            table = pd.DataFrame({"doc_id": sorted(frame["doc_id"].unique())})
        docs = self.ctx.documents_for(table["doc_id"].tolist())
        meta = docs["meta"].tolist()
        info = pd.DataFrame({
            "doc_id": docs["doc_id"].to_numpy(), "label": docs["label"].astype(int).to_numpy(),
            "cluster_id": docs["cluster_id"].astype(str).to_numpy(), "source": docs["source"].to_numpy(),
            "lang_stratum": docs["lang_stratum"].fillna("unk").to_numpy() if "lang_stratum" in docs else "unk",
            "subset": [m.get("subset") for m in meta], "stratum": [m.get("stratum") for m in meta],
        })
        out = info.merge(table, on="doc_id", how="inner")
        return out.sort_values("doc_id", kind="stable").reset_index(drop=True)

    # -- latency ----------------------------------------------------------------------------------------------------
    def latency_ms(self, fitted: FittedDetector, ws: str | WindowSet = "test:deep", n_docs: int = LATENCY_DOCS,
                   guards: bool = False) -> dict[str, Any] | None:
        """Milliseconds per document for the full pipeline (counts -> nose -> code -> readout) on the first
        ``n_docs`` documents of a set (ТЗ Этап 4 "Задержка"). Guards are timed only with ``guards=True`` (a real
        forward pass without the score cache) because it is expensive; otherwise ``None`` is returned for them."""
        ws = self.window_set(ws)
        ids = sorted(ws.frame["doc_id"].unique())[:n_docs]
        sub = ws.frame[ws.frame["doc_id"].isin(set(ids))]
        texts = sub["text"].astype(str).tolist()
        spec = fitted.spec
        if spec.kind == "guard":
            if not guards or not fitted.available:
                return None
            gm = self.guard(spec.guard, use_cache=False)
            gm.load()
            t0 = time.perf_counter()
            gm.score(texts)
        else:
            t0 = time.perf_counter()
            if spec.kind == "regex":
                fitted.model.score(texts)
            else:
                bins = self.d_glom if spec.nose == "n51_hash" else self.bins
                X = nose.char_ngram_hash_counts(texts, sizes=self.ngram_sizes, bins=bins, seed=self.seeds["nose"])
                if spec.nose == "n51_hash":
                    U = self.n51_hash.transform(X, centered=True)
                else:
                    U = self.n16k.transform(X)
                    if spec.nose == "n51_svd":
                        U = self.n51_svd.transform(U, centered=True)
                if spec.kind == "fly" and spec.matrix is not None:
                    fitted.model.score(self._code_batch(spec.code_key, U))
                else:
                    fitted.model.score(U)
        secs = time.perf_counter() - t0
        return {"ms_per_doc": 1000.0 * secs / max(len(ids), 1), "n_docs": len(ids), "n_windows": len(texts),
                "seconds": secs}

    # -- housekeeping -----------------------------------------------------------------------------------------------
    def cache_bytes(self) -> int:
        return self._cache_bytes()

    def cleanup(self) -> int:
        """Delete this seed's feature cache directory and drop the in-memory caches; returns bytes freed."""
        freed = self._cache_bytes()
        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._counts.clear()
        self._features.clear()
        self._codes.clear()
        self._guard_scores.clear()
        return freed


# ----------------------------------------------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------------------------------------------
class Evaluator:
    """Document-level statistics of ТЗ Этап 4 with the seed's bootstrap settings (``n`` from config / smoke,
    seed = the ``bootstrap`` child, percentile intervals, clusters = ``cluster_id`` inside each source)."""

    def __init__(self, fc: FeatureContext) -> None:
        self.fc = fc
        st = fc.cfg.default["stats"]
        self.n_boot = fc.ctx.bootstrap_n()
        self.alpha = float(st["bootstrap"]["alpha"])
        self.tost_level = float(st["tost"]["ci"])
        self.seed = fc.seeds["bootstrap"]

    # -- AUC ----------------------------------------------------------------------------------------------------------
    def auc_ci(self, df: pd.DataFrame, score: str = "score") -> CI:
        return macro_auc_bootstrap({"s": df}, self.n_boot, self.seed, self.alpha, score=score)

    def macro_auc_ci(self, by_source: Mapping[str, pd.DataFrame], score: str = "score") -> CI:
        return macro_auc_bootstrap(dict(by_source), self.n_boot, self.seed, self.alpha, score=score)

    def diff(self, by_source: Mapping[str, pd.DataFrame], a: str, b: str) -> dict[str, Any]:
        """Paired difference macroAUC(a) − macroAUC(b) (one source -> AUC difference) on the same cluster draws:
        ``{"ci95": CI, "ci90": CI, "p": two-sided percentile-bootstrap p}``; the 90 % interval is cut from the
        same draws as the 95 % one, so TOST and the verdict p-value agree."""
        ci = macro_auc_bootstrap(dict(by_source), self.n_boot, self.seed, self.alpha, score=a, reference=b)
        ci90 = percentile_ci(ci.samples, ci.point, 1.0 - self.tost_level, ci.n, ci.n_clusters)
        return {"ci95": ci, "ci90": ci90, "p": bootstrap_p(ci.samples)}

    # -- rates at thresholds ----------------------------------------------------------------------------------------
    @staticmethod
    def _slim(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
        """The cluster column and the score columns only: ``cluster_bootstrap`` re-indexes the frame once per
        draw, and a document table carries a dozen detector and string columns that the rate statistic never
        reads (the E1 smoke profile spent a third of the evaluation in that ``iloc``). Same rows, same draws,
        same numbers."""
        keep = ["cluster_id", *dict.fromkeys(cols)]
        return df[keep].reset_index(drop=True)

    def rate_ci(self, df: pd.DataFrame, tau: float, score: str = "score") -> CI:
        """Share of documents with ``score >= tau`` (TPR on positives, FPR on negatives) with a cluster CI."""
        return cluster_bootstrap(self._slim(df, score), lambda d: fpr_at_threshold(d[score].to_numpy(), tau),
                                 self.n_boot, self.seed, self.alpha)

    def rate_diff_ci(self, df: pd.DataFrame, a: str, tau_a: float, b: str, tau_b: float) -> CI:
        """Paired difference of two detectors' rates at their own thresholds on the same documents (H2)."""
        return paired_cluster_bootstrap(self._slim(df, a, b), lambda d: fpr_at_threshold(d[a].to_numpy(), tau_a),
                                        lambda d: fpr_at_threshold(d[b].to_numpy(), tau_b), self.n_boot, self.seed,
                                        self.alpha)

    def fpr_target(self) -> float | None:
        """The E0 FPR target: ``fpr_target`` of ``results/power.json`` when it exists, else the ТЗ Этап 0 rule on
        |P_test| (``eval.thresholds.fpr_target_for_pool``); ``None`` withdraws TPR@FPR (AUC only)."""
        path = results_mod.power_path(self.fc.ctx.root, self.fc.ctx.smoke)
        if path.exists():
            try:
                val = read_json(path).get("fpr_target")
                return None if val is None else float(val)
            except (OSError, ValueError):
                pass
        return fpr_target_for_pool(self.fc.ctx.p_test_size, self.fc.cfg)

    def tau_fpr(self, p_val_scores: np.ndarray, fpr: float | None = None) -> dict[str, Any] | None:
        """Frozen τ_FPR on P_val (ТЗ 2.5) at the E0 target; ``None`` when the target is withdrawn."""
        target = self.fpr_target() if fpr is None else fpr
        if target is None:
            return None
        return tau_fpr_record(np.asarray(p_val_scores, dtype=float), fpr=target, source="P_val", cfg=self.fc.cfg,
                              n_test_pool=self.fc.ctx.p_test_size)

    def tau_tpr(self, pos_scores: np.ndarray, source: str, tpr: float | None = None) -> dict[str, Any]:
        return tau_tpr_record(np.asarray(pos_scores, dtype=float), source, tpr=tpr, cfg=self.fc.cfg)


def by_source(frame: pd.DataFrame, sources: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """Split a document frame into ``{source: frame}`` (only sources with both classes when ``sources`` is None)."""
    out = {}
    for s, g in frame.groupby("source", sort=True):
        if sources is not None and s not in sources:
            continue
        if sources is None and g["label"].nunique() < 2:
            continue
        out[str(s)] = g.reset_index(drop=True)
    return out


DEFAULT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("auc/deep", "tfidf_lr", "protectai_v2"), ("auc/dojo", "tfidf_lr", "protectai_v2"),                 # H1a template
    ("auc/para_deep", "protectai_v2", "tfidf_lr"), ("auc/bipia", "protectai_v2", "tfidf_lr"),            # H1a semantic
    ("auc/dyn", "protectai_v2", "tfidf_lr"),
    ("macro_auc", "real_fly_linear", "lr_svd"), ("macro_auc", "flyhash_linear", "tfidf_lr"),            # H1b (i)
    ("macro_auc", "real_fly_bloom", "tfidf_lr"), ("macro_auc", "flyhash_bloom", "tfidf_lr"),            # H1b (ii) full
    ("macro_auc", "real_fly_bloom", "protectai_v2"), ("macro_auc", "real_fly_linear", "protectai_v2"),
)
H2_PAIRS: tuple[tuple[str, str], ...] = (("real_fly_bloom", "protectai_v2"), ("real_fly_linear", "protectai_v2"),
                                         ("protectai_v2", "piguard"))


def standard_evaluation(fc: FeatureContext, fitted: Mapping[str, FittedDetector], sources: Sequence[str] | None = None,
                        pairs: Sequence[tuple[str, str, str]] | None = None, h2_pairs: Sequence[tuple[str, str]] | None = None,
                        validation: bool = True, latency: bool = True, latency_guards: bool = False,
                        notinject: bool = True) -> dict[str, Any]:
    """The E1-style evaluation of a set of fitted detectors (design_experiments §2 step 6) -> ``{"numbers",
    "tables", "thresholds", "notes", "doc_tables"}`` in the results format.

    Steps: score every test source (dedup-excluded windows dropped), P_val and (optionally) the validation sets;
    per-source AUC and macroAUC with CIs; paired differences for ``pairs`` (default: the hypothesis pairs of ТЗ Этап 4
    that both detectors allow); τ_FPR per detector on P_val with TPR@τ per positive source and FPR on P_test; τ_90 on
    deep (and dojo) test positives; FPR on NotInject at τ_FPR and τ_90(deep) overall, by subset and language stratum,
    with the H2 paired differences; validated hyperparameters, latency and state size. ``para_deep`` is the deep
    paraphrase stratum evaluated as its own source (H1a semantic half).
    """
    ev = Evaluator(fc)
    ctx = fc.ctx
    numbers: dict[str, Any] = {}
    tables: dict[str, list[dict[str, Any]]] = {"detectors": [], "sources": []}
    thresholds: dict[str, Any] = {}
    notes: list[str] = []
    sources = list(sources if sources is not None else ctx.test_sources)
    dets = {n: f for n, f in fitted.items() if f.available}
    for n, f in fitted.items():
        if not f.available:
            notes.append(f"{n}: unavailable ({f.status.get('path')}), skipped")
    for n, f in fitted.items():
        row = {"detector": n, "kind": f.spec.kind, "nose": f.spec.nose, "matrix": f.spec.matrix,
               "readout": f.spec.readout, "available": f.available, "fit_seconds": f.fit_seconds,
               "n_train_windows": f.n_train, "train_set": f.train_set}
        row.update({k: v for k, v in f.choices.items() if not isinstance(v, (dict, list))})
        tables["detectors"].append(row)
        for hp in ("gamma", "C"):
            if hp in f.choices:
                numbers[f"hyper/{n}/{hp}"] = results_mod.number(f.choices[hp], note=str(f.choices.get(f"{hp}_source")))
        if f.available:
            numbers[f"state_bytes/{n}"] = results_mod.number(f.state_size_bytes())

    # -- scores per test source -------------------------------------------------------------------------------------
    doc_tables: dict[str, pd.DataFrame] = {}
    for s in sources:
        ws = fc.window_set(f"test:{s}")
        scores = fc.score_many(dets, ws)
        df = fc.doc_frame(scores, ws)
        doc_tables[s] = df
        tables["sources"].append({"source": s, "n_docs": int(len(df)), "n_pos": int((df["label"] == 1).sum()),
                                  "n_neg": int((df["label"] == 0).sum()), "n_clusters": int(df["cluster_id"].nunique()),
                                  "n_windows": ws.n})
    if "para" in doc_tables:
        deep = doc_tables["para"][doc_tables["para"]["stratum"] == "deep"]
        if len(deep):
            doc_tables["para_deep"] = deep.reset_index(drop=True)
    pos_sources = {s: df for s, df in doc_tables.items() if s in POSITIVE_SOURCES and df["label"].nunique() == 2}

    # -- AUC by source and macroAUC ---------------------------------------------------------------------------------
    for det in dets:
        for s, df in doc_tables.items():
            if s == "notinject" or det not in df.columns or df["label"].nunique() < 2:
                continue
            numbers[f"auc/{s}/{det}"] = results_mod.number(None, ev.auc_ci(df, det))
        present = {s: df for s, df in pos_sources.items() if det in df.columns}
        if present:
            numbers[f"macro_auc/{det}"] = results_mod.number(None, ev.macro_auc_ci(present, det),
                                                             note="sources=" + ",".join(sorted(present)))

    # -- paired differences -----------------------------------------------------------------------------------------
    for metric, a, b in (pairs if pairs is not None else DEFAULT_PAIRS):
        if a not in dets or b not in dets:
            continue
        if metric == "macro_auc":
            frames = {s: df for s, df in pos_sources.items() if a in df.columns and b in df.columns}
        else:
            s = metric.split("/", 1)[1]
            df = doc_tables.get(s)
            frames = {s: df} if df is not None and df["label"].nunique() == 2 and a in df.columns and b in df.columns else {}
        if not frames:
            continue
        d = ev.diff(frames, a, b)
        numbers[f"diff/{metric}/{a}-{b}"] = results_mod.number(None, d["ci95"], p=d["p"])
        numbers[f"diff90/{metric}/{a}-{b}"] = results_mod.number(None, d["ci90"])

    # -- thresholds: τ_FPR on P_val, TPR@FPR on test positives, FPR on P_test, τ_90 --------------------------------
    p_val_ws = fc.window_set("p_val")
    p_val_df = fc.doc_frame(fc.score_many(dets, p_val_ws), p_val_ws)
    p_test_ids = ctx.p_test_doc_ids
    target = ev.fpr_target()
    if target is None:
        notes.append(f"TPR@FPR withdrawn: |P_test|={ctx.p_test_size} below the E0 carrier rule (AUC only)")
    taus: dict[str, dict[str, float]] = {}
    for det in dets:
        if det not in p_val_df.columns:
            continue
        taus[det] = {}
        rec = ev.tau_fpr(p_val_df[det].to_numpy(), target)
        if rec is not None:
            thresholds[f"tau_fpr/{det}"] = rec
            taus[det]["tau_fpr"] = rec["value"]
            for s, df in pos_sources.items():
                if det in df.columns:
                    pos = df[df["label"] == 1]
                    numbers[f"tpr_at_fpr/{s}/{det}"] = results_mod.number(None, ev.rate_ci(pos, rec["value"], det))
            pool = pd.concat([df[df["doc_id"].isin(p_test_ids)] for s, df in doc_tables.items()
                              if s != "para_deep" and det in df.columns], ignore_index=True)
            if len(pool):
                numbers[f"fpr_ptest/{det}"] = results_mod.number(None, ev.rate_ci(pool, rec["value"], det))
        for s, key in (("deep", "tau90_deep"), ("dojo", "tau90_dojo")):
            df = doc_tables.get(s)
            if df is not None and det in df.columns and (df["label"] == 1).any():
                rec90 = ev.tau_tpr(df.loc[df["label"] == 1, det].to_numpy(), s)
                thresholds[f"{key}/{det}"] = rec90
                taus[det][key] = rec90["value"]

    # -- NotInject --------------------------------------------------------------------------------------------------
    ni = doc_tables.get("notinject") if notinject else None
    if ni is not None and len(ni):
        for det, tmap in taus.items():
            if det not in ni.columns:
                continue
            for tkey, tau in tmap.items():
                if tkey == "tau90_dojo":
                    continue
                numbers[f"fpr_notinject/{tkey}/{det}"] = results_mod.number(None, ev.rate_ci(ni, tau, det))
                for col in ("subset", "lang_stratum"):
                    for val, g in ni.groupby(col, sort=True):
                        if val is None or len(g) == 0:
                            continue
                        numbers[f"fpr_notinject/{tkey}/{det}/{val}"] = results_mod.number(None, ev.rate_ci(g, tau, det))
        for a, b in (h2_pairs if h2_pairs is not None else H2_PAIRS):
            if a in taus and b in taus and "tau90_deep" in taus[a] and "tau90_deep" in taus[b]:
                ci = ev.rate_diff_ci(ni, a, taus[a]["tau90_deep"], b, taus[b]["tau90_deep"])
                numbers[f"diff/fpr_notinject/tau90_deep/{a}-{b}"] = results_mod.number(None, ci, p=bootstrap_p(ci.samples))

    # -- validation (preconditions of H2 / H3) ----------------------------------------------------------------------
    if validation:
        val_frames: dict[str, pd.DataFrame] = {}
        for s in ("deep", "bipia", "dojo"):
            ws = fc.window_set(f"val:{s}")
            if ws.n == 0:
                continue
            df = fc.doc_frame(fc.score_many(dets, ws), ws)
            if df["label"].nunique() == 2:
                val_frames[s] = df
        for det in dets:
            present = {s: df for s, df in val_frames.items() if det in df.columns}
            for s, df in present.items():
                numbers[f"val_auc/{s}/{det}"] = results_mod.number(auc_point(df[det].to_numpy(), df["label"].to_numpy()),
                                                                   n=len(df))
            if present:
                numbers[f"val_macro_auc/{det}"] = results_mod.number(
                    macro_auc({s: auc_point(df[det].to_numpy(), df["label"].to_numpy()) for s, df in present.items()}),
                    note="sources=" + ",".join(sorted(present)))

    # -- latency ----------------------------------------------------------------------------------------------------
    if latency:
        lat_set = "test:deep" if "deep" in sources else f"test:{sources[0]}"
        for det, f in dets.items():
            lat = fc.latency_ms(f, lat_set, guards=latency_guards)
            if lat is not None:
                numbers[f"latency_ms/{det}"] = results_mod.number(lat["ms_per_doc"], n=lat["n_docs"])
            elif f.spec.kind == "guard":
                notes.append(f"latency of {det} not measured (guards={latency_guards})")
    notes.append(f"test reads: {len(ctx.test_reads)}; bootstrap n={ev.n_boot}, alpha={ev.alpha}; fpr_target={target}")
    return {"numbers": numbers, "tables": tables, "thresholds": thresholds, "notes": notes, "doc_tables": doc_tables}


# ----------------------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------------------
class ResultBuilder:
    """Accumulates the four result sections; ``merge`` takes the output of :func:`standard_evaluation`."""

    def __init__(self) -> None:
        self.numbers: dict[str, Any] = {}
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.thresholds: dict[str, Any] = {}
        self.notes: list[str] = []
        self.extra: dict[str, Any] = {}

    def add_number(self, key: str, value: Any, ci: Any = None, n: int | None = None, note: str | None = None,
                   **extra: Any) -> None:
        self.numbers[results_mod.check_key(key)] = results_mod.number(value, ci, n, note, **extra)

    def add_ci(self, key: str, ci: CI | Mapping[str, Any], note: str | None = None, **extra: Any) -> None:
        self.add_number(key, None, ci, note=note, **extra)

    def add_table(self, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        self.tables.setdefault(name, []).extend(dict(r) for r in rows)

    def add_threshold(self, name: str, record: Mapping[str, Any]) -> None:
        self.thresholds[name] = dict(record)

    def note(self, text: str) -> None:
        self.notes.append(str(text))

    def merge(self, out: Mapping[str, Any]) -> None:
        for k, v in (out.get("numbers") or {}).items():
            self.numbers[results_mod.check_key(k)] = v
        for k, rows in (out.get("tables") or {}).items():
            self.add_table(k, rows)
        self.thresholds.update(out.get("thresholds") or {})
        self.notes.extend(out.get("notes") or [])


class Runner:
    """Runs one experiment body for one global seed and writes ``results/<E>/<seed>.json``.

    ``body(fc, rb)`` receives the :class:`FeatureContext` and a :class:`ResultBuilder`. The run is skipped when the
    result file exists with the current ``config_hash`` (``run_all.sh`` idempotency) unless ``force``; the seed's
    feature cache is deleted afterwards unless ``keep_cache``; test reads are journaled with the purpose
    ``"<experiment> seed=<seed>"``.
    """

    def __init__(self, experiment: str, seed: int, smoke: bool = False, root: Path = ROOT, cfg: Configs | None = None,
                 ctx: Context | None = None, keep_cache: bool = False, force: bool = False,
                 access_log: Callable | None = None, guard_factory: Callable | None = None,
                 cache: bool = True) -> None:
        self.experiment, self.seed, self.smoke, self.root = str(experiment), int(seed), bool(smoke), Path(root)
        self.cfg = cfg or (ctx.cfg if ctx is not None else None)
        self._ctx = ctx
        self.keep_cache, self.force, self.cache = bool(keep_cache), bool(force), bool(cache)
        self._access_log, self._guard_factory = access_log, guard_factory
        self.purpose = f"{self.experiment} seed={self.seed}"

    @property
    def ctx(self) -> Context:
        if self._ctx is None:
            self._ctx = Context(self.cfg, smoke=self.smoke, root=self.root, access_log=self._access_log)
        return self._ctx

    def should_skip(self) -> bool:
        return results_mod.is_current(self.experiment, self.seed, self.smoke, self.root)

    def result_path(self) -> Path:
        return results_mod.result_path(self.experiment, self.seed, self.smoke, self.root)

    def run(self, body: Callable[[FeatureContext, ResultBuilder], Any]) -> Path:
        if not self.force and self.should_skip():
            return self.result_path()
        fc = FeatureContext(self.ctx, self.seed, purpose=self.purpose, cache=self.cache,
                            guard_factory=self._guard_factory)
        rb = ResultBuilder()
        t0 = time.perf_counter()
        n_reads_before = len(self.ctx.test_reads)
        try:
            body(fc, rb)
            timing = {"seconds": time.perf_counter() - t0, "cache_bytes": fc.cache_bytes(),
                      "test_reads": [s for s, _ in self.ctx.test_reads[n_reads_before:]]}
            rb.notes.append(f"seed children: {fc.seeds}")
            return results_mod.write_result(self.experiment, self.seed, rb.numbers, rb.tables, rb.thresholds, rb.notes,
                                            self.smoke, root=self.root, seeds=fc.seeds, timing=timing,
                                            extra=rb.extra or None)
        finally:
            if not self.keep_cache:
                fc.cleanup()
