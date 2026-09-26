"""Industrial detectors of ТЗ 3.2: ProtectAI v2 (comparator), PIGuard, Prompt Guard 2 (optional), inference only.

``GuardModel(name, cfg)`` loads a local HF snapshot from ``cfg.default['baselines']['transformers']['models']``
with ``from_pretrained(local_dir, local_files_only=True)`` (PIGuard needs ``trust_remote_code=True``; its
``modeling_piguard.py`` ships in the snapshot), runs CPU batches, truncates at ``max_length`` and returns the
softmax probability of the positive label. Scores are cached by ``text_hash`` in an append-only parquet under
``cache_dir`` (ТЗ 3.2 "оценки кешируются по хешу окна") so that ten seeds and E6 never re-run a transformer on the
same window. Main runs use the common 256-character windows (ТЗ 1.3, one window scheme for every detector);
``token_windows`` builds the 512-token windows that only E6 uses. ``torch``/``transformers`` are imported lazily
so that importing this module (and the test suite) stays cheap and network-free.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flyguard.config import ROOT, Configs
from flyguard.io import atomic_write_bytes
from flyguard.baselines.common import clip01, text_hash

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                "pytorch_model.bin.index.json")


def resolve_positive_index(id2label: Mapping[Any, Any], positive_label: str | None) -> tuple[int, str, str]:
    """Index of the positive (injection) class from ``config.json``'s ``id2label`` -> ``(index, label, how)``.

    ТЗ 3.2 scores are "the probability of the positive label"; the configured ``positive_label`` is verified
    against the model's own label map instead of trusting an index. Resolution order, recorded in ``how``:
    ``exact`` (configured name found), ``case_insensitive`` (same name up to case), ``contains_inject`` (the single
    label containing "inject", case-insensitively — the fallback for a model card that renames its labels),
    otherwise ``ValueError`` so a mis-configured comparator can never be scored silently.
    """
    labels = {int(k): str(v) for k, v in id2label.items()}
    if positive_label is not None:
        exact = [i for i, lab in labels.items() if lab == str(positive_label)]
        if len(exact) == 1:
            return exact[0], labels[exact[0]], "exact"
        ci = [i for i, lab in labels.items() if lab.lower() == str(positive_label).lower()]
        if len(ci) == 1:
            return ci[0], labels[ci[0]], "case_insensitive"
    inj = [i for i, lab in labels.items() if "inject" in lab.lower()]
    if len(inj) == 1:
        return inj[0], labels[inj[0]], "contains_inject"
    raise ValueError(f"cannot resolve positive label {positive_label!r} in id2label {labels!r}")


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64 (numerically shifted): the ТЗ 3.2 score is the softmax probability of the
    positive label; done in numpy so the stubbed tests need no torch."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def token_windows(text: str, tokenizer: Any, max_length: int, stride_ratio: float) -> list[tuple[int, int, str]]:
    """512-token windows for E6 (ТЗ 1.3 "512-токенные окна трансформеров только в E6") -> ``[(start, end, text)]``.

    The window holds ``max_length - num_special_tokens`` content tokens so that the model sees a full window plus
    its ``[CLS]``/``[SEP]`` without truncation; the stride is ``round(size * stride_ratio)`` with ``stride_ratio =
    windows.stride / windows.size`` from the config (192/256 = 0.75, the character scheme's ratio, so token and
    character windows overlap by the same fraction). Windows are cut with the fast tokenizer's offset mapping and
    returned as slices of the *original* text (so ``text_hash`` of a window is reproducible and no decoding
    artefacts appear); a tokenizer without offsets falls back to ``decode``. A text of at most ``size`` tokens is
    one window; otherwise starts are ``0, stride, 2*stride, ...`` plus a final window ending exactly at the last
    token, so the tail is always covered by a full-size window.
    """
    n_special = int(tokenizer.num_special_tokens_to_add()) if hasattr(tokenizer, "num_special_tokens_to_add") else 2
    size = max(1, int(max_length) - n_special)
    stride = max(1, int(round(size * float(stride_ratio))))
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping") if hasattr(enc, "get") else None
    ids = enc["input_ids"]
    n = len(ids)
    if n <= size:
        return [(0, len(text), text)]
    starts = list(range(0, n - size, stride))
    if not starts or starts[-1] != n - size:
        starts.append(n - size)
    windows: list[tuple[int, int, str]] = []
    for s in starts:
        e = s + size
        if offsets is not None:
            cs, ce = int(offsets[s][0]), int(offsets[e - 1][1])
            windows.append((cs, ce, text[cs:ce]))
        else:  # pragma: no cover - slow tokenizers only
            windows.append((s, e, tokenizer.decode(ids[s:e])))
    return windows


class ScoreCache:
    """Append-only parquet cache ``text_hash -> score`` (design §6: dedup by hash, never rewritten destructively).

    Columns: ``text_hash`` (str), ``score`` (float64), ``model`` (str), ``max_length`` (int32), ``created_at``
    (UTC ISO string). ``append`` keeps every existing row, drops incoming hashes that are already present (first
    value wins), and replaces the file atomically through ``flyguard.io``.
    """

    COLUMNS = ("text_hash", "score", "model", "max_length", "created_at")

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                                 zip(self.COLUMNS, ("string", "float64", "string", "int32", "string"))})
        return pq.read_table(self.path).to_pandas()

    def load(self) -> dict[str, float]:
        df = self.read()
        if df.empty:
            return {}
        return dict(zip(df["text_hash"].astype(str).tolist(), df["score"].astype(float).tolist()))

    def get_many(self, hashes: Sequence[str]) -> dict[str, float]:
        want = set(hashes)
        return {h: s for h, s in self.load().items() if h in want}

    def append(self, scores: Mapping[str, float], model: str, max_length: int) -> int:
        """Add new hashes; returns the number of rows written (0 when everything was already cached)."""
        existing = self.read()
        known = set(existing["text_hash"].astype(str).tolist()) if not existing.empty else set()
        rows = [(h, float(s)) for h, s in scores.items() if h not in known]
        if not rows:
            return 0
        seen: set[str] = set()
        rows = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new = pd.DataFrame({"text_hash": pd.Series([r[0] for r in rows], dtype="string"),
                            "score": np.array([r[1] for r in rows], dtype=np.float64),
                            "model": pd.Series([model] * len(rows), dtype="string"),
                            "max_length": np.full(len(rows), int(max_length), dtype=np.int32),
                            "created_at": pd.Series([stamp] * len(rows), dtype="string")})
        merged = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), buf, compression="zstd")
        atomic_write_bytes(self.path, buf.getvalue())
        return len(rows)


class GuardModel:
    """Transformer prompt-injection classifier as a ``WindowScorer`` over strings (ТЗ 3.2).

    ``name`` is a key of ``cfg.default['baselines']['transformers']['models']`` (``protectai_v2`` is the
    comparator fixed before any test read; ``piguard``; ``prompt_guard_2`` optional). ``available`` is False when
    the snapshot directory or its weights are missing (Prompt Guard 2 without an HF token, BLOCKERS B2): the
    engine records the skip instead of failing. Loading is lazy and injectable (``loader(self) -> (tokenizer,
    model)``) so tests stub the model. ``score(texts, hashes=None)`` returns the positive-class probability per
    text, served from the parquet cache when the ``text_hash`` is known and appended otherwise. Batches of
    ``batch_size`` texts are sorted by length to reduce padding; ``torch.set_num_threads`` uses
    ``cfg.operator['compute']['cpu_cores']``.

    Main runs score the common 256-character windows (ТЗ 1.3), each truncated by the tokenizer at ``max_length``
    (a 256-character window is far below 512 tokens, so nothing is cut). E6 only (``score_long``) re-windows a
    document with the model's own tokenizer: with the configured ``max_length`` 512 and the two special tokens
    of these encoders, a token window holds 510 content tokens plus ``[CLS]``/``[SEP]`` (510 + 2), the stride is
    ``round(510 * 192/256) = round(382.5) = 382`` tokens (Python rounds the half to even; overlap 128 tokens),
    starts are ``0, 382, 764, ...`` plus a last window ending on the final token, and the document score is the
    max over its token windows (ТЗ 1.3 document rule). Token windows go through the same hash cache as any text.
    """

    def __init__(self, name: str, cfg: Configs, loader: Callable[["GuardModel"], tuple[Any, Any]] | None = None,
                 root: Path = ROOT, cache_dir: str | Path | None = None, use_cache: bool = True):
        tcfg = cfg.default["baselines"]["transformers"]
        if name not in tcfg["models"]:
            raise KeyError(f"unknown transformer baseline {name!r}; configured: {sorted(tcfg['models'])}")
        self.name = name
        self.spec = dict(tcfg["models"][name])
        path = Path(self.spec["path"])
        self.path = path if path.is_absolute() else Path(root) / path
        self.max_length = int(self.spec["max_length"])
        self.positive_label = self.spec.get("positive_label")
        self.trust_remote_code = bool(self.spec.get("trust_remote_code", False))
        self.optional = bool(self.spec.get("optional", False))
        self.batch_size = int(tcfg.get("batch_size", 32))
        self.num_threads = int(cfg.operator.get("compute", {}).get("cpu_cores", 0) or 0)
        win = cfg.default["windows"]
        self.stride_ratio = float(win["stride"]) / float(win["size"])
        cdir = Path(cache_dir) if cache_dir is not None else Path(tcfg["cache_dir"])
        cdir = cdir if cdir.is_absolute() else Path(root) / cdir
        self.cache: ScoreCache | None = ScoreCache(cdir / f"{name}.parquet") if use_cache else None
        self._loader = loader or _default_loader
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.id2label: dict[int, str] | None = None
        self.positive_index: int | None = None
        self.positive_label_resolved: str | None = None
        self.positive_how: str | None = None
        cfg_file = self.path / "config.json"
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as fh:
                raw = json.load(fh).get("id2label") or {}
            self.id2label = {int(k): str(v) for k, v in raw.items()}
            if self.id2label:
                self.positive_index, self.positive_label_resolved, self.positive_how = resolve_positive_index(
                    self.id2label, self.positive_label)

    # -- availability -------------------------------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when the local snapshot has a config and weights (a missing optional model is a recorded skip)."""
        return self.path.is_dir() and (self.path / "config.json").exists() and any(
            (self.path / f).exists() for f in WEIGHT_FILES)

    def status(self) -> dict[str, Any]:
        """What the results file records about this detector (path, availability, label resolution)."""
        return {"name": self.name, "path": str(self.path), "available": self.available, "optional": self.optional,
                "max_length": self.max_length, "positive_label_config": self.positive_label,
                "positive_index": self.positive_index, "positive_label": self.positive_label_resolved,
                "positive_how": self.positive_how, "id2label": self.id2label,
                "cache": str(self.cache.path) if self.cache else None}

    # -- protocol -----------------------------------------------------------------------------------------------
    def fit(self, X_train: Any = None, y_train: Any = None, X_val: Any = None, y_val: Any = None,
            groups: Any = None) -> "GuardModel":
        """No training: ТЗ 3.2 evaluates the released checkpoints as they are (``groups`` accepted, unused)."""
        return self

    def load(self) -> "GuardModel":
        if self._model is None:
            if not self.available:
                raise RuntimeError(f"{self.name}: snapshot missing or incomplete at {self.path}")
            self._tokenizer, self._model = self._loader(self)
            if self.positive_index is None:
                id2label = getattr(getattr(self._model, "config", None), "id2label", None) or {}
                self.id2label = {int(k): str(v) for k, v in id2label.items()}
                self.positive_index, self.positive_label_resolved, self.positive_how = resolve_positive_index(
                    self.id2label, self.positive_label)
        return self

    def load_tokenizer(self) -> Any:
        """Tokenizer only (for ``token_windows`` without paying for the weights)."""
        if self._tokenizer is None:
            if self._model is not None:
                return self._tokenizer
            if self._loader is _default_loader:
                self._tokenizer = _default_tokenizer(self)
            else:
                self.load()
        return self._tokenizer

    def token_windows(self, text: str) -> list[tuple[int, int, str]]:
        """E6 windows of this model's tokenizer (``max_length`` tokens, config stride ratio)."""
        return token_windows(text, self.load_tokenizer(), self.max_length, self.stride_ratio)

    def score(self, X: Sequence[str], hashes: Sequence[str] | None = None) -> np.ndarray:
        """Positive-class probability per text; cached by ``text_hash`` (pass ``hashes`` to skip re-hashing)."""
        texts = [str(t) for t in X]
        keys = list(hashes) if hashes is not None else [text_hash(t) for t in texts]
        if len(keys) != len(texts):
            raise ValueError("hashes and texts differ in length")
        known: dict[str, float] = self.cache.get_many(keys) if self.cache is not None else {}
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in known and k not in todo:
                todo[k] = t
        if todo:
            fresh = self._predict_many(list(todo.values()))
            new_scores = dict(zip(todo.keys(), fresh.tolist()))
            if self.cache is not None:
                self.cache.append(new_scores, self.name, self.max_length)
            known.update(new_scores)
        return clip01([known[k] for k in keys])

    def score_long(self, X: Sequence[str]) -> np.ndarray:
        """E6 helper: max over the model's 512-token windows of each text (document rule of ТЗ 1.3).

        The token windows of *all* documents are collected first and scored in one ``score`` call, so the cache
        is read and appended once per call and the batches are filled across documents; the per-document max is
        then taken with ``np.maximum.at``. A document without windows (never produced by ``token_windows``, which
        always returns at least one) would score 0.
        """
        texts = [str(t) for t in X]
        windows: list[str] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            for _, _, w in self.token_windows(t):
                windows.append(w)
                owner.append(i)
        out = np.zeros(len(texts), dtype=np.float64)
        if windows:
            np.maximum.at(out, np.asarray(owner, dtype=np.int64), self.score(windows))
        return clip01(out)

    # -- inference ----------------------------------------------------------------------------------------------
    def _predict_many(self, texts: list[str]) -> np.ndarray:
        self.load()
        order = np.argsort([-len(t) for t in texts], kind="stable")
        probs = np.empty(len(texts), dtype=np.float64)
        for i in range(0, len(texts), self.batch_size):
            idx = order[i:i + self.batch_size]
            probs[idx] = self._predict_batch([texts[j] for j in idx])
        return probs

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        enc = self._tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                              return_tensors="pt")
        ctx = self._torch.inference_mode() if self._torch is not None else contextlib.nullcontext()
        with ctx:
            out = self._model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().float().numpy()
        return softmax(np.asarray(logits))[:, int(self.positive_index)]


def _set_threads(gm: GuardModel, torch_mod: Any) -> None:
    n = gm.num_threads
    if n > 0:
        torch_mod.set_num_threads(min(n, os.cpu_count() or n))


def _default_tokenizer(gm: GuardModel) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(gm.path), local_files_only=True, trust_remote_code=gm.trust_remote_code)


def _default_loader(gm: GuardModel) -> tuple[Any, Any]:
    """Real loader: local snapshot only, CPU, eval mode, threads from ``compute.cpu_cores`` (no network)."""
    import torch
    from transformers import AutoModelForSequenceClassification
    _set_threads(gm, torch)
    tokenizer = _default_tokenizer(gm)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(gm.path), local_files_only=True, trust_remote_code=gm.trust_remote_code)
    model.eval()
    gm._torch = torch
    return tokenizer, model
