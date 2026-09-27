"""Experiment context: the tables and manifests of design §2 by role, with test windows behind one journaled door.

Test discipline (ТЗ "Честность эксперимента", CLAUDE.md): the non-test frames (``documents``, ``windows``) are read
from the parquet files with a ``split != "test"`` filter, so no test row is ever in memory until
:meth:`Context.load_test_windows` is called. That method is the *only* accessor of test windows and documents: it
reads one source's test rows, memoises them and journals the read once per source through
``flyguard.netlog.log_data_access(path, "test", purpose)`` (``access_log`` is injectable so tests never touch
``logs/data_access.log``). Everything a validation choice may see -- train, validation, C_unl, P_val -- comes from
the non-test frames.

Roles (ТЗ 1.8–1.10, ``splits.json`` / ``pools.json``):

* ``train_windows``            windows of ``e1.train`` (deepset train, the only labels of E1);
* ``val_windows(sources)``     windows of ``e1.val``; the default is deepset val only, because hyperparameters are
  chosen on labels from deepset train (ТЗ 1.10) -- BIPIA-val and AgentDojo-val documents are available with
  ``sources=("deep", "bipia", "dojo")`` for the validation macroAUC of the H3 precondition and E0's spreads;
* ``c_unl_windows``            texts of ``c_unl`` (train + val documents, labels unused) for SVD / idf / centring;
* ``p_val_windows``            the negative pool P_val (all non-test) for the frozen τ_FPR;
* ``load_test_windows(source)`` / ``test_documents(source)`` / ``p_test_doc_ids`` for the test side.

Smoke mode (``smoke=True``) reads ``data/processed/smoke`` and ``data/manifests/smoke`` (``data.build.output_dirs``).
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

from flyguard.config import ROOT, Configs, load_configs
from flyguard.data.build import output_dirs
from flyguard.io import read_json

AccessLog = Callable[[Path, str, str], None]
SOURCES = ("deep", "bipia", "dojo", "dyn", "para", "notinject")
DOC_META_COLUMNS = ("doc_id", "source", "split", "label", "cluster_id", "lang", "lang_stratum")


def _default_access_log(path: Path, split: str, purpose: str) -> None:
    from flyguard.netlog import log_data_access

    log_data_access(path, split=split, purpose=purpose)


def _read_table(path: Path, filters: list | None) -> pd.DataFrame:
    return pq.read_table(path, filters=filters).to_pandas()


def _postprocess_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Same shape as ``data.build.read_documents``: ``spans`` as pairs, ``meta`` as dicts."""
    df = df.reset_index(drop=True)
    if "spans" in df.columns:
        df["spans"] = [[(int(s["start"]), int(s["end"])) for s in (sp if sp is not None else [])] for sp in df["spans"]]
    df["meta"] = [json.loads(m) if isinstance(m, str) and m else {} for m in df.get("meta_json", [""] * len(df))]
    df["label"] = df["label"].astype(int)
    return df


class Context:
    """Tables, manifests, connectome and guard availability for one run (real or smoke)."""

    def __init__(self, cfg: Configs | None = None, smoke: bool = False, root: Path = ROOT,
                 access_log: AccessLog | None = None) -> None:
        self.cfg = cfg or load_configs()
        self.smoke = bool(smoke)
        self.root = Path(root)
        self.processed_dir, self.manifests_dir = output_dirs(self.root, self.smoke)
        self._access_log = access_log or _default_access_log
        self.documents_path = self.processed_dir / "documents.parquet"
        self.windows_path = self.processed_dir / "windows.parquet"
        self.episodes_path = self.processed_dir / "episodes.parquet"
        self.splits: dict[str, Any] = read_json(self.manifests_dir / "splits.json")
        self.pools: dict[str, Any] = read_json(self.manifests_dir / "pools.json")
        dedup_path = self.manifests_dir / "dedup.json"
        self.dedup: dict[str, Any] | None = read_json(dedup_path) if dedup_path.exists() else None
        nontest = [("split", "!=", "test")]
        self.documents: pd.DataFrame = _postprocess_documents(_read_table(self.documents_path, nontest))
        self.windows: pd.DataFrame = _read_table(self.windows_path, nontest).reset_index(drop=True)
        if (self.documents["split"] == "test").any() or (self.windows["split"] == "test").any():
            raise RuntimeError("test rows leaked into the non-test frames")
        self.episodes: pd.DataFrame | None = (_read_table(self.episodes_path, None)
                                              if self.episodes_path.exists() else None)
        self._test_windows: dict[str, pd.DataFrame] = {}
        self._test_documents: dict[str, pd.DataFrame] = {}
        self.test_reads: list[tuple[str, str]] = []
        self._connectome: tuple | None = None

    # -- sizes and ids (no data read) -------------------------------------------------------------------------------
    @property
    def test_sources(self) -> list[str]:
        """Sources present in the E1 test split (``splits.json``), in the design's order."""
        present = set(self.splits["e1"]["test"])
        return [s for s in SOURCES if s in present] + sorted(present - set(SOURCES))

    def test_doc_ids(self, source: str) -> list[str]:
        return list(self.splits["e1"]["test"].get(source, []))

    @property
    def p_test_doc_ids(self) -> set[str]:
        return set(self.pools["p_test"]["doc_ids"])

    @property
    def p_val_doc_ids(self) -> set[str]:
        return set(self.pools["p_val"]["doc_ids"])

    @property
    def p_test_size(self) -> int:
        """|P_test| -- the pool size the ТЗ Этап 0 carrier rule (1 % / 5 % / AUC only) is defined on."""
        return int(self.pools["p_test"]["n"])

    def bootstrap_n(self) -> int:
        d = self.cfg.default
        return int(d["smoke"]["bootstrap"] if self.smoke else d["stats"]["bootstrap"]["n"])

    def n_null(self) -> int:
        d = self.cfg.default
        return int(d["smoke"]["n_null"] if self.smoke else d["expansion"]["curveball"]["n_null"])

    # -- non-test roles ---------------------------------------------------------------------------------------------
    def _windows_of(self, doc_ids: Iterable[str]) -> pd.DataFrame:
        ids = set(doc_ids)
        return self.windows[self.windows["doc_id"].isin(ids)].reset_index(drop=True)

    @cached_property
    def train_windows(self) -> pd.DataFrame:
        """Windows of ``e1.train`` (deepset train only; ТЗ 1.10)."""
        w = self._windows_of(self.splits["e1"]["train"])
        if len(w) and not (w["source"] == "deep").all():
            raise ValueError("E1 train windows must come from deepset only (ТЗ 1.10)")
        return w

    def val_windows(self, sources: Sequence[str] | str | None = ("deep",)) -> pd.DataFrame:
        """Windows of ``e1.val``; ``sources=None`` or ``"all"`` returns every validation source."""
        w = self._windows_of(self.splits["e1"]["val"])
        if sources is None or sources == "all":
            return w
        keep = [sources] if isinstance(sources, str) else list(sources)
        return w[w["source"].isin(keep)].reset_index(drop=True)

    @cached_property
    def val_windows_deep(self) -> pd.DataFrame:
        return self.val_windows(("deep",))

    @cached_property
    def c_unl_windows(self) -> pd.DataFrame:
        """C_unl (ТЗ 1.9): train + val texts, never test clusters; labels are not used by any caller."""
        w = self._windows_of(self.splits["c_unl"])
        if len(w) and (w["split"] == "test").any():
            raise ValueError("C_unl contains test windows")
        return w

    @cached_property
    def p_val_windows(self) -> pd.DataFrame:
        """Windows of the P_val documents (all non-test by construction, ТЗ 1.8)."""
        w = self._windows_of(self.p_val_doc_ids)
        missing = self.p_val_doc_ids - set(w["doc_id"])
        if missing:
            raise ValueError(f"{len(missing)} P_val documents are not in the non-test windows")
        return w

    # -- the test door ----------------------------------------------------------------------------------------------
    def load_test_windows(self, source: str, purpose: str = "", doc_ids: Iterable[str] | None = None,
                          name: str | None = None) -> pd.DataFrame:
        """The ONLY accessor of test windows (design_experiments §1): reads the test rows of ``source`` from
        ``windows.parquet`` and ``documents.parquet``, memoises them and journals the read once per source with
        ``log_data_access(path, "test", purpose)``. By default the rows are the ``e1.test[source]`` documents;
        ``doc_ids`` with a ``name`` selects another test-only list of that source under its own memo key (E6's
        BIPIA variants, ``splits.json["bipia"]["e6_docs"]["test"]``; E3 folds). Rows excluded by dedup (ТЗ 1.7)
        are returned with their flag; the engine drops them before scoring."""
        if source not in self.splits["e1"]["test"]:
            raise KeyError(f"{source!r} is not a test source of splits.json (present: {self.test_sources})")
        if doc_ids is not None and not name:
            raise ValueError("an explicit doc_ids list needs a name for its memo key")
        key = source if doc_ids is None else f"{source}#{name}"
        if key not in self._test_windows:
            why = f"experiments.context.load_test_windows({key}): {purpose or 'experiment run'}"
            filters = [("split", "==", "test"), ("source", "==", source)]
            self._access_log(self.windows_path, "test", why)
            self._access_log(self.documents_path, "test", why)
            self.test_reads.append((key, why))
            wins = _read_table(self.windows_path, filters).reset_index(drop=True)
            docs = _postprocess_documents(_read_table(self.documents_path, filters))
            ids = set(self.test_doc_ids(source) if doc_ids is None else doc_ids)
            self._test_windows[key] = wins[wins["doc_id"].isin(ids)].reset_index(drop=True)
            self._test_documents[key] = docs[docs["doc_id"].isin(ids)].reset_index(drop=True)
        return self._test_windows[key]

    def test_documents(self, source: str, purpose: str = "", name: str | None = None) -> pd.DataFrame:
        """Test documents of ``source`` (labels, clusters, strata, meta); loads through the same door."""
        key = source if name is None else f"{source}#{name}"
        if key not in self._test_documents:
            if name is not None:
                raise KeyError(f"test list {key!r} was not loaded through load_test_windows")
            self.load_test_windows(source, purpose)
        return self._test_documents[key]

    def documents_for(self, doc_ids: Iterable[str]) -> pd.DataFrame:
        """Document rows (design §2 columns + ``meta``) for the given ids, from the non-test frame and every test
        source already loaded through :meth:`load_test_windows`; ids of unloaded test sources raise."""
        ids = list(dict.fromkeys(doc_ids))
        frames = [self.documents] + list(self._test_documents.values())
        docs = pd.concat(frames, ignore_index=True) if len(frames) > 1 else self.documents
        sub = docs[docs["doc_id"].isin(set(ids))]
        missing = set(ids) - set(sub["doc_id"])
        if missing:
            raise KeyError(f"{len(missing)} documents are unknown or belong to a test source that was not loaded")
        return sub.set_index("doc_id").loc[ids].reset_index()

    # -- connectome and guards --------------------------------------------------------------------------------------
    @property
    def connectome_path(self) -> Path:
        return self.root / "data" / "processed" / "connectome" / "malecns_R.npz"

    def connectome(self) -> tuple:
        """``(M, meta)`` of :func:`flyguard.connectome.load_malecns` (binary [n_kc, n_glomeruli]; weighted in meta)."""
        if self._connectome is None:
            from flyguard.connectome import load_malecns

            self._connectome = load_malecns(self.connectome_path)
        return self._connectome

    def guard_names(self) -> list[str]:
        return list(self.cfg.default["baselines"]["transformers"]["models"])

    def guard(self, name: str, **kwargs: Any):
        """A :class:`flyguard.baselines.transformers_guard.GuardModel` rooted at this context's root."""
        from flyguard.baselines.transformers_guard import GuardModel

        return GuardModel(name, self.cfg, root=self.root, **kwargs)

    def guard_status(self, name: str) -> dict[str, Any]:
        return self.guard(name).status()

    def available_guards(self) -> list[str]:
        return [n for n in self.guard_names() if self.guard(n).available]


def load_test_windows(ctx: Context, source: str, purpose: str = "") -> pd.DataFrame:
    """Module-level alias of the single test accessor (design_experiments §1 names it as a function)."""
    return ctx.load_test_windows(source, purpose)
