"""Synthetic fixtures for flyguard.eval tests: no real data, deterministic seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_source(n: int, n_clusters: int, seed: int, shift: float = 1.0, shift_ref: float | None = None,
                icc: float = 0.3) -> pd.DataFrame:
    """Documents with labels constant inside clusters, a shared cluster effect and two detector scores
    (``score``, ``ref``) on the same documents. True AUC of ``score`` is Φ(shift/√2) under unit variances."""
    rng = np.random.default_rng(seed)
    cluster_of = np.arange(n) % n_clusters
    labels_by_cluster = (np.arange(n_clusters) % 2).astype(int)
    y = labels_by_cluster[cluster_of]
    b = rng.normal(0, np.sqrt(icc), n_clusters)[cluster_of]
    e1 = rng.normal(0, np.sqrt(1 - icc), n)
    e2 = 0.5 * e1 + np.sqrt(0.75) * rng.normal(0, np.sqrt(1 - icc), n)
    shift_ref = shift if shift_ref is None else shift_ref
    return pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(n)],
        "label": y,
        "cluster_id": [f"c{c}" for c in cluster_of],
        "score": shift * y + b + e1,
        "ref": shift_ref * y + b + e2,
    })


@pytest.fixture
def source_factory():
    """The ``make_source`` helper as a fixture, so test modules need no import of this file."""
    return make_source


@pytest.fixture
def three_sources() -> dict[str, pd.DataFrame]:
    return {"deep": make_source(120, 120, 1, shift=1.2), "bipia": make_source(200, 40, 2, shift=0.8),
            "dojo": make_source(300, 30, 3, shift=1.5)}


@pytest.fixture
def ci_dict():
    def _mk(point: float, low: float, high: float, level: float = 0.95) -> dict:
        return {"point": point, "low": low, "high": high, "level": level, "n_boot": 1000, "n": 100}

    return _mk
