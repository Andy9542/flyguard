"""Synthetic fixtures for the fly-model tests (ТЗ 2.6). No real data: texts are random pseudo-words, the
'connectome' is a small random binary matrix with varied degrees."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo",
         "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor",
         "whiskey", "xray", "yankee", "zulu", "please", "ignore", "table", "email", "code", "summary", "answer"]


def make_texts(n: int = 160, seed: int = 1234) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        words = rng.choice(WORDS, int(rng.integers(5, 40)))
        text = " ".join(words)
        if i % 7 == 0:
            text += " Ärger Übung straße café"
        out.append(text)
    return out


def make_matrix(n_cells: int = 60, d: int = 12, seed: int = 7) -> sp.csr_matrix:
    """Binary [n_cells, d] csr with in-degrees cycling through 2..6 (varied rows, so Curveball trades happen)."""
    rng = np.random.default_rng(seed)
    rows = [np.sort(rng.choice(d, 2 + i % 5, replace=False)) for i in range(n_cells)]
    lens = np.array([len(r) for r in rows])
    indptr = np.concatenate([[0], np.cumsum(lens)])
    M = sp.csr_matrix((np.ones(int(lens.sum()), dtype=np.float32), np.concatenate(rows).astype(np.int32), indptr),
                      shape=(n_cells, d))
    M.sort_indices()
    return M


def random_codes(n: int, m: int, k: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Binary csr with exactly k random active cells per row."""
    cols = np.concatenate([np.sort(rng.choice(m, k, replace=False)) for _ in range(n)])
    indptr = np.arange(0, n * k + 1, k)
    return sp.csr_matrix((np.ones(n * k, dtype=np.float32), cols.astype(np.int32), indptr), shape=(n, m))


@pytest.fixture(scope="session")
def texts() -> list[str]:
    return make_texts()


@pytest.fixture(scope="session")
def small_M() -> sp.csr_matrix:
    return make_matrix()
