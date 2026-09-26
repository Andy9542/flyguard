"""Shared fixtures for tests/agentdojo_io: real configs (not data), synthetic trace trees under tmp_path."""
from __future__ import annotations

from pathlib import Path

import pytest

from flyguard.config import load_configs

from .fixtures import synthetic as S

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return load_configs()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def traces_tree(tmp_path) -> tuple[Path, dict[str, Path]]:
    """`<tmp>/traces/agentdojo/toy-model/toy/...` with every synthetic log; returns (traces root, name -> path)."""
    root = tmp_path / "traces"
    return root, S.write_tree(root)
