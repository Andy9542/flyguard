"""config.config_hash: execution-only keys stay out of the hash, everything that can change a result stays in."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from flyguard.config import ROOT, UNHASHED_KEYS, config_hash, hashed_view


@pytest.fixture
def cfg_root(tmp_path: Path) -> Path:
    """A copy of the repository's configs/ tree (configs only, no data)."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    return tmp_path


def _edit(root: Path, rel: str, dotted: str, value) -> None:
    path = root / rel
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = data
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node[part]
    node[leaf] = value
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _get(data, dotted: str):
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return KeyError
        data = data[part]
    return data


def test_every_unhashed_key_exists_in_the_repository_configs():
    """Guards the list against rot: a renamed key would silently re-enter the hash."""
    for rel, keys in UNHASHED_KEYS.items():
        data = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        for dotted in keys:
            assert _get(data, dotted) is not KeyError, f"{rel}: {dotted} not found"


@pytest.mark.parametrize("rel,dotted,value", [
    ("configs/default.yaml", "baselines.transformers.num_threads", 16),
    ("configs/default.yaml", "baselines.transformers.batch_size", 8),
    ("configs/default.yaml", "baselines.transformers.cache_dir", "elsewhere/scores"),
    ("configs/operator.yaml", "compute.cpu_cores", 64),
    ("configs/operator.yaml", "compute", {"cpu_cores": 2, "ram_gb": 8, "gpu": True, "wall_clock_hours": 1}),
    ("configs/operator.yaml", "llm_api.budget_usd", 99.0),
    ("configs/operator.yaml", "llm_api.reserve_for_paraphrases_usd", 0.0),
    ("configs/operator.yaml", "llm_api.prices_usd_per_million", {}),
    ("configs/operator.yaml", "llm_api.key_file", "/elsewhere/.env"),
    ("configs/operator.yaml", "llm_api.avoid_peak_hours", False),
    ("configs/operator.yaml", "hf.token_env", "OTHER_TOKEN"),
])
def test_execution_only_keys_leave_the_hash_unchanged(cfg_root, rel, dotted, value):
    before = config_hash(cfg_root)
    _edit(cfg_root, rel, dotted, value)
    assert config_hash(cfg_root) == before


@pytest.mark.parametrize("rel,dotted,value", [
    ("configs/default.yaml", "stats.tost.delta_rel", 0.06),
    ("configs/default.yaml", "windows.size", 300),
    ("configs/default.yaml", "baselines.transformers.comparator", "piguard"),
    ("configs/default.yaml", "smoke.n_null", 5),
    ("configs/operator.yaml", "shared.traces_dir", "/elsewhere/traces"),
    ("configs/operator.yaml", "llm_api.agent_models", ["other-model"]),
    ("configs/experiments/E4.yaml", "n_curveball", 100),
])
def test_result_affecting_keys_change_the_hash(cfg_root, rel, dotted, value):
    before = config_hash(cfg_root)
    _edit(cfg_root, rel, dotted, value)
    assert config_hash(cfg_root) != before


def test_text_files_hash_by_bytes_and_yaml_comments_do_not_count(cfg_root):
    before = config_hash(cfg_root)
    default = cfg_root / "configs/default.yaml"
    default.write_text(default.read_text(encoding="utf-8") + "\n# an operator note, not a setting\n", encoding="utf-8")
    assert config_hash(cfg_root) == before  # YAML by parsed content
    regex = cfg_root / "configs/regex_patterns.txt"
    regex.write_bytes(regex.read_bytes() + b"\n")
    after_regex = config_hash(cfg_root)
    assert after_regex != before  # regexes byte for byte
    prompt = sorted((cfg_root / "configs/prompts").glob("*.txt"))[0]
    prompt.write_bytes(prompt.read_bytes() + b" ")
    assert config_hash(cfg_root) != after_regex  # prompts byte for byte
    (cfg_root / "configs/experiments/E6.yaml").unlink()
    assert config_hash(cfg_root) not in (before, after_regex)


def test_missing_configs_and_hashed_view(tmp_path):
    assert config_hash(tmp_path) == config_hash(tmp_path)  # every file "<missing>", still deterministic
    a = hashed_view("configs/default.yaml", b"b: 1\na: {x: 2}\n")
    b = hashed_view("configs/default.yaml", b"# note\na:\n  x: 2\nb: 1\n")
    assert a == b == b'{"a":{"x":2},"b":1}'
    assert hashed_view("configs/regex_patterns.txt", b"x\n") == b"x\n"
    drop = hashed_view("configs/default.yaml", b"baselines: {transformers: {num_threads: 4, comparator: p}}\n")
    assert drop == b'{"baselines":{"transformers":{"comparator":"p"}}}'
