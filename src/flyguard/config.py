"""Configs (configs/*.yaml), derived seeds and the frozen-config hash."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = ("configs/operator.yaml", "configs/default.yaml", "configs/regex_patterns.txt")
# Execution-only keys: they steer speed, parallelism, cache location, credentials or the API budget of a stage and
# never a number in results/*.json, so config_hash leaves them out. Tuning them (e.g. ``num_threads`` after the
# smoke run) must not make every result stale and restart the seeds of a run in progress. The transformer score
# cache is keyed by the window's text hash alone, so batch size and threads are already outside the identity of a
# score. Everything else stays in the hash; YAML comments and formatting do not count (see hashed_view).
UNHASHED_KEYS: dict[str, tuple[str, ...]] = {
    "configs/default.yaml": ("baselines.transformers.batch_size", "baselines.transformers.num_threads",
                             "baselines.transformers.cache_dir"),
    "configs/operator.yaml": ("compute", "hf.token_env", "llm_api.key_file", "llm_api.budget_usd",
                              "llm_api.reserve_for_paraphrases_usd", "llm_api.prices_usd_per_million",
                              "llm_api.avoid_peak_hours"),
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(ROOT / path if not Path(path).is_absolute() else path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Configs:
    operator: dict[str, Any]
    default: dict[str, Any]
    experiments: dict[str, dict[str, Any]]

    def exp(self, name: str) -> dict[str, Any]:
        return self.experiments[name]


def load_configs(root: Path = ROOT) -> Configs:
    exps = {p.stem: load_yaml(p) for p in sorted((root / "configs" / "experiments").glob("*.yaml"))}
    return Configs(load_yaml(root / "configs/operator.yaml"), load_yaml(root / "configs/default.yaml"), exps)


def child_seeds(global_seed: int, names: list[str]) -> dict[str, int]:
    """One global seed -> named child seeds via numpy SeedSequence (ТЗ "Сиды"), in configured order."""
    children = np.random.SeedSequence(int(global_seed)).spawn(len(names))
    return {name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(names, children)}


def seeds_for(cfg: Configs, global_seed: int) -> dict[str, int]:
    return child_seeds(global_seed, list(cfg.default["seeds"]["children"]))


def _drop_key(data: Any, dotted: str) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        node.pop(leaf, None)


def hashed_view(rel: str, raw: bytes) -> bytes:
    """What ``config_hash`` digests of one config file: YAML by content (parsed, minus :data:`UNHASHED_KEYS`,
    canonical sorted JSON, so comments and formatting do not count), any other file by its bytes."""
    if not rel.endswith((".yaml", ".yml")):
        return raw
    data = yaml.safe_load(raw.decode("utf-8"))
    if isinstance(data, dict) and rel in UNHASHED_KEYS:
        data = copy.deepcopy(data)
        for dotted in UNHASHED_KEYS[rel]:
            _drop_key(data, dotted)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def config_hash(root: Path = ROOT) -> str:
    """sha256 over the frozen config files, every experiment config and the prompts, in a fixed order.

    ТЗ "Честность эксперимента": the config is frozen before the final run and a result must carry the hash of the
    config it ran with; ``run_all.sh`` re-runs a stage whose result carries another hash. YAML files enter by parsed
    content without the execution-only keys of :data:`UNHASHED_KEYS` (see :func:`hashed_view`); regexes and prompts
    enter byte for byte. A missing file hashes as ``<missing>``."""
    h = hashlib.sha256()
    files = [root / f for f in CONFIG_FILES] + sorted((root / "configs" / "experiments").glob("*.yaml"))
    files += sorted((root / "configs" / "prompts").glob("*.txt"))
    for f in files:
        rel = f.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(hashed_view(rel, f.read_bytes()) if f.exists() else b"<missing>")
    return h.hexdigest()


def git_commit(root: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
