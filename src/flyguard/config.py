"""Configs (configs/*.yaml), derived seeds and the frozen-config hash."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = ("configs/operator.yaml", "configs/default.yaml", "configs/regex_patterns.txt")


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


def config_hash(root: Path = ROOT) -> str:
    """sha256 over the frozen config files and every experiment config (bytes, in a fixed order)."""
    h = hashlib.sha256()
    files = [root / f for f in CONFIG_FILES] + sorted((root / "configs" / "experiments").glob("*.yaml"))
    files += sorted((root / "configs" / "prompts").glob("*.txt"))
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()


def git_commit(root: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
