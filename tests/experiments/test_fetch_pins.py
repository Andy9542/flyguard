"""scripts/fetch.sh --pins-only (ТЗ Этап 6 section 2 "коммиты харнессов"): the pins of an existing
data/manifests/sources.json are rewritten from the constants of fetch.sh without any download or rehash, and the file
list with its "generated" stamp is kept (a full rerun would drop the sha256 of the deleted MaleCNS weights, A10)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
OLD = {"generated": "2026-09-26T09:00:00Z",
       "pins": {"agentdojo": {"package": "0.1.35", "repo": "https://github.com/ethz-spylab/agentdojo"}},
       "files": [{"path": "data/ext/FlyHash-Connectome/data/raw/connectome-weights.feather", "sha256": "ab" * 32,
                  "bytes": 1, "url": "flypath build"}]}


def run_fetch(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(root / "scripts/fetch.sh"), *args], cwd=root, capture_output=True, text=True,
                          timeout=60)


@pytest.mark.skipif(shutil.which("python3") is None, reason="fetch.sh writes the manifest with python3")
def test_pins_only_rewrites_pins_and_keeps_files(tmp_path):
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPTS / "fetch.sh", root / "scripts/fetch.sh")
    man = root / "data/manifests/sources.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps(OLD), encoding="utf-8")
    p = run_fetch(root, "--pins-only")
    assert p.returncode == 0, p.stderr[:200]
    new = json.loads(man.read_text(encoding="utf-8"))
    assert new["files"] == OLD["files"] and new["generated"] == OLD["generated"] and new["pins_written"]
    assert new["pins"]["agentdojo"]["commit"] == "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"      # ASSUMPTIONS A53
    assert new["pins"]["agentdojo"]["tag"] == "v0.1.35"
    assert set(new["pins"]) == {"agentdojo", "agentdyn", "flyhash_connectome", "malecns"}      # check_acceptance.REQUIRED_PINS
    assert not (root / "logs/network.log").exists()                                              # nothing downloaded
    assert not (root / "data/raw").exists() or not any((root / "data/raw").iterdir())
    assert run_fetch(root, "--bogus").returncode == 2
    man.unlink()
    assert run_fetch(root, "--pins-only").returncode == 1                                         # nothing to rewrite
