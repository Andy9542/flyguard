"""Two append-only journals: logs/network.log (every outgoing request) and logs/data_access.log (test-file reads).

The ТЗ has no host allow-list; the journal exists so that the acceptance check can show that the only outgoing
data flows are downloads and the two permitted LLM streams, and that regex_patterns.txt was committed before
the first read of a test file.
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from flyguard.config import ROOT

NETWORK_LOG = ROOT / "logs" / "network.log"
DATA_ACCESS_LOG = ROOT / "logs" / "data_access.log"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_request(method: str, url: str, purpose: str, path: Path = NETWORK_LOG) -> None:
    parts = urllib.parse.urlsplit(url)
    clean = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{_stamp()}\t{parts.hostname or '?'}\t{method}\t{clean}\t{purpose}\n")


def log_data_access(file: str | Path, split: str, purpose: str, path: Path = DATA_ACCESS_LOG) -> None:
    """Record a read of a data file with its split ('train', 'val', 'test', 'unlabeled')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rel = Path(file)
    try:
        rel = rel.resolve().relative_to(ROOT)
    except ValueError:
        pass
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{_stamp()}\t{split}\t{rel}\t{purpose}\n")
