"""Fill the transformer score caches once, one process per model, before the per-seed experiment runs.

Why: the three DeBERTa guards are the most expensive part of the pipeline on CPU (~10 windows/s per model at 8
threads) and their scores do not depend on the seed. ``GuardModel`` caches scores by ``text_hash``; running this
command for the three models in parallel processes (each writes only its own ``<model>.parquet``, and the writer is
locked anyway) means every later E1/E6/contract process only reads the cache, so seeds can run concurrently.

What is scored: every window of ``windows.parquet`` (all sources and splits, dedup-excluded windows included because
the contract scores every window of a step, ASSUMPTIONS A35) and, unless ``--no-tok512``, the 512-token windows of
the documents E6 re-windows (E1 test documents of every source, P_val documents, BIPIA E6 variants). No metric is
computed and nothing is printed but counts; the test read is journaled once per file (split ``test``).

Resumable: the texts are scored in chunks of ``--chunk`` (default 1 000) and ``GuardModel.score`` appends each
chunk's scores to the locked cache when the chunk ends, so an interrupted run loses at most one chunk per model and a
restart scores only the hashes still missing (``score`` skips cached hashes). One progress line per chunk goes to
stderr (counts only).

    python -m flyguard.experiments.prescore --model protectai_v2 [--smoke] [--no-tok512] [--chunk 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from flyguard.baselines.transformers_guard import GuardModel
from flyguard.config import ROOT, load_configs
from flyguard.data.build import output_dirs
from flyguard.netlog import log_data_access


CHUNK = 1000


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def _progress(model: str, stage: str, done: int, total: int, t0: float) -> None:
    print(json.dumps({"model": model, "stage": stage, "done": done, "total": total,
                      "seconds": round(time.time() - t0, 1)}), file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-tok512", action="store_true")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="texts per cache append (resume granularity)")
    args = ap.parse_args(argv)
    size = max(1, int(args.chunk))
    cfg = load_configs()
    gm = GuardModel(args.model, cfg)
    if not gm.available:
        print(json.dumps({"model": args.model, "available": False}))
        return 0
    processed, manifests = output_dirs(ROOT, args.smoke)
    purpose = f"guard prescoring ({args.model}): scores cached by text hash, no metric computed"
    wpath, dpath = processed / "windows.parquet", processed / "documents.parquet"
    log_data_access(wpath, "test", purpose)
    windows = pd.read_parquet(wpath, columns=["text", "text_hash"]).drop_duplicates("text_hash")
    t0 = time.time()
    texts, hashes = windows["text"].astype(str).tolist(), windows["text_hash"].astype(str).tolist()
    for a, b in _chunks(len(texts), size):
        gm.score(texts[a:b], hashes=hashes[a:b])
        _progress(args.model, "windows", b, len(texts), t0)
    out = {"model": args.model, "windows_unique": int(len(windows)), "windows_seconds": round(time.time() - t0, 1)}
    if not args.no_tok512:
        splits = json.loads((manifests / "splits.json").read_text())
        pools = json.loads((manifests / "pools.json").read_text())
        ids = {d for v in splits["e1"]["test"].values() for d in v}
        ids |= set(pools.get("p_val", {}).get("doc_ids", []))
        ids |= set(splits.get("bipia", {}).get("e6_docs", {}).get("test", []))
        log_data_access(dpath, "test", purpose + " (512-token windows of E6)")
        docs = pd.read_parquet(dpath, columns=["doc_id", "text"])
        docs = docs[docs["doc_id"].isin(ids)].drop_duplicates("text")
        t1 = time.time()
        dtexts = docs["text"].astype(str).tolist()
        for a, b in _chunks(len(dtexts), size):
            gm.score_long(dtexts[a:b])
            _progress(args.model, "tok512", b, len(dtexts), t1)
        out.update({"tok512_documents_unique": int(len(docs)), "tok512_seconds": round(time.time() - t1, 1)})
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
