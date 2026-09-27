"""Fill the transformer score caches once, one process per model, before the per-seed experiment runs.

Why: the three DeBERTa guards are the most expensive part of the pipeline on CPU (~10 windows/s per model at 4
threads) and their scores do not depend on the seed. ``GuardModel`` caches scores by ``text_hash``; running this
command for the three models in parallel processes (each writes only its own ``<model>.parquet``, and the writer is
locked anyway) means every later E1/E6 process only reads the cache, so seeds can run concurrently.

What is scored -- exactly the texts a guard is ever evaluated on, read from the engine's own window sets (review
finding: scoring every row of ``windows.parquet`` spent most of the smoke budget on windows no guard reads):

* ``val_all``           every E1 validation window (the H2/H3 validation AUCs of ``standard_evaluation``);
* ``p_val``             the P_val windows (τ_FPR of every detector);
* ``test:<source>``     the E1 test windows of every test source in ``splits.json`` (NotInject included), with the
  dedup-excluded windows dropped exactly as ``FeatureContext.window_set`` drops them;
* ``bipia_all``         the E6 BIPIA variant windows (:func:`flyguard.experiments.e6.bipia_all_frame`), only when the
  E6 part ``bipia_all`` runs in this mode;
* 512-token windows     of the documents :func:`flyguard.experiments.e6.tok512_documents` returns, only when the E6
  part ``tok512`` runs in this mode (and ``--no-tok512`` is not given).

"Runs in this mode" is :func:`flyguard.experiments.e6.resolve_parts` with ``smoke`` -- the E6.yaml flags, restricted
to ``smoke.e6_parts`` in smoke mode (ASSUMPTIONS A54) -- so the smoke prescore scores neither the E6 BIPIA variants
nor the token windows. Nothing else calls a guard: E0, E2–E5 fit no guard, E3's default references are the regexes
only, and the contract has no guard variant (its every-window step scores concern the fly and TF-IDF). Guard latency
in E1 is a forward pass without the cache and does not read it. Train windows are never scored.

Test reads go through the context door (``Context.load_test_windows``, journaled once per source with split ``test``
and a prescoring purpose); no metric is computed and nothing is printed but counts.

Resumable: the hashes already in the model's cache are skipped up front, the rest are scored in chunks of
``--chunk`` (default 1 000) and ``GuardModel.score`` appends each chunk's scores to the locked cache when the chunk
ends, so an interrupted run loses at most one chunk per model and a restart scores only the hashes still missing.
Token windows are chunked by document (``score_long`` skips cached token windows the same way). One progress line
per chunk goes to stderr (counts only).

    python -m flyguard.experiments.prescore --model protectai_v2 [--smoke] [--no-tok512] [--chunk 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from flyguard.config import ROOT, Configs, load_configs
from flyguard.experiments import e6
from flyguard.experiments.context import Context
from flyguard.experiments.engine import FeatureContext

CHUNK = 1000


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(n, start + size)


def _progress(model: str, stage: str, done: int, total: int, t0: float) -> None:
    print(json.dumps({"model": model, "stage": stage, "done": done, "total": total,
                      "seconds": round(time.time() - t0, 1)}), file=sys.stderr, flush=True)


def purpose_for(model: str) -> str:
    return f"guard prescoring ({model}): scores cached by text hash, no metric computed"


def guard_window_frames(fc: FeatureContext, parts: Sequence[str]) -> dict[str, pd.DataFrame]:
    """The window sets guards are scored on, by role (module docstring), as the engine builds them."""
    ctx = fc.ctx
    frames: dict[str, pd.DataFrame] = {"val_all": fc.window_set("val_all").frame, "p_val": fc.window_set("p_val").frame}
    for s in ctx.test_sources:
        frames[f"test:{s}"] = fc.window_set(f"test:{s}").frame
    if "bipia_all" in parts:
        extra = e6.bipia_all_frame(fc)
        if extra is not None:
            frames["bipia_all"] = extra
    return frames


def guard_windows(fc: FeatureContext, parts: Sequence[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Unique ``(text_hash, text)`` rows of :func:`guard_window_frames` and the window count per role."""
    frames = guard_window_frames(fc, parts)
    counts = {role: int(len(f)) for role, f in frames.items()}
    parts_ = [f[["text_hash", "text"]] for f in frames.values() if len(f)]
    if not parts_:
        return pd.DataFrame({"text_hash": pd.Series(dtype=str), "text": pd.Series(dtype=str)}), counts
    allw = pd.concat(parts_, ignore_index=True)
    allw["text_hash"] = allw["text_hash"].astype(str)
    return allw.drop_duplicates("text_hash").sort_values("text_hash", kind="stable").reset_index(drop=True), counts


def tok512_texts(fc: FeatureContext) -> tuple[list[str], dict[str, int]]:
    """Distinct document texts ``part_tok512`` re-windows (:func:`flyguard.experiments.e6.tok512_documents`) and the
    document count per role."""
    by_source, p_val = e6.tok512_documents(fc)
    counts = {f"test:{s}": int(len(d)) for s, d in by_source.items()}
    counts["p_val"] = int(len(p_val))
    texts = pd.concat([d["text"] for d in list(by_source.values()) + [p_val]], ignore_index=True).astype(str)
    return sorted(set(texts.tolist())), counts


def prescore(model: str, cfg: Configs | None = None, smoke: bool = False, root: Path = ROOT, chunk: int = CHUNK,
             tok512: bool = True, guard_factory: Callable[..., Any] | None = None,
             access_log: Callable | None = None, ctx: Context | None = None) -> dict[str, Any]:
    """Score every text a guard is evaluated on into ``model``'s cache (module docstring); returns counts.

    ``guard_factory(name)`` (tests: a stub model) replaces ``GuardModel(name, cfg, root=root)``; ``access_log``
    replaces the journal writer of the context door (tests never touch ``logs/data_access.log``)."""
    root = Path(root)
    cfg = cfg or (ctx.cfg if ctx is not None else load_configs(root))
    size = max(1, int(chunk))
    if guard_factory is None:
        from flyguard.baselines.transformers_guard import GuardModel

        gm = GuardModel(model, cfg, root=root)
    else:
        gm = guard_factory(model)
    if not gm.available:
        return {"model": model, "available": False}
    parts = e6.resolve_parts(cfg, None, smoke=smoke)
    ctx = ctx or Context(cfg, smoke=smoke, root=root, access_log=access_log)
    fc = FeatureContext(ctx, 0, purpose=purpose_for(model), cache=False, guard_factory=guard_factory)
    out: dict[str, Any] = {"model": model, "available": True, "smoke": bool(smoke), "e6_parts": list(parts)}

    windows, by_role = guard_windows(fc, parts)
    known = set(gm.cache.load()) if gm.cache is not None else set()
    todo = windows[~windows["text_hash"].isin(known)].reset_index(drop=True)
    texts, hashes = todo["text"].astype(str).tolist(), todo["text_hash"].tolist()
    t0 = time.time()
    for a, b in _chunks(len(texts), size):
        gm.score(texts[a:b], hashes=hashes[a:b])
        _progress(model, "windows", b, len(texts), t0)
    out.update({"windows_by_role": by_role, "windows_unique": int(len(windows)),
                "windows_cached_before": int(len(windows) - len(todo)), "windows_scored": int(len(todo)),
                "windows_seconds": round(time.time() - t0, 1)})

    if not tok512:
        out["tok512"] = "skipped (--no-tok512)"
    elif "tok512" not in parts:
        out["tok512"] = "not needed: E6 part tok512 does not run in this mode" + (" (smoke.e6_parts)" if smoke else "")
    else:
        dtexts, doc_counts = tok512_texts(fc)
        t1 = time.time()
        for a, b in _chunks(len(dtexts), size):
            gm.score_long(dtexts[a:b])
            _progress(model, "tok512", b, len(dtexts), t1)
        out.update({"tok512": "scored", "tok512_documents_by_role": doc_counts,
                    "tok512_documents_unique": int(len(dtexts)), "tok512_seconds": round(time.time() - t1, 1)})
    return out


def main(argv: Sequence[str] | None = None, guard_factory: Callable[..., Any] | None = None,
         access_log: Callable | None = None, root: Path = ROOT, cfg: Configs | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-tok512", action="store_true")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="texts per cache append (resume granularity)")
    args = ap.parse_args(argv)
    root = Path(root)
    if cfg is None:   # a test root holds data only; its configs are this checkout's
        cfg = load_configs(root) if (root / "configs" / "default.yaml").exists() else load_configs()
    out = prescore(args.model, cfg, smoke=args.smoke, root=root, chunk=args.chunk,
                   tok512=not args.no_tok512, guard_factory=guard_factory, access_log=access_log)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
