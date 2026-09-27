"""E6 -- sensitivity of the E1 conclusions to one factor at a time (ТЗ Этап 4 "E6 чувствительность",
docs/design_experiments.md §3, ``configs/experiments/E6.yaml``).

Every part varies exactly one setting of the E1 protocol and keeps everything else (deepset-train labels, the
E1 validation choices, the cross-dataset test sources, the cluster bootstrap of ТЗ Этап 4):

* ``k``          -- k-WTA at ``inhibition.e6_k_frac`` (2.5 % and 10 % of m) instead of 5 %, real fly, both readouts
  (``real_fly_bloom_k2.5`` ...). The FlyHash copies are behind the opt-in part ``k_flyhash`` because every FlyHash
  code key is a full coding pass over all test windows (~16 ms per window at 20x).
* ``gamma``      -- the Bloom readout at every γ of ``readout.bloom.gammas`` fixed instead of validated
  (``real_fly_bloom_gamma0.99``, ``flyhash_bloom_gamma0.5`` ...; ТЗ 2.4 "γ по валидации" -> here the whole grid).
* ``weighted``   -- the measured matrix with synapse counts (``connectome.load_malecns`` meta ``weighted``) instead
  of the binary M (ТЗ 2.2 "взвешенная в E6").
* ``normalized`` -- the Bloom variant F_c[i] = γ^{n_c(i) N_min / N_c} (ТЗ 2.4) instead of subsampling to the minority.
* ``flyhash40``  -- FlyHash at the non-primary ``expansion.flyhash.expansions`` entries (m = 40 · 16 384, ТЗ 2.2).
* ``tau80``      -- τ_80(deep) and τ_80(dojo) next to τ_90 (ТЗ 2.5), with the NotInject FPR at τ_80 and the H2 pair
  differences at that operating point.
* ``bipia_all``  -- all BIPIA attack names x positions (documents with meta ``variant == "e6"``, ТЗ 1.4) as the
  source ``bipia_all``: overall AUC, per-position and per-attack marginals with CIs, attack x position cells.
* ``tok512``     -- the transformer detectors on their own 512-token windows (``GuardModel.score_long``; ТЗ 1.3
  "512-токенные окна трансформеров только в E6") as ``<guard>_tok512``, paired with the 256-character version.
* ``para_shallow`` -- the shallow paraphrase stratum as its own source (``auc/para_shallow/<det>``) and a macroAUC
  over {deep, bipia, dojo, dyn, para_deep, para_shallow} (``macro_auc_strata/<det>``).

Keys follow :mod:`flyguard.experiments.results`; every variant is paired with its E1 base detector on the same
cluster draws (``diff/macro_auc/<variant>-<base>``), so a sensitivity is read as an effect with an interval, not
as two point estimates. ``run(ctx, seed, smoke, parts=...)`` selects parts (E6 is first on the ТЗ cut list); the
default parts are the flags set in ``E6.yaml``, and in smoke mode only those of them listed in ``smoke.e6_parts``
(ASSUMPTIONS A54: the transformer-heavy ``bipia_all`` and ``tok512`` and the FlyHash-40 coding pass run in the real
run only). :func:`resolve_parts` is the single rule of which parts run -- the guard prescoring stage reads it to
decide whether the E6 BIPIA variants and the 512-token windows need scores. Every part gets a row of the table
``parts`` (``выполнено`` / ``не выполнено в смоуке`` / ``выключено в E6.yaml`` / ``не запрошено`` / ``не хватило
данных`` with the reason), mirrored in the top-level field ``e6_parts``, so the report renders the parts a smoke run
left out. Test windows are opened only through the context door (the E6 BIPIA variants under the memo key
``bipia#e6``); this module never reads a data file itself. :func:`bipia_all_frame` and :func:`tok512_documents` are
the exact window / document sets ``bipia_all`` and ``tok512`` score, shared with the prescoring stage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from flyguard.config import ROOT, Configs, load_configs
from flyguard.eval.metrics import auc as auc_point, macro_auc
from flyguard.eval.tost import bootstrap_p
from flyguard.experiments import results as results_mod
from flyguard.experiments.context import Context
from flyguard.experiments.engine import (DEFAULT_PAIRS, DETECTORS, H2_PAIRS, POSITIVE_SOURCES, DetectorSpec, Evaluator,
                                         FeatureContext, FittedDetector, ResultBuilder, Runner, fly_spec,
                                         standard_evaluation)

EXPERIMENT = "E6"
BASE_DETECTORS = ("real_fly_bloom", "real_fly_linear", "flyhash_bloom", "flyhash_linear", "tfidf_lr", "lr_svd")
PART_FLAGS: dict[str, str] = {
    "k": "k_frac", "gamma": "gammas", "weighted": "weighted_malecns", "normalized": "normalized_bloom",
    "tau80": "tau_tpr", "bipia_all": "bipia_all_attacks_positions", "tok512": "transformer_windows_512_tokens",
    "flyhash40": "flyhash_expansion_40", "para_shallow": "shallow_stratum_as_source",
}
"""Part name -> the ``E6.yaml`` flag that enables it by default."""
OPT_IN_PARTS = ("k_flyhash",)
ALL_PARTS = tuple(PART_FLAGS) + OPT_IN_PARTS
STRATA_SOURCES = ("deep", "bipia", "dojo", "dyn", "para_deep", "para_shallow")
DERIVED_TABLES = ("para_deep", "para_shallow", "bipia_all")
TOK512 = "_tok512"
STATUS_RUN = "выполнено"
STATUS_SMOKE = "не выполнено в смоуке"
STATUS_OFF = "выключено в E6.yaml"
STATUS_NOT_REQUESTED = "не запрошено"
STATUS_NO_DATA = "не хватило данных"


# ----------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------
def e6_config(cfg: Configs) -> dict[str, Any]:
    """The E6 constants: ``E6.yaml`` values when present, else the frozen ``default.yaml`` entries."""
    exp = cfg.experiments.get(EXPERIMENT, {}) if isinstance(cfg.experiments, Mapping) else {}
    d = cfg.default
    return {
        "k_fracs": [float(k) for k in (exp.get("k_frac") or d["inhibition"].get("e6_k_frac") or [])],
        "gammas": [float(g) for g in (exp.get("gammas") or d["readout"]["bloom"]["gammas"])],
        "tau_tpr": float(exp.get("tau_tpr") or d["thresholds"]["tau_tpr"].get("e6_tpr", 0.8)),
        "expansions": [int(e) for e in d["expansion"]["flyhash"]["expansions"]
                       if int(e) != int(d["expansion"]["flyhash"]["primary_expansion"])],
        "flags": {part: bool(exp.get(flag, False)) for part, flag in PART_FLAGS.items()},
    }


def _validated(parts: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for p in parts:
        if p not in ALL_PARTS:
            raise KeyError(f"unknown E6 part {p!r}; known: {ALL_PARTS}")
        if p not in out:
            out.append(p)
    return tuple(out)


def smoke_parts(cfg: Configs) -> tuple[str, ...] | None:
    """``smoke.e6_parts`` (validated), or ``None`` when the smoke profile does not restrict E6."""
    listed = cfg.default.get("smoke", {}).get("e6_parts")
    return None if listed is None else _validated(str(p) for p in listed)


def resolve_parts(cfg: Configs, parts: Iterable[str] | None = None, smoke: bool = False) -> tuple[str, ...]:
    """The E6 parts a run executes: the given names (validated, order kept) when ``parts`` is not ``None``;
    otherwise the parts enabled in ``E6.yaml``, and in smoke mode only those of them in ``smoke.e6_parts``
    (ASSUMPTIONS A54). The guard prescoring stage calls this with ``parts=None`` to learn whether ``bipia_all`` and
    ``tok512`` will ask for transformer scores."""
    if parts is not None:
        return _validated(parts)
    flags = e6_config(cfg)["flags"]
    enabled = tuple(p for p in PART_FLAGS if flags.get(p))
    allowed = smoke_parts(cfg) if smoke else None
    return enabled if allowed is None else tuple(p for p in enabled if p in allowed)


def part_status(cfg: Configs, run_parts: Sequence[str], smoke: bool, explicit: bool,
                outcomes: Mapping[str, str | None] | None = None) -> list[dict[str, Any]]:
    """One row ``{part, status, reason}`` per known part (the ``parts`` table): why a part did or did not run.
    ``outcomes`` maps a run part to the reason it produced nothing (``не хватило данных``), ``None`` when it ran."""
    flags = e6_config(cfg)["flags"]
    allowed = smoke_parts(cfg) if smoke else None
    outcomes = dict(outcomes or {})
    rows: list[dict[str, Any]] = []
    for part in ALL_PARTS:
        flag = PART_FLAGS.get(part)
        if part in run_parts:
            why = outcomes.get(part)
            status, reason = (STATUS_RUN, None) if why is None else (STATUS_NO_DATA, why)
        elif explicit:
            status, reason = STATUS_NOT_REQUESTED, "not in the parts requested for this run"
        elif flag is None:
            status, reason = STATUS_NOT_REQUESTED, "opt-in part (run(..., parts=[...]) / --parts)"
        elif not flags.get(part):
            status, reason = STATUS_OFF, f"E6.yaml {flag}: false"
        elif allowed is not None and part not in allowed:
            status, reason = STATUS_SMOKE, "not in smoke.e6_parts (ASSUMPTIONS A54); runs in the real run"
        else:   # pragma: no cover - resolve_parts returns every enabled, allowed part
            status, reason = STATUS_NOT_REQUESTED, None
        rows.append({"part": part, "status": status, "flag": flag, "reason": reason})
    return rows


def _tag(value: float) -> str:
    return f"{value:g}"


def e6_detectors(cfg: Configs, parts: Sequence[str], guard_names: Sequence[str] = ()) -> tuple[dict[str, DetectorSpec],
                                                                                              dict[str, str],
                                                                                              list[dict[str, Any]]]:
    """The E6 detector set: the E1 bases (fly, TF-IDF, LR-svd, available guards) plus the variants of ``parts``.
    Returns ``(specs, base_of, variants_table)``; variant names never contain ``-`` (the ``diff`` keys split on
    it) and carry the setting in their suffix."""
    c = e6_config(cfg)
    specs: dict[str, DetectorSpec] = {n: DETECTORS[n] for n in BASE_DETECTORS}
    for g in guard_names:
        if g in DETECTORS:
            specs[g] = DETECTORS[g]
    base_of: dict[str, str] = {}
    table: list[dict[str, Any]] = []

    def add(name: str, spec: DetectorSpec, base: str, part: str, **setting: Any) -> None:
        specs[name] = spec
        base_of[name] = base
        table.append({"detector": name, "base": base, "part": part, **setting})

    if "k" in parts or "k_flyhash" in parts:
        families = []
        if "k" in parts:
            families += [("real_fly", "n51_svd", "measured")]
        if "k_flyhash" in parts:
            families += [("flyhash", "n16k", "flyhash")]
        for kf in c["k_fracs"]:
            for fam, nose, matrix in families:
                for ro in ("bloom", "linear"):
                    name = f"{fam}_{ro}_k{_tag(100 * kf)}"
                    add(name, fly_spec(name, nose, matrix, ro, k_frac=kf), f"{fam}_{ro}", "k", k_frac=kf)
    if "gamma" in parts:
        for g in c["gammas"]:
            for fam, nose, matrix in (("real_fly", "n51_svd", "measured"), ("flyhash", "n16k", "flyhash")):
                name = f"{fam}_bloom_gamma{_tag(g)}"
                add(name, fly_spec(name, nose, matrix, "bloom", gamma=g), f"{fam}_bloom", "gamma", gamma=g)
    if "weighted" in parts:
        for ro in ("bloom", "linear"):
            name = f"real_fly_{ro}_weighted"
            add(name, fly_spec(name, "n51_svd", "weighted", ro), f"real_fly_{ro}", "weighted", matrix="weighted")
    if "normalized" in parts:
        for fam, nose, matrix in (("real_fly", "n51_svd", "measured"), ("flyhash", "n16k", "flyhash")):
            name = f"{fam}_bloom_normalized"
            add(name, fly_spec(name, nose, matrix, "bloom", normalized=True), f"{fam}_bloom", "normalized",
                normalized=True)
    if "flyhash40" in parts:
        for exp in c["expansions"]:
            for ro in ("bloom", "linear"):
                name = f"flyhash{exp}_{ro}"
                add(name, fly_spec(name, "n16k", f"flyhash{exp}", ro), f"flyhash_{ro}", "flyhash40", expansion=exp)
    return specs, base_of, table


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------
def _two_class(df: pd.DataFrame | None, col: str | None = None) -> bool:
    return df is not None and len(df) > 0 and (col is None or col in df.columns) and df["label"].nunique() == 2


def _add_diff(rb: ResultBuilder, ev: Evaluator, key: str, frames: Mapping[str, pd.DataFrame], a: str, b: str) -> None:
    frames = {s: df for s, df in frames.items() if _two_class(df, a) and b in df.columns}
    if not frames:
        return
    d = ev.diff(frames, a, b)
    rb.add_number(f"diff/{key}/{a}-{b}", None, d["ci95"], p=d["p"])
    rb.add_number(f"diff90/{key}/{a}-{b}", None, d["ci90"])


def _notinject_fpr(rb: ResultBuilder, ev: Evaluator, ni: pd.DataFrame, tkey: str, det: str, tau: float) -> None:
    rb.add_number(f"fpr_notinject/{tkey}/{det}", None, ev.rate_ci(ni, tau, det))
    for col in ("subset", "lang_stratum"):
        if col not in ni.columns:
            continue
        for val, g in ni.groupby(col, sort=True):
            if val is None or len(g) == 0:
                continue
            rb.add_number(f"fpr_notinject/{tkey}/{det}/{val}", None, ev.rate_ci(g, tau, det))


def _positive_frames(doc_tables: Mapping[str, pd.DataFrame], col: str) -> dict[str, pd.DataFrame]:
    return {s: df for s, df in doc_tables.items() if s in POSITIVE_SOURCES and _two_class(df, col)}


def _rederive_para(doc_tables: dict[str, pd.DataFrame]) -> None:
    """``para_deep`` / ``para_shallow`` are slices of ``para``; recut them after columns were added to ``para``."""
    para = doc_tables.get("para")
    if para is None or "stratum" not in para.columns:
        return
    for stratum in ("deep", "shallow"):
        sub = para[para["stratum"] == stratum]
        if len(sub):
            doc_tables[f"para_{stratum}"] = sub.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------------------------------------------------
def part_tau80(rb: ResultBuilder, ev: Evaluator, doc_tables: Mapping[str, pd.DataFrame], dets: Sequence[str],
               tpr: float, h2_pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """τ_80 on the deep and dojo test positives (ТЗ 2.5 "τ_80 в E6"), NotInject FPR at τ_80(deep) and the H2
    pair differences at that point. Returns ``{det: tau80_deep}``."""
    taus: dict[str, float] = {}
    for det in dets:
        for s, key in (("deep", "tau80_deep"), ("dojo", "tau80_dojo")):
            df = doc_tables.get(s)
            if df is None or det not in df.columns or not (df["label"] == 1).any():
                continue
            rec = ev.tau_tpr(df.loc[df["label"] == 1, det].to_numpy(), s, tpr=tpr)
            rb.add_threshold(f"{key}/{det}", rec)
            if key == "tau80_deep":
                taus[det] = rec["value"]
    ni = doc_tables.get("notinject")
    if ni is not None and len(ni):
        for det, tau in taus.items():
            if det in ni.columns:
                _notinject_fpr(rb, ev, ni, "tau80_deep", det, tau)
        for a, b in h2_pairs:
            if a in taus and b in taus and a in ni.columns and b in ni.columns:
                ci = ev.rate_diff_ci(ni, a, taus[a], b, taus[b])
                rb.add_number(f"diff/fpr_notinject/tau80_deep/{a}-{b}", None, ci, p=bootstrap_p(ci.samples))
    return taus


def part_para_shallow(rb: ResultBuilder, ev: Evaluator, doc_tables: dict[str, pd.DataFrame], dets: Sequence[str],
                      pairs: Sequence[tuple[str, str, str]]) -> str | None:
    """The shallow paraphrase stratum as its own source and the macroAUC over the strata-split sources; returns the
    reason when there is nothing to evaluate, ``None`` otherwise."""
    _rederive_para(doc_tables)
    shallow = doc_tables.get("para_shallow")
    if shallow is None:
        rb.note("para_shallow: no shallow paraphrase documents in the test tables; part skipped")
        return "no shallow paraphrase documents in the test tables"
    for det in dets:
        if _two_class(shallow, det):
            rb.add_number(f"auc/para_shallow/{det}", None, ev.auc_ci(shallow, det))
        frames = {s: doc_tables[s] for s in STRATA_SOURCES if s in doc_tables and _two_class(doc_tables[s], det)}
        if frames:
            rb.add_number(f"macro_auc_strata/{det}", None, ev.macro_auc_ci(frames, det),
                          note="sources=" + ",".join(sorted(frames)))
    for metric, a, b in pairs:
        if metric == "macro_auc":
            _add_diff(rb, ev, "macro_auc_strata", {s: doc_tables[s] for s in STRATA_SOURCES if s in doc_tables}, a, b)
        elif metric == "auc/para_deep":
            _add_diff(rb, ev, "auc/para_shallow", {"para_shallow": shallow}, a, b)
    return None


def bipia_all_frame(fc: FeatureContext) -> pd.DataFrame | None:
    """The windows ``bipia_all`` scores: the main BIPIA test windows plus the E6 variants of the test contexts
    (``splits.json bipia.e6_docs.test``, opened through the door under ``bipia#e6``), dedup-excluded windows dropped
    on both halves; ``None`` without BIPIA test documents or E6 variants. Shared with the guard prescoring stage."""
    ctx = fc.ctx
    ids = list((ctx.splits.get("bipia") or {}).get("e6_docs", {}).get("test", []))
    if "bipia" not in ctx.test_sources or not ids:
        return None
    extra = ctx.load_test_windows("bipia", fc.purpose, doc_ids=ids, name="e6")
    if "dedup_excluded" in extra.columns:
        extra = extra[~extra["dedup_excluded"].fillna(False).astype(bool)]
    main = fc.window_set("test:bipia").frame
    return pd.concat([main, extra], ignore_index=True).drop_duplicates("window_id")


def tok512_documents(fc: FeatureContext) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The documents ``tok512`` re-windows with each model's tokenizer: per E1 test source the test documents with at
    least one window left after dedup (exactly the documents of that source's E1 document table, since document
    scores are taken over non-excluded windows), and every P_val document (the τ_FPR pool). Frames ``doc_id, text``
    (``text`` = the normalised document text of ``documents.parquet``). The E6 BIPIA variants are not re-windowed.
    Shared with the guard prescoring stage."""
    ctx = fc.ctx
    by_source: dict[str, pd.DataFrame] = {}
    for s in ctx.test_sources:
        ids = set(fc.window_set(f"test:{s}").frame["doc_id"])
        docs = ctx.test_documents(s, fc.purpose)
        by_source[s] = docs.loc[docs["doc_id"].isin(ids), ["doc_id", "text"]].reset_index(drop=True)
    p_val = ctx.documents_for(sorted(ctx.p_val_doc_ids))[["doc_id", "text", "cluster_id"]].reset_index(drop=True)
    return by_source, p_val


def part_bipia_all(fc: FeatureContext, rb: ResultBuilder, ev: Evaluator, fitted: Mapping[str, FittedDetector],
                   doc_tables: dict[str, pd.DataFrame], pairs: Sequence[tuple[str, str, str]]) -> str | None:
    """All BIPIA attack names x positions (ТЗ 1.4, meta ``variant == "e6"``) as the source ``bipia_all``.

    The set is the main test pairs (clean + the sampled attack of every context) plus the E6 variants of the same
    contexts, opened through the door under ``bipia#e6``; dedup-excluded windows are dropped as in every test set
    (:func:`bipia_all_frame`). Negatives of every marginal and cell are all clean documents; positives are the
    attacked documents of that position / attack name / cell. Marginals carry cluster-bootstrap CIs, cells are point
    AUCs (``bipia_attacks``). Returns the reason when the part has no data, ``None`` otherwise.
    """
    ctx = fc.ctx
    frame = bipia_all_frame(fc)
    if frame is None:
        rb.note("bipia_all: no E6 BIPIA variants listed in splits.json (bipia.e6_docs.test); part skipped")
        return "no E6 BIPIA variants in splits.json (bipia.e6_docs.test)"
    ws = fc.register_set("bipia_all", frame)
    df = fc.doc_frame(fc.score_many(fitted, ws), ws)
    meta = ctx.documents_for(df["doc_id"].tolist())["meta"].tolist()
    df["attack"] = [m.get("attack") for m in meta]
    df["position"] = [m.get("position") for m in meta]
    df["variant"] = [m.get("variant", "main") for m in meta]
    doc_tables["bipia_all"] = df
    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]
    rb.add_table("sources", [{"source": "bipia_all", "n_docs": int(len(df)), "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                              "n_clusters": int(df["cluster_id"].nunique()), "n_windows": ws.n,
                              "n_e6_variant_docs": int((df["variant"] == "e6").sum()),
                              "n_attacks": int(pos["attack"].nunique()), "n_positions": int(pos["position"].nunique())}])
    dets = [n for n, f in fitted.items() if f.available and n in df.columns]
    rows: list[dict[str, Any]] = []

    def marginal(det: str, sub_pos: pd.DataFrame, attack: str | None, position: str | None, ci: bool) -> None:
        sub = pd.concat([neg, sub_pos], ignore_index=True)
        row = {"detector": det, "attack": attack, "position": position, "n_pos": int(len(sub_pos)), "n_neg": int(len(neg))}
        if not _two_class(sub, det):
            rows.append({**row, "auc": None, "ci_low": None, "ci_high": None})
            return
        if ci:
            c = ev.auc_ci(sub, det)
            rows.append({**row, "auc": c.point, "ci_low": c.low, "ci_high": c.high})
        else:
            rows.append({**row, "auc": auc_point(sub[det].to_numpy(), sub["label"].to_numpy()), "ci_low": None, "ci_high": None})

    for det in dets:
        if _two_class(df, det):
            rb.add_number(f"auc/bipia_all/{det}", None, ev.auc_ci(df, det))
        for position, g in pos.groupby("position", sort=True):
            marginal(det, g, None, str(position), ci=True)
        for attack, g in pos.groupby("attack", sort=True):
            marginal(det, g, str(attack), None, ci=True)
        for (attack, position), g in pos.groupby(["attack", "position"], sort=True):
            marginal(det, g, str(attack), str(position), ci=False)
    rb.add_table("bipia_attacks", rows)
    for metric, a, b in pairs:
        if metric in ("auc/bipia", "macro_auc") and a in dets and b in dets:
            _add_diff(rb, ev, "auc/bipia_all", {"bipia_all": df}, a, b)
    return None


def part_tok512(fc: FeatureContext, rb: ResultBuilder, ev: Evaluator, fitted: Mapping[str, FittedDetector],
                doc_tables: dict[str, pd.DataFrame], tpr80: float | None) -> str | None:
    """The transformer detectors on their own 512-token windows (``GuardModel.score_long``, document score = max
    over token windows), named ``<guard>_tok512`` and paired with the 256-character version of the same model.

    Whole documents are re-windowed by the model's tokenizer, so a document's 256-character windows that dedup
    excluded are still part of its token windows (noted): the comparison is between the two windowings of the
    same documents, the ТЗ 1.3 rule being character windows for every detector and token windows only here. The
    documents are :func:`tok512_documents` (the set the prescoring stage caches). Returns the reason when the part
    has nothing to score, ``None`` otherwise.
    """
    ctx = fc.ctx
    guards = [n for n, f in fitted.items() if f.available and f.spec.kind == "guard"]
    if not guards:
        rb.note("tok512: no available transformer detector; part skipped")
        return "no available transformer detector"
    sources = [s for s in doc_tables if s not in DERIVED_TABLES]
    test_docs, p_val_docs = tok512_documents(fc)
    p_val = pd.DataFrame({"doc_id": p_val_docs["doc_id"].to_numpy(), "label": 0,
                          "cluster_id": p_val_docs["cluster_id"].astype(str).to_numpy()})
    names: dict[str, str] = {}
    for g in guards:
        gm = fc.guard(fitted[g].spec.guard)
        name = f"{g}{TOK512}"
        names[name] = g
        for s in sources:
            docs = test_docs[s]
            docs = docs[docs["doc_id"].isin(set(doc_tables[s]["doc_id"]))]
            scores = pd.Series(gm.score_long(docs["text"].astype(str).tolist()), index=docs["doc_id"].to_numpy())
            doc_tables[s][name] = doc_tables[s]["doc_id"].map(scores).astype(float).to_numpy()
        p_val[name] = np.asarray(gm.score_long(p_val_docs["text"].astype(str).tolist()), dtype=float)
    _rederive_para(doc_tables)
    rb.note(f"tok512: {sorted(names)} scored on whole documents re-windowed by the tokenizer "
            f"(dedup-excluded character windows are not removed from token windows)")
    pos_frames = {s: df for s, df in doc_tables.items() if s in POSITIVE_SOURCES and _two_class(df)}
    ni = doc_tables.get("notinject")
    target = ev.fpr_target()
    p_test_ids = ctx.p_test_doc_ids
    for name, base in names.items():
        for s, df in doc_tables.items():
            if s != "notinject" and _two_class(df, name):
                rb.add_number(f"auc/{s}/{name}", None, ev.auc_ci(df, name))
                if base in df.columns and s not in ("para_shallow", "bipia_all"):
                    _add_diff(rb, ev, f"auc/{s}", {s: df}, name, base)
        present = {s: df for s, df in pos_frames.items() if name in df.columns}
        if present:
            rb.add_number(f"macro_auc/{name}", None, ev.macro_auc_ci(present, name), note="sources=" + ",".join(sorted(present)))
            _add_diff(rb, ev, "macro_auc", present, name, base)
        taus: dict[str, float] = {}
        rec = ev.tau_fpr(p_val[name].to_numpy(), target)
        if rec is not None:
            rb.add_threshold(f"tau_fpr/{name}", rec)
            taus["tau_fpr"] = rec["value"]
            for s, df in present.items():
                rb.add_number(f"tpr_at_fpr/{s}/{name}", None, ev.rate_ci(df[df["label"] == 1], rec["value"], name))
            pool = pd.concat([df[df["doc_id"].isin(p_test_ids)] for s, df in doc_tables.items()
                              if s not in DERIVED_TABLES and name in df.columns], ignore_index=True)
            if len(pool):
                rb.add_number(f"fpr_ptest/{name}", None, ev.rate_ci(pool, rec["value"], name))
        for s, key, tpr in (("deep", "tau90_deep", None), ("dojo", "tau90_dojo", None), ("deep", "tau80_deep", tpr80),
                            ("dojo", "tau80_dojo", tpr80)):
            df = doc_tables.get(s)
            if key.startswith("tau80") and tpr is None:
                continue
            if df is not None and name in df.columns and (df["label"] == 1).any():
                r = ev.tau_tpr(df.loc[df["label"] == 1, name].to_numpy(), s, tpr=tpr)
                rb.add_threshold(f"{key}/{name}", r)
                taus[key] = r["value"]
        if ni is not None and len(ni) and name in ni.columns:
            for tkey, tau in taus.items():
                if tkey.endswith("dojo"):
                    continue
                _notinject_fpr(rb, ev, ni, tkey, name, tau)
                base_rec = rb.thresholds.get(f"{tkey}/{base}")
                if base_rec is not None and base in ni.columns:
                    ci = ev.rate_diff_ci(ni, name, tau, base, float(base_rec["value"]))
                    rb.add_number(f"diff/fpr_notinject/{tkey}/{name}-{base}", None, ci, p=bootstrap_p(ci.samples))
    return None


# ----------------------------------------------------------------------------------------------------------------
# Body and entry points
# ----------------------------------------------------------------------------------------------------------------
def e6_body(fc: FeatureContext, rb: ResultBuilder, parts: Iterable[str] | None = None) -> None:
    """The E6 body for one seed: fit the E1 bases and the requested variants on deepset train (validation choices
    on deepset val, exactly as E1), run the standard evaluation with variant-vs-base pairs, then the parts that
    need more than a detector name (τ_80, para_shallow, bipia_all, tok512) and the latency of the variants whose
    pipeline cost differs from their base."""
    cfg, ctx = fc.cfg, fc.ctx
    explicit = parts is not None
    parts = resolve_parts(cfg, parts, smoke=ctx.smoke)
    c = e6_config(cfg)
    specs, base_of, variants = e6_detectors(cfg, parts, ctx.guard_names())
    rb.add_table("variants", variants)
    rb.note(f"E6 parts: {list(parts)}; k_fracs={c['k_fracs']}, gammas={c['gammas']}, tau_tpr={c['tau_tpr']}, "
            f"expansions={c['expansions']}")
    outcomes: dict[str, str | None] = {}
    fitted = fc.fit_many(list(specs.values()))
    pairs = list(DEFAULT_PAIRS) + [("macro_auc", v, b) for v, b in base_of.items()]
    h2 = list(H2_PAIRS) + [(v, b) for v, b in base_of.items()]
    h2 += [(v, "protectai_v2") for v, b in base_of.items() if b.startswith("real_fly")]
    out = standard_evaluation(fc, fitted, pairs=pairs, h2_pairs=h2, latency=False)
    rb.merge(out)
    doc_tables: dict[str, pd.DataFrame] = dict(out["doc_tables"])
    ev = Evaluator(fc)
    dets = [n for n, f in fitted.items() if f.available]
    tpr80 = c["tau_tpr"] if "tau80" in parts else None
    if "tau80" in parts:
        part_tau80(rb, ev, doc_tables, dets, tpr80, h2)
    if "para_shallow" in parts:
        outcomes["para_shallow"] = part_para_shallow(rb, ev, doc_tables, dets, pairs)
    if "bipia_all" in parts:
        outcomes["bipia_all"] = part_bipia_all(fc, rb, ev, fitted, doc_tables, pairs)
    if "tok512" in parts:
        outcomes["tok512"] = part_tok512(fc, rb, ev, fitted, doc_tables, tpr80)
    lat_set = "test:deep" if "deep" in ctx.test_sources else f"test:{ctx.test_sources[0]}"
    for row in variants:
        if row["part"] in ("k", "weighted", "flyhash40") and fitted[row["detector"]].available:
            lat = fc.latency_ms(fitted[row["detector"]], lat_set)
            if lat is not None:
                rb.add_number(f"latency_ms/{row['detector']}", lat["ms_per_doc"], n=lat["n_docs"])
    rb.note(f"E6 detectors: {len(dets)} available of {len(specs)}; variants: {len(variants)}")
    status = part_status(cfg, parts, ctx.smoke, explicit, outcomes)
    rb.add_table("parts", status)
    rb.extra["e6_parts"] = {"run": list(parts), "smoke": bool(ctx.smoke), "explicit": bool(explicit),
                            "status": {r["part"]: r["status"] for r in status}}
    skipped = [r["part"] for r in status if r["status"] == STATUS_SMOKE]
    if skipped:
        rb.note(f"E6 smoke: parts {skipped} не выполнено в смоуке (smoke.e6_parts, ASSUMPTIONS A54); their numbers "
                f"come from the real run")


def run(ctx: Context | None = None, seed: int = 0, smoke: bool = False, *, root: Path = ROOT, cfg: Configs | None = None,
        parts: Iterable[str] | None = None, keep_cache: bool = False, force: bool = False,
        guard_factory: Any = None, access_log: Any = None, cache: bool = True) -> Path:
    """Run E6 for one global seed and write ``results/E6/<seed>.json`` (``results/smoke/E6/`` in smoke mode);
    skipped when the file is current (same ``config_hash``) unless ``force``."""
    runner = Runner(EXPERIMENT, seed, smoke=smoke, root=root, cfg=cfg, ctx=ctx, keep_cache=keep_cache, force=force,
                    access_log=access_log, guard_factory=guard_factory, cache=cache)
    return runner.run(lambda fc, rb: e6_body(fc, rb, parts))


def default_seeds(cfg: Configs, smoke: bool) -> list[int]:
    seeds = [int(s) for s in cfg.default["seeds"]["global"]]
    return seeds[: int(cfg.default["smoke"]["seeds"])] if smoke else seeds


def parse_parts(text: str | None) -> tuple[str, ...] | None:
    """``"k,gamma"`` -> ``("k", "gamma")``; ``None`` / ``"all"`` -> the E6.yaml default parts."""
    if text is None or text.strip().lower() in ("", "all", "default"):
        return None
    return tuple(t.strip() for t in text.split(",") if t.strip())


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Plug-in hook of :mod:`flyguard.experiments.run` (``run E6 --parts k,gamma``)."""
    parser.add_argument("--parts", default=None, help=f"comma-separated subset of {ALL_PARTS} (default: E6.yaml flags)")


def configs_for(args: argparse.Namespace) -> Configs | None:
    """``args.cfg`` if given, else the configs of ``args.root`` when that tree is a checkout (has ``configs/``),
    else ``None`` (:class:`Context` then loads this checkout's configs; test roots hold data only)."""
    cfg = getattr(args, "cfg", None)
    if cfg is not None:
        return cfg
    root = Path(getattr(args, "root", ROOT))
    return load_configs(root) if (root / "configs" / "default.yaml").exists() else None


def run_cli(args: argparse.Namespace) -> list[Path]:
    """Plug-in hook of :mod:`flyguard.experiments.run`: every seed of ``args.seeds`` on one shared context (test
    windows are journaled once per process), then ``results/E6/summary.json``. The context (and with it the
    parquet tables) is built only when at least one seed is not current, so a rerun of ``run_all.sh`` over
    finished seeds touches no data."""
    root = Path(getattr(args, "root", ROOT))
    cfg = configs_for(args)
    parts = parse_parts(getattr(args, "parts", None))
    seeds = [int(s) for s in args.seeds]
    todo = [s for s in seeds if args.force or not results_mod.is_current(EXPERIMENT, s, args.smoke, root)]
    ctx = Context(cfg, smoke=args.smoke, root=root) if todo else None
    paths = [run(ctx, s, args.smoke, root=root, cfg=cfg, parts=parts, keep_cache=args.keep_cache, force=args.force)
             for s in seeds]
    results_mod.summarize(EXPERIMENT, args.smoke, root)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E6 sensitivity (ТЗ Этап 4)")
    p.add_argument("--seed", type=int, action="append", dest="seed", help="global seed (repeatable; default: all configured)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-cache", dest="keep_cache", action="store_true")
    add_cli_arguments(p)
    a = p.parse_args(argv)
    a.seeds = a.seed or default_seeds(configs_for(a) or load_configs(), a.smoke)
    for path in run_cli(a):
        print(f"E6: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
