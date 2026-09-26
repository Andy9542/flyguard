"""Missing inputs fail loudly (review: no silent fallbacks), the CLI's exit codes, the budget stop of `all` with a
clean resume, the manifest never re-opening the trace logs, and the acceptance invariants of paraphrases.csv."""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict

import pytest

from flyguard.gen import paraphrases as P
from flyguard.io import read_jsonl


def test_missing_inputs_fail_loudly_unless_allow_partial(make_rt, T):
    rt = make_rt()
    assert P.check_inputs(rt) == []
    rt.paths.deepset_test.unlink()
    with pytest.raises(P.MissingInput, match="deepset test"):
        P.build_bases(rt)
    assert not rt.paths.bases.exists() and not rt.paths.bases_check.exists()   # nothing frozen on failure
    with pytest.raises(P.MissingInput):
        P.deepset_bases(rt)
    partial = make_rt(allow_partial=True)
    bases, st = P.build_bases(partial)
    assert len(bases) == 8 and {b.kind for b in bases} == {P.KIND_TEMPLATE} and len(st["partial"]) == 1
    sidecar = json.loads(partial.paths.bases_check.read_text(encoding="utf-8"))
    assert sidecar["allow_partial"] is True and "deepset test" in sidecar["partial"][0]
    partial.paths.bases.unlink(), partial.paths.bases_check.unlink()
    (rt.paths.meta_dir / "agentdojo_alpha.json").unlink()
    with pytest.raises(P.MissingInput, match="meta"):
        P.build_bases(rt)
    with pytest.raises(P.MissingInput, match="meta"):
        P.template_bases(rt)
    assert P.template_bases(partial) == ([], {"composed_equals_log": 0, "composed_differs_from_log": 0, "from_log": 0, "composed": 0})
    with pytest.raises(P.MissingInput, match="no bases"):
        P.build_bases(partial)                                                 # even --allow-partial needs one base
    shutil.rmtree(rt.paths.traces_dir)
    assert any("trace directory" in p for p in P.check_inputs(rt))


def test_offline_stages_refuse_to_run_without_bases(make_rt, T):
    rt = make_rt()
    for fn in (P.load_bases, P.filter_stage, P.finalize_stage, P.write_manifest, P.judge_stage):
        with pytest.raises(P.MissingInput):
            fn(rt)
    with pytest.raises(P.MissingInput):
        P.generate_stage(make_rt(cfg=T.make_cfg()), None)                     # bases=None -> load_bases -> missing
    rt.paths.bases.parent.mkdir(parents=True)
    rt.paths.bases.write_text("")
    with pytest.raises(P.MissingInput, match="empty"):
        P.load_bases(rt)
    assert not rt.paths.csv.exists() and not rt.paths.manifest.exists() and not rt.paths.filtered.exists()


def test_banned_list_requires_train_unless_allow_partial(make_rt):
    rt = make_rt()
    rt.paths.deepset_train.unlink()
    with pytest.raises(P.MissingInput, match="train"):
        P.final_banned_list(rt)
    assert not rt.paths.banned.exists()
    d = P.final_banned_list(make_rt(allow_partial=True))
    assert d["chi2_extra"] == [] and d["partial"] is True and d["source"] is None and d["final"] == d["starter"]


def test_manifest_uses_the_sidecar_and_never_reopens_the_logs(make_rt, T):
    rt = make_rt()
    T.freeze_three(rt)
    P.run_all(rt)
    n_lines = len(rt.paths.data_access_log.read_text(encoding="utf-8").splitlines())
    shutil.rmtree(rt.paths.traces_dir)
    m = P.write_manifest(rt)
    assert m["counts"]["template_verification"] == {"composed_equals_log": 1, "composed_differs_from_log": 0, "from_log": 1, "composed": 7}
    assert len(rt.paths.data_access_log.read_text(encoding="utf-8").splitlines()) == n_lines
    rt.paths.bases_check.unlink()
    with pytest.raises(P.MissingInput, match="bases_check"):
        P.write_manifest(rt)


def test_run_all_stops_at_the_budget_and_resumes_without_repeating_calls(make_rt, T):
    """ТЗ "Бюджет API" / operator run: `all` stops cleanly (CSV + manifest written, spend.json refreshed, total
    <= budget) and a second run after a top-up performs only the missing calls."""
    rt = make_rt(budget=0.021)                                                 # room for 10 of the 14 calls
    T.freeze_three(rt)
    out = P.run_all(rt)
    assert out["stopped"] and "budget" in out["stopped"] and len(rt.transport.calls) == 10
    assert P.spend_total(rt.paths.spend_dir) <= 0.021 and rt.spend_json_calls == [rt.cfg]
    assert rt.paths.csv.exists() and rt.paths.manifest.exists() and rt.paths.results_json.exists()
    rt2 = make_rt(budget=5.0, transport=T.FakeTransport())
    out2 = P.run_all(rt2)
    assert out2["stopped"] is None and len(rt2.transport.calls) == 14 - 10
    calls = [(c["base_id"], c["generator"], c["call_index"]) for c in read_jsonl(rt2.paths.calls) if c["status"] in ("ok", "refusal")]
    judged = [(j["cand_id"], j["judge"]) for j in read_jsonl(rt2.paths.judgements) if j["status"] in ("ok", "refusal")]
    assert len(calls) == 6 == len(set(calls)) and len(judged) == 8 == len(set(judged))
    assert len(read_jsonl(rt2.paths.spend)) == 14
    rt3 = make_rt(budget=5.0, transport=T.FakeTransport())
    assert P.run_all(rt3)["processed"] == 0 and rt3.transport.calls == []


def test_main_exit_codes_and_spend_refresh(make_rt, T, project, monkeypatch, capsys):
    """CLI: 2 when bases are missing, 3 on a budget stop (spend.json refreshed in `finally`), 0 after a top-up."""
    budgets = {"usd": 0.021}
    transports: list = []
    refreshed: list = []

    def fake_transport(rt):
        transports.append(T.FakeTransport())
        return transports[-1]

    monkeypatch.setattr(P, "load_configs", lambda: T.make_cfg(budget=budgets["usd"]))
    monkeypatch.setattr(P.Paths, "from_cfg", classmethod(lambda cls, cfg, root=None: T.make_paths(project)))
    monkeypatch.setattr(P, "make_transport", fake_transport)
    monkeypatch.setattr(P, "refresh_spend_json", lambda rt: refreshed.append(rt.cfg))
    assert P.main(["filter"]) == 2 and "missing input" in capsys.readouterr().err
    T.freeze_three(make_rt())
    assert P.main(["all"]) == 3
    printed = json.loads(capsys.readouterr().out)
    assert printed["stopped"] and "budget" in printed["stopped"] and len(refreshed) >= 1
    assert "text" not in json.dumps(printed) and len(transports[0].calls) == 10
    budgets["usd"] = 5.0
    assert P.main(["all"]) == 0 and len(transports[1].calls) == 4
    assert P.main(["manifest"]) == 0 and P.main(["all"]) == 0 and transports[2].calls == []


def test_paraphrases_csv_acceptance_invariants(make_rt, T):
    """ТЗ acceptance: every accepted positive has Jaccard <= 0.5 with its base (recomputed from the texts) and
    the deep stratum has no word of the final banned list; both strata per base when both passed; para_id
    enumerates per base."""
    rt = make_rt()
    T.freeze_three(rt)
    P.run_all(rt)
    bases = {b.base_id: b for b in P.load_bases(rt)}
    final = P.BannedMatcher(P.final_banned_list(rt)["final"])
    f = rt.pcfg["filters"]
    with open(rt.paths.csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and list(rows[0]) == P.CSV_COLUMNS
    by = defaultdict(list)
    for r in rows:
        base = bases[r["base_id"]]
        j = P.jaccard_texts(base.text, r["text"], int(f["shingle"]))
        assert j <= float(f["jaccard_max"]) and j == pytest.approx(float(r["jaccard_to_base"]), abs=1e-3)
        assert r["label"] == str(base.label) and r["kind"] == base.kind
        if r["stratum"] == "deep":
            assert j <= float(f["deep_max"])
            if base.label == 1:
                assert final.hits(r["text"]) == []
        else:
            assert r["stratum"] == "shallow"
        assert P.detect_language(P.normalize_text(r["text"])) == "en"
        by[r["base_id"]].append(r)
    for base_id, rs in by.items():
        assert [r["para_id"] for r in rs] == [f"para:{base_id}:{k}" for k in range(len(rs))]
        assert len(rs) <= int(rt.pcfg["max_accepted_per_base"]) and {r["stratum"] for r in rs} == {"deep", "shallow"}
    assert set(by) == set(bases)
