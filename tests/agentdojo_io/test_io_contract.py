"""contract.py: the 16-column CSV (contract §8) round-trip, validator rejections, and the split manifest (§6)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from flyguard.agentdojo_io import contract as C

MODEL = "MODEL"


def _row(**kw) -> C.ContractRow:
    base = dict(episode_id=f"workspace/user_task_2/injection_task_3/important_instructions/{MODEL}", suite="workspace",
                user_task="user_task_2", injection_task="injection_task_3", attack="important_instructions", model=MODEL,
                episode_class="hijacked", injection_step=2, first_harmful_step=4, match="full", detector="flyguard",
                variant="real_fly/bloom", alarm_step=3, max_score=0.91, threshold=0.42, threshold_n_benign=65)
    base.update(kw)
    return C.ContractRow(**base)


def _benign(**kw) -> C.ContractRow:
    base = dict(episode_id=f"workspace/user_task_5/none/none/{MODEL}", suite="workspace", user_task="user_task_5",
                injection_task=None, attack=None, model=MODEL, episode_class="benign", injection_step=None,
                first_harmful_step=None, match=None, detector="flyguard", variant="real_fly/bloom", alarm_step=None,
                max_score=0.12, threshold=0.42, threshold_n_benign=65)
    base.update(kw)
    return C.ContractRow(**base)


def test_header_order_and_round_trip(tmp_path):
    path = C.write_csv([_row(), _benign()], tmp_path / "flyguard.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ("episode_id,suite,user_task,injection_task,attack,model,episode_class,injection_step,"
                        "first_harmful_step,match,detector,variant,alarm_step,max_score,threshold,threshold_n_benign")
    assert lines[1] == f"workspace/user_task_2/injection_task_3/important_instructions/{MODEL},workspace,user_task_2," \
                       f"injection_task_3,important_instructions,{MODEL},hijacked,2,4,full,flyguard,real_fly/bloom,3,0.91,0.42,65"
    assert lines[2] == f"workspace/user_task_5/none/none/{MODEL},workspace,user_task_5,,,{MODEL},benign,,,,flyguard,real_fly/bloom,,0.12,0.42,65"
    assert C.validate_csv(path) == []
    assert C.parse_rows(path) == [_row(), _benign()]
    assert len(C.COLUMNS) == 16 and C.DETECTOR == "flyguard" and "tfidf_lr" in C.VARIANTS


def test_from_episode_handles_missing_values():
    ep = {"episode_id": f"workspace/user_task_5/none/none/{MODEL}", "suite": "workspace", "user_task": "user_task_5",
          "injection_task": None, "attack": pd.NA, "model": MODEL, "episode_class": "benign",
          "injection_step": pd.NA, "first_harmful_step": float("nan"), "match": None}
    row = C.ContractRow.from_episode(ep, "tfidf_lr", alarm_step=None, max_score=0.1, threshold=0.5, threshold_n_benign=20)
    assert row.injection_task is None and row.attack is None and row.injection_step is None and row.first_harmful_step is None
    ep2 = {**ep, "episode_id": f"workspace/user_task_2/injection_task_3/tool_knowledge/{MODEL}", "injection_task": "injection_task_3",
           "attack": "tool_knowledge", "episode_class": "hijacked", "injection_step": 1, "first_harmful_step": 2.0, "match": "full"}
    row = C.ContractRow.from_episode(ep2, "real_fly/linear", alarm_step=1, max_score=0.9, threshold=0.5, threshold_n_benign=20)
    assert row.first_harmful_step == 2 and row.alarm_step == 1 and row.variant == "real_fly/linear"


def test_mapping_rows_with_numpy_values_write_plain_cells(tmp_path):
    import numpy as np
    record = {**_row().__dict__, "injection_step": np.int64(2), "first_harmful_step": np.int64(4),
              "alarm_step": np.int64(3), "max_score": np.float64(0.91), "threshold": np.float64(0.42),
              "threshold_n_benign": np.int64(65)}
    frame_row = pd.DataFrame([record]).to_dict("records")[0]
    path = C.write_csv([record, {**frame_row, "variant": "tfidf_lr"}], tmp_path / "np.csv")
    text = path.read_text(encoding="utf-8")
    assert "np." not in text and ",3,0.91,0.42,65" in text
    assert C.validate_csv(path) == []
    assert C.parse_rows(path)[0] == _row()


@pytest.mark.parametrize("rows,needle", [
    ([_row(episode_class="error")], "episode_class"),
    ([_benign(injection_step=0)], "benign row has non-empty injection_step"),
    ([_benign(match="unmatched")], "benign row has non-empty match"),
    ([_row(attack=None, episode_id=f"workspace/user_task_2/injection_task_3/none/{MODEL}")], "empty attack"),
    ([_row(match="unmatched")], "unmatched row must have empty first_harmful_step"),
    ([_row(first_harmful_step=None)], "requires first_harmful_step"),
    ([_row(match="partial", first_harmful_step=None)], "match 'partial'"),
    ([_row(detector="other")], "detector must be 'flyguard'"),
    ([_row(variant="fly/unknown")], "unknown variant"),
    ([_row(alarm_step=3, max_score=0.2, threshold=0.5)], "alarm_step set but max_score"),
    ([_row(alarm_step=None, max_score=0.9, threshold=0.5)], "no alarm_step but max_score"),
    ([_row(), _row()], "duplicate row"),
    ([_row(suite="banking")], "suite='banking' disagrees"),
    ([_row(injection_task="injection_task_9")], "injection_task disagrees"),
    ([_row(episode_id="workspace/user_task_2/injection_task_3")], "malformed episode_id"),
    ([_row(threshold_n_benign=None)], "threshold_n_benign is required"),
    ([_row(threshold=None)], "threshold must be a finite number"),
])
def test_validator_rejections(tmp_path, rows, needle):
    path = C.write_csv(rows, tmp_path / "bad.csv")
    problems = C.validate_csv(path)
    assert problems and any(needle in p for p in problems), problems


def test_validator_raw_text_problems(tmp_path):
    good = C.rows_to_csv_text([_row()])
    p = tmp_path / "x.csv"
    p.write_text(good.replace("episode_id,suite", "suite,episode_id"), encoding="utf-8")
    assert any("header mismatch" in m for m in C.validate_csv(p))
    p.write_text(good.rstrip("\n") + ",extra\n", encoding="utf-8")
    assert any("wrong number of cells" in m for m in C.validate_csv(p))
    p.write_text(good.replace(",3,0.91", ",three,0.91"), encoding="utf-8")
    assert any("alarm_step must be a non-negative integer" in m for m in C.validate_csv(p))
    p.write_text("", encoding="utf-8")
    assert C.validate_csv(p) == ["empty file"]
    assert C.validate_csv(tmp_path / "nope.csv")[0].startswith("file not found")
    # another team's vocabulary is accepted when the vocabulary checks are switched off
    other = C.write_csv([_row(detector="mushka", variant="behavioural/v1")], tmp_path / "mushka.csv")
    assert C.validate_csv(other) and C.validate_csv(other, detector=None, variants=None) == []


def _episodes():
    def ep(uid, it, atk, cls, split, bench="agentdojo"):
        return {"episode_id": f"toy/{uid}/{it or 'none'}/{atk or 'none'}/{MODEL}", "episode_class": cls,
                "contract_split": split, "attack": atk, "benchmark": bench}
    return [ep("user_task_1", None, None, "benign", "test"),
            ep("user_task_1", "injection_task_0", "important_instructions", "hijacked", "test"),
            ep("user_task_0", None, None, "benign", "train"),
            ep("user_task_3", None, None, "benign", "val"),
            ep("user_task_3", "injection_task_0", "tool_knowledge", "injection_ignored", "val"),
            ep("user_task_0", "injection_task_1", "injecagent", "hijacked", "train"),
            ep("user_task_0", "injection_task_1", "important_instructions", "hijacked", "excluded"),
            ep("user_task_2", "injection_task_2", "important_instructions", "error", "test"),
            ep("user_task_9", None, None, "benign", "train", bench="agentdyn")]


def test_split_manifest_shape_and_counts(tmp_path):
    m = C.write_split_manifest(_episodes(), tmp_path / "split_manifest.json", rule={"hash": "crc32", "mod": 3, "rem": 2,
                                                                                    "test_attack": "important_instructions",
                                                                                    "val_fraction": 0.2}, val_seed=7)
    assert set(m) == {"test", "observation", "validation_clean", "validation_attacks", "train_attacks", "rule", "counts"}
    assert m["test"] == [f"toy/user_task_1/injection_task_0/important_instructions/{MODEL}", f"toy/user_task_1/none/none/{MODEL}"]
    assert m["observation"] == [f"toy/user_task_0/none/none/{MODEL}", f"toy/user_task_9/none/none/{MODEL}"]
    assert m["validation_clean"] == [f"toy/user_task_3/none/none/{MODEL}"]
    assert m["validation_attacks"] == [f"toy/user_task_3/injection_task_0/tool_knowledge/{MODEL}"]
    assert m["train_attacks"] == [f"toy/user_task_0/injection_task_1/injecagent/{MODEL}"]
    assert m["counts"]["error"] == 1 and m["counts"]["excluded"] == 1 and m["counts"]["test"] == 2
    assert m["counts"]["by_benchmark"]["agentdyn"] == {"observation": 1}
    assert m["rule"]["mod"] == 3 and m["rule"]["val_seed"] == 7 and "crc32" in m["rule"]["test_task"]
    on_disk = json.loads((tmp_path / "split_manifest.json").read_text())
    assert on_disk == m and C.validate_split_manifest(on_disk) == []
    bad = {**m, "observation": m["observation"] + [m["test"][0]]}
    assert any("in both" in p for p in C.validate_split_manifest(bad))
    assert any("missing list" in p for p in C.validate_split_manifest({"test": []}))
