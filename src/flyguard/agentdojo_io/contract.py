"""Contract §8 result CSV (16 columns) with its validator, and `results/shared/split_manifest.json` (contract §6,
§1) — design §3 `contract.py`.

The CSV is exchanged with the second team and checked by a validator common to both; the column order, the
empty-cell convention and the `detector`/`variant` vocabulary are therefore fixed here and nowhere else.
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from flyguard.agentdojo_io import labels as L
from flyguard.agentdojo_io.parse import NONE_TOKEN, split_episode_id
from flyguard.io import atomic_write_json, atomic_write_text

COLUMNS = ("episode_id", "suite", "user_task", "injection_task", "attack", "model", "episode_class",
           "injection_step", "first_harmful_step", "match", "detector", "variant", "alarm_step", "max_score",
           "threshold", "threshold_n_benign")
"""Contract §8 header, exactly in this order."""

DETECTOR = "flyguard"
VARIANTS = ("real_fly/bloom", "real_fly/linear", "flyhash/linear", "tfidf_lr")
"""FlyGuard's submitted variants (contract §8: three fly variants and the reference `tfidf_lr`)."""

MATCHES = (L.MATCH_FULL, L.MATCH_NAME_ONLY, L.MATCH_UNMATCHED)
_INT_COLUMNS = ("injection_step", "first_harmful_step", "alarm_step", "threshold_n_benign")
_FLOAT_COLUMNS = ("max_score", "threshold")


@dataclass
class ContractRow:
    """One CSV line (contract §8). `None` is written as an empty cell."""

    episode_id: str
    suite: str
    user_task: str
    injection_task: str | None
    attack: str | None
    model: str
    episode_class: str
    injection_step: int | None
    first_harmful_step: int | None
    match: str | None
    detector: str
    variant: str
    alarm_step: int | None
    max_score: float | None
    threshold: float | None
    threshold_n_benign: int | None

    @classmethod
    def from_episode(cls, episode: Mapping[str, Any], variant: str, alarm_step: int | None, max_score: float | None,
                     threshold: float | None, threshold_n_benign: int | None, detector: str = DETECTOR) -> "ContractRow":
        """Build a row from an `episodes.parquet` record (design §2 columns) plus the detector's outputs."""
        return cls(episode_id=str(episode["episode_id"]), suite=str(episode["suite"]), user_task=str(episode["user_task"]),
                   injection_task=_none(episode.get("injection_task")), attack=_none(episode.get("attack")),
                   model=str(episode["model"]), episode_class=str(episode["episode_class"]),
                   injection_step=_int_or_none(episode.get("injection_step")),
                   first_harmful_step=_int_or_none(episode.get("first_harmful_step")), match=_none(episode.get("match")),
                   detector=detector, variant=variant, alarm_step=_int_or_none(alarm_step),
                   max_score=(None if max_score is None else float(max_score)),
                   threshold=(None if threshold is None else float(threshold)),
                   threshold_n_benign=_int_or_none(threshold_n_benign))


def _none(value: Any) -> Any:
    """Missing values from pandas (`None`, NaN, pd.NA) and the `none` token become None."""
    if value is None:
        return None
    try:
        if value != value:  # NaN / pd.NA
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, str) and value.strip().lower() in ("", NONE_TOKEN):
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    value = _none(value)
    return None if value is None else int(value)


def _cell(value: Any) -> str:
    value = _none(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):  # includes numpy.float64: plain shortest repr, never `np.float64(...)`
        return repr(float(value)) if value == value else ""
    if hasattr(value, "item") and not isinstance(value, str):  # numpy integers from DataFrame records
        return _cell(value.item())
    return str(value)


def rows_to_csv_text(rows: Iterable[ContractRow | Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        record = asdict(row) if isinstance(row, ContractRow) else dict(row)
        writer.writerow([_cell(record.get(col)) for col in COLUMNS])
    return buf.getvalue()


def write_csv(rows: Iterable[ContractRow | Mapping[str, Any]], path: str | Path) -> Path:
    """Write `<detector>.csv` (contract §8): header in `COLUMNS` order, one line per episode and variant, empty
    cells for missing values (an empty `alarm_step` means no alarm). Atomic, UTF-8, `\\n` line ends."""
    path = Path(path)
    atomic_write_text(path, rows_to_csv_text(rows))
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Raw rows (all cells as strings, empty for missing) — validation and metrics parse them themselves."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_rows(path: str | Path) -> list[ContractRow]:
    """Typed rows of a CSV that passed `validate_csv`."""
    out = []
    for raw in read_csv(path):
        kw: dict[str, Any] = {}
        for f in fields(ContractRow):
            cell = raw.get(f.name, "")
            if cell == "":
                kw[f.name] = None
            elif f.name in _INT_COLUMNS:
                kw[f.name] = int(cell)
            elif f.name in _FLOAT_COLUMNS:
                kw[f.name] = float(cell)
            else:
                kw[f.name] = cell
        out.append(ContractRow(**kw))
    return out


def _is_int(cell: str) -> bool:
    return cell.isdigit()


def _is_float(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return cell.lower() not in ("nan", "inf", "-inf", "+inf")


def validate_rows(header: list[str], rows: list[dict[str, Any]], detector: str | None = DETECTOR,
                  variants: Iterable[str] | None = VARIANTS) -> list[str]:
    """Problems of a parsed CSV (empty list = valid). Checks: exact header; 16 cells per row; episode id format
    and agreement with the id columns; class in contract §2; benign rows have no injection task, attack,
    injection_step, first_harmful_step or match, attacked rows have all of injection task and attack; integer
    steps; `match` vocabulary and its coupling with `first_harmful_step` (§4); detector/variant vocabulary;
    numeric scores; `alarm_step` consistent with `max_score` vs `threshold`; one row per (variant, episode)."""
    problems: list[str] = []
    if header != list(COLUMNS):
        problems.append(f"header mismatch: expected {','.join(COLUMNS)}; got {','.join(header)}")
        return problems
    variant_set = set(variants) if variants is not None else None
    seen: set[tuple[str, str]] = set()
    for n, row in enumerate(rows, start=2):
        where = f"line {n}"
        if row.get("_n_cells", len(COLUMNS)) != len(COLUMNS):
            problems.append(f"{where}: wrong number of cells ({row['_n_cells']} instead of {len(COLUMNS)})")
            continue
        eid = row["episode_id"]
        try:
            parts = split_episode_id(eid)
        except ValueError:
            problems.append(f"{where}: malformed episode_id {eid!r}")
            continue
        for col in ("suite", "user_task", "model"):
            if row[col] != parts[col]:
                problems.append(f"{where}: {col}={row[col]!r} disagrees with episode_id")
        if (row["injection_task"] or None) != parts["injection_task"]:
            problems.append(f"{where}: injection_task disagrees with episode_id")
        if (row["attack"] or None) != parts["attack"]:
            problems.append(f"{where}: attack disagrees with episode_id")
        cls = row["episode_class"]
        if cls not in L.CONTRACT_CLASSES:
            problems.append(f"{where}: episode_class {cls!r} not in {L.CONTRACT_CLASSES}")
        if cls == L.CLASS_BENIGN:
            for col in ("injection_task", "attack", "injection_step", "first_harmful_step", "match"):
                if row[col] != "":
                    problems.append(f"{where}: benign row has non-empty {col}")
        else:
            for col in ("injection_task", "attack"):
                if row[col] == "":
                    problems.append(f"{where}: {cls} row has empty {col}")
            if row["match"] not in MATCHES:
                problems.append(f"{where}: match {row['match']!r} not in {MATCHES}")
        for col in _INT_COLUMNS:
            if row[col] != "" and not _is_int(row[col]):
                problems.append(f"{where}: {col} must be a non-negative integer or empty, got {row[col]!r}")
        if row["match"] == L.MATCH_UNMATCHED and row["first_harmful_step"] != "":
            problems.append(f"{where}: unmatched row must have empty first_harmful_step")
        if row["match"] in (L.MATCH_FULL, L.MATCH_NAME_ONLY) and row["first_harmful_step"] == "":
            problems.append(f"{where}: match={row['match']} requires first_harmful_step")
        if row["match"] == "" and row["first_harmful_step"] != "":
            problems.append(f"{where}: first_harmful_step without match")
        if detector is not None and row["detector"] != detector:
            problems.append(f"{where}: detector must be {detector!r}, got {row['detector']!r}")
        if row["variant"] == "":
            problems.append(f"{where}: empty variant")
        elif variant_set is not None and row["variant"] not in variant_set:
            problems.append(f"{where}: unknown variant {row['variant']!r}")
        for col in _FLOAT_COLUMNS:
            if row[col] == "" or not _is_float(row[col]):
                problems.append(f"{where}: {col} must be a finite number, got {row[col]!r}")
        if row["threshold_n_benign"] == "":
            problems.append(f"{where}: threshold_n_benign is required")
        if all(_is_float(row[c]) for c in _FLOAT_COLUMNS) and all(row[c] != "" for c in _FLOAT_COLUMNS):
            score, thr = float(row["max_score"]), float(row["threshold"])
            if row["alarm_step"] != "" and score < thr:
                problems.append(f"{where}: alarm_step set but max_score {score} < threshold {thr}")
            if row["alarm_step"] == "" and score > thr:
                problems.append(f"{where}: no alarm_step but max_score {score} > threshold {thr}")
        key = (row["variant"], eid)
        if key in seen:
            problems.append(f"{where}: duplicate row for variant {row['variant']!r} and episode {eid!r}")
        seen.add(key)
    return problems


def validate_csv(path: str | Path, detector: str | None = DETECTOR, variants: Iterable[str] | None = VARIANTS) -> list[str]:
    """Validate a `<detector>.csv` file (contract §8); returns the list of problems, empty when the file is valid.
    Pass `detector=None` / `variants=None` to validate another team's file with its own vocabulary."""
    path = Path(path)
    if not path.exists():
        return [f"file not found: {path}"]
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return ["empty file"]
        raw_rows = list(reader)
    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        row: dict[str, Any] = dict(zip(COLUMNS, r))
        if len(r) != len(COLUMNS):
            row["_n_cells"] = len(r)
        rows.append(row)
    return validate_rows(header, rows, detector=detector, variants=variants)


# ------------------------------------------------------------------------------------------- split manifest

MANIFEST_LISTS = ("test", "observation", "validation_clean", "validation_attacks", "train_attacks")


def split_manifest(episodes: Iterable[Mapping[str, Any]], rule: Mapping[str, Any] | None = None,
                   val_seed: int | None = None) -> dict[str, Any]:
    """Contract §6 split manifest from episode records (design §2 columns `episode_id`, `episode_class`,
    `contract_split`, `attack`): `test` = episodes with `contract_split == test`; `observation` = clean (`benign`)
    train episodes; `validation_clean` / `validation_attacks` = benign / attacked val episodes; `train_attacks` =
    attacked train episodes. `error` episodes and `excluded` ones appear only in `counts`. Exactly these five
    lists plus `rule` and `counts`, sorted ids: the shape the second team's validator accepts (design §3)."""
    lists: dict[str, list[str]] = {k: [] for k in MANIFEST_LISTS}
    counts: dict[str, Any] = {"error": 0, "excluded": 0, "by_split": {}, "by_class": {}, "by_benchmark": {}}
    for ep in episodes:
        eid = str(ep["episode_id"])
        cls = str(ep["episode_class"])
        split = str(ep["contract_split"])
        counts["by_class"][cls] = counts["by_class"].get(cls, 0) + 1
        bench = str(ep.get("benchmark", "unknown"))
        counts["by_benchmark"].setdefault(bench, {})
        if cls == L.CLASS_ERROR:
            counts["error"] += 1
            continue
        counts["by_split"][split] = counts["by_split"].get(split, 0) + 1
        clean = cls == L.CLASS_BENIGN
        if split == L.SPLIT_TEST:
            bucket = "test"
        elif split == L.SPLIT_VAL:
            bucket = "validation_clean" if clean else "validation_attacks"
        elif split == L.SPLIT_TRAIN:
            bucket = "observation" if clean else "train_attacks"
        else:
            counts["excluded"] += 1
            continue
        lists[bucket].append(eid)
        counts["by_benchmark"][bench][bucket] = counts["by_benchmark"][bench].get(bucket, 0) + 1
    for k in lists:
        lists[k] = sorted(set(lists[k]))
        counts[k] = len(lists[k])
    rule = dict(rule or L.contract_rule())
    rule_out = {"test_task": f"{rule.get('hash', 'crc32')}(user_task_id) mod {rule['mod']} == {rule['rem']}",
                "test_episodes": f"clean runs of test tasks and `{rule.get('test_attack')}` attacks on them",
                "excluded": f"`{rule.get('test_attack')}` on non-test tasks (held out of training and validation); "
                            "other templates on test tasks",
                "validation": f"{rule.get('val_fraction')} of non-test tasks per benchmark and suite, deterministic "
                              "from global seed 0 (child seed `subsample`)",
                **({"val_seed": int(val_seed)} if val_seed is not None else {}), **rule}
    return {**lists, "rule": rule_out, "counts": counts}


def write_split_manifest(episodes: Iterable[Mapping[str, Any]], path: str | Path, rule: Mapping[str, Any] | None = None,
                         val_seed: int | None = None) -> dict[str, Any]:
    """Write `results/shared/split_manifest.json` (contract §1, §6) and return it."""
    manifest = split_manifest(list(episodes), rule=rule, val_seed=val_seed)
    atomic_write_json(path, manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Shape check of a split manifest: the five lists present, ids well-formed, lists pairwise disjoint."""
    problems = []
    for k in MANIFEST_LISTS:
        if k not in manifest or not isinstance(manifest[k], list):
            problems.append(f"missing list {k!r}")
    for k in ("rule", "counts"):
        if k not in manifest:
            problems.append(f"missing {k!r}")
    if problems:
        return problems
    seen: dict[str, str] = {}
    for k in MANIFEST_LISTS:
        for eid in manifest[k]:
            try:
                split_episode_id(str(eid))
            except ValueError:
                problems.append(f"{k}: malformed episode id {eid!r}")
            if eid in seen and seen[eid] != k:
                problems.append(f"episode {eid!r} in both {seen[eid]} and {k}")
            seen[eid] = k
    return problems
