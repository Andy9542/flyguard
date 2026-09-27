#!/usr/bin/env bash
# scripts/run_all.sh — the whole FlyGuard pipeline in stage order (docs/design_experiments.md §4, ТЗ "Задача
# выполнена, когда" 3: reproduces every table and figure from scratch and continues an interrupted run without
# recomputing what is done).
#
#   scripts/run_all.sh                      full run, every stage, seeds from configs/default.yaml (seeds.global)
#   scripts/run_all.sh --smoke              smoke subsets (data/processed/smoke, results/smoke, 1 seed) = scripts/smoke.sh
#   scripts/run_all.sh --from e4            start at a stage (earlier stages are not looked at)
#   scripts/run_all.sh --only report        run one stage, ignoring its "done" predicate
#   scripts/run_all.sh --skip e6,e2         leave stages out (ТЗ "Порядок отрезания": E6, E2, E3, E5)
#   scripts/run_all.sh --seeds 0,1          override the global seeds of the experiment stages
#   scripts/run_all.sh --list               print the stage names and exit
#   scripts/run_all.sh --dry-run            evaluate the "done" predicates and print what would run, run nothing
#
# Stages, in order:
#   setup_env fetch gen_traces gen_paraphrases build_stage1 e0_stage1 build_full e0_stage2
#   e1 e4 e5 e3 e2 e6 contract verdicts report check
#
# Idempotency. Every stage has a "done" predicate that is evaluated before it runs (see stage_done below): a virtual
# environment that imports, the fetched files, a frozen manifest, a splits.json that already covers the sources on
# disk, results/<E>/<seed>.json present with the current config_hash (flyguard.experiments.results.is_current) for
# every requested seed, results/power.json frozen at stage 2 with the current hash. The experiment CLI applies the
# same rule internally (Runner.should_skip), so a stage that is re-entered after an interruption only computes the
# seeds that are missing. To recompute an experiment on purpose, delete its results/<E>/<seed>.json files.
#
# Order note. design_experiments §4 lists "E1 cheap detectors -> guards -> E0 stage 2 -> E1"; here E1 runs once,
# after E0 stage 2 has frozen power.json: an early E1 pass would write results/E1/<seed>.json and make the full E1
# skip itself, and any E1 test read before the freeze fails the acceptance criterion "power.json written before the
# final run". The ТЗ interleaving covered the wait for API generation, which precedes the CPU stages here; guard
# scores are cached by text_hash inside GuardModel, so nothing is computed twice.
#
# Timing log: logs/run_all.log, one tab-separated line per stage event
#   <utc time>  <mode: real|smoke>  <stage>  <status: start|done|skip|fail|TOTAL>  <seconds>  <note>
# scripts/check_acceptance.py derives the 15-minute smoke criterion from these lines: the time of a clean smoke run
# = the sum, over the stages build_stage1 .. report, of each stage's most recent smoke "done" duration (a re-entered
# or --from run skips finished stages, so the TOTAL of the last run alone would understate a clean run).
#
# Exit status: non-zero on the first failing stage (set -euo pipefail); in --smoke the final check stage is
# reported but not fatal (the acceptance criteria describe the real run) and runs without pytest, the log line says
# which. A clean machine needs the frozen traces on disk (data/traces, or shared.traces_dir in configs/operator.yaml)
# for the full build: traces are never regenerated once results/shared/traces_manifest.json exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/run_all.log"; mkdir -p "$ROOT/logs"

STAGES=(setup_env fetch gen_traces gen_paraphrases build_stage1 e0_stage1 build_full e0_stage2
        e1 e4 e5 e3 e2 e6 contract verdicts report check)
# One entry point for every experiment (src/flyguard/experiments/run.py):
#   run E0 --stage 1|2 [--seed s]   run E1..E6 [--seeds 0,1,...]   run contract [--seeds s]   run verdicts   (+ --smoke)
EXP_CMD=(-m flyguard.experiments.run)

SMOKE=0; FROM=""; ONLY=""; SKIP=""; SEEDS=""; DRY=0
usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --from) FROM="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    --skip) SKIP="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    --list) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
MODE=$([[ $SMOKE == 1 ]] && echo smoke || echo real)
SMOKE_FLAG=(); [[ $SMOKE == 1 ]] && SMOKE_FLAG=(--smoke)
is_stage() { local s; for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
for s in $FROM $ONLY ${SKIP//,/ }; do is_stage "$s" || { echo "unknown stage: $s (see --list)" >&2; exit 2; }; done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
logline() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$MODE" "$1" "$2" "${3:-}" "${4:-}" >> "$LOG"; }
say() { echo "[run_all $(ts)] $*"; }

# ---------------------------------------------------------------------------------------------------------------
# Seeds: --seeds, else seeds.global (smoke: the first smoke.seeds of them)
# ---------------------------------------------------------------------------------------------------------------
seeds_from_config() {
  "$PY" - "$SMOKE" <<'PY'
import sys
from flyguard.config import load_configs
d = load_configs().default
seeds = list(d["seeds"]["global"])
if sys.argv[1] == "1":
    seeds = seeds[: int(d["smoke"]["seeds"])]
print(",".join(str(s) for s in seeds))
PY
}

# ---------------------------------------------------------------------------------------------------------------
# "done" predicates: print "done" or "todo"
# ---------------------------------------------------------------------------------------------------------------
stage_done() {
  local stage="$1"
  case "$stage" in
    setup_env)
      [[ -x "$PY" && -x .venv-flypath/bin/python && -x .venv-agentdyn/bin/python ]] \
        && "$PY" -c 'import flyguard, torch, transformers, sklearn, matplotlib' >/dev/null 2>&1 && echo done || echo todo ;;
    fetch)
      local f ok=1
      for f in data/manifests/sources.json data/raw/deepset/train.parquet data/raw/deepset/test.parquet \
               data/raw/bipia/email/test.jsonl data/raw/notinject/NotInject_one.json \
               data/raw/models/protectai_v2/config.json data/raw/models/piguard/config.json \
               data/processed/connectome/malecns_R.npz; do
        [[ -e "$f" ]] || ok=0
      done
      [[ $ok == 1 ]] && echo done || echo todo ;;
    gen_traces) [[ -f results/shared/traces_manifest.json ]] && echo done || echo todo ;;
    gen_paraphrases) [[ -f data/paraphrases/paraphrases_manifest.json && -f data/paraphrases/paraphrases.csv ]] && echo done || echo todo ;;
    report|check) echo todo ;;
    *) "$PY" - "$stage" "$SMOKE" "$SEEDS" <<'PY'
import json, sys
from pathlib import Path
from flyguard.config import ROOT, config_hash
from flyguard.data.build import output_dirs
from flyguard.experiments import results as R

stage, smoke, seeds = sys.argv[1], sys.argv[2] == "1", [int(s) for s in sys.argv[3].split(",") if s]
root = ROOT
current = config_hash(root)

def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def present_sources():
    """Sources the full build can add: dojo/dyn when the frozen manifest and the trace logs (data/traces, or the
    operator's shared.traces_dir) are on disk; para when paraphrases.csv exists."""
    import yaml
    have = {"deep", "bipia", "notinject"}
    op = yaml.safe_load((root / "configs/operator.yaml").read_text(encoding="utf-8")) or {}
    dflt = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8")) or {}
    shared = (op.get("shared") or {}).get("traces_dir")
    logdir = Path(shared) if shared else root / dflt["traces"]["logdir"]
    if (root / "results/shared/traces_manifest.json").exists() and logdir.is_dir():
        have |= {"dojo", "dyn"}
    if (root / "data/paraphrases/paraphrases.csv").exists():
        have.add("para")
    return have

def splits():
    _, manifests = output_dirs(root, smoke)
    return load(manifests / "splits.json")

def done_e(name):
    return bool(seeds) and all(R.is_current(name, s, smoke, root) for s in seeds)

def power():
    return load(R.power_path(root, smoke))

def frozen(p):
    return bool(p) and (p.get("stage") == 2 or p.get("frozen") is True) and p.get("config_hash") == current

if stage == "build_stage1":
    ok = splits() is not None
elif stage == "build_full":
    sp = splits()
    ok = sp is not None and present_sources() <= set(sp.get("e1", {}).get("test", {}))
elif stage == "e0_stage1":
    ok = power() is not None                    # stage 2 supersedes stage 1: never rerun stage 1 over a frozen file
elif stage == "e0_stage2":
    ok = frozen(power())
elif stage in ("e1", "e2", "e3", "e4", "e5", "e6"):
    ok = done_e(stage.upper())
elif stage == "contract":
    c = load(R.results_dir(root, smoke) / "contract.json")
    ok = bool(c) and c.get("config_hash") == current
elif stage == "verdicts":
    vp = R.results_dir(root, smoke) / "verdicts.json"
    v = load(vp)
    ok = bool(v) and v.get("config_hash") == current
    if ok:
        newest = max([R.summary_path(e, smoke, root).stat().st_mtime for e in ("E1", "E2", "E4")
                      if R.summary_path(e, smoke, root).exists()] + [0.0])
        ok = vp.stat().st_mtime >= newest
else:
    ok = False
print("done" if ok else "todo")
PY
    ;;
  esac
}

# ---------------------------------------------------------------------------------------------------------------
# Stage bodies
# ---------------------------------------------------------------------------------------------------------------
yaml_get() { "$PY" -c 'import sys, yaml; d = yaml.safe_load(open(sys.argv[1])) or {}
for k in sys.argv[2].split("."):
    d = (d or {}).get(k)
print("" if d is None else d)' "$1" "$2"; }

run_setup_env() { scripts/setup_env.sh; }
run_fetch() { scripts/fetch.sh; }

run_gen_traces() {
  if [[ $SMOKE == 1 ]]; then
    say "smoke: traces are not generated (ТЗ: smoke runs on already generated episodes); results/shared/traces_manifest.json is missing"
    return 0
  fi
  local traces_dir; traces_dir="$(yaml_get configs/operator.yaml shared.traces_dir)"
  if [[ -z "$traces_dir" ]]; then
    [[ -f results/pilot.json ]] || scripts/gen_traces.sh pilot
    scripts/gen_traces.sh run
  else
    say "operator traces in $traces_dir: freezing the manifest only"
  fi
  scripts/gen_traces.sh freeze
}

run_gen_paraphrases() {
  if [[ $SMOKE == 1 ]]; then
    say "smoke: paraphrases are not generated (no API calls in smoke); data/paraphrases/paraphrases_manifest.json is missing"
    return 0
  fi
  local rc=0
  scripts/gen_paraphrases.sh all || rc=$?
  if [[ $rc == 3 ]]; then
    say "gen_paraphrases stopped by the API budget (exit 3): continuing with the frozen subset (see DEVIATIONS)"
  elif [[ $rc != 0 ]]; then
    return "$rc"
  fi
  [[ -f data/paraphrases/paraphrases_manifest.json ]] || { say "paraphrases_manifest.json missing after gen_paraphrases"; return 1; }
}

run_build_stage1() { "$PY" -m flyguard.data.build --without-traces "${SMOKE_FLAG[@]}"; }
run_build_full() { "$PY" -m flyguard.data.build "${SMOKE_FLAG[@]}"; }
run_e0_stage1() { "$PY" "${EXP_CMD[@]}" E0 --stage 1 --seed "${SEEDS%%,*}" "${SMOKE_FLAG[@]}"; }
run_e0_stage2() { "$PY" "${EXP_CMD[@]}" E0 --stage 2 --seed "${SEEDS%%,*}" "${SMOKE_FLAG[@]}"; }
run_experiment() { "$PY" "${EXP_CMD[@]}" "$1" --seeds "$SEEDS" "${SMOKE_FLAG[@]}"; }
run_e1() { run_experiment E1; }
run_e2() { run_experiment E2; }
run_e3() { run_experiment E3; }
run_e4() { run_experiment E4; }
run_e5() { run_experiment E5; }
run_e6() { run_experiment E6; }
run_contract() { "$PY" "${EXP_CMD[@]}" contract --seeds "${SEEDS%%,*}" "${SMOKE_FLAG[@]}"; }   # one CSV, first seed (contract_run)
run_verdicts() { "$PY" "${EXP_CMD[@]}" verdicts "${SMOKE_FLAG[@]}"; }
run_report() { "$PY" scripts/make_report.py "${SMOKE_FLAG[@]}"; }
run_check() {
  # smoke: the suite (about 3 min) is not part of the 15-minute skeleton run; the real run executes it inline
  if [[ $SMOKE == 1 ]]; then "$PY" scripts/check_acceptance.py --smoke --pytest skip; else "$PY" scripts/check_acceptance.py; fi
}

# ---------------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------------
in_list() { local x; for x in ${2//,/ }; do [[ "$x" == "$1" ]] && return 0; done; return 1; }
T0=$(date +%s)
RUN_STATUS=ok
[[ $DRY == 1 ]] || logline RUN start 0 "args: smoke=$SMOKE from=${FROM:-} only=${ONLY:-} skip=${SKIP:-} seeds=${SEEDS:-config}"
started=$([[ -z "$FROM" ]] && echo 1 || echo 0)
for stage in "${STAGES[@]}"; do
  [[ -n "$ONLY" && "$stage" != "$ONLY" ]] && continue
  [[ "$stage" == "$FROM" ]] && started=1
  [[ $started == 1 ]] || continue
  if in_list "$stage" "$SKIP"; then [[ $DRY == 1 ]] || logline "$stage" skip 0 "--skip"; say "skip $stage (--skip)"; continue; fi
  # seeds (and the venv python) are needed from the data stages on; setup_env/fetch/gen_* run without them
  case "$stage" in setup_env|fetch|gen_traces|gen_paraphrases) ;; *) [[ -n "$SEEDS" ]] || SEEDS="$(seeds_from_config)" ;; esac
  if [[ -z "$ONLY" && "$(stage_done "$stage")" == "done" ]]; then
    [[ $DRY == 1 ]] || logline "$stage" skip 0 "done"; say "skip $stage (done)"; continue
  fi
  if [[ $DRY == 1 ]]; then say "would run $stage"; continue; fi
  logline "$stage" start 0
  say "start $stage"
  t=$(date +%s); rc=0
  ( "run_$stage" ) || rc=$?
  dt=$(( $(date +%s) - t ))
  if [[ $rc == 0 ]]; then
    logline "$stage" done "$dt"; say "done $stage in ${dt}s"
  elif [[ "$stage" == check && $SMOKE == 1 ]]; then
    logline "$stage" fail "$dt" "exit $rc (not fatal in smoke)"; say "check_acceptance reported failures (exit $rc); smoke continues"
    RUN_STATUS="ok (check reported failures)"
  else
    logline "$stage" fail "$dt" "exit $rc"
    logline RUN TOTAL $(( $(date +%s) - T0 )) "failed at $stage"
    say "FAILED at $stage (exit $rc) after $(( $(date +%s) - T0 ))s"
    exit "$rc"
  fi
done
TOTAL=$(( $(date +%s) - T0 ))
[[ $DRY == 1 ]] && { say "dry run finished (seeds: ${SEEDS:-config})"; exit 0; }
logline RUN TOTAL "$TOTAL" "$RUN_STATUS"
say "all stages finished in ${TOTAL}s (mode $MODE)"
