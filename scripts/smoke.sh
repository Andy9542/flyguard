#!/usr/bin/env bash
# scripts/smoke.sh = scripts/run_all.sh --smoke (ТЗ "Бюджет времени": steps 2–13 on 1 seed, 20 curveball nulls,
# 200 documents per source, 100 bootstrap draws, 20 already generated episodes and 10 paraphrase bases; must finish
# in <= 15 minutes on this machine and produce every section of the report).
#
# Smoke artefacts never shadow the real ones: data/processed/smoke, data/manifests/smoke, results/smoke/<E>/,
# results/smoke/power.json, results/smoke/REPORT.md and results/smoke/figures/. No API call is made: traces and
# paraphrases are taken from the frozen files (a missing manifest is reported, not generated).
#
#   scripts/smoke.sh                 everything (setup_env / fetch are skipped when their outputs exist)
#   scripts/smoke.sh --from e1       the usual run_all.sh options are passed through
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_all.sh" --smoke "$@"
