#!/usr/bin/env bash
# ТЗ 1.6: paraphrase bases -> generation -> filters -> judges -> finalize -> manifest, idempotent per base.
#   scripts/gen_paraphrases.sh all                    # everything that is missing, then paraphrases.csv + manifest
#   scripts/gen_paraphrases.sh generate --limit 20    # one stage, at most 20 bases in this invocation
#   scripts/gen_paraphrases.sh bases --allow-partial  # build bases although an input is missing (recorded in the manifest)
# Stages: bases | generate | filter | judge | finalize | manifest | all.
# Exit codes: 3 = stopped by the API budget (rerun after a top-up resumes without repeating a single call);
#             2 = a required input or an earlier stage's output is missing.
# results/spend.json is refreshed after every API stage. Smoke runs use build.py's subset of the frozen
# paraphrases.csv (smoke.paraphrase_bases); this script has no smoke mode and makes no calls for one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source scripts/env.sh
exec .venv/bin/python -m flyguard.gen.paraphrases "${@:-all}"
