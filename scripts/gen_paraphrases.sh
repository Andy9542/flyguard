#!/usr/bin/env bash
# ТЗ 1.6: paraphrase bases -> generation -> filters -> judges -> finalize -> manifest, idempotent per base.
#   scripts/gen_paraphrases.sh all              # everything that is missing, then paraphrases.csv + manifest
#   scripts/gen_paraphrases.sh all --smoke      # smoke.paraphrase_bases bases per kind -> data/paraphrases/smoke/
#   scripts/gen_paraphrases.sh generate --limit 20   # one stage, at most 20 bases in this invocation
# Stages: bases | generate | filter | judge | finalize | manifest | all. Exit code 3 = stopped by the API budget.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source scripts/env.sh
exec .venv/bin/python -m flyguard.gen.paraphrases "${@:-all}"
