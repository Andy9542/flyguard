#!/usr/bin/env bash
# ТЗ 1.5: pilot -> model choice -> prioritised, sharded, idempotent generation -> freeze.
#   scripts/gen_traces.sh pilot      # 40 attacked + 10 clean workspace episodes per candidate model
#   scripts/gen_traces.sh run        # all priorities that fit the budget projection (resumable)
#   scripts/gen_traces.sh freeze     # traces_manifest.json + copy for the second team
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
source scripts/env.sh
exec .venv/bin/python -m flyguard.gen.traces "$@"
