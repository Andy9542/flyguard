#!/usr/bin/env bash
# Export the LLM provider key(s). The operator keeps DEEPSEEK_API_KEY in llm_api.key_file (configs/operator.yaml);
# a repo-local .env (gitignored) overrides. Usage: source scripts/env.sh
_FG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_FG_KEY_FILE="$(python3 -c 'import yaml,sys; print((yaml.safe_load(open(sys.argv[1])).get("llm_api") or {}).get("key_file") or "")' "$_FG_ROOT/configs/operator.yaml" 2>/dev/null || true)"
if [[ -n "$_FG_KEY_FILE" && -f "$_FG_KEY_FILE" ]]; then
  while IFS= read -r line; do
    case "$line" in DEEPSEEK_API_KEY=*) export "${line//\"/}" ;; esac
  done < "$_FG_KEY_FILE"
fi
if [[ -f "$_FG_ROOT/.env" ]]; then set -a; . "$_FG_ROOT/.env"; set +a; fi
unset _FG_ROOT _FG_KEY_FILE
