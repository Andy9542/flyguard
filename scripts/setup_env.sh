#!/usr/bin/env bash
# Create the three virtual environments (ТЗ "Окружение"): .venv (main, CPU only), .venv-agentdyn (AgentDyn fork,
# same package name as agentdojo), .venv-flypath (FlyHash-Connectome, only `flypath build` is used).
#   scripts/setup_env.sh          create / sync
#   scripts/setup_env.sh --lock   re-resolve requirements.lock from pyproject.toml first
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
LOG="$ROOT/logs/network.log"; mkdir -p logs data/ext
netlog() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" >> "$LOG"; }
PY=3.11
uv python install "$PY" >/dev/null 2>&1 || true
AGENTDYN_REPO=https://github.com/SaFo-Lab/AgentDyn.git
AGENTDYN_COMMIT=5353cf7615b135cace8d07c8f12dac53a16b6db3
FLYPATH_REPO=https://github.com/ssenge/FlyHash-Connectome.git
FLYPATH_COMMIT=91caf384abe291814d58e9563aa9713386d65a25

# --- main environment -------------------------------------------------------------------------
if [[ "${1:-}" == "--lock" || ! -f requirements.lock ]]; then
  netlog pypi.org GET https://pypi.org/simple/ "resolve requirements.lock (uv pip compile)"
  netlog download.pytorch.org GET https://download.pytorch.org/whl/cpu "resolve torch CPU wheel"
  uv pip compile pyproject.toml --python-version 3.11 --generate-hashes --no-header -o requirements.lock
fi
[[ -x .venv/bin/python ]] || uv venv --python "$PY" .venv
netlog pypi.org GET https://pypi.org/simple/ "install requirements.lock into .venv"
netlog files.pythonhosted.org GET https://files.pythonhosted.org/ "download wheels for .venv"
netlog download.pytorch.org GET https://download.pytorch.org/whl/cpu "download torch CPU wheel"
uv pip sync --python .venv/bin/python --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match requirements.lock
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/src" > "$SITE/flyguard.pth"

# --- AgentDyn (fork of agentdojo, pinned commit) ----------------------------------------------
if [[ ! -d data/ext/AgentDyn/.git ]]; then
  netlog github.com GIT "$AGENTDYN_REPO" "clone AgentDyn (blobless), pinned commit $AGENTDYN_COMMIT"
  git clone -q --filter=blob:none --no-checkout "$AGENTDYN_REPO" data/ext/AgentDyn
fi
( cd data/ext/AgentDyn
  git sparse-checkout set --no-cone '/src/' '/pyproject.toml' '/README.md' '/LICENSE' '/uv.lock' '/tests/' >/dev/null
  git -c advice.detachedHead=false checkout -q "$AGENTDYN_COMMIT" )
[[ -x .venv-agentdyn/bin/python ]] || uv venv --python "$PY" .venv-agentdyn
netlog pypi.org GET https://pypi.org/simple/ "install AgentDyn and its dependencies into .venv-agentdyn"
uv pip install -q --python .venv-agentdyn/bin/python -e data/ext/AgentDyn
uv pip install -q --python .venv-agentdyn/bin/python openai pyyaml

# --- FlyHash-Connectome (flypath build only) --------------------------------------------------
if [[ ! -d data/ext/FlyHash-Connectome/.git ]]; then
  netlog github.com GIT "$FLYPATH_REPO" "clone FlyHash-Connectome, pinned commit $FLYPATH_COMMIT"
  git clone -q "$FLYPATH_REPO" data/ext/FlyHash-Connectome
fi
( cd data/ext/FlyHash-Connectome && git -c advice.detachedHead=false checkout -q "$FLYPATH_COMMIT" )
[[ -x .venv-flypath/bin/python ]] || uv venv --python "$PY" .venv-flypath
netlog pypi.org GET https://pypi.org/simple/ "install FlyHash-Connectome dependencies into .venv-flypath"
uv pip install -q --python .venv-flypath/bin/python numpy scipy pandas pyarrow pyyaml

.venv/bin/python -c 'import numpy, scipy, pandas, pyarrow, sklearn, xxhash, datasketch, langdetect, datasets, torch, transformers, openai, agentdojo; print("main env ok: torch", torch.__version__, "transformers", transformers.__version__, "agentdojo", agentdojo.__version__ if hasattr(agentdojo, "__version__") else "0.1.35")'
.venv-agentdyn/bin/python -c 'import agentdojo, sys; print("agentdyn env ok:", agentdojo.__file__)'
.venv-flypath/bin/python -c 'import numpy, pandas, pyarrow, scipy, yaml; print("flypath env ok")'
echo "setup_env done"
