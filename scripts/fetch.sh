#!/usr/bin/env bash
# Download and checksum all data and models (ТЗ 1.1), build the MaleCNS graph with `flypath build`,
# and write data/manifests/sources.json. Idempotent: existing files with a matching sha256 are kept.
#
#   scripts/fetch.sh               download, verify, build, write the manifest (files + pins)
#   scripts/fetch.sh --pins-only   rewrite only the "pins" of an existing sources.json from the constants below
#                                  (no download, no rehash, the file list and its "generated" stamp are kept)
#
# Why --pins-only: the pins (harness commits and versions, cited in REPORT.md section 2) can change after the files
# were fetched (ASSUMPTIONS A53 added the AgentDojo commit). A full rerun is not a safe way to refresh them: the
# 1 GB MaleCNS weights file is deleted after hashing (A10), so a rerun would drop its sha256 entry and rebuild R.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
LOG="$ROOT/logs/network.log"; mkdir -p logs data/raw data/manifests
netlog() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" "$4" >> "$LOG"; }
MANIFEST=data/manifests/sources.json
TMP_MANIFEST="$(mktemp)"; echo "[]" > "$TMP_MANIFEST"

# write_manifest <files.json> [pins-only]: data/manifests/sources.json = the pins below + the file list
write_manifest() {
  python3 - "$1" "$MANIFEST" "${2:-}" <<'PY'
import json, sys, datetime
tmp, out, mode = sys.argv[1:4]
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
d = {"generated": now,
     "pins": {"agentdojo": {"package": "0.1.35", "repo": "https://github.com/ethz-spylab/agentdojo", "benchmark_version": "v1.2.2",
                            "tag": "v0.1.35", "commit": "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"},   # release tag of the installed package
              "agentdyn": {"repo": "https://github.com/SaFo-Lab/AgentDyn", "commit": "5353cf7615b135cace8d07c8f12dac53a16b6db3", "benchmark_version": "v1.2.2"},
              "flyhash_connectome": {"repo": "https://github.com/ssenge/FlyHash-Connectome", "commit": "91caf384abe291814d58e9563aa9713386d65a25"},
              "malecns": {"version": "v1.0", "minconf": 0.5, "bucket": "gs://flyem-male-cns/v1.0/"}},
     "files": json.load(open(tmp))}
if mode == "pins-only":   # keep the fetch stamp of the file list; record when the pins were rewritten
    d["generated"] = json.load(open(out)).get("generated")
    d["pins_written"] = now
json.dump(d, open(out, "w"), indent=1, sort_keys=True)
print("manifest:", out, len(d["files"]), "files", "(pins only)" if mode == "pins-only" else "")
PY
}

if [[ "${1:-}" == --pins-only ]]; then
  [[ -f "$MANIFEST" ]] || { echo "$MANIFEST missing: run scripts/fetch.sh first" >&2; exit 1; }
  python3 -c 'import json, sys; json.dump(json.load(open(sys.argv[1]))["files"], open(sys.argv[2], "w"))' "$MANIFEST" "$TMP_MANIFEST"
  write_manifest "$TMP_MANIFEST" pins-only
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "usage: scripts/fetch.sh [--pins-only]" >&2; exit 2
fi

# dl <url> <dest> <purpose> [expected_sha256]
dl() {
  local url="$1" dest="$2" purpose="$3" want="${4:-}" host
  host="$(printf '%s' "$url" | sed -E 's#^[a-z]+://([^/]+)/.*#\1#')"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && -n "$want" && "$(sha256sum "$dest" | cut -d' ' -f1)" == "$want" ]]; then
    :
  elif [[ -f "$dest" && -z "$want" ]]; then
    :
  else
    netlog "$host" GET "$url" "$purpose"
    curl -fsSL --http1.1 --retry 5 --retry-all-errors --retry-delay 5 -m 7200 -C - ${DL_AUTH:+-H "Authorization: Bearer $DL_AUTH"} -o "$dest.part" "$url" && mv "$dest.part" "$dest"
  fi
  local got; got="$(sha256sum "$dest" | cut -d' ' -f1)"
  if [[ -n "$want" && "$got" != "$want" ]]; then echo "sha256 mismatch for $dest: $got != $want" >&2; exit 1; fi
  python3 - "$TMP_MANIFEST" "$dest" "$url" "$got" <<'PY'
import json, sys, os, datetime
m, dest, url, sha = sys.argv[1:5]
d = json.load(open(m)); d = [e for e in d if e["path"] != dest]
d.append({"path": dest, "url": url, "sha256": sha, "bytes": os.path.getsize(dest),
          "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
json.dump(d, open(m, "w"), indent=1, sort_keys=True)
PY
}

# --- deepset/prompt-injections --------------------------------------------------------------
HF=https://huggingface.co
dl $HF/datasets/deepset/prompt-injections/resolve/main/data/train-00000-of-00001-9564e8b05b4757ab.parquet data/raw/deepset/train.parquet "deepset train"
dl $HF/datasets/deepset/prompt-injections/resolve/main/data/test-00000-of-00001-701d16158af87368.parquet  data/raw/deepset/test.parquet  "deepset test"

# --- BIPIA (context files in the repo; qa and abstract need external corpora, see audit) -----
BIPIA=https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark
for t in email table code; do for s in train test; do
  dl $BIPIA/$t/$s.jsonl data/raw/bipia/$t/$s.jsonl "BIPIA $t $s contexts"
done; done
for f in text_attack_train text_attack_test code_attack_train code_attack_test; do
  dl $BIPIA/$f.json data/raw/bipia/$f.json "BIPIA attacks $f"
done
for t in qa abstract; do dl $BIPIA/$t/index.json data/raw/bipia/$t/index.json "BIPIA $t sample index"; dl $BIPIA/$t/md5.txt data/raw/bipia/$t/md5.txt "BIPIA $t md5"; done

# --- NotInject + PIGuard open training set --------------------------------------------------
PIG=https://raw.githubusercontent.com/leolee99/PIGuard/main/datasets
for f in NotInject_one NotInject_two NotInject_three; do dl $PIG/$f.json data/raw/notinject/$f.json "NotInject subset $f"; done
dl $PIG/train.json data/raw/piguard_train/train.json "PIGuard open training set (contamination audit)"

# --- Guard models (inference only) ------------------------------------------------------------
model_files() { # <repo> <dir> <extra files...>
  local repo="$1" dir="$2"; shift 2
  for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json spm.model "$@"; do
    dl $HF/$repo/resolve/main/$f data/raw/models/$dir/$f "model $repo: $f"
  done
}
model_files protectai/deberta-v3-base-prompt-injection-v2 protectai_v2
model_files leolee99/PIGuard piguard modeling_piguard.py __init__.py
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN set: fetching Prompt Guard 2 (gated; the token is sent only to huggingface.co)"
  for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
    DL_AUTH="$HF_TOKEN" dl $HF/meta-llama/Llama-Prompt-Guard-2-86M/resolve/main/$f data/raw/models/prompt_guard_2/$f "model meta-llama/Llama-Prompt-Guard-2-86M (gated): $f" || { echo "Prompt Guard 2 download failed (access not granted yet?)"; break; }
  done
else
  echo "HF_TOKEN not set: Prompt Guard 2 skipped (BLOCKERS)"
fi

# --- MaleCNS via flypath build ------------------------------------------------------------------
FP=data/ext/FlyHash-Connectome
if [[ "${FETCH_SKIP_MALECNS:-0}" == 1 ]]; then echo "MaleCNS step skipped (FETCH_SKIP_MALECNS=1)"; else
if [[ ! -f $FP/data/graph/edges.npz ]]; then
  [[ -x .venv-flypath/bin/python ]] || { echo "run scripts/setup_env.sh first" >&2; exit 1; }
  for f in connectome-weights-male-cns-v1.0-minconf-0.5.feather body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather; do
    netlog storage.googleapis.com GET "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/$f" "MaleCNS v1.0 via flypath build"
  done
  ( cd $FP && ../../../.venv-flypath/bin/python -m flypath build )
fi
( cd $FP && ../../../.venv-flypath/bin/python - <<'PY'
import hashlib, pathlib, sys
want = {l.split()[1]: l.split()[0] for l in open("DATA_CHECKSUMS.txt") if l.strip() and not l.startswith("#")}
bad = 0
for p in sorted(pathlib.Path("data/raw").glob("*.feather")):
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    ok = want.get(p.name) == h
    bad += not ok
    print(("ok  " if ok else "BAD ") + p.name, h)
sys.exit(1 if bad else 0)
PY
)
# glomerulus x KC projection for FlyGuard (FlyHash-Connectome selection), then drop the 1 GB raw weights file
( cd $FP && ../../../.venv-flypath/bin/python ../../../scripts/malecns_matrix.py ../../processed/connectome R )
for f in $FP/data/raw/*.feather $FP/data/graph/* data/processed/connectome/malecns_R.*; do
  python3 - "$TMP_MANIFEST" "$f" <<'PY'
import json, sys, os, hashlib
m, dest = sys.argv[1:3]
d = json.load(open(m)); d = [e for e in d if e["path"] != dest]
h = hashlib.sha256(); 
with open(dest, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
d.append({"path": dest, "url": "flypath build (ssenge/FlyHash-Connectome@91caf384)", "sha256": h.hexdigest(), "bytes": os.path.getsize(dest)})
json.dump(d, open(m, "w"), indent=1, sort_keys=True)
PY
done
if [[ "${FETCH_KEEP_RAW_MALECNS:-0}" != 1 ]]; then rm -f $FP/data/raw/connectome-weights-*.feather; fi
fi

# --- Published runs of AgentDojo / AgentDyn as the reserve (ТЗ "Трассы", резерв) ------------
# The mushka project on this machine already holds a sparse checkout of both `runs/` at the pinned commits.
for r in agentdojo AgentDyn; do
  if [[ ! -e data/ext/published_runs_$r && -d /home/ubuntu/projects/mushka/data/ext/$r/runs ]]; then
    ln -s /home/ubuntu/projects/mushka/data/ext/$r data/ext/published_runs_$r
  fi
done

write_manifest "$TMP_MANIFEST"
echo "fetch done"
