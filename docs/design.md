# FlyGuard design and interfaces

This is the contract between the modules. Each section names the ТЗ items the module owns. Types are Python
3.11; arrays are numpy unless stated; sparse matrices are `scipy.sparse.csr_matrix`. Everything is driven by
`configs/default.yaml` (`cfg.default`) and `configs/operator.yaml` (`cfg.operator`), loaded by
`flyguard.config.load_configs()`. Per-seed randomness comes only from `flyguard.config.seeds_for(cfg, seed)`.

## 1. Layout and ownership

```
src/flyguard/
  config.py io.py netlog.py                 (done) configs, seeds, atomic IO, journals
  gen/harness_run.py gen/traces.py gen/spend.py   (done) trace generation through DeepSeek, pilot, freeze
  gen/paraphrases.py                        §7   paraphrase generation, filters, judges, manifest
  agentdojo_io/{parse.py,labels.py,contract.py}   §3   trace logs -> step documents, episode classes, contract CSV
  data/{normalize.py,windows.py,dedup.py,loaders.py,splits.py,pools.py,audit.py,build.py}   §2, §4
  nose.py connectome.py fly.py readout.py   §5   the fly
  baselines/{regex.py,lexical.py,transformers_guard.py,common.py}   §6
  eval/{metrics.py,thresholds.py,bootstrap.py,tost.py,power.py,verdicts.py}   §8
  experiments/{engine.py,e0.py,e1.py,e2.py,e3.py,e4.py,e5.py,e6.py,contract_run.py}   §9
scripts/{fetch.sh,setup_env.sh,gen_traces.sh,gen_paraphrases.sh,smoke.sh,run_all.sh,make_report.py,check_acceptance.py}
tests/<module>/test_*.py
```

Paths on disk (all relative to the repo root; `data/raw` is written only by `scripts/fetch.sh`):

```
data/raw/deepset/{train,test}.parquet            columns: text, label
data/raw/bipia/{email,table,code}/{train,test}.jsonl  {context, question?, ideal}; text_attack_*.json {name: [5 strings]}
data/raw/notinject/NotInject_{one,two,three}.json  [{prompt, word_list, category}]
data/raw/models/{protectai_v2,piguard}/          HF snapshots (local `from_pretrained`)
data/processed/connectome/malecns_R.{npz,json}   m: synapse counts [51 glomeruli, 1886 KC] (float64), glomeruli list
data/processed/meta/{agentdojo,agentdyn}_<suite>.json   tools, user-task prompts, injection goals, ground-truth calls
data/traces/{agentdojo,agentdyn}/<model>/<suite>/<user_task>/<attack|none>/<injection_task|none>.json
data/paraphrases/{bases.jsonl,candidates.jsonl,judgements.jsonl,paraphrases.csv,paraphrases_manifest.json}
data/processed/{documents.parquet,windows.parquet,episodes.parquet}
data/manifests/{sources.json,audit.md,splits.json,pools.json,dedup.json,contamination.json,traces_extraction.json}
results/{E0..E6}/<seed>.json  results/power.json  results/spend.json  results/pilot.json
results/shared/{traces_manifest.json,split_manifest.json,<detector>.csv,traces/}
```

## 2. Documents and windows (owner: `flyguard.data`) — ТЗ 1.2, 1.3, 1.4, 1.7, 1.8, 1.9, 1.10

`documents.parquet`, one row per document:

| column | type | meaning |
|---|---|---|
| doc_id | str | unique, prefixed by source: `deep:train:<i>`, `deep:test:<i>`, `bipia:<task>:<ctx>:<atk|clean>`, `dojo:<episode_id>#<step>`, `dyn:<episode_id>#<step>`, `para:<base_id>:<k>`, `notinject:<subset>:<i>` |
| source | str | `deep`, `bipia`, `dojo`, `dyn`, `para`, `notinject` |
| split | str | E1 role: `train`, `val`, `test` |
| label | int8 | 1 = injection, 0 = clean |
| text | str | normalized (NFKC, runs of whitespace -> one space, stripped) |
| text_orig | str | as loaded |
| lang | str | langdetect code, `unk` on failure; `lang_stratum` = `en` / `non-en` |
| cluster_id | str | ТЗ 1.10: deep -> `deep:<split>:<i>` (each doc its own cluster); bipia -> the base context; dojo/dyn -> `<suite>/<user_task>`; para -> the base id; notinject -> its own id |
| spans | list<struct{start:int32, end:int32}> | injection spans in `text` coordinates; empty for span-less sources |
| meta_json | str | JSON with source-specific fields (suite, user_task, injection_task, attack, episode_id, step, bipia task/attack/position, paraphrase stratum/base, notinject subset/word_list) |

`windows.parquet`, one row per window (ТЗ 1.3): `window_id` (= `<doc_id>#w<k>`), `doc_id`, `source`, `split`,
`start`, `end`, `text`, `label` (1 if the window covers ≥ 64 chars of a span or a whole shorter span; document label
for span-less sources), `cluster_id`, `text_hash` (xxhash64 hex of `text`, for score caches), `dedup_excluded`
(bool), `dup_of` (window_id or null). Windows: size 256, stride 192; a document shorter than 256 chars is one window.

`episodes.parquet` (dojo/dyn, from `agentdojo_io`): `episode_id` (contract format
`<suite>/<user_task>/<injection_task|none>/<attack|none>/<model>`), `benchmark`, `suite`, `user_task`,
`injection_task`, `attack`, `model`, `episode_class` (`benign`/`injection_ignored`/`hijacked`/`error`),
`utility`, `security`, `n_steps`, `injection_step` (int or null), `first_harmful_step` (int or null), `match`
(`full`/`name_only`/`unmatched`/null), `contract_split` (`test`/`val`/`train`/`excluded`), `e1_val_task`
(bool, crc32(user_task_id) % 5 == 0), `log_path`, `sha256`.

Dedup (ТЗ 1.7): MinHash (datasketch, `num_perm` from config) over character 5-gram shingles of the window text,
within class, across all sources; Jaccard ≥ 0.8 between a test window and any train/val window excludes the test
window (`dedup_excluded=True`, `dup_of` set); a positive document whose positive windows are all excluded leaves
the positives; BIPIA pairs are never collapsed. `data/manifests/dedup.json` records counts. `tests/data/test_dedup.py`
checks: no cluster contains both labels, no BIPIA pair is broken, exclusions only remove test windows.

Splits (`data/manifests/splits.json`):

```
{"e1": {"train": [doc_id...], "val": [doc_id...], "test": {"deep": [...], "bipia": [...], "dojo": [...], "dyn": [...], "para": [...], "notinject": [...]}},
 "c_unl": [doc_id...],                       # texts of e1 train + val (no labels used), never test clusters
 "e3": {"cross_template": [{"fold": "<held-out template>", "train": [...], "test": [...]}],
        "cross_suite": [...], "double_holdout": [{"fold": "<suite>x<template>", ...}]},
 "bipia": {"val_contexts": [...], "test_contexts": [...]},
 "rules": {...copied from cfg.default["splits"]...}, "template_names": {"important_instructions": ..., ...}}
```
E1: labels only from deepset train; `val` = stratified 20 % of deepset train (seed `subsample` of global seed 0 is
fixed for the split itself — the split is data, not an experiment). BIPIA contexts 20/80 val/test by cluster.
AgentDojo tasks with crc32(user_task_id) % 5 == 0 are validation (their clean outputs feed P_val); the rest test.
AgentDyn, paraphrases and NotInject are test only.

Pools (`data/manifests/pools.json`): `P_val` = benign deepset train (non-val part? no: the whole benign part of
deepset train is negatives for training; P_val for thresholds uses the *val* benign deepset documents), BIPIA-val
clean documents, clean AgentDojo outputs of validation tasks; `P_test` = benign deepset test, BIPIA-test clean,
remaining clean AgentDojo outputs, clean AgentDyn outputs, negative paraphrases. Record sizes by source.

`build.py` orchestrates: load raw -> normalize -> audit -> documents -> windows -> dedup -> splits -> pools ->
manifests. It must run in two stages: `--without-traces` (deepset, BIPIA, NotInject) and full (adds dojo/dyn/para
when their files exist); both are idempotent and rewrite the manifests. `audit.md` lists documents, classes,
languages, lengths, windows per source, share of German in deepset, NotInject composition, BIPIA tasks dropped.

Smoke mode (`cfg.default["smoke"]`): `build.py --smoke` limits documents per source to `docs_per_source`
(deterministic head after sorting by doc_id) and writes to `data/processed/smoke/` and `data/manifests/smoke/`.

## 3. Trace logs -> documents (owner: `flyguard.agentdojo_io`) — ТЗ 1.5 "Извлечение", contract §2–§6, §8

`parse.py`: `read_log(path) -> TraceLog` with fields of the harness JSON (`suite_name`, `pipeline_name`,
`user_task_id`, `injection_task_id`, `attack_type`, `injections: dict[str,str]`, `messages`, `error`, `utility`,
`security`, `duration`, `benchmark_version`). `tool_steps(log) -> list[Step]`: step *i* = the i-th tool call in
the episode (count from 0, contract §3) with `tool` (name), `args` (dict), `output_text` (the tool message
content: string, or the concatenated `content` blocks' `content`/`text`), `error`.
`episode_id(log) -> str` in the contract format (model = `pipeline_name`).

`labels.py`:
- `injection_spans(step_text, injections, fill) -> list[(start,end)]`: exact match of each injection string after
  the same normalization as documents (NFKC + whitespace collapse) — placeholders `{user}`/`{model}` are already
  filled in the log's `injections`; the harness's own values (`Emma Johnson`, `DeepSeek`) are recorded in
  `traces_extraction.json`. A step is labelled 1 if any injection string occurs in it. `injection_step` = first
  such step, null if none (then the episode is `error`-like: recorded, excluded from positives, counted).
- `episode_class(log)`: contract §2 — no injection task -> `benign`; `security` true -> `hijacked`; else
  `injection_ignored`; non-empty `error` -> `error` (excluded from all metrics, counted in the audit).
- `first_harmful_step(steps, refs) -> (step | None, match)`: contract §4 with `configs/harm_references.yaml`
  (the second team's frozen file, ASSUMPTIONS A11) supplemented by `data/processed/meta/*.json` ground truth:
  a call matches `full` when the function name (or an `alt_functions` entry) matches and every `key` value is
  contained in the corresponding argument (strings compared stripped/lower-cased, URLs without scheme/`www.`/trailing
  slash, IBANs without spaces; lists become sets); `name_only` when only the name matches; else `unmatched`
  (`first_harmful_step` null). Injection tasks with `calls: []` are `no_tool_reference` -> `unmatched`.
- `contract_split(user_task_id, attack)`: crc32(user_task_id.encode()) % 3 == 2 -> test task. Test episodes = clean
  runs of test tasks + `important_instructions` attacks on them. Non-test tasks: clean runs and other templates ->
  train/val (20 % of non-test tasks to val, deterministic from global seed 0); `important_instructions` on non-test
  tasks -> `excluded` from FlyGuard training/validation (contract §6) but still test material for E1 `dojo` source?
  **No**: E1's `dojo` test source is *all* AgentDojo episodes of non-validation tasks (ТЗ 1.10); the contract split
  is a separate labelling used only by `contract_run.py`. Keep both columns.

`contract.py`: `write_csv(rows, path)` / `validate_csv(path) -> list[str]` for the 16 columns of contract §8
exactly in this order: `episode_id,suite,user_task,injection_task,attack,model,episode_class,injection_step,first_harmful_step,match,detector,variant,alarm_step,max_score,threshold,threshold_n_benign`;
empty cells for missing values; `detector = flyguard`; variants `real_fly/bloom`, `real_fly/linear`,
`flyhash/linear`, `tfidf_lr`. `write_split_manifest(episodes, path)` writes `results/shared/split_manifest.json`
with keys `test`, `observation` (clean train episodes), `validation_clean`, `validation_attacks`, `train_attacks`
(lists of episode ids; this shape passes the second team's validator) plus `rule` and `counts`.
Metrics of contract §9 live in `eval/metrics.py::contract_metrics`.

`extract.py::build_episode_documents(cfg, benchmark) -> (episodes DataFrame, documents DataFrame)` produces the
dojo/dyn rows of §2 (`doc_id = <src>:<episode_id>#<step>`, `label` = step label, `spans`, `cluster_id =
<suite>/<user_task>`, `split` per E1 rules) and `data/manifests/traces_extraction.json` (counts by class, steps,
labelled steps, unmatched, errors, fill strings). Tests use synthetic logs under `tests/agentdojo_io/fixtures/`.

## 4. Sources (owner: `flyguard.data.loaders`) — ТЗ 1.1–1.4

- deepset: parquet `text`, `label`; ids by row order; train/test as published.
- BIPIA (ТЗ 1.4): tasks `email`, `table`, `code` from the repo files (`qa`, `abstract` dropped -> audit, BLOCKERS B4);
  text tasks use `text_attack_*.json`, code uses `code_attack_*.json`; each attack name has 5 strings ->
  `attack_id = <name>-<k>`. For each *test* context: the clean document (0) and one attacked document (1) with one
  attack and one position (`start`/`middle`/`end`) drawn from seed `subsample` (global seed 0); insertion follows
  `bipia/data/utils.py` (`insert_start`: attack + "\n" + context; `insert_end`: context + "\n" + attack;
  `insert_middle`: at a sentence boundary chosen by the seeded RNG — replicate the published logic, cite it). The
  span is the inserted attack string. E6 uses all 15 attacks × 3 positions. Contexts split 20/80 val/test by
  cluster (context id). Train contexts of BIPIA are not used for labels (labels come only from deepset train).
- NotInject: 339 prompts, `subset` in {one,two,three}, `category`; label 0; test only, FPR only.
- Paraphrases: `data/paraphrases/paraphrases.csv` from §7 (test only, `cluster_id = base_id`, `stratum`).

## 5. The fly (owners: `nose.py`, `connectome.py`, `fly.py`, `readout.py`) — ТЗ 2.1–2.6

```python
# nose.py
def char_ngram_hash_counts(texts: Sequence[str], sizes=(3,4,5), bins=16384, seed=0) -> csr_matrix  # xxhash64(g, seed) % bins on lower-cased text; counts
class N16k:      fit(X_counts_unl) -> self; transform(X_counts) -> csr   # log1p then rows sum to 1; centering mean kept from C_unl
class N51Svd:    __init__(rank=51, seed_svd, seed_perm); fit(X16k_unl) -> self; transform(X16k) -> ndarray[n,51]   # TruncatedSVD(random_state=seed_svd) on C_unl; standardize per component by C_unl mean/std; then permute columns by pi (seed_perm); mean-centred on C_unl
class N51Hash:   transform(X_counts_51) -> ndarray[n,51]   # bins=51, x_b = c_b / sum c; centred on C_unl mean
def center(X, mean) -> X - mean
```
The 16 384-bin counts are computed once per (seed, window set) and cached under `data/processed/features/<seed>/`.

```python
# connectome.py — M is binary float32 csr [m_cells, d_inputs]
def load_malecns(path) -> tuple[csr_matrix, dict]      # from malecns_R.npz: binary (m>0), transposed to [1886, 51]; meta incl. weighted matrix for E6
def random_same_density(M, seed) -> csr_matrix         # each cell keeps its in-degree; inputs uniform without replacement
def curveball(M, n_trades, seed) -> csr_matrix         # Strona 2014 trades between random cell pairs; preserves both degree sequences; n_trades = swaps_per_edge * nnz
def flyhash_matrix(d, expansion, fan_in, seed) -> csr_matrix   # [expansion*d, d], fan_in distinct inputs per cell
def dense_gaussian_sign_matrix(d, nnz, seed) -> ndarray  # [round(nnz/d), d] N(0,1): same multiply-adds as M (ТЗ 3.3)
def indegree_stats(M) -> dict                           # mean/min/max, check 4-8
```
```python
# fly.py
def kwta(A: ndarray[n, m], k: int) -> csr_matrix[n, m] binary   # exactly k winners per row, ties -> lowest index (stable argsort of -A)
def expand(U: ndarray|csr [n, d], M: csr [m, d]) -> ndarray[n, m]   # U @ M.T (dense for m<=1886*? use sparse products for FlyHash sizes)
def fly_code(U, M, k) -> csr_matrix   # kwta(expand(U, M), k)
def sign_code(U, G) -> csr_matrix     # binary (G @ u > 0)
```
```python
# readout.py
class BloomReadout:   # ТЗ 2.4 Bloom (FlyNN)
    def __init__(self, m: int, k: int, gamma: float, seed_subsample: int, normalized: bool = False)
    def fit(self, Z: csr[n, m], y: ndarray) -> self       # balance classes by subsampling the majority to the minority (seed); F_c[i] <- gamma * F_c[i] for active i; normalized variant F_c[i] = gamma ** (n_c(i) * N_min / N_c)
    def score(self, Z) -> ndarray[n] in [0,1]              # phi_c = 1 - mean_{i in supp z} F_c[i]; s = (phi_1 - phi_0 + 1)/2
def make_logistic(C, seed, cfg=None) -> LogisticRegression   # THE logistic regression (readout.linear in config: lbfgs, l2, balanced, max_iter 2000, tol 1e-4); baselines.common.make_logreg wraps it
class LinearReadout:  # ТЗ 2.4 MBON: make_logistic(C) on the KC code; score = predict_proba[:,1]
def select_gamma(...) / select_C(...)   # on validation AUC only; grids from config
```
Unit tests of ТЗ 2.6 owned here: kwta gives exactly k; determinism per seed; different `nose`/`perm` seeds give
different codes; Bloom changes only active cells; γ=0 after 200 examples -> mean F ≤ 0.02; balancing equalises
classes; curveball preserves degrees; measured matrix in-degree recorded.

## 6. Baselines (owner: `flyguard.baselines`) — ТЗ 3.1, 3.2

Common protocol (`common.py`):
```python
class WindowScorer(Protocol):
    name: str
    def fit(self, X_train, y_train, X_val=None, y_val=None, groups=None) -> "WindowScorer"   # X = features of the detector's input space; groups = cluster_id (else doc_id) per train window, required when no two-class validation set is given (grouped CV)
    def score(self, X) -> ndarray[float]   # per window in [0,1] (or monotone score; document score = max over windows)
```
- `regex.py`: `configs/regex_patterns.txt` (one Python regex per line, `#` comments), compiled case-insensitively;
  window score = 1 if any pattern matches else 0. The pattern file is authored from deepset **train** statistics
  only and committed before any test read (acceptance criterion); keep 15–30 patterns.
- `lexical.py`: `TfidfLR` (idf fitted on C_unl N16k counts, LR with C from grid by val AUC), `KNN(k, cosine)`,
  `NearestCentroid`, `LRSvd` (LR on N51-svd features).
- `transformers_guard.py`: `GuardModel(name, cfg)`: local `from_pretrained` (PIGuard with `trust_remote_code=True`),
  CPU, batch inference, truncation at `max_length`; score = softmax probability of the positive label; results cached
  by `text_hash` in `data/processed/scores_cache/<name>.parquet` (append-only, dedup by hash). 512-token windows
  (E6): a separate windowing by the model's tokenizer, only for E6. Prompt Guard 2 is optional (skipped when the
  directory is missing; recorded).

## 7. Paraphrases (owner: `flyguard.gen.paraphrases`) — ТЗ 1.6, Appendix A

Pipeline `python -m flyguard.gen.paraphrases {bases,generate,filter,judge,finalize,manifest}` (idempotent per base;
`scripts/gen_paraphrases.sh` runs all). Bases (`bases.jsonl`, `source_id`/`base_id`, `kind` in
{`template`, `deepset_injection`, `deepset_benign`}): 4 templates × 35 goals with the harness's fill strings
(read the actual attack strings from the frozen trace logs' `injections`, dedup by text; ASSUMPTIONS A5),
deepset test injections, deepset test benign documents (reading deepset **test** for the bases is a permitted,
logged test read: the bases are inputs to generation, not evaluation).
Generation: DeepSeek chat completions with `response_format={"type":"json_object"}`, `thinking` disabled,
temperature 0.9, prompts filled from `configs/prompts/*.txt` (`===SYSTEM===`/`===USER===` split), `calls_per_base`
calls per base; every call logged to `logs/network.log` and metered into `results/spend/paraphrases.jsonl`
(`stream: paraphrases`, same record shape as the trace runner). Responses parsed by strict JSON schema; anything
else is a refusal (counted by generator and by template kind). Never print candidate texts.
Filters (code): langdetect `en`; length 0.5–2× base; Jaccard over character 5-gram shingle sets with the base
≤ 0.5 (shallow 0.3–0.5, deep ≤ 0.3 and no banned word/phrase, matched case-insensitively on word boundaries, any
inflection approximated by prefix match of the stem in `banned_words.txt`); required entities of the base
(emails, URLs, IBAN-like tokens, amounts, file names, quoted strings, capitalised names) present verbatim; dedup
within the set at Jaccard 0.8. The final banned list = starter list + 30 words with the highest χ² between
injection and benign documents of deepset train (lower-cased whitespace tokens); written to the manifest.
Judges: `deepseek-v4-pro`, temperature 0, prompts A.4/A.5; positive accepted if `same_action` and
`still_instruction`; negative if `meaning_preserved` and not `contains_instruction_to_ai`; `judge_agreement`
recorded (single judge: D2). Up to 5 accepted per base, both strata and (if several) both generators represented,
deterministic order from seed `paraphrase`. Outputs `paraphrases.csv` (`para_id, base_id, kind, label, stratum,
text, jaccard_to_base, generator, judge_confidence`) and `paraphrases_manifest.json` (models, dates, prompt
sha256s, banned list, acceptance and refusal rates by stratum/kind/generator, counts).

## 8. Statistics (owner: `flyguard.eval`) — ТЗ Этап 4 "Метрики", "Статистика", "Правила вердиктов", Этап 0

```python
# metrics.py
def doc_scores(window_scores: DataFrame[window_id, score], windows: DataFrame) -> DataFrame[doc_id, score]   # max over non-excluded windows
def auc(scores, labels) -> float
def macro_auc(per_source: dict[str, float]) -> float                  # equal weight over present sources
def tpr_at_fpr(pos_scores, neg_pool_scores, fpr) -> tuple[float, float]  # (tpr, threshold)
def fpr_at_threshold(scores, tau) -> float
def tau_for_tpr(pos_scores, tpr=0.90) -> float                         # τ_90(S)
def contract_metrics(rows: list[ContractRow]) -> dict                   # contract §9 (stopped-before-harm, FA/100, delay, ignored alarms, unmatched)
# thresholds.py: threshold records {value, source, target, n} (ТЗ 2.5)
# bootstrap.py
def cluster_bootstrap(df: DataFrame[doc_id, cluster_id, label, score...], stat: Callable, n: int, seed: int, alpha=0.05) -> CI(point, low, high, samples)
def paired_cluster_bootstrap(df, stat_a, stat_b, n, seed) -> CI of (a - b)
def macro_auc_bootstrap(by_source: dict[str, DataFrame], n, seed) -> CI    # resample clusters within each source
def two_stage_bootstrap_h3(...)  # clusters x null matrices x pi permutations
# tost.py
def tost_equivalent(diff_ci90: CI, delta: float) -> bool; def holm(pvalues: dict) -> dict; def randomization_p(observed, nulls) -> float
# power.py  (E0)
def power_table(cfg, sizes_by_source, val_runs) -> dict   # binormal simulation at AUC in {0.75,0.85,0.95}, observed cluster-size distribution, cluster bootstrap -> MDD of AUC difference and TOST power at ±δ; carrier rule: |P_test| >= 2000 -> TPR@1%, 500-1999 -> 5%, else AUC only; NotInject 339-pair CI width of a proportion difference
# verdicts.py
def verdict_h1a(results, cfg=None, power=None) -> Verdict; verdicts_h1b(results, cfg, power) -> {real_fly, flyhash}; verdict_h2; verdict_h3(results, cfg, power)   # the E0 gate is mandatory: pass power=<results/power.json dict> (or a carrier flag per row), otherwise ValueError; statuses: подтверждена / опровергнута / не хватило данных / предусловие не выполнено, with effect and CI
# tau_fpr_record(neg_pool_scores, fpr=None, source='P_val', cfg=None, *, n_test_pool=None): the 1 %/5 %/AUC-only target follows |P_test| (or power.json fpr_target), never |P_val|
# doc_scores(window_scores, windows, strict=True): raises when a non-excluded window has no score
```
CI type: `{"point": float, "low": float, "high": float, "level": 0.95, "n_boot": int, "n": int}`.

## 9. Experiments and results (owner: `flyguard.experiments`) — ТЗ Этап 4, Этап 5, Этап 6

`engine.py` builds, per global seed, the feature context: window counts (16k) for the union of windows in play,
noses fitted on C_unl, matrices (measured, random, FlyHash, curveball set), detectors fitted on train windows
(deepset train windows only for E1), validation-based choices (γ, C, thresholds), then scores on val/test windows,
document scores, metrics with CIs. Heavy things are cached: counts per seed, transformer scores by `text_hash`.
Results file format (every experiment, every seed):

```
results/<E>/<seed>.json = {"experiment": "E1", "seed": 0, "config_hash": "...", "git_commit": "...",
  "seeds": {...children...}, "smoke": false,
  "numbers": {"<key>": {"value": x, "ci_low": ..., "ci_high": ..., "n": ..., "note": ...}},
  "tables": {"<name>": [row dicts]}, "thresholds": {"<name>": {"value":..., "source":..., "target":..., "n":...}},
  "notes": [...]}
```
Keys are stable strings like `auc/deep/tfidf_lr`, `macro_auc/real_fly_bloom`, `tpr_at_fpr/dojo/protectai_v2`,
`fpr_notinject/tau90_deep/piguard`. `results/<E>/summary.json` aggregates over seeds (mean, sd, per-seed values).
`make_report.py` renders every number as `value [low, high]` with a `results/<E>/summary.json#<key>` reference.
Smoke mode: one seed, 20 curveballs, 200 docs per source, 100 bootstrap draws, from `data/processed/smoke/`.
`run_all.sh` runs stages in order and skips stages whose outputs exist (`results/<E>/<seed>.json` present and
`config_hash` matches); `smoke.sh` = `run_all.sh --smoke`.

## 10. Frozen artefacts and honesty checks

- `configs/regex_patterns.txt`, `configs/*.yaml`, prompts and `configs/harm_references.yaml` are committed before
  the first test read (`logs/data_access.log` + git history; `scripts/check_acceptance.py` verifies).
- `results/power.json` is written by E0 before the final run and re-written once after traces/paraphrases arrive.
- Traces and paraphrases are frozen by manifests; `gen/traces.py verify` re-hashes.
- The comparator `protectai_v2` is fixed in `configs/default.yaml`.
