# FlyGuard experiments layer — detailed contract (wave 2)

Complements `docs/design.md` §9–§10. Owners: `src/flyguard/experiments/*`, `scripts/{run_all.sh,smoke.sh,make_report.py,check_acceptance.py}`, `tests/experiments/*`.
Inputs are the wave-1 modules exactly as they exist in the tree (read their docstrings; the public APIs are summarised in
`/home/ubuntu/.claude/projects/-home-ubuntu-projects-flyguard/8d58349d-5f53-407e-ae70-4d0e426fa951/tool-results/b4s72hyv6.txt`, but the code is authoritative).

## 1. Data the engine consumes

`data/processed/documents.parquet`, `windows.parquet`, `episodes.parquet` (design §2), `data/manifests/{splits.json,pools.json,dedup.json}`,
`data/processed/connectome/malecns_R.npz`, guard models under `data/raw/models/`, `configs/regex_patterns.txt`.
Smoke: the same files under `data/processed/smoke/` and `data/manifests/smoke/` (built by `python -m flyguard.data.build --smoke`).
Test documents are read exactly once per experiment run through a single accessor `experiments.context.load_test_windows(cfg, source)` that calls
`flyguard.netlog.log_data_access(path, "test", purpose)`; validation choices never see them.

## 2. Per-seed pipeline (`experiments/engine.py`)

For global seed `s` (children via `flyguard.config.seeds_for`):

1. **Counts.** `nose.char_ngram_hash_counts` on the texts of every window in play (train + val + test + P_val/P_test pools), seed `nose`,
   cached at `data/processed/features/seed<s>/counts16k.npz` and `counts51.npz` (via `cached_char_ngram_hash_counts`, keyed by window ids).
2. **Noses** fitted on C_unl windows only (`splits.json["c_unl"]` documents → their windows): `N16k`, `N51Svd(seed_svd, seed_perm)`, `N51Hash`.
3. **Matrices.** `connectome.load_malecns` (measured, binary; `meta["weighted"]` for E6), `random_same_density(M, seed_projection)`,
   `flyhash_matrix(16384, expansion, 6, seed_projection)`, `dense_gaussian_sign_matrix(51, nnz, seed_projection)`, `curveball_nulls(M, n_null, seed_curveball)`.
4. **Codes.** `fly.fly_code(U, M, k)` with `k = fly.k_for(m)`; N16k → FlyHash path passes `mean=nose.mean_`; batch over windows.
5. **Detectors** (train on deepset-train windows of the E1 split; validation = E1 `val` windows; C_unl for idf):

   | name | input | model |
   |---|---|---|
   | `regex` | window text | `baselines.regex.RegexScorer` |
   | `tfidf_lr` | N16k rows | `baselines.lexical.TfidfLR(X_unl=C_unl counts)` |
   | `knn1`, `knn5`, `centroid` | N16k rows | `baselines.lexical.KNN`, `NearestCentroid` |
   | `lr_svd` | N51-svd | `baselines.lexical.LRSvd` |
   | `real_fly_bloom`, `real_fly_linear` | N51-svd → measured M → k-WTA | `readout.BloomReadout`, `readout.LinearReadout` |
   | `flyhash_bloom`, `flyhash_linear` | N16k → FlyHash M (20×) → k-WTA | same readouts |
   | `protectai_v2`, `piguard`, `prompt_guard_2` | window text | `baselines.transformers_guard.GuardModel` (cache by `text_hash`; scored once per window set, not per seed) |

   γ (Bloom) and C (linear/LR) are chosen on validation AUC (`readout.select_gamma/select_C`, `baselines.common.select_c` with groups = cluster_id).
   Few-shot variants (E2, H1b(ii)) choose γ on validation separately for 1 and 10 shots.
6. **Scores.** Window scores → `eval.metrics.doc_scores` (max over non-excluded windows) → per-source AUC, macroAUC, TPR@FPR on P_test,
   FPR on NotInject at τ_FPR and τ_90(deep), threshold records (`eval.thresholds`), cluster bootstrap CIs (`eval.bootstrap`), paired differences
   for the hypothesis pairs, TOST/Holm/randomisation (`eval.tost`).
7. **Results** → `results/<E>/<seed>.json` (design §9 format) with `config_hash`, `git_commit`, `seeds`, `smoke`, timing and state-size numbers
   (`readout.*.state_size_bytes`, latency per document measured on 200 test documents).

All heavy artefacts are cached and re-used when the seed, config hash and window set match; `run_all.sh` skips an experiment/seed whose result file
exists with the current `config_hash`.

## 3. Experiments

- **E0 (`e0.py`)** — `eval.power.power_table` with sizes from `splits.json`/`pools.json`, validation macroAUC spread over 20 curveball nulls and
  10 perm seeds (computed on validation only, Bloom full training), the H2 339-pair CI width. Writes `results/power.json` (+ `results/E0/summary.json`).
  Runs twice: `--stage 1` before traces/paraphrases are in the tables, `--stage 2` (final, frozen) after.
- **E1 (`e1.py`)** — the table above on the cross-dataset split; numbers keyed `auc/<source>/<detector>`, `macro_auc/<detector>`,
  `tpr_at_fpr/<source>/<detector>`, `fpr_notinject/<tau>/<detector>[/<subset>|/<lang>]`, `diff/<metric>/<a>-<b>` with CIs, latency/state size.
  Verdict inputs for H1a, H1b(i) and (ii, full training part), H2 come from E1.
- **E2 (`e2.py`)** — learning curves: shots ∈ {1, 10, 100, full} × 10 subsamples (seed `subsample`), detectors real_fly_bloom, flyhash_bloom,
  real_fly_linear, tfidf_lr, knn1, centroid; macroAUC with bands; H1b(ii) few-shot comparison Bloom vs kNN(1) at 1 and 10 shots.
- **E3 (`e3.py`)** — folds from `splits.json["e3"]`; metrics per ТЗ 1.10 (cross-template: TPR at the fold's negative-based threshold + AUC;
  cross-suite: FPR on the unfamiliar suite; double hold-out: both with n_pos). If only one template exists (D6), record "не хватило данных".
- **E4 (`e4.py`)** — measured M vs 200 curveball, random-same-density, dense sign code; primary Bloom full, secondary linear and Bloom-10-shot;
  randomisation p, Holm over secondary sources/metrics, two-stage bootstrap (`eval.bootstrap.two_stage_bootstrap_h3`) over clusters × null matrices × π perms
  (π perms = the 10 global seeds' `perm` children); TOST at ±δ. H3 inputs.
- **E5 (`e5.py`)** — ablation grid nose {n51_svd, n51_hash, n16k} × expansion {none, measured (51-channel noses only), random_1886, random_327680}
  × readout {linear, bloom (only with an expansion)}; contributions table (nose, expansion, learning rule, wiring) per ТЗ "Вклады".
- **E6 (`e6.py`)** — sensitivity: k ∈ {2.5 %, 10 %}, all γ, weighted M, normalised Bloom, τ_80, all BIPIA attacks × positions (`meta variant e6`),
  512-token windows for transformers (`GuardModel.score_long`), FlyHash 40×, shallow paraphrase stratum as its own source.
- **Contract (`contract_run.py`)** — ТЗ Этап 5: train the submitted variants on windows of non-test tasks and non-`important_instructions`
  templates of AgentDojo (γ and C on validation episodes, C_unl from the same episodes), threshold per contract §7 on validation benign episodes
  (`eval.thresholds.contract_threshold`), alarm_step = first step whose max window score ≥ τ, rows for `real_fly/bloom`, `real_fly/linear`,
  `flyhash/linear`, `tfidf_lr` → `results/shared/flyguard.csv` (16 columns, `agentdojo_io.contract.write_csv`), metrics of §9
  (`eval.metrics.contract_metrics_by_variant`) → `results/contract.json`. With only `important_instructions` generated (D6) the training set
  contains no attacked windows of other templates: train on clean non-test AgentDojo windows + deepset train labels as the fallback, and say so
  in `notes` and DEVIATIONS (proposal; the orchestrator writes the journal).

## 4. Scripts

- `scripts/run_all.sh [--smoke] [--from <stage>]`: setup_env → fetch → (gen_traces run/freeze if `shared.traces_dir` is null and the manifest is
  missing) → gen_paraphrases → data build (stage 1 / full) → E0 stage 1 → E1 cheap detectors → guards → E0 stage 2 (freeze `power.json`) →
  E1 → E4 → E5 → E3 → E2 → E6 → contract → report → check_acceptance. `set -euo pipefail`, timing log `logs/run_all.log`, each stage idempotent.
- `scripts/smoke.sh` = `run_all.sh --smoke` on the smoke subsets (1 seed, 20 nulls, 200 docs/source, 100 bootstrap draws, 20 episodes, 10 bases)
  and must finish in ≤ 15 minutes on this machine (16 cores) producing every REPORT section.
- `scripts/make_report.py [--smoke]` renders `REPORT.md` (Russian; sections 1–10 of ТЗ Этап 6) from `results/**/summary.json`, `power.json`,
  `spend.json`, `pilot.json`, manifests and journals; every number as `value [low, high]` followed by a `(results/<file>#<key>)` reference;
  figures under `results/figures/` (matplotlib, Agg backend): ROC by source, learning curves with bands, curveball histograms with the measured M,
  NotInject FPR at τ_90 by detector and subset, AUC by paraphrase stratum, ablation grid as a text table.
- `scripts/check_acceptance.py`: machine checks of "Критерии приёмки": sources.json sha256 present for every file; network.log hosts ⊆ downloads +
  api.deepseek.com; traces manifest re-hash; paraphrase manifest rules (Jaccard ≤ 0.5, double yes / single judge per D2, banned words absent in deep);
  spend ≤ budget; smoke timing; pytest; dedup invariants; KC in-degree 4–8; power.json written before E1 test results (mtimes + `frozen` flag);
  one window scheme (E6 flag only); threshold records complete; config hash in results equals current; regex commit older than the first test read
  in `logs/data_access.log` (git log date vs journal); contract CSV validates; split_manifest rule; both manifests present. Prints a checklist with
  ✅/❌ and exits non-zero on any ❌.

## 5. Verdict rendering

`experiments/verdicts_run.py` assembles the inputs of `eval.verdicts.verdict_h1a/h1b/h2/h3` from `results/{E1,E2,E4}/summary.json` and
`results/power.json` (carrier flags, preconditions), writes `results/verdicts.json`; `make_report.py` renders section 1 from it.
