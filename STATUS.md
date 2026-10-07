# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-07 (training speed, step 0: old-format baseline read; waiting on commit + new-format baseline log).

**Nothing is running.** The user is not running training or evaluation now, so code is not frozen.

**Logging cleanup (2026-10-07):** INFO reports stage/layer/node timings and shapes;
training `-verbose` enables DEBUG, `-quiet` keeps warnings/errors. Evaluation always
preserves warnings. Details and log capture: `scripts/README.md` § Train.

**Next session: training speed (planned, not started).** Benchmark = BC5CDR-disease, `train_plus_ctd.pubtator`,
`configs/xmr4el_bc5cdr_config.json` (depth 2: root -> 16 leaves of ~734 labels, 17 nodes, 89,929 rows):
tree `13-47-38` trained in 567 s (PECOS on the same features: 34 s). Same config without CTD (664 labels): 15 s.
0. Before any edit: (a) the user commits the current tree so the pre-change code has a commit for the
   bit-identical check (`git worktree add <dir> <commit>`); (b) baseline log with the new logging.
   `outputs/logs/speed_baseline_bc5cdr_ctd.log` (2026-10-07 14:11, 577.7 s, mps) uses the OLD log format
   (`Training ML: Number 0`, no elapsed fields); breakdown from timestamps: encoding + PIFA 241 s
   (14:11:10.9 -> 14:15:12.0, not split), root 80 s (clustering 38 s, matcher 40 s), 16 leaves 243 s
   (11-22 s each, matcher dominated; leaf clustering <0.1 s). Encoding and leaves are each ~42%.
   The old log cannot split encoding into SapBERT / char TF-IDF+SVD / context TF-IDF+SVD / PIFA, so the
   user reruns it with the new logging into a new file (old log kept). Compare with
   `grep -E "Transformer completed|Encoding completed|Label embeddings completed|Clustering completed|Node completed|Layer completed|Hierarchy completed|Run completed"`.
   Do not compare message names across the two formats; compare stage spans. Command:
   `.venv/bin/python scripts/train.py -train_path datasets/BC5CDR/disease/train_plus_ctd.pubtator -model_config configs/xmr4el_bc5cdr_config.json 2>&1 | tee outputs/logs/speed_baseline2_bc5cdr_ctd.log`
1. Drop the three `gc.collect()` in `MLModel.train` (`node.py:261,304,354`; `import gc` at `node.py:2`);
   `tree.py` has none. Refcounting frees the arrays.
2. Parallelize nodes within a layer (`tree.py:306` loop over `inputs`) with joblib loky, `n_jobs=n_workers`,
   only for layers with >1 node. Worker = module-level fn: build MLModel, train, raise-flag if
   empty internal node, `prepare_layer` (`tree.py:204`; make it a static/module fn), save to `ml_dir`, return
   `(ml_path, raw_children)`. Parent keeps input order, so `ml_list`/`next_inputs`/
   `layer_child_maps` and the load-time `child_index_map` guard are unchanged.
   BLAS: `classifiers.py:343` lifts the cap to `os.cpu_count()` (`classifiers.py:174` OvR also uses all cores);
   make it a module value (`BLAS_THREADS`) the worker sets to `max(1, cpu_count // n_jobs)`. Rankers inside
   workers get `n_label_workers=1`. Add `n_jobs` to the parent's existing layer summary.
   Risks: per-worker CUDA context for internal-layer k-means (fallback: `device` cpu);
   memory = n_jobs node copies + loky memmaps (`/dev/shm` on Linux).
3. Verify: pytest suite + selfchecks (`docs/pipeline.md`); synthetic flag-1 train/predict with `n_workers=1`
   bit-identical to the pre-change commit; `n_workers=4` same top-k, scores within ~1e-5 (BLAS thread count
   changes summation order). User reruns the step-0 command (log to `speed_after_bc5cdr_ctd.log`) and the
   beam-10 dev eval of the new tree; acc@1 must stay 0.845 +- noise.

**Done: XMR4EL vs PECOS on BC5CDR** (`docs/results.md` § BC5CDR, § BC5CDR-disease dev/test).
- Data: `datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/`; disease-only splits `datasets/BC5CDR/disease/{train,dev,test}.pubtator`;
  CTD MEDIC dictionary `datasets/CTD/ctd_disease.pubtator` (`scripts/dict_to_pubtator.py`), merged as
  `train_plus_ctd.pubtator` (11,744 labels). Reader drops `-1` ids, splits composites via column 7
  (`tests/data/test_readers.py::test_bc5cdr_ids`).
- PECOS needs `-threshold 0` (default 0.1 cost it 0.075 on dev). On Mac: `docker build/run --platform linux/amd64`
  (`docs/full_cuda_comparison.md`).
- Selected: `configs/xmr4el_bc5cdr_config.json` (depth 2) on train + CTD, beam 10, hybrid (dictionary if string seen).
  Depth 3 (`xmr4el_bc5cdr_dict_config.json`) loses ~0.08-0.09. Test: XMR4EL 0.860 / hybrid 0.906 vs PECOS
  0.862 / 0.904 (tie). Literature test acc@1 BioSyn 93.2, SapBERT 93.5 (protocol not matched).
- Open accuracy gaps (after speed): unseen strings 0.64; tree below the dictionary on seen 1-label strings
  (0.930 vs 0.985); optional CTD cost on the 3631 shared dev rows.
- Earlier MedMentions PECOS rows used `threshold` 0.1: rerun with 0 before claiming parity or a win there.

**Current best, MedMentions (session 11-12):** base config = flag 6 (SapBERT mention | char TF-IDF->SVD 768 |
context window 10 TF-IDF->SVD 768) + `abbrev_expansion` append + matcher `jointlogisticregression`. Eval with
`-beam_size 2 -alpha 0 -path_score`. Dev, beam 2: 500 labels 0.862 / MRR 0.896 (PECOS 0.861, `threshold` 0.1);
1000 labels 0.843 / 0.884 (PECOS 0.838, `threshold` 0.1). Weak group: unseen strings 0.53-0.58.
Train 375.7 s at 1000 labels vs PECOS 12.4 s.

## Future work

- Fused matcher/ranker cluster scores (`MLModel.fused_predict` -> `scoring.fused_cluster_scores`): removed
  2026-10-07, it had no callers. Recover with `git show b5c56a8:xmr4el/learning/scoring.py`
  (`fused_cluster_scores`) and `git show b5c56a8:xmr4el/hierarchy/node.py` (`fused_predict`).
- Unseen strings (abbreviations not defined in the doc, generic labels); a candidate reranker on top of
  matcher x path score (must beat 0.843 at 1000 labels); encoder swap (flat screen first); training-time
  stage breakdown; full-label MedMentions run (user does not want it yet).

## Scope and working rules

- Train/evaluate on local PubTator or grouped TSV + label files. UMLS/KB integration is out of scope.
- The user runs corpus training/evaluation. Agents may run synthetic checks and provide commands.
- Commands: `.venv/bin/python` on the user's machine, `python3` inside Docker containers.
- Preserve datasets and `outputs/`. Dependencies are managed with `uv`.
- No compatibility code for old saved trees: a tree the current code cannot load is retrained.
- Use 500–1000 labels for diagnosis, not for claims about full-label-space performance.
- Change one experimental factor at a time. Record bundled changes as a bundle without attributing
  the outcome to one component.

## Interpretation rules

- Report uniform random-cluster chance beside routing: top-1 is `1/K`, top-2 is `2/K`
  for a single root with `K` clusters. At `K=6`, these are 0.167 and 0.333.
  The diagnostic also reports a majority-cluster reference to expose mention-frequency skew.
- `-topk 0` removes the final per-leaf cut, but the internal leaf matcher still selects at most
  100 clusters/labels. Candidate recall is a pure routing ceiling only when that limit is inactive.
- The evaluator's random-ordering line is approximate when candidate counts vary. For exact
  comparisons, average the per-query expectation `hit_i * min(k, n_i) / n_i`, zero for empty rows.
  Do not diagnose a broken ranker from a small gap to the approximate line.
- Metrics exclude out-of-vocabulary gold labels. Record kept/total alongside every row and compare
  different vocabulary sizes separately. Coverage times conditional accuracy gives all-mention
  accuracy when excluded labels count as misses.
