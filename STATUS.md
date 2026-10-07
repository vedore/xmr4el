# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-07 (legacy removal; work order reset).

**Nothing is running.** The user is not running training or evaluation now, so code is not frozen.

**In flight: training speed (planned, not started).** From the log of an earlier
`xmr4el_full_cuda_config.json` run: leaves take ~0.25 s Clustering->Matcher and ~1.5-2 s
Matcher->next node; depth 4 x 8 clusters = up to 585 nodes. Leaf k-means is already skipped
(`node.py:248`); the leaf's 0.25 s gap is `gc.collect()`.
1. Drop the three `gc.collect()` in `MLModel.train` (`node.py:261,304,354`); `tree.py` has none.
   Refcounting frees the arrays.
2. Parallelize nodes within a layer (`tree.py:308` loop) with joblib loky, `n_jobs=n_workers`,
   only for layers with >1 node. Worker = module-level fn: build MLModel, train, raise-flag if
   empty internal node, `prepare_layer` (make it a static/module fn), save to `ml_dir`, return
   `(ml_path, raw_children)`. Parent keeps input order, so `ml_list`/`next_inputs`/
   `layer_child_maps` and the load-time `child_index_map` guard are unchanged.
   BLAS: `classifiers.py:343` lifts the cap to `os.cpu_count()`; make it a module value
   (`BLAS_THREADS`) the worker sets to `max(1, cpu_count // n_jobs)`. Rankers inside workers get
   `n_label_workers=1`. Parent logs `layer L: N nodes, n_jobs J` (also fixes "Number 0").
   Risks: per-worker CUDA context for internal-layer k-means (fallback: `device` cpu);
   memory = n_jobs node copies + loky memmaps (`/dev/shm` on Linux).
3. Verify: pytest suite; synthetic flag-1 train/predict with `n_workers=1` bit-identical to
   pre-change; `n_workers=4` same top-k, scores within ~1e-5 (BLAS thread count changes
   summation order). User reruns one corpus train to compare wall time.

**Next: compare XMR4EL with PECOS on BC5CDR.**
- Data: `datasets/BC5CDR/CDR_Data/CDR.Corpus.v010516/CDR_{Training,Development,Test}Set.PubTator.txt`
  (GitHub mirror JHnlp/BioCreative-V-CDR-Corpus; official URL 403). `load_pubtator_file` drops `-1` ids
  (also as composite parts), splits composite `D1|D2` mentions via column 7, drops composites without
  column 7 (`tests/data/test_readers.py::test_bc5cdr_ids`). Rows: train 9396 / 1327 ids, dev 9613, test 9762.
- Done (dev, `docs/results.md` § BC5CDR): tree `16-44-38`, `configs/xmr4el_bc5cdr_config.json` (base with
  `n_clusters` 16: root -> 16 leaves of ~83 labels), PECOS `-nr_splits 16 -max_leaf_size 100`, export
  `outputs/pecos/bc5cdr_dev`. XMR4EL 0.921 vs PECOS 0.847 with default `threshold` 0.1, 0.922 with `-threshold 0` (tie).
- Prepared, not run: CTD MEDIC dictionary `datasets/CTD/ctd_disease.pubtator` (`scripts/dict_to_pubtator.py`;
  85,693 names, 11,742 MeSH ids); disease-only splits `datasets/BC5CDR/disease/{train,dev,test}.pubtator`
  (awk-filtered) and `train_plus_ctd.pubtator` (11,744 labels; dev/test gold in vocab 0.9998/0.9975);
  config `configs/xmr4el_bc5cdr_dict_config.json` (bc5cdr config, depth 3: 16 -> 8 per node with cut_half
  -> 128 leaves; PECOS counterpart `-nr_splits 16 -max_leaf_size 100`).
- To do: train/eval the dict config, PECOS with `-threshold 0` on the same export, test split only for the
  selected configuration. Literature BC5CDR-disease test acc@1: BioSyn 93.2, SapBERT 93.5.
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
