# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-06 (session 13, restructure). A new session starts from this block; update it at the
end of every step and before the user resets the chat.

**Nothing in flight. Code not frozen.** Restructure steps 0-7, 9, 10 done (session 13, commits
`2e068a1`..HEAD). Waiting on the user: step 8 (`rm -rf test`; old trees no longer load) and step 11
(retrain one base-config tree, command below). When step-11 output is pasted: compare with the session-12
1000-label joint row (0.8436 / MRR 0.8827); a match within noise closes the restructure.

```bash
rm -rf test && python3 scripts/train.py -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt -model_config configs/xmr4el_base_config.json -ds_len 1000 && python3 scripts/evaluate.py -xmodel_path "$(ls -td outputs/saved_trees/xmodel_* | head -1)" -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt -beam_size 2 -topk 0 -alpha 0 -path_score
```

Restructure notes: `prepare_data_older` was the live flattener, renamed `prepare_data` (not deleted).
`numba` is now an explicit dep (`kmeans_pytorch` imports it; `umap-learn` used to pull it in). Synthetic
baseline (flag-1 XModel train/save/load/predict, 2 configs) matched bit-for-bit after every step, except the
ranker-scored path (`train_rankers` + `alpha 0.5`), which is nondeterministic run to run before the
restructure too (max |diff| 0.42, argmax stable): parallel ranker training, not investigated.

**Current best (session 11-12):** base config = flag 6 (SapBERT mention | char TF-IDF->SVD 768 | context
window 10 TF-IDF->SVD 768) + `abbrev_expansion` append + matcher `jointlogisticregression`. Eval with
`-beam_size 2 -alpha 0 -path_score`. Dev, beam 2: 500 labels 0.862 / MRR 0.896 (PECOS 0.861); 1000 labels
0.843 / 0.884 (PECOS 0.838). Weak group: unseen strings 0.53-0.58. Train 375.7 s at 1000 labels vs PECOS
12.4 s; stage breakdown not yet seen.

## Restructure plan (user decision 2026-10-06)

No compat layer: old saved trees are deleted, not migrated. `docs/results.md` rows stay as records; their
trees cannot be reloaded after step 4 (pickles store module paths), so reproducing them needs a retrain.

Target:

```
xmr4el/
  __init__.py      logger
  data.py          Preprocessor + PubTator split functions + __main__ CLI (preprocessor.py + utils/pubtator_splits.py)
  encoder.py       TextEncoder + PIFA (text_encoder.py + label_embedding_factory.py)
  vectorizers.py   Tfidf + TruncatedSVD (vectorizers.py + dimension_model.py)
  transformers.py
  classifiers.py   used backends + Matcher + MatcherTrainer (classifier_model.py + matcher/*)
  clusterers.py    used backends + Clustering + ClusteringTrainer (clustering_model.py + clustering/*)
  ranker.py        Ranker + RankerTrainer (ranker/*)
  node.py          MLModel (xmr/base.py, first part)
  tree.py          HierarchicalMLModel (xmr/base.py, second part; typo fixed)
  xmodel.py        XModel (xmr/model.py)
  temp_store.py
  eval.py          gold_rank, split_by_ranker, string_breakdown (from test_evaluate_pipeline.py)
scripts/   train.py, evaluate.py, diagnose_routing.py, screen_features.py, screen_leaf_scorer.py, pecos_run.py, pecos_compare.py
tests/     the test_* files as pytest tests
configs/   *.json (was .models/)
outputs/   saved_trees/, ablation/, pecos/ (gitignored)
```

Steps (one commit each, only while no user run is in flight; must preserve the loader ordering guard,
sorted label order, leaf label-level matching, ranker warm-starting, small-cluster reassignment):

0. Commit the uncommitted session-12 fixes as they are.
1. Baseline: agent runs a small synthetic train + eval, saves scores to the scratchpad.
2. Delete unused code: classifier backends RandomForest/SVC/LightGBM; cluster backends FAISS/MiniBatch/
   Agglomerative; `force_multi_core_processing_clustering_model`; `umap-learn` dep (then `uv lock`);
   tracked `test/xmr4el/test.log`; `prepare_data_older` after checking its call in `xmr/model.py`.
3. Merge `utils/pubtator_splits.py` into `data.py` (nothing imports it; its docstring names a wrong file;
   run as `python -m xmr4el.data`).
4. Flatten the package to the target tree (`git mv` + merges), fix imports, keep registry `type` strings
   so configs load; check `node.py` <-> `classifiers.py`/`clusterers.py` import cycles.
5. Split `base.py` into `node.py` / `tree.py`; rename to `HierarchicalMLModel` everywhere.
6. Move eval metrics into `xmr4el/eval.py`.
7. `test/xmr4el/` -> `scripts/` + `tests/` (drop `sys.path` hacks, pytest `def test_*`); `.models/` ->
   `configs/`; `test/test_data/{ablation,pecos}` -> `outputs/`; gitignore `outputs/`.
8. User deletes old trees: `rm -rf test/test_data`.
9. Agent reruns step-1 baseline + `pytest tests/`; scores must match.
10. Update README, AGENTS.md, CLAUDE.md (paths, entry points, saved-trees rule), `docs/pipeline.md`
    paths, a note in `docs/results.md` that its trees were deleted, this block.
11. User retrains one base-config tree into `outputs/saved_trees/` (agent gives the command).

## Research next (after the restructure)

User does not want the full-label run yet. Candidates: unseen strings (abbreviations not defined in the
doc, generic labels), candidate reranker on top of matcher x path score (must beat 0.843 at 1000 labels),
encoder swap (flat screen first), training-time stage breakdown.

## Scope and working rules

- Train/evaluate on local PubTator or grouped TSV + label files. UMLS/KB integration is out of scope.
- The user runs corpus training/evaluation. Agents may run synthetic checks and provide commands.
- Preserve datasets and `outputs/`. Dependencies are managed with `uv`.
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
