# Status and next steps

History: `docs/status_log.md`. Valid results: `docs/results.md`.
Implemented behavior, synthetic checks, pipeline observations, PECOS notes: `docs/pipeline.md`.
Commands: `docs/results.md` § Commands.

## Resume here

Last updated 2026-10-06 (four-domain implementation).

**Current task: four-domain restructuring implemented.** Source now lives in `xmr4el/data/`,
`features/`, `learning/`, and `hierarchy/`; `XModel` remains the public entry point.
Scripts and tests follow the accepted layout below. Imports and current documentation were updated.
The previous flattening plan is historical, not the current target.

Validation: baseline suite 13 passed; final suite 18 passed. Synthetic flag-1 train/save/load/predict
agrees with the pre-move baseline within 1e-12 across 12 alpha/fusion/path-score combinations,
including loading the pre-move tree. CLI self-checks, compileall and wheel build pass.
macOS/joblib reports physical-core detection and worker-shutdown warnings (shutdown warnings also
occurred in the baseline). `JOBLIB_MULTIPROCESSING=0 .venv/bin/python -m pytest tests/ -q` completes
with 18 passed and no warnings. No corpus training or dataset/artifact deletion was performed.

Implementation notes: texts are owned directly by XModel; encoder components are serialized once;
confirmed unused helpers/caching were removed. Scoring/fusion and label/PIFA, SVD and matcher logic
were extracted; routing/ranker augmentation is shared. `predict(scorer="cosine" | "ranker")` replaces
evaluation's leaf mutation; ranker errors log their cosine fallback. Input lengths are validated,
fitting copies caller configs, and the balanced PyTorch backend loads only when selected.
Legacy flat imports and the otherwise empty `temp_store.py` class remain for existing pickle paths;
removing that file would break pre-move XModel state. PubTator splitting now uses
`python scripts/split_pubtator.py`.

Follow-up: user-run saved-tree corpus evaluation against the existing results; investigate existing
joblib/macOS warnings and ranker-training nondeterminism separately if they obstruct experiments.

**Previous flattening status.** Restructure steps 0-7, 9, 10 done (session 13, commits
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

## Four-domain organization plan (accepted and implemented 2026-10-06)

Scope: small, behavior-preserving moves and targeted extractions; no pipeline rewrite or new
dependencies. Keep `XModel` as the public entry point. Keep package `__init__.py` files minimal.

Target:

```text
xmr4el/
  __init__.py
  xmodel.py                    # public pipeline orchestration
  eval.py                      # shared evaluation functions
  data/
    __init__.py
    readers.py                 # Preprocessor: readers, grouping, text preparation
    splits.py                  # PubTator splitting/export operations
  features/
    __init__.py
    encoder.py                 # TextEncoder: text feature composition
    vectorizers.py             # TF-IDF
    reduction.py               # dimensionality reduction / SVD
    transformers.py
    label_embeddings.py        # label matrices, binarization, PIFA
  learning/
    __init__.py
    classifiers.py             # reusable classifier backends
    matcher.py                 # Matcher and MatcherTrainer
    ranker.py                  # ranker training and model collection
    scoring.py                 # label scoring and fusion extracted from node.py
  hierarchy/
    __init__.py
    clusterers.py              # label clustering
    node.py                    # MLModel: node training and coordination
    tree.py                    # HierarchicalMLModel: construction and traversal
scripts/
  train.py
  evaluate.py
  split_pubtator.py             # CLI extracted from data.py
  diagnose_routing.py
  experiments/
    beam_sweep.py
    screen_features.py
    screen_leaf_scorer.py
  baselines/
    pecos_compare.py
    pecos_run.py
tests/
  data/
  features/
  learning/
  hierarchy/
  integration/                 # full XModel train/save/load/predict checks
  test_eval_quiet.py
configs/
docs/
data/                          # ignored local inputs; preserve
datasets/                      # ignored local inputs; preserve
outputs/                       # ignored experiment artifacts; preserve
```

Root packaging files, Dockerfiles, README, STATUS and agent instructions stay in place.
Tests follow domain ownership, not a mandatory one-test-file-per-source-file rule.
Dependency direction: data and features feed the hierarchy; hierarchy nodes use learning components;
`xmodel.py` coordinates them. Keep data/features independent of hierarchy/learning.

Review findings to address (prioritized; sampled static inspection, not runtime validation):

1. Temporary ownership: TempVarStore shares a class-level directory; cleanup affects all instances.
   Node/ranker imports create unused temporary directories; the tree constructor's directory is
   replaced by its training context. Store training texts directly in XModel and remove temp_store.py
   after checking its callers; retain the tree's scoped TemporaryDirectory.
2. Persistence duplication: XModel.save retains `_text_encoder` after saving it separately;
   TextEncoder.save pops public property names instead of `_vectorizer_model` / `_dimension_model`.
   Correct state exclusion and keep explicit component save/load ownership.
3. Hidden scoring: evaluate.py selects cosine by mutating leaves; node.py broadly catches ranker
   errors and substitutes cosine. Expose scorer selection in prediction arguments and report failures.
4. Oversized prediction: node.py has 706 lines (predict: 243); tree.py has 637 (predict: 267).
   Extract scoring/fusion, share training/inference feature augmentation, and avoid whole-query
   densification where sparse operations suffice. Preserve existing numerical behavior first.
5. Mixed responsibilities: move label/PIFA logic out of encoder.py, SVD out of vectorizers.py,
   matcher logic out of classifiers.py, and splitting/CLI out of data.py as shown above.
6. Dead machinery: repository searches found no callers for node/ranker temporary-save helpers,
   ranker cosine/IP helpers, or component argument-config readers. Ranker's cached function is
   constructed but never called by training. Recheck callers before deletion; keep useful backends.
7. Input/config/import behavior: prepare_data silently truncates mismatched inputs through zip;
   feature/clustering fitting mutates caller configs; XModel import changes environment/warnings
   and eagerly imports the PyTorch clustering backend. Validate lengths, copy configs before updates,
   and scope runtime setup to where it is needed.
8. Evaluation/tests/docs: share repeated metric aggregation without changing intentional tie policies;
   move model persistence and transformer OOM tests out of test_data_loading.py, consolidate production
   self-checks into domain tests, fix README's nonexistent `xmr4el.xmr.model` import and stale global
   reranking description.

Migration order (completed under explicit implementation authorization):

1. Run `python -m pytest tests/` and capture a synthetic train/save/load/predict baseline.
2. Fix temporary ownership and duplicated persistence separately, with focused regression checks.
3. Remove confirmed unused machinery.
4. Move/extract source modules one at a time, update imports, preserve registry `type` strings, and
   check import cycles. Keep old import aliases where existing saved trees must remain loadable:
   pickles reference module paths. The historical no-compat decision below concerned the earlier
   flattening, not permission to delete current outputs.
5. Move scripts/tests; update subprocess paths, split entry points, README, AGENTS.md, CLAUDE.md,
   and docs/pipeline.md. Replace `python -m xmr4el.data` splitting usage with the new split CLI.
6. Refactor scoring and shared augmentation separately from file moves; verify numerical equivalence.
7. Rerun synthetic checks. Prefer saved-tree evaluation for behavior checks; the user runs corpus
   training, and retraining is needed only if artifact compatibility requires it.

Preserve label/column ordering, layer/model load ordering, child-label mapping guards, leaf label-level
matching, ranker warm-starting, small-cluster reassignment, datasets and outputs. No deletion or
corpus retraining is authorized by saving this plan. No generic serialization framework is needed.

## Previous flattening plan (session 13; historical)

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
