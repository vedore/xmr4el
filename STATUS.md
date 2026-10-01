# Status and next steps

## Resume here

Last updated 2026-10-01 (session 7). A new session starts from this block; update it at the
end of every step and before the user resets the chat.

Code not frozen (no user run in flight).

Done (session 7, `docs/results.md` Session 7): flag-4 `16-37-01` alpha sweep + diagnostic; flag-1
`15-16-12` diagnostic + alpha 0 / 1; leaf-Z fix retrain `xmodel_2026-10-01_11-20-27` with alpha
sweep, diagnostic and `-scorer cosine` at alpha 0.5 / 1. Uncommitted: leaf-Z fix (`prepare_layer`),
`test_prepare_layer.py`, eval `-scorer cosine` switch (`MLModel.predict` reads `cosine_scorer`).

State (flag 4, 500 labels): hierarchy + cosine leaf scorer acc@1 0.764 (alpha 1) / 0.748 (0.5),
MRR 0.822, = flat nearest-label 0.767, > dictionary 0.717. recall@cand 0.949 is the remaining cap.
Trained per-label rankers anti-rank (defect 2: under-trained, uncalibrated; more epochs alone fail).
Step 4 exit reached: one attributable result (leaf-Z fix) + retain decision.

User decisions (2026-10-01): commit; rankers left out of prediction (eval `-scorer` defaults to
`cosine`; `-scorer ranker` reproduces the old rows); user will redesign rankers later. Leaf ranker
training still runs (cost only).

Code not frozen. Latest (results Session 7, "Mention-string breakdown"): tree 0.764, hybrid
(dict if string seen, else tree) 0.805, beam 3 no acc@1 gain. 72% of tree errors are unseen mention
strings (acc@1 0.343); seen strings are near-solved by the dictionary.

Next decision (proposed to user): attack unseen strings. First look at what they are (abbreviations,
synonyms, morphology) with an error sample, then one feature change (e.g. char n-gram TF-IDF on the
mention, or per-block normalisation so context counts), one retrain, same breakdown. The hybrid is
kept as a reported line; ranker redesign can use the string prior as a feature.

Commands: `docs/results.md` § Commands. Old invalid runs: `docs/results_archive.md`.

## Overview

**Order: label mapping (done) → saved-tree diagnostics (done; flag-4 tree invalid) → retrain the
flag-4 tree + controlled scoring ablation on the flag-1 tree (in flight) → choose further model changes.**
Only `docs/results.md` holds valid rows (Session 6 on). Flag 1 is valid; the old flag-4 tree
`15-58-50` is invalid (row-permuted training features).

## Scope and working rules

- Train/evaluate on local PubTator or grouped TSV + label files. UMLS/KB integration is out of scope.
- The user runs corpus training/evaluation. Agents may run synthetic checks and provide commands.
- Preserve datasets and saved trees. Use `.venv/bin/python`; dependencies are managed with `uv`.
- Use 500–1000 labels for diagnosis, not for claims about full-label-space performance.
- `README.md` describes usage; `AGENTS.md` and `CLAUDE.md` describe conventions;
  `docs/results.md` holds valid results; `docs/results_archive.md` holds invalid history.
- Change one experimental factor at a time. Record bundled changes as a bundle without attributing
  the outcome to one component.

## What is implemented

| Area | Current behavior | Evidence |
|---|---|---|
| Model loading | Numeric `ml_<n>` order; guard checks parent/child label sets | `HierarchicaMLModel.load`, `xmr4el/xmr/base.py` |
| Label mapping | Training keeps `MultiLabelBinarizer.classes_` (empty label groups have no column); load sorts legacy first-seen lists and asserts label count equals `Z` rows | `XModel._fit/load`, `xmr4el/xmr/model.py` |
| Evaluation | Hierarchy scores via `per_leaf`; acc@1, MRR, recall@k, candidate recall, vocabulary coverage | `test/xmr4el/test_evaluate_pipeline.py` |
| Features | Flag 4: transformer mention + reduced TF-IDF context, then row normalization | `TextEncoder.encode/predict`, base config |
| Clustering | Seeded balanced k-means; undersized clusters reassigned instead of dropping labels | `xmr4el/models/cluster_wrapper/clustering_model.py`, `xmr4el/clustering/train.py` |
| Ranker training | Epochs reuse models and vary sampling seeds; config uses `E_warm=1`, `log_loss`, `neg_mult=5` | `xmr4el/ranker/train.py`, `.models/xmr4el_base_config.json` |
| Leaf matching | Newly trained leaf matchers are label-level; leaf early stopping is disabled for singleton positives | `MLModel.train`, `xmr4el/xmr/base.py` |
| Missing rankers | Prediction uses a cosine fallback | `MLModel.predict`, `xmr4el/xmr/base.py` |

Implemented does not mean validated on dev. Saved trees retain their trained classifiers and leaf
structure; loading them with current code does not apply training changes retroactively.

## Evidence we can use

Three row/label alignment failures invalidated earlier evaluation:

1. Layer models loaded in filesystem order, so traversal visited the wrong child models.
   Numeric loading order and the parent/child invariant guard now address this.
2. `Y` columns follow sorted label names, but `initial_labels` held first-seen names. The recorded
   500-label check found zero matching positions. Training now keeps the binarizer's `classes_`.
3. Transformer embedding batches were reassembled in lexicographic file order (`batch10` before
   `batch2`), permuting every row past batch 1 when a corpus spans more than 10 batches. This hit
   every flag 2-4 training run (23.5k rows = 12 batches); flag 1 is TF-IDF only. Now sorted
   numerically; files are keyed by start row so OOM recovery cannot skip rows. The earlier train-side figures (4.79x, 3.81x) came from permuted flag-4 features
   and are withdrawn.

Valid now (flag 1, 500 labels, `docs/results.md` Session 6): acc@1 0.113 at 23x the random line;
dev root routing 0.546 vs majority cluster 0.598; flat nearest-label 0.267; mention dictionary 0.718.

Synthetic checks passed during the latest review:

```bash
.venv/bin/python test/xmr4el/test_data_loading.py
.venv/bin/python test/xmr4el/test_evaluate_pipeline.py -selfcheck
.venv/bin/python test/xmr4el/diagnose_routing.py -selfcheck
.venv/bin/python test/xmr4el/test_prepare_layer.py
```

`test_data_loading.py` covers the mapping through the real `XModel.train/save/load` paths
(encoder and hierarchy stubbed): unsorted labels plus an empty group, legacy first-seen metadata,
and a legacy empty group, which load refuses. It fails when any of the mapping fixes is removed.

## Next steps, in order

### 1. Label mapping — done

`initial_labels` is the binarizer's `classes_`, not a re-sorted input list; see the regression above.

### 2. Corrected saved-tree diagnostics

The recorded trees are flag 1 (`15-16-12`) and flag 4 (`15-58-50`), each trained with 500 labels.
These runs are diagnostics, not a feature ablation: their training history is uncertain (historical
notes disagree about ranker loss) and is not worth reconstructing. Inspect only the minimal metadata
needed to interpret them: the label vocabulary, and whether each leaf matcher is label-level
(classes = leaf labels) or cluster-level. That decides what "matcher-only" means in step 3.
The 500-label vocabulary (~18% dev coverage) limits generalization, not a controlled comparison.

The user runs:

```bash
for t in xmodel_2026-09-29_15-16-12 xmodel_2026-09-29_15-58-50; do
  .venv/bin/python test/xmr4el/test_evaluate_pipeline.py \
    -xmodel_path "test/test_data/saved_trees/$t" \
    -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt \
    -beam_size 2 -topk 0

  .venv/bin/python test/xmr4el/diagnose_routing.py \
    -xmodel_path "test/test_data/saved_trees/$t" \
    -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
    -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt \
    -max_rows 100000   # all in-vocabulary dev rows, identical to the eval rows
done
```

The diagnostic samples up to 5000 rows per split with seed 0. It reports root matcher vs cosine
routing on train/dev, flat nearest-label accuracy, a mention dictionary baseline, and flag-4 block
contributions. Compare models on the same label vocabulary and dev examples.

Interpretation rules:

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

**Done (Session 6).** Flag-1 rows are valid. The flag-4 tree must be retrained after the batch-order
fix; retraining with current code is a new baseline, not a single-factor delta against `15-58-50`.

**Exit (original):** corrected dev rows for both trees plus both diagnostic reports in `docs/results.md`.
Record tree ID, code revision/diff, actual saved configuration, split, vocabulary, seed, beam,
top-k, candidate counts, metrics, and chance references. Mark old rows explicitly invalid.

### 3. Controlled scoring ablation

Keep the saved tree, queries, beam, candidate set and fusion mode fixed; compare matcher-only
(`alpha=0`) with current fusion (`alpha=0.5`) without a final per-leaf cut.
Control added: `-alpha` on `test_evaluate_pipeline.py` (passed through `XModel.predict`). The eval
prints acc@1/MRR split by whether the gold label's ranker scored, and counts rankers that fell back
to cosine at predict time (`MLModel.ranker_failed`). The flag-1 tree has 435/500 trained rankers.

Scope of the claim:

- `alpha=0.5` fuses learned rankers **and** cosine fallbacks, so a gain does not show that learned
  rankers helped. Report results separately for gold labels with and without a trained ranker.
- A trained ranker that raises at predict time silently falls back to cosine
  (`except Exception`, `MLModel.predict`, `base.py`). Count those fallbacks; "with ranker" means
  the ranker actually scored.
- `alpha=0` still computes ranker outputs and weights them to zero, so this measures inference
  contribution, not inference cost or the effect of removing ranker training.
- Whether the ranker is the thesis's main contribution depends on agreed thesis scope.

**Exit:** one attributable result and a retain/revert decision for ranker fusion.

### 4. Choose further model changes from the corrected evidence

These are observations and hypotheses, not a mandatory sequence of fixes.

| Pipeline stage | Confirmed behavior | Follow-up if implicated |
|---|---|---|
| Data | `-ds_len` keeps the first N label groups; the historical 500-label vocabulary covered about 18.4% of dev mentions | Keep vocabulary fixed for debugging; use a seeded, recorded sample or full vocabulary for broader claims |
| Features | Flag 4 normalizes after concatenation; raw block norms determine relative contribution | If block diagnostics suggest a problem, compare one normalization change with a retrained control; dominance alone is not proof of harm |
| Label embeddings / hierarchy | PIFA uses full mention+context features | If flat retrieval works but centroid routing fails, isolate a label-representation or clustering change; PIFA itself is not a bug |
| Matcher | Child models train on in-cluster rows but receive beam-routed queries | If root routing works and child rejection fails, measure out-of-cluster errors before adding routed negatives |
| Ranker | Each per-label linear model sees a constant `Z_label` block, contributing an intercept-like term | Measure scoring contribution before redesigning; constant features are not evidence of failure |
| Traversal / scoring | Ancestor `path_logscore` prunes the beam but is omitted from final leaf scores | If retrieval is good and cross-leaf ordering is poor, test route-weighted scoring on the same saved tree and candidates |
| Training computation | `fused_predict` applies the first available ranker to every pair; leaf training still calls it | Trace consumers before changing it. It is not dead code, although final-layer child preparation appears unnecessary for prediction |

Start route-score experiments with a fixed rule, such as adding ancestor log probabilities to
log leaf scores. A learned weight needs separate justification and dev tuning. Neither route
weighting nor `log_loss` alone establishes calibration across leaves.

**Exit:** one attributable result and a retain/revert decision. If a new tree is needed, separate
the current-code baseline from subsequent changes; old trees do not test the new leaf matcher.

### 5. External baseline and scale, after measurement is trustworthy

- Keep the per-label ranker as a research hypothesis and measure its contribution within XMR4EL.
- Then run a PECOS baseline on the same split, label mapping and instance features. Record hierarchy,
  negative sampling, scoring, search budget, runtime and memory differences. Similar parameter names
  do not establish equivalent algorithms.
- Custom layer feature augmentation remains in `prepare_layer`; the proposed scoring/matcher changes
  do not make XMR4EL differ from PECOS only by its ranker. Treat this as a system comparison unless
  other differences are explicitly controlled.
- PECOS setup is pending. Earlier notes reported a macOS wheel limitation; verify package/runtime
  compatibility when scheduling it. Linux/Docker is an option, not an already-tested environment.
- Keep st21pv as the target dataset. Report actual training-label count and dev/test coverage;
  the dataset's advertised label count is not automatically the model vocabulary. Reserve test
  evaluation for the selected configuration. Consider MedMentions/full only if scaling is needed.

No broad rewrite, learned score calibration, or additional encoder dependency is scheduled.
