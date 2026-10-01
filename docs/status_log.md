# Status log

Completed steps and history moved out of `STATUS.md`. Not a work order; results live in
`docs/results.md` (valid) and `docs/results_archive.md` (invalid).

## Session 7 (2026-10-01)

Done (`docs/results.md` Session 7): flag-4 `16-37-01` alpha sweep + diagnostic; flag-1
`15-16-12` diagnostic + alpha 0 / 1; leaf-Z fix retrain `xmodel_2026-10-01_11-20-27` with alpha
sweep, diagnostic and `-scorer cosine` at alpha 0.5 / 1. Committed in `f24b35c` (leaf-Z fix in
`prepare_layer`, `test_prepare_layer.py`, eval `-scorer cosine`) and `31d5486` (mention-string
breakdown, dictionary hybrid).

State (flag 4, 500 labels): hierarchy + cosine leaf scorer acc@1 0.764 (alpha 1) / 0.748 (0.5),
MRR 0.822, = flat nearest-label 0.767, > dictionary 0.717. recall@cand 0.949 is the remaining cap.
Trained per-label rankers anti-rank (defect 2: under-trained, uncalibrated; more epochs alone fail).
Step 4 exit reached: one attributable result (leaf-Z fix) + retain decision.

Mention-string breakdown: tree 0.764, hybrid (dict if string seen, else tree) 0.805, beam 3 no
acc@1 gain. 72% of tree errors are unseen mention strings (acc@1 0.343); seen strings are
near-solved by the dictionary.

User decisions (2026-10-01): rankers left out of prediction (eval `-scorer` defaults to `cosine`;
`-scorer ranker` reproduces the old rows); user will redesign rankers later. Leaf ranker training
still runs (cost only). Explore featurization before scaling labels.

## Alignment failures that invalidated earlier evaluation

1. Layer models loaded in filesystem order, so traversal visited the wrong child models.
   Numeric loading order and the parent/child invariant guard now address this.
2. `Y` columns follow sorted label names, but `initial_labels` held first-seen names. The recorded
   500-label check found zero matching positions. Training now keeps the binarizer's `classes_`.
3. Transformer embedding batches were reassembled in lexicographic file order (`batch10` before
   `batch2`), permuting every row past batch 1 when a corpus spans more than 10 batches. This hit
   every flag 2-4 training run (23.5k rows = 12 batches); flag 1 is TF-IDF only. Now sorted
   numerically; files are keyed by start row so OOM recovery cannot skip rows. The earlier
   train-side figures (4.79x, 3.81x) came from permuted flag-4 features and are withdrawn.

Valid after the fixes (flag 1, 500 labels, `docs/results.md` Session 6): acc@1 0.113 at 23x the
random line; dev root routing 0.546 vs majority cluster 0.598; flat nearest-label 0.267; mention
dictionary 0.718.

## Step 1. Label mapping — done

`initial_labels` is the binarizer's `classes_`, not a re-sorted input list; regression in
`test/xmr4el/test_data_loading.py`.

## Step 2. Corrected saved-tree diagnostics — done (Session 6)

Trees: flag 1 (`15-16-12`) and flag 4 (`15-58-50`), each 500 labels. Diagnostics, not a feature
ablation; training history uncertain. Flag-1 rows are valid. The flag-4 tree was invalid
(row-permuted features) and was retrained after the batch-order fix as a new baseline.

Commands used:

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

Exit: corrected dev rows for both trees plus both diagnostic reports in `docs/results.md`, with
tree ID, code revision, saved config, split, vocabulary, seed, beam, top-k, candidate counts,
metrics and chance references; old rows marked invalid.

## Step 3. Controlled scoring ablation — done (Session 7)

Fixed tree, queries, beam, candidates and fusion mode; matcher-only (`alpha=0`) vs fusion
(`alpha=0.5`), no final per-leaf cut. `-alpha` on `test_evaluate_pipeline.py`; eval splits
acc@1/MRR by whether the gold label's ranker scored and counts predict-time fallbacks
(`MLModel.ranker_failed`). Flag-1 tree had 435/500 trained rankers.

Claim limits: `alpha=0.5` fuses learned rankers and cosine fallbacks; a ranker raising at predict
time silently falls back to cosine; `alpha=0` still computes ranker outputs (contribution, not cost).

Outcome: rankers anti-rank; removed from prediction (`-scorer cosine` default).
