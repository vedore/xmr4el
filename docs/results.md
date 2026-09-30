# Results

One row per change, with the exact command that produced it. This table is the ablation appendix.

## Metric choice

Each mention carries exactly one gold CUI (`featurization/preprocessor.py:113` → one-hot `Y` rows),
so this is extreme multi-**class**, not multi-label. Reported metrics are **acc@1, MRR, recall@k**.
PECOS's precision@k / propensity-scored suite is **not** reused: with a single gold label,
precision@k is just recall@k / k and carries no extra information.

`recall@candidates` is the binary `gold in cand` hit count. `In-vocabulary mentions` is the vocabulary ceiling — mentions whose gold CUI is
absent from the training label space are deleted before scoring
(`test/xmr4el/test_evaluate_pipeline.py`), so every row below is an in-vocabulary upper bound.

## Session 6 (2026-09-30): first dev rows with the corrected label mapping

Eval-only on the saved 2026-09-29 trees; code = `9556856` + uncommitted `classes_` label mapping.
st21pv dev, 500-label vocabulary (first 500 label groups), in-vocabulary 7530/40884 (18.4%),
347 gold CUIs kept. `-beam_size 2 -topk 0`, K=6 root clusters, leaf limit 100. Both trees have
label-level leaf matchers (OvR classes = leaf labels, 83-85 per leaf) and 66-83 rankers per leaf.

| Tree | flag | recall@cand | acc@1 | MRR | R@5 | R@20 | R@100 | cand/query | random acc@1 |
|---|---|---|---|---|---|---|---|---|---|
| `15-16-12` | 1 | 0.823 | 0.113 | 0.217 | 0.339 | 0.501 | 0.705 | 166.6 | 0.0049 |
| `15-58-50` | 4 | **invalid** (see below) | 0.124 | 0.144 | 0.163 | 0.171 | 0.227 | 166.6 | 0.0025 |

`diagnose_routing.py` (5000 sampled rows per split, seed 0):

| Tree | split | majority cluster | root matcher | cosine/all | flat nearest-label acc@1 |
|---|---|---|---|---|---|
| `15-16-12` | train | 0.558 | 0.640 | 0.649 | 0.304 |
| `15-16-12` | dev | 0.598 | 0.546 | 0.600 | 0.267 |
| `15-58-50` | train | 0.301 | 0.806 | 0.632 | 0.364 |
| `15-58-50` | dev | 0.316 | 0.283 | 0.241 | 0.106 |

Dictionary baseline (dev, mention string -> most frequent train CUI): acc@1 0.718, string coverage 0.743.

**Flag-4 tree is invalid: its training features are row-permuted.** `Transformer._predict` read its
on-disk embedding batches with a lexicographic sort (`batch10` before `batch2`). The 23512 training
rows span 12 batches of 2000, so every row from 4000 on got another row's mention embedding. Checked
on the saved `X`: repeated mention strings share an identical mention vector for 1.00 of pairs under
the lexicographic order and 0.44 under the true order. Dev eval (<=10 batches) was unaffected. Flag 1 (TF-IDF only) never calls the transformer. Fixed: batch files are keyed by start row
and read in numeric order, which also fixes OOM recovery (shrinking the batch size used to skip rows).
The flag-4 tree must be retrained. The earlier train-side figures (matcher 4.79x chance, cosine 3.81x)
came from this tree's permuted `X` and are withdrawn.

Flag-1 reading (valid, 500 labels only):
- Scoring beats random ordering (acc@1 23x the random line), so the leaf scores carry signal. This does
  not show that rankers improve on matcher-only scores; that is the `-alpha 0` vs `0.5` comparison.
- Root routing top-1 is below the majority-cluster reference on dev (0.546 vs 0.598). Clusters have
  similar label counts but skewed mention frequencies. How much of recall@cand 0.82 at beam 2 is skew
  needs the fixed top-2-cluster reference (now printed by `diagnose_routing.py`).
- Hierarchy acc@1 0.113 (7530 rows) vs flat nearest-label 0.267 and mention dictionary 0.718 (5000-row
  sample). Not yet on identical rows; rerun the diagnostic with `-max_rows 100000` to use all 7530.
  All figures cover only the 18.4% of dev mentions inside the 500-label vocabulary.
- The permutation invalidates flag 4; only the corrected retrain can show how much of its train/dev
  gap it explains.
- Mention-block share 0.994-0.998 of the squared row norm in flag 4 (valid: it does not depend on row
  order) confirms that the context block contributes almost nothing after concatenation.

## Commands

Older, invalid runs: `docs/results_archive.md`.

```bash
# train (500-label diagnosis scale)
.venv/bin/python test/xmr4el/test_train_pipeline.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config .models/xmr4el_base_config.json -ds_len 500

# eval (all in-vocabulary dev rows; -alpha 0 = matcher only)
.venv/bin/python test/xmr4el/test_evaluate_pipeline.py -xmodel_path test/test_data/saved_trees/<run> \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -beam_size 2 -topk 0 -alpha 0.5

# routing / flat / dictionary diagnostic on the same rows as eval
.venv/bin/python test/xmr4el/diagnose_routing.py -xmodel_path test/test_data/saved_trees/<run> \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -max_rows 100000
```

Synthetic checks: see `STATUS.md`.
