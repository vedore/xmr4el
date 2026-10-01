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

## Session 7 (2026-10-01): flag-4 retrain + flag-1 alpha ablation

Code = `47ee08b` (batch-order + OOM fixes, `classes_` label mapping). Same split/vocabulary as Session 6:
st21pv dev, 500 labels, in-vocabulary 7530/40884, 347 gold CUIs. `-beam_size 2 -topk 0`, K=6.
Flag-4 tree `xmodel_2026-09-30_16-37-01` (`-ds_len 500`, base config, X 23512x2268) replaces the
invalid `15-58-50`. Flag-1 tree `15-16-12` unchanged (TF-IDF only, unaffected by the batch fix).

| Tree | flag | alpha | recall@cand | acc@1 | MRR | R@5 | R@20 | R@100 | cand/query | random acc@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `16-37-01` | 4 | 0.5 | 0.949 | 0.529 | 0.671 | 0.841 | 0.854 | 0.914 | 166.3 | 0.0057 |
| `16-37-01` | 4 | 0 | 0.949 | 0.510 | 0.686 | 0.909 | 0.929 | 0.941 | 166.3 | 0.0057 |
| `16-37-01` | 4 | 1 | 0.949 | **0.001** | 0.015 | 0.003 | 0.051 | 0.299 | 166.3 | 0.0057 |
| `15-16-12` | 1 | 0 | 0.823 | 0.143 | 0.283 | 0.456 | 0.707 | 0.797 | 166.6 | 0.0049 |
| `15-16-12` | 1 | 0.5 | 0.823 | 0.113 | 0.217 | 0.339 | 0.501 | 0.705 | 166.6 | 0.0049 (Session 6) |
| `15-16-12` | 1 | 1 | 0.823 | **0.000** | 0.009 | 0.0001 | 0.003 | 0.331 | 166.6 | 0.0049 |

Gold with / without a trained ranker (acc@1): flag 4 alpha 0.5 0.528 (n=7397) / 0.609 (n=133);
flag 4 alpha 0 0.511 / 0.496; flag 4 alpha 1 **0.000** / 0.068; flag 1 alpha 0 0.144 / 0.000 (n=37). 0 cosine fallbacks at predict. Distinct scores/query: flag 1
alpha 0 52.4 of 167 (matcher ties), alpha 1 165.2.

`diagnose_routing.py -max_rows 100000` (all rows; dev rows identical to eval):

| Tree | split | majority | fixed top-2 | root matcher top-1 | top-2 | cosine/all | flat nearest-label acc@1 |
|---|---|---|---|---|---|---|---|
| `16-37-01` | train | 0.352 | 0.616 | 0.903 | 0.968 | 0.768 | 0.793 |
| `16-37-01` | dev | 0.351 | 0.608 | 0.879 | 0.949 | 0.756 | 0.767 |
| `15-16-12` | train | 0.568 | 0.713 | 0.624 | 0.880 | 0.632 | 0.298 |
| `15-16-12` | dev | 0.603 | 0.731 | 0.547 | 0.823 | 0.600 | 0.258 |

Flag 4 extra: mention-block share 0.994 (train) / 0.998 (dev); cosine/mention = cosine/all
(dev 0.756, flat 0.766); cosine/context dev 0.407, flat 0.202. Dictionary baseline (dev, same rows):
acc@1 0.7165, string coverage 0.743.

Reading (500 labels, in-vocabulary rows only):
- **Rankers anti-rank on flag 1.** Ranker-only (alpha 1) puts gold below random ordering: R@50 0.029
  vs random line 0.247, R@100 0.331 vs 0.494, MRR 0.009. acc@1 falls monotonically with alpha
  (0.143 -> 0.113 -> 0.000). Ranker scores carry signal with the wrong sign or are miscalibrated
  across per-label models; cause not yet traced. Flag 4 hints the same: labels scored by cosine
  fallback beat ranker-scored labels (0.609 vs 0.528).
- **Feature-independent: flag 4 alpha 1 is also below random** (R@50 0.101 vs 0.285, R@100 0.299
  vs 0.570). Among 7397 gold labels with a ranker, acc@1 is exactly 0.000. Matcher-only (alpha 0) has
  the best MRR/R@5 (0.686 / 0.909); alpha 0.5 trades R@5 for +0.019 acc@1.
- **Cause, two training-side defects** (checked on the saved trees, `16-37-01` and `15-16-12`):
  1. *Leaf label embeddings are erased.* `prepare_layer` appended mean/sum/max of `X_node . Z` to each
     child z; the sum grows with node size, so after L2 norm those 3 dims hold 1.000 of every leaf z's
     squared norm (both trees, all checked leaves). Leaf cosine fallback and ranker hard-negative
     mining (cos/ip to z) therefore ignore the label. Fixed (zero pad, width kept);
     regression `test/xmr4el/test_prepare_layer.py` (cos 0.0009 before, 1.0 after).
  2. *Per-label rankers are under-trained and not comparable across labels* (not fixed). Leaf rankers
     of `16-37-01`: median ||w_X|| 0.13-0.34, so half output ~sigmoid(bias) 0.47-0.50 for any query;
     better-trained rankers have strongly negative bias (corr(||w_X||, bias) -0.75 to -0.85, 1:5
     negatives). The gold label's ranker is usually trained, so its score on its own mention falls
     below the ~0.5 crowd of untrained rankers. Synthetic reproduction (SGD log_loss, l1,
     eta0 0.01, 3 partial_fit epochs): n_pos=2 outputs 0.47 for positive and OOD queries alike;
     n_pos=300 gives its positive 0.46. 20 epochs do not fix it (n_pos=20 positive 0.31 < n_pos=2
     baseline 0.36), so more epochs alone are not the fix.
- **Flag 4 routing is solved at this scale; leaf scoring is the bottleneck.** Root matcher 0.879 vs
  majority 0.351 (flag 1: 0.547 vs 0.603, below majority). recall@cand 0.949 caps acc@1 at 0.949, but
  the hierarchy gets 0.529 while flat nearest-label on the same features gets 0.767 and the
  dictionary 0.717.
- Flag 4 vs flag 1 (alpha 0.5, identical rows): acc@1 0.529 vs 0.113, recall@cand 0.949 vs 0.823.
  This bundles features, clustering and every trained component; no single factor is attributed.
- The flag-4 mention block still carries ~all of the row norm; context alone is weak (flat 0.202).

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
