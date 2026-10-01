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

### Leaf-Z fix retrain (single factor)

Tree `xmodel_2026-10-01_11-20-27` (flag 4, `-ds_len 500`, base config); code = `47ee08b` + leaf-Z
fix in `prepare_layer` (zero pad). Checked on the saved tree: pad share of leaf z norm 0.0000 in all 6
leaves (old trees: 1.000). Same rows/flags as above.

| Tree | flag | alpha | recall@cand | acc@1 | MRR | R@5 | R@20 | R@100 | gold w/ ranker acc@1 (n=7383) | gold w/o ranker acc@1 (n=147) |
|---|---|---|---|---|---|---|---|---|---|---|
| `11-20-27` | 4 | 0 | 0.949 | 0.510 | 0.686 | 0.909 | 0.929 | 0.941 | 0.510 | 0.517 |
| `11-20-27` | 4 | 0.5 | 0.949 | 0.501 | 0.673 | 0.887 | 0.896 | 0.916 | 0.498 | **0.680** |
| `11-20-27` | 4 | 1 | 0.949 | 0.014 | 0.032 | 0.030 | 0.084 | 0.264 | 0.003 | 0.537 |
| `11-20-27` cosine | 4 | 0.5 | 0.949 | 0.748 | 0.821 | 0.914 | 0.942 | 0.948 | 0.749 | 0.667 |
| `11-20-27` cosine | 4 | 1 | 0.949 | **0.764** | 0.822 | 0.888 | 0.936 | 0.948 | 0.765 | 0.735 |

384 rankers (was 387). `diagnose_routing.py` output is identical to `16-37-01` (root not touched).

Reading:
- Control holds: alpha 0 reproduces `16-37-01` exactly (0.510 / 0.686 / 0.909); root and leaf matcher
  do not use leaf z, so training is deterministic and the fix is the only factor.
- The restored leaf z works as a scorer: labels scored by cosine (no ranker, n=147) go 0.068 -> 0.537
  at alpha 1, and fusing cosine with the matcher lifts them 0.517 (alpha 0) -> 0.680 (alpha 0.5).
- Trained rankers still anti-rank (alpha 1 gold-with-ranker 0.003), and alpha 0.5 is now slightly worse
  than before (0.501 vs 0.529): changing hard negatives did not fix ranker calibration (defect 2).
- `-scorer cosine` (eval-only; every leaf label scored by cosine to its leaf z, rankers ignored):
  acc@1 0.510 -> 0.764 (alpha 1) / 0.748 (alpha 0.5), MRR 0.686 -> 0.822. "gold w/ ranker" here only
  marks labels that have a (unused) trained ranker.
- Hierarchy + cosine now matches flat nearest-label on the same rows (0.764 vs 0.767) and beats the
  dictionary (0.717). Leaf z = parent z with a zero pad and the 3 query extras only rescale x, so the
  leaf cosine orders like flat nearest-label within the candidates; acc@1 within candidates
  0.764/0.949 = 0.805. The remaining gap to flat is routing (recall@cand 0.949).
- Matcher fusion (alpha 0.5) trades -0.016 acc@1 for +0.026 R@5. Alpha was picked on dev; treat
  0.5 vs 1 as unresolved until a held-out check.
- Ranker contribution, measured: negative at every alpha. Retain the leaf-Z fix; leaf scoring by
  cosine beats trained rankers by 0.25 acc@1 on this split.

### Mention-string breakdown and beam 3 (eval-only, `11-20-27`, `-scorer cosine -alpha 1`)

Leaf scoring here is cosine only (alpha 1 gives the matcher weight 0); trained rankers unused.
`-train_path` breakdown; dict = most frequent train label for the exact (lower-cased) mention string.

| group | n | share | tree acc@1 | dict acc@1 | either | tree errors (share of 1776) |
|---|---|---|---|---|---|---|
| all | 7530 | 1.000 | 0.764 | 0.717 | 0.811 | 1776 |
| seen, 1 label | 3631 | 0.482 | 0.952 | 0.984 | 0.989 | 175 (0.10) |
| seen, >1 label | 1964 | 0.261 | 0.832 | 0.928 | 0.943 | 330 (0.19) |
| unseen string | 1935 | 0.257 | 0.343 | 0.000 | 0.343 | 1271 (0.72) |

Hybrid (dict if string seen, else tree): **acc@1 0.805** (+0.041 over the tree, training-free; the
"either" oracle is 0.811). Consistency: "all" reproduces tree 0.764 and dict 0.7165.

Beam 3 (`-beam_size 3`): recall@cand 0.949 -> 0.975, candidates 166 -> 249, acc@1 0.764 -> 0.763,
MRR 0.822 -> 0.821, R@20 0.936 -> 0.951.

Reading:
- Routing is not what limits top-1: beam 3 recovers 2.6% more golds but they rank low; acc@1 flat.
- 72% of tree errors are mentions whose string never appears in train (acc@1 0.343). After the hybrid,
  ~86% of remaining errors are unseen strings. That is where the next gain is: the mention encoder
  (flag 4 = TF-IDF/SVD 1500 + `sentencetbiobert`; mention block holds 0.998 of the row norm).
- On seen strings the most-frequent-label prior beats the tree (0.928 vs 0.832 ambiguous, 0.984 vs
  0.952 unambiguous). The hybrid captures that; a reranker could learn it as a feature.

## Session 8 (2026-10-01): feature screen by flat retrieval (`screen_features.py`)

No training. Vocabulary and dev rows of tree `11-20-27` (500 labels, 7530 in-vocabulary dev rows,
23512 train rows). Mention-only features; label = PIFA centroid of its train rows;
score = cosine to every label (flat, no hierarchy). Breakdown groups as in Session 7.

| features | acc@1 | MRR | hybrid | seen, 1 label | seen, >1 label | unseen string |
|---|---|---|---|---|---|---|
| `sbiobert` (current flag-4 mention block) | 0.766 | 0.825 | 0.804 | 0.956 | 0.833 | 0.340 |
| char 2-4 TF-IDF (`char_wb`, sublinear) | 0.743 | 0.797 | 0.808 | 0.926 | 0.789 | 0.355 |
| `sbiobert` + char, per-block L2 | **0.784** | **0.847** | **0.832** | 0.954 | 0.799 | **0.448** |

Consistency: `sbiobert` 0.766 reproduces tree 0.764 / flat 0.767, so the mention block alone is the
flag-4 feature.

Round 2 (tie-corrected `screen_features.py`: expected acc@1/MRR under random tie-breaking; breakdown
uses one random tie-break; round-1 `max`/`word` output was inflated by ties and is discarded):

| features | scorer | acc@1 | MRR | hybrid | seen, 1 | seen, >1 | unseen | gold tied at top |
|---|---|---|---|---|---|---|---|---|
| `sbiobert` | centroid | 0.766 | 0.825 | 0.804 | 0.956 | 0.833 | 0.340 | 0.000 |
| `sbiobert` | max | 0.695 | 0.796 | 0.825 | 0.983 | 0.427 | 0.423 | 0.254 |
| char | centroid | 0.743 | 0.797 | 0.808 | 0.926 | 0.789 | 0.355 | 0.000 |
| char | max | 0.678 | 0.782 | 0.815 | 0.984 | 0.421 | 0.385 | 0.279 |
| word TF-IDF | centroid | 0.726 | 0.785 | 0.798 | 0.911 | 0.790 | 0.315 | 0.093 |
| word TF-IDF | max | 0.647 | 0.741 | 0.787 | 0.980 | 0.422 | 0.272 | 0.373 |
| `sbiobert` + char | centroid | **0.784** | **0.847** | 0.832 | 0.954 | 0.799 | 0.448 | 0.000 |
| `sbiobert` + char | max | 0.704 | 0.803 | 0.832 | 0.984 | 0.431 | 0.451 | 0.262 |
| `sbiobert` + charsvd 768 | centroid | 0.775 | 0.837 | 0.826 | 0.949 | 0.794 | 0.427 | 0.000 |
| `sbiobert` + charsvd 768 | max | 0.706 | 0.804 | **0.835** | 0.984 | 0.420 | **0.461** | 0.262 |

Round 3, SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, [CLS] pooling, max 25 tokens;
user decision 2026-10-01). Same rows; `sbiobert+charsvd` repeated as the in-run reference.

| features | scorer | acc@1 | MRR | hybrid | seen, 1 | seen, >1 | unseen | gold tied at top |
|---|---|---|---|---|---|---|---|---|
| `sapbert` | centroid | 0.780 | 0.842 | 0.820 | 0.953 | 0.833 | 0.404 | 0.000 |
| `sapbert` | max | 0.729 | 0.820 | **0.839** | 0.984 | 0.514 | 0.476 | 0.215 |
| `sbiobert` + charsvd 768 | centroid | 0.775 | 0.837 | 0.826 | 0.949 | 0.794 | 0.427 | 0.000 |
| `sbiobert` + charsvd 768 | max | 0.707 | 0.805 | 0.836 | 0.984 | 0.453 | 0.463 | 0.261 |
| `sapbert` + charsvd 768 | centroid | 0.793 | 0.851 | 0.836 | 0.961 | 0.804 | 0.466 | 0.000 |
| `sapbert` + charsvd 768 | max | 0.696 | 0.799 | 0.833 | 0.984 | 0.404 | 0.453 | 0.277 |
| `sapbert` + char | centroid | **0.799** | **0.857** | 0.841 | 0.962 | 0.804 | **0.486** | 0.000 |
| `sapbert` + char | max | 0.693 | 0.797 | 0.829 | 0.984 | 0.394 | 0.439 | 0.278 |

Reproducibility: `sbiobert+charsvd / centroid` repeats round 2 exactly (0.7745). `max` rows move by
up to 0.009 in their breakdown between runs (one random tie-break; ~26% of rows tied).

Reading (round 3):
- SapBERT beats S-BioBert as the mention encoder, alone (0.780 vs 0.766; unseen 0.404 vs 0.340) and
  with char-SVD (0.793 vs 0.775; unseen 0.466 vs 0.427). Tree-feasible winner:
  `sapbert` + charsvd 768 = `.models/xmr4el_flag5_sapbert_config.json`.
- Char n-grams still add on top of SapBERT (+0.013 with SVD, +0.019 raw), so they are complementary.
- Seen strings with >1 label stay at 0.80-0.83 for every encoder (dictionary 0.928): a mention-only
  feature cannot separate labels that share a string; that needs context or a frequency prior.
- `max` still loses overall (ties); its best hybrid (0.839) is within 0.003 of the centroid winner.

Reading (rounds 1-2):
- Character n-grams are the first feature gain at this scale: on unseen strings `sbiobert` + char
  reaches 0.448 vs 0.340 (+0.108); overall +0.018, hybrid +0.028. Char alone is weaker than
  `sbiobert` overall, so the blocks are complementary, not a replacement.
- Seen ambiguous strings drop (0.833 -> 0.799), which the hybrid's dictionary covers.
- 30 sampled unseen-string `sbiobert` errors (label shown by its most frequent train mention):
  most golds are broad CUIs whose train mentions are heterogeneous ("findings" x6, "proteins" x4,
  "embase" x3, "population", "regions", "area", "infection", "cells", "words"); a centroid
  represents those poorly. About 6/30 are abbreviations (IHCC, BarR, FP, hUCB, CHD) that need local
  context, which no current feature provides (the context block is the whole document, ~0 weight).
- Raw char TF-IDF cannot go into the tree: tree training densifies X (`base.py` `X.toarray()`).
  `sbiobert` + charsvd 768 / centroid keeps half the overall gain (+0.009 vs +0.018) and most of the
  unseen gain (+0.087 of +0.108). It is the tree-feasible winner -> `emb_flag` 5.
- `max` (1-NN) loses overall: a quarter of dev rows tie at the top (one string under several
  labels, no frequency prior), collapsing seen ambiguous strings to ~0.42. On unseen strings it
  gains for `sbiobert` (+0.083) but only +0.003-0.034 once char is in; hybrids differ by <= 0.009.
  Centroid stays; `max` is not a feature fix.
- Word TF-IDF on the mention is worst on unseen strings (0.315 / 0.272): no subword overlap.

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
# leaf scorer defaults to cosine to leaf z; -scorer ranker uses the trained rankers (Session 7 rows before the cosine rows)

# routing / flat / dictionary diagnostic on the same rows as eval
.venv/bin/python test/xmr4el/diagnose_routing.py -xmodel_path test/test_data/saved_trees/<run> \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -max_rows 100000

# flat feature screen, no training (vocabulary/rows of the given tree)
.venv/bin/python test/xmr4el/screen_features.py -xmodel_path test/test_data/saved_trees/<run> \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -features <list>
```

Synthetic checks: see `STATUS.md`.
