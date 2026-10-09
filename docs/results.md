# Results

One row per change, with the exact command that produced it. This table is the ablation appendix.

The trees behind the rows recorded before the 2026-10-06 restructure were deleted in it (their pickles store the old
module paths and no longer load). Rows stay as records; reproducing one needs a retrain. Paths in rows are
as they were at the time; the Commands section uses the current layout.

## Metric choice

Each mention carries exactly one gold CUI (`Preprocessor.load_pubtator_file` in `xmr4el/data/readers.py` → one-hot `Y` rows),
so this is extreme multi-**class**, not multi-label. Reported metrics are **acc@1, MRR, recall@k**.
PECOS's precision@k / propensity-scored suite is **not** reused: with a single gold label,
precision@k is just recall@k / k and carries no extra information.

`recall@candidates` is the binary `gold in cand` hit count. `In-vocabulary mentions` is the vocabulary ceiling — mentions whose gold CUI is
absent from the training label space are deleted before scoring
(`scripts/evaluate.py`), so every row below is an in-vocabulary upper bound.

## Session 6 (2026-09-30): first dev rows with the corrected label mapping

Eval-only on the saved 2026-09-29 tree; code = `9556856` + uncommitted `classes_` label mapping.
st21pv dev, 500-label vocabulary (first 500 label groups), in-vocabulary 7530/40884 (18.4%),
347 gold CUIs kept. `-beam_size 2 -topk 0`, K=6 root clusters, leaf limit 100. Both trees have
label-level leaf matchers (OvR classes = leaf labels, 83-85 per leaf) and 66-83 rankers per leaf.
Only the flag-1 tree is listed: the flag-4 tree of that day had row-permuted training features
(transformer batch-order bug, fixed) and its rows were deleted.

| Tree | flag | recall@cand | acc@1 | MRR | R@5 | R@20 | R@100 | cand/query | random acc@1 |
|---|---|---|---|---|---|---|---|---|---|
| `15-16-12` | 1 | 0.823 | 0.113 | 0.217 | 0.339 | 0.501 | 0.705 | 166.6 | 0.0049 |

`diagnose_routing.py` (5000 sampled rows per split, seed 0):

| Tree | split | majority cluster | root matcher | cosine/all | flat nearest-label acc@1 |
|---|---|---|---|---|---|
| `15-16-12` | train | 0.558 | 0.640 | 0.649 | 0.304 |
| `15-16-12` | dev | 0.598 | 0.546 | 0.600 | 0.267 |

Dictionary baseline (dev, mention string -> most frequent train CUI): acc@1 0.718, string coverage 0.743.

Flag-1 reading (500 labels only):
- Scoring beats random ordering (acc@1 23x the random line), so the leaf scores carry signal. This does
  not show that rankers improve on matcher-only scores; that is the `-alpha 0` vs `0.5` comparison.
- Root routing top-1 is below the majority-cluster reference on dev (0.546 vs 0.598). Clusters have
  similar label counts but skewed mention frequencies. How much of recall@cand 0.82 at beam 2 is skew
  needs the fixed top-2-cluster reference (now printed by `diagnose_routing.py`).
- Hierarchy acc@1 0.113 (7530 rows) vs flat nearest-label 0.267 and mention dictionary 0.718 (5000-row
  sample). Not yet on identical rows; rerun the diagnostic with `-max_rows 100000` to use all 7530.
  All figures cover only the 18.4% of dev mentions inside the 500-label vocabulary.

## Session 7 (2026-10-01): flag-4 retrain + flag-1 alpha ablation

Code = `47ee08b` (batch-order + OOM fixes, `classes_` label mapping). Same split/vocabulary as Session 6:
st21pv dev, 500 labels, in-vocabulary 7530/40884, 347 gold CUIs. `-beam_size 2 -topk 0`, K=6.
Flag-4 tree `xmodel_2026-09-30_16-37-01` (`-ds_len 500`, base config, X 23512x2268), first valid
flag-4 tree. Flag-1 tree `15-16-12` unchanged (TF-IDF only, unaffected by the batch fix).

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

### Flag-5 SapBERT tree `xmodel_2026-10-01_15-10-04` (bundle)

Config `.models/xmr4el_flag5_sapbert_config.json`, checked on the saved tree: `emb_flag` 5,
`sapbert`, char_wb 2-4 TF-IDF -> SVD 768, `train_rankers` false; otherwise as `11-20-27` (500 labels,
depth 2, K 6). Eval: dev, beam 2, `-topk 0`, `-scorer cosine -alpha 1`, `-train_path` breakdown.
Bundle vs `11-20-27`: encoder S-BioBert -> SapBERT, flag 4 -> 5 (char block, per-block L2, document
context dropped), rankers not trained (prediction-neutral, `test_no_rankers.py`).

| tree | acc@1 | MRR | R@5 | R@20 | recall@cand | cand/query | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| `11-20-27` (flag 4, S-BioBert) | 0.764 | 0.822 | - | 0.936 | 0.949 | 166 | 0.952 | 0.832 | 0.343 | 0.805 |
| `15-10-04` (flag 5, SapBERT) | **0.790** | **0.844** | 0.915 | 0.935 | 0.942 | 167 | 0.956 | 0.804 | **0.464** | **0.836** |
| flat screen, `sapbert` + charsvd | 0.793 | 0.851 | | | | 500 | 0.961 | 0.804 | 0.466 | 0.836 |

7530/7530 in-vocabulary dev rows in both trees; dictionary 0.7165 (same rows).

Reading:
- Retain. +0.026 acc@1, +0.022 MRR, +0.121 on unseen strings, +0.031 hybrid; the gain is where the
  screen predicted (unseen strings), and the tree lands 0.003 under its flat ceiling.
- Routing is slightly worse (recall@cand 0.949 -> 0.942), so the hierarchy now costs ~0.003-0.007;
  still not the bottleneck.
- Seen ambiguous strings fall 0.832 -> 0.804 (dropping document context and S-BioBert): mention-only
  features cannot separate labels that share a string. Dictionary gets 0.928 there; the hybrid
  recovers it. Next feature lever: local context.
- Attribution is to the bundle; the screen attributes the encoder (+0.018 flat) and the char block
  (+0.013 flat on SapBERT) separately.

## Session 9 (2026-10-01): context screen (flat, `screen_features.py`)

Same vocabulary/rows as Session 8 (500 labels, 7530 dev rows), centroid scorer. Context blocks are
added to `sapbert` + charsvd 768; every block L2-normalised, then scaled by `*w`. `ctxwin` = word
TF-IDF (English stop words, sublinear) of +-10 words around the mention, mention excluded (PubTator
offsets, verified 122241/122241 train and 40884/40884 dev); `ctxdoc` = same on the whole document;
`winsbiobert` = S-BioBert of the window; `svd` = SVD 768.

| features | acc@1 | MRR | hybrid | seen, 1 | seen, >1 | unseen |
|---|---|---|---|---|---|---|
| `sapbert` + charsvd (baseline) | 0.793 | 0.851 | 0.836 | 0.961 | 0.804 | 0.466 |
| + ctxwin * 0.3 | 0.795 | 0.852 | 0.838 | 0.961 | 0.807 | 0.471 |
| + ctxwin * 0.5 | 0.803 | 0.857 | 0.839 | 0.962 | 0.833 | 0.475 |
| + ctxwin * 1 | **0.805** | **0.860** | 0.839 | 0.963 | **0.835** | 0.478 |
| + ctxdoc * 0.5 | 0.803 | 0.858 | 0.838 | 0.962 | 0.834 | 0.474 |
| + winsbiobert * 0.5 | 0.800 | 0.856 | **0.840** | 0.962 | 0.815 | 0.479 |
| + ctxwinsvd * 0.5 | 0.803 | 0.858 | 0.838 | 0.962 | 0.832 | 0.474 |

Reading:
- Context adds +0.010-0.012 acc@1, almost all on seen ambiguous strings (0.804 -> 0.835), the
  group it was meant for. Unseen strings gain +0.008-0.013.
- The hybrid barely moves (+0.002-0.004): on seen strings the dictionary's frequency prior already
  gets 0.928, so context and dictionary fix the same rows. Context matters when there is no
  dictionary (tree alone) and at larger vocabularies where strings are more ambiguous (untested).
- A 10-word window and the whole document give the same result at weight 0.5 (0.803); TF-IDF beats
  the S-BioBert window embedding on ambiguous strings (0.833 vs 0.815). SVD 768 loses nothing
  (0.803 = raw at 0.5), so the tree-feasible block is `ctxwinsvd`.
- Weight: 1 (equal block weight, the untuned default, as in flag 5) >= 0.5 > 0.3 on raw TF-IDF.
  Chosen for the tree: equal weights, so no dev-tuned parameter enters.

### Flag-6 SapBERT tree `xmodel_1000_flag6_sapbert_sgd`, 1000 labels

Config `.models/xmr4el_flag6_sapbert_config.json`, checked on the saved tree: `emb_flag` 6,
`context_window` 10, 1000 labels, K 6. 11661/40884 in-vocabulary dev rows. New vocabulary: compare
only within 1000 labels.

| features (flat, centroid) | acc@1 | MRR | hybrid | seen, 1 | seen, >1 | unseen |
|---|---|---|---|---|---|---|
| `sapbert` + charsvd (flag 5) | 0.781 | 0.838 | 0.822 | 0.951 | 0.796 | 0.435 |
| + ctxwinsvd (flag 6, tree features) | **0.800** | **0.852** | **0.827** | 0.955 | **0.840** | 0.453 |

n: seen 1 = 5532, seen >1 = 3281, unseen = 2848. Dictionary 0.716 (string coverage 0.756).
Routing (dev): root matcher cluster acc 0.884, top-2 0.946.

| tree (beam 2, `-alpha 1`) | acc@1 | MRR | R@5 | R@20 | recall@cand | cand/query | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| `xmodel_1000_flag6_sapbert_sgd` (flag 6) | 0.797 | 0.844 | 0.905 | 0.932 | 0.943 | 200 | 0.951 | 0.844 | 0.442 | 0.824 |
| flat screen, flag-6 features | 0.800 | 0.852 | | | | 1000 | 0.955 | 0.840 | 0.453 | 0.827 |

The first eval was run at `-alpha 0.5` (acc@1 0.759, MRR 0.819). That mixes the leaf matcher's cluster probability into
the score; it cost 0.038.

Reading (tree):
- The tree lands 0.003 under its flat ceiling, the same gap as flag 5 at 500 labels. At 1000 labels the hierarchy
  still costs almost nothing; recall@cand 0.943 is the cap.
- The tree beats the dictionary on all rows (0.797 vs 0.716), but not on seen strings (ambiguous 0.844 vs 0.893).
  The hybrid adds +0.027.

Reading (flat):
- Context gains more at 1000 labels (+0.019) than at 500 (+0.012), nearly all on seen ambiguous strings
  (+0.044), as expected for a larger vocabulary.

## Session 11 (2026-10-02): abbreviation expansion (flat, `screen_features.py`)

Vocabulary/rows of tree `xmodel_1000_flag6_sapbert_sgd` (1000 labels, 11661 in-vocabulary dev rows), features `sapbert` + charsvd + ctxwinsvd
(flag 6), centroid scorer. A mention that is a short form defined as "long form (SF)" in its own document (Schwartz-Hearst,
first definition wins) is expanded in train and dev; context blocks unchanged. Expanded rows: train 1593/36377, dev 487/11661
(349 of the 2848 unseen-string rows). Detector precision: 30/30 sampled dev pairs correct (full files: 11595/122241 train,
3895/40884 dev rows). Groups keyed by the raw mention string.

| abbrev | acc@1 | MRR | hybrid (raw key) | hybrid (expanded key) | seen, 1 | seen, >1 | unseen | abbrev rows | abbrev, unseen |
|---|---|---|---|---|---|---|---|---|---|
| none (baseline) | 0.800 | 0.852 | 0.827 | | 0.955 | 0.840 | 0.453 | 0.386 | 0.181 |
| replace | 0.809 | 0.860 | 0.833 | 0.833 | 0.958 | 0.842 | 0.480 | 0.550 | 0.390 |
| append ("SF long form") | **0.810** | **0.861** | 0.834 | **0.835** | **0.960** | 0.842 | **0.482** | **0.556** | **0.398** |

n: seen 1 = 5532, seen >1 = 3281, unseen = 2848, abbrev rows = 487 (unseen 349).

Reading:
- Append wins: +0.010 acc@1, +0.009 MRR, +0.008 hybrid. Abbreviation rows 0.386 -> 0.556; unseen abbreviations 0.181 -> 0.398.
- Append >= replace everywhere (small): keeping the short form keeps its char n-grams.
- Dictionary key barely matters (0.834 raw vs 0.835 expanded); the loader expands both sides, so eval uses the expanded key.
- Remaining abbreviation errors are mostly generic gold labels ("findings", "regions", "embase") and short forms
  not defined in the document.

Retained: `abbrev_expansion: "append"` (now `.models/xmr4el_base_config.json`); not yet in a tree.

## Session 11 (2026-10-02): XMR4EL vs PECOS XR-Linear, 500 labels

Tree `xmodel_500_flag6_sapbert_abbrev_sgd`: base config (flag 6, SapBERT + charsvd + ctxwinsvd, `abbrev_expansion` append,
window 10), `-ds_len 500` -> 500 labels, 23512 train rows; 7530/40884 in-vocabulary dev rows (347 distinct gold labels).
New vocabulary and new features: compare only within this table.

PECOS 1.2.8 XR-Linear (`pecos.dockerfile`, `test/xmr4el/pecos_run.py`, defaults otherwise): PIFA label embeddings,
hierarchical k-means nr_splits 8 / max leaf 100 -> 8 clusters -> labels, top-100 output. `ours` = the tree's own X/Y
and dev features (same rows, same label order; `pecos_compare.py export`); `tfidf` = PECOS's default TF-IDF of the same
"mention [SEP] window" texts. Metrics from the same code as eval (`pecos_compare.py score`).

| system | acc@1 | MRR | R@5 | R@20 | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|
| XMR4EL tree (beam 2, cosine leaf, `-alpha 1`) | 0.817 | 0.867 | 0.928 | 0.947 | 0.964 | 0.839 | 0.521 | 0.851 |
| PECOS, our features, beam 2 | **0.861** | 0.891 | 0.928 | 0.939 | **0.977** | **0.933** | **0.571** | **0.865** |
| PECOS, our features, beam 10 | **0.861** | **0.896** | **0.939** | **0.955** | **0.977** | **0.933** | **0.571** | **0.865** |
| PECOS, PECOS TF-IDF, beam 2 | 0.507 | 0.628 | 0.776 | 0.825 | 0.583 | 0.623 | 0.251 | 0.782 |
| PECOS, PECOS TF-IDF, beam 10 | 0.510 | 0.639 | 0.799 | 0.862 | 0.587 | 0.625 | 0.252 | 0.783 |
| dictionary | 0.716 | | | | 0.984 | 0.932 | 0 | |

n: seen 1 = 3638, seen >1 = 1946, unseen = 1946. Tree: 40 candidates/query, recall@cand 0.951.
PECOS time (Docker, amd64 emulation): train 7.3 s / predict 0.7 s (ours, beam 2); XMR4EL train time not recorded.

Reading:
- Same features, same rows: PECOS +0.044 acc@1 over the tree. The features are not the gap; the label scorer is.
  The tree scores leaf labels by cosine to the PIFA centroid (rankers off); PECOS trains one linear classifier per label
  against the other labels of its cluster.
- The gain is largest on seen ambiguous strings (0.839 -> 0.933, = dictionary 0.932): a per-label classifier can use the
  context block, a centroid cannot separate labels that share a mention. Unseen +0.050, seen single +0.013.
- PECOS alone (0.861) beats our hybrid (0.851); its hybrid adds only 0.004.
- Beam 2 vs 10 does not change acc@1 (0.861): routing is not PECOS's limit either.
- Our own leaf matcher as the scorer (`-alpha 0`, beam 2): acc@1 0.394, MRR 0.607, but R@5 0.917 (seen 1 0.450,
  seen >1 0.320, unseen 0.362). Gold is near the top, rarely first. From code: each leaf's matcher is a multilabel
  OneVsRest (fitted on a 0/1 indicator matrix), so `predict_proba` is one sigmoid per label, trained only against the
  other labels of that leaf; the tree merges the beam's leaves by max score with no root/cluster factor
  (`_predict_one_leaf`). A wrong leaf's labels never saw the query's true neighbours as negatives, so beam-2 scores from
  two leaves are not on one scale. PECOS multiplies the cluster score into each label score along the path.
  (Corrected 2026-10-06: an earlier version of this note said OneVsRest renormalises rows within a leaf; it does not in
  multilabel mode, checked on the saved matchers: `multilabel_` True.)
- Confirmed with `-alpha 0 -beam_size 1` (one leaf, no merge): acc@1 0.794, MRR 0.839, recall@cand 0.898 (seen 1 0.934,
  seen >1 0.812, unseen 0.515). Within the routed leaf the matcher is right 0.794 / 0.898 = 0.88 of the time it can be,
  vs cosine 0.817 / 0.951 = 0.86 and PECOS ~0.90. The leaf classifiers are fine; the beam merge is the bug.
- With the fix (`-alpha 0 -beam_size 2 -path_score`): acc@1 0.801, MRR 0.858, R@5 0.926, recall@cand 0.945, hybrid
  0.856 (seen 1 0.935, seen >1 0.812, unseen 0.540). Merge fixed (0.394 -> 0.801), but the leaf matcher stays under
  cosine (0.817) and PECOS (0.861) on seen strings (PECOS 0.977 / 0.933); it beats cosine on unseen (0.540 vs 0.521).
  Same routing, so the remaining gap is the leaf classifier itself (SGD log_loss, L1, `class_weight` balanced vs PECOS
  L2 squared hinge, unweighted).

Leaf scorer screen (`screen_leaf_scorer.py`, no training): the tree's 6 leaves (83-85 labels), exported features, oracle
routing, so acc@1 = within-leaf accuracy (end-to-end ~ this x routing recall).

| leaf scorer | within-leaf acc@1 | seen, 1 | seen, >1 | unseen | hybrid | fit time |
|---|---|---|---|---|---|---|
| cosine to PIFA centroid (tree default) | 0.866 | 0.977 | 0.857 | 0.666 | 0.890 | 2 s |
| logreg liblinear L2 C=1 | 0.891 | 0.957 | 0.945 | 0.712 | 0.902 | 53 s |
| logreg liblinear L2 C=1, class_weight balanced | 0.920 | 0.987 | 0.947 | 0.769 | 0.916 | 104 s |
| LinearSVC squared hinge C=1 (PECOS loss) | **0.921** | **0.989** | 0.945 | **0.771** | **0.917** | 108 s |

- liblinear L2 beats cosine by +0.055 within the leaf; balanced weights help (+0.029), so `class_weight` was not the
  problem: the SGD L1 solution was. svm = logreg_bal within 0.001.
- Retained: logreg liblinear balanced (gives probabilities, needed for the path product). Base config matcher (all
  layers) is now `sklearnlogisticregression` {solver liblinear, C 1, class_weight balanced}; tree `xmodel_500_flag6_sapbert_abbrev_sgd` was trained
  with the previous SGD matcher (its pickle keeps that config).
- PECOS with its default word TF-IDF on "mention [SEP] window" is weak (0.51): the context words dominate the mention.
  The useful comparison is the same-features row.

## Session 11 (2026-10-02): liblinear leaf matcher + path score, 500 and 1000 labels

Trees trained with the base config (flag 6 SapBERT + charsvd + ctxwinsvd, abbrev append, window 10, matcher
`sklearnlogisticregression` liblinear L2 C 1 balanced at every layer). Eval beam 2; `-alpha 0 -path_score` = leaf matcher
probability x routing path probability; `-alpha 1` = cosine to leaf z on the same tree.

| tree | labels | dev rows | scorer | acc@1 | MRR | R@5 | recall@cand | seen, 1 | seen, >1 | unseen | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `xmodel_500_flag6_sapbert_abbrev_logreg` | 500 | 7530 | matcher x path | **0.862** | **0.896** | **0.941** | 0.956 | **0.976** | 0.931 | **0.578** | **0.866** |
| same | 500 | 7530 | cosine | 0.816 | 0.867 | 0.929 | 0.956 | 0.962 | 0.840 | 0.520 | 0.851 |
| PECOS XR-Linear, same features (beam 2) | 500 | 7530 | | 0.861 | 0.891 | 0.928 | | 0.977 | **0.933** | 0.571 | 0.865 |
| `xmodel_500_flag6_sapbert_abbrev_sgd` (SGD matcher) | 500 | 7530 | cosine | 0.817 | 0.867 | 0.928 | 0.951 | 0.964 | 0.839 | 0.521 | 0.851 |
| `xmodel_1000_flag6_sapbert_abbrev_logreg` | 1000 | 11661 | matcher x path | **0.843** | **0.884** | **0.933** | 0.952 | **0.973** | **0.905** | **0.525** | **0.845** |
| same | 1000 | 11661 | cosine | 0.808 | 0.856 | 0.920 | 0.951 | 0.960 | 0.843 | 0.480 | 0.834 |
| `xmodel_1000_flag6_sapbert_abbrev_joint` (joint L-BFGS matcher) | 1000 | 11661 | matcher x path | **0.844** | 0.883 | 0.930 | 0.948 | | | | |
| same | 1000 | 11661 | cosine (path score on) | 0.803 | 0.845 | 0.896 | 0.947 | | | | |
| PECOS XR-Linear, same features (beam 2; 2 -> 16 clusters -> labels) | 1000 | 11661 | | 0.838 | 0.871 | 0.911 | | 0.967 | 0.894 | 0.529 | 0.846 |
| PECOS, same, beam 10 | 1000 | 11661 | | 0.839 | 0.878 | 0.926 | | 0.966 | 0.894 | 0.535 | 0.847 |
| `xmodel_1000_flag6_sapbert_sgd` (no abbrev, SGD) | 1000 | 11661 | cosine | 0.797 | 0.844 | 0.905 | 0.943 | 0.951 | 0.844 | 0.442 | 0.824 |

n (500): seen 1 = 3638, seen >1 = 1946, unseen = 1946. n (1000, abbrev keys): 5518 / 3259 / 2884 (raw keys: 5532 / 3281 /
2848). Dictionary: 0.716 (500), 0.715 (1000).

Reading:
- 500 labels: the tree now matches PECOS on the same features (0.862 vs 0.861; MRR 0.896 vs 0.891). The two changes were
  the leaf classifier (liblinear L2 instead of SGD L1) and combining leaves by path probability.
- 1000 labels: 0.843 vs 0.797 for the previous 1000-label tree (+0.046). Of that, abbreviation expansion is ~+0.011 (cosine
  on the new tree 0.808 vs 0.797, matching the flat screen +0.010); the matcher + path score is +0.035.
- Seen ambiguous strings: the tree now beats the dictionary at 1000 (0.905 vs 0.895) and ties it at 500 (0.931 vs 0.932),
  so the hybrid adds only +0.002-0.004. The remaining weak group is unseen strings (0.53-0.58).
- Cosine on a logreg tree equals cosine on the SGD tree (0.816 vs 0.817): the matcher change does not hurt routing.
- 1000 labels: the tree beats PECOS on the same features (0.843 vs 0.838 at beam 2; MRR 0.884 vs 0.871). PECOS's
  hierarchy at 1000 labels is 2 -> 16 clusters (nr_splits 8, max leaf 100), not the tree's 6 leaves.
- Training time (user): our tree ~100 s (500 labels) vs PECOS 7.1 s (features precomputed for PECOS). Profile of a
  500-label training (cProfile, self time; 195 s under the profiler): SapBERT forward ~91 s (47%), waiting on the
  OneVsRest liblinear worker processes ~67 s (34%), rest (SVD, I/O, gc, imports) ~30 s. Clustering 1.4 s.
- Leaf classifier speed screen (`screen_leaf_scorer.py`, 1000 labels, oracle routing, 6 leaves of 166-170 labels):
  logreg_bal 0.8974 in 238 s; liblinear tol 1e-2: 0.8974 in 197 s; char/context SVD cut to 256 dims: 0.8919 in 143 s
  (unseen -0.013, rejected); **joint** (same objective, all labels of a leaf in one L-BFGS problem on BLAS,
  `JointOvRLogistic`): 0.8974 in **19 s**, identical in every group. On one 167-label leaf it agreed with liblinear's top-1
  on every dev row. Retained: base config matcher `jointlogisticregression` {C 1, class_weight balanced, tol 1e-4} (1e-3 since 2026-10-08, `11-04-56`).
  Cost per label of the old path: liblinear 0.45 s single-core; 8 OneVsRest worker processes only reach 0.12-0.16 s.
- Speed changes (prediction-neutral up to float noise): `Transformer.transform` embeds each distinct text once (500
  labels: 23512 rows -> 5937 strings; measured 13.2 s -> 6.2 s on 3000 rows, max abs diff 6e-6) and runs on the Apple
  GPU (MPS) when available (1.4x on 25-token mentions, max abs diff 9e-6, cosine 1.000000).
- Joint matcher retrain (session 12, 2026-10-06, Linux): `xmodel_1000_flag6_sapbert_abbrev_joint` 0.8436 / MRR 0.8827
  vs liblinear tree 0.843 / 0.884: same objective, same result within noise. Train 375.7 s (6m34 real, 158 min CPU) vs
  PECOS 12.4 s; no stage breakdown yet. Group columns not reported (compact eval report).

## 2026-10-06: BC5CDR dev, XMR4EL vs PECOS XR-Linear

Tree `xmodel_2026-10-06_16-44-38`, `configs/xmr4el_bc5cdr_config.json` (base, `n_clusters` 16: root -> 16 leaves),
trained on `CDR_TrainingSet` (reader drops `-1`, splits composites; 1327 labels). Dev `CDR_DevelopmentSet`:
7524/9613 rows have a gold label seen in training (78.3%, 694 labels); all metrics are over those rows.
PECOS: `pecos_run.py -nr_splits 16 -max_leaf_size 100` (16 clusters -> labels), XR-Linear defaults otherwise;
`ours` = the tree's exported features. Eval `-topk 0 -alpha 0 -path_score`.

| system | beam | acc@1 | MRR | R@5 | R@10 | R@20 | seen, 1 (n=5983) | seen, >1 (n=61) | unseen (n=1480) | hybrid |
|---|---|---|---|---|---|---|---|---|---|---|
| XMR4EL | 2 | 0.921 | 0.943 | 0.967 | 0.972 | 0.975 | 0.977 | 0.656 | 0.705 | 0.934 |
| XMR4EL | 10 | **0.922** | **0.944** | **0.969** | **0.975** | **0.979** | **0.977** | 0.656 | **0.707** | **0.934** |
| PECOS, our features | 2 | 0.847 | 0.878 | 0.912 | 0.927 | 0.941 | 0.912 | **0.672** | 0.593 | 0.912 |
| PECOS, our features | 10 | 0.847 | 0.878 | 0.913 | 0.928 | 0.942 | 0.912 | **0.672** | 0.592 | 0.911 |
| PECOS, our features, `threshold` 0 | 2 | **0.922** | 0.945 | 0.972 | 0.976 | 0.979 | 0.975 | **0.672** | **0.720** | **0.937** |
| PECOS, our features, `threshold` 0 | 10 | **0.922** | **0.946** | **0.975** | **0.979** | **0.983** | 0.975 | **0.672** | **0.720** | **0.937** |
| PECOS, PECOS TF-IDF | 2 | 0.487 | 0.558 | 0.642 | 0.656 | 0.664 | 0.546 | 0.377 | 0.250 | 0.844 |
| PECOS, PECOS TF-IDF | 10 | 0.503 | 0.597 | 0.713 | 0.742 | 0.761 | 0.564 | 0.393 | 0.259 | 0.846 |

XMR4EL recall@cand 0.983 (beam 2, 167 cand/query) / 0.997 (beam 10, 830). Dictionary acc@1 0.795 (seen 1-label 0.993).
Coverage x acc@1 (all 9613 dev rows): XMR4EL 0.721, PECOS-ours 0.663 (`threshold` 0: 0.722). Eval 20 s (beam 2) / 72 s (beam 10);
train times not recorded.

- PECOS's default weight pruning (`threshold` 0.1) was the whole +0.074 gap: with `threshold` 0
  (`XLinearModel.train(..., threshold=0.0)`, same chain, train 3.4 s) PECOS ties XMR4EL (0.922 vs 0.921/0.922).
  PECOS is +0.015 on unseen strings, XMR4EL +0.002 on seen 1-label. Beam does not matter for either system.
- Every earlier PECOS row (MedMentions, Session 11) used the default `threshold` 0.1, so those PECOS numbers
  are likely low too; rerun with `threshold` 0 before claiming parity or a win there.
- 21.7% of dev rows have a gold label absent from training; neither system can get them.

## 2026-10-07: BC5CDR-disease dev, CTD MEDIC dictionary, depth 2 vs 3, vs PECOS

Disease-only splits `datasets/BC5CDR/disease/{train,dev}.pubtator`; `train_plus_ctd.pubtator` = train + CTD MEDIC
names (`scripts/dict_to_pubtator.py`, 11,744 labels). Dev has 4306 rows. Eval `-beam_size 2 -topk 0 -alpha 0 -path_score`.
PECOS: `pecos_run.py -features ours -nr_splits 16 -max_leaf_size 100 -threshold 0` on the export of tree `12-38-23`
(`outputs/pecos/bc5cdr_dict_dev`). `cov x acc` = acc@1 x in-vocabulary rows / 4306.

| system | train data | depth | beam | in vocab | acc@1 | MRR | R@5 | R@cand (cand) | seen, 1 | seen, >1 | unseen | hybrid | cov x acc | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XMR4EL `12-38-23` | train + CTD | 3 | 2 | 4305 | 0.747 | 0.789 | 0.841 | 0.891 (184) | 0.842 | 0.679 | 0.448 | 0.855 | 0.747 | 567 s |
| XMR4EL `13-47-38` | train + CTD | 2 | 2 | 4305 | 0.838 | 0.874 | 0.919 | 0.943 (200) | 0.921 | 0.759 | 0.581 | 0.884 | 0.837 | 567 s |
| XMR4EL `13-47-38` | train + CTD | 2 | 10 | 4305 | 0.845 | 0.887 | 0.938 | 0.992 (1000) | 0.928 | **0.778** | 0.587 | 0.886 | 0.845 | 567 s |
| XMR4EL `15-15-09` (torch L-BFGS) | train + CTD | 2 | 10 | 4305 | 0.845 | 0.887 | 0.938 | 0.992 (1000) | 0.928 | 0.778 | 0.587 | 0.886 | 0.845 | 454 s |
| XMR4EL `17-28-48` (own balanced k-means) | train + CTD | 2 | 10 | 4305 | **0.853** | **0.895** | **0.950** | 0.994 (1000) | **0.930** | **0.778** | **0.615** | **0.892** | **0.853** | 179 s* |
| XMR4EL `11-04-56` (scaled L-BFGS, tol 1e-3) | train + CTD | 2 | 10 | 4305 | 0.851 | 0.895 | 0.951 | 0.994 (1000) | 0.929 | 0.774 | 0.613 | 0.891 | 0.851 | 132 s* |
| XMR4EL `11-09-56` (128 leaves, tol 1e-3) | train + CTD | 2 | 10 | 4305 | **0.856** | **0.898** | 0.947 | 0.992 (917) | **0.941** | 0.764 | 0.595 | 0.887 | **0.855** | 106 s* |
| XMR4EL `13-49-39` | train | 3 | 2 | 3631 | 0.801 | 0.824 | 0.847 | 0.881 (10) | 0.901 | 0.672 | 0.528 | 0.862 | 0.675 | 41 s |
| XMR4EL `13-55-09` | train | 2 | 2 | 3631 | 0.879 | 0.910 | 0.947 | 0.965 (82) | 0.954 | 0.672 | 0.681 | 0.902 | 0.741 | 15 s |
| PECOS, our features | train + CTD | 8 -> 128 -> labels | 2 | 4305 | 0.838 | 0.864 | 0.892 | (100) | 0.923 | 0.755 | 0.576 | 0.883 | 0.837 | 34.6 s |
| PECOS, our features | train + CTD | 8 -> 128 -> labels | 10 | 4305 | 0.847 | 0.884 | 0.928 | (100) | **0.930** | 0.755 | 0.594 | 0.886 | 0.846 | 34.6 s |

Dictionary acc@1 (most frequent train label for the exact string): 0.756 with CTD (seen 1-label 0.978), 0.726 without.
Rows with the same train data share the same row set; the no-CTD rows cover 3631 rows and are compared only by `cov x acc`.

- With CTD, depth 3 loses 0.091 acc@1 to depth 2 (0.747 vs 0.838) and 0.052 of candidate recall: depth 3 at beam 2
  prunes the gold. Without CTD, depth 3 loses 0.078 (0.801 vs 0.879).
- XMR4EL depth 2 ties PECOS: beam 2 0.838 vs 0.838, beam 10 0.845 vs 0.847 (hybrid 0.886 both). XMR4EL is ahead on
  MRR/R@5 at beam 10 (0.887 / 0.938 vs 0.884 / 0.928).
- On seen 1-label strings both trees are below the exact-string dictionary (0.928 / 0.930 vs 0.978); the hybrid
  (dictionary if the string was seen, else tree) is +0.041 over the XMR4EL tree at beam 10.
- CTD at depth 2: `cov x acc` 0.741 -> 0.837 (hybrid 0.760 -> 0.884), all from the 674 rows whose gold is not in
  the BC5CDR train labels. Whether CTD costs accuracy on the 3631 rows both trees cover is not measured
  (no-CTD tree 0.879 there; the CTD tree's acc@1 on 4305 rows is not comparable).
- Literature BC5CDR-disease **test** acc@1: BioSyn 93.2, SapBERT 93.5; these are dev rows.
- `15-15-09` = `13-47-38` config with the joint matcher solved by torch L-BFGS instead of scipy L-BFGS-B: same
  dev metrics, hierarchy 330.5 -> 197.9 s (16 leaves 251.4 -> 118.9 s, root 78.7 s unchanged; logs
  `outputs/logs/speed_baseline2_bc5cdr_ctd.log`, `speed_torchlbfgs_bc5cdr_ctd.log`). SapBERT 197 / 222 s run to run (mps).
- `17-28-48` = `15-15-09` plus the SapBERT embedding cache and the own balanced k-means (`balancedkmeans`, numpy:
  recursive balanced 2-means, then joint balanced refinement; replaces kmeans-pytorch). *179 s is a cache-hit run
  (encoding 25.2 s); hierarchy 148.4 s (root clustering 3.0 s, root node 34.9 s). Log
  `outputs/logs/bkmeans_bc5cdr_ctd.log`. Same code with seed 1: acc@1 0.843, MRR 0.889, hybrid 0.887, so the
  seed 0 gain over PECOS (+0.006) is within the seed spread; kmeans-pytorch had 0.845-0.847 at seed 0.
  The test section below still uses `13-47-38`; not rerun with this clustering.
- `11-04-56` = `17-28-48` with the joint matcher's objective divided by n, so the max|grad| <= tol stop is reached
  (before, every fit ran to the float32 loss stall whatever tol was), and tol 1e-3: acc@1 -0.0017, hierarchy
  148.4 -> 100.8 s (L-BFGS iterations root 258 -> 118, leaves 83-129 -> 62-93). *Cache-hit run. Log
  `outputs/logs/tol1e3s_bc5cdr_ctd.log`.
- `11-09-56` = `11-04-56` with root `n_clusters` 128 (leaves of 91-92 labels, PECOS's leaf size): acc@1 +0.005,
  seen 1-label +0.012, but unseen -0.018, hybrid -0.004, R@20-R@100 lower (R@100 0.977 vs 0.987). Hierarchy
  100.8 -> 75.1 s: root clustering 17.6 s (3.0 s at 16), root matcher 22.5 s, 128 leaves 34.9 s. *Cache-hit run.
  Log `outputs/logs/k128_bc5cdr_ctd.log`. Beam 20: R@cand 0.9940 (1835 cand), all ranked metrics identical, so
  the lower R@20-R@100 is cross-leaf scoring, not routing. Rerun `11-31-05` with the vectorized balanced assignment:
  identical metrics, root clustering 17.6 -> 5.0 s, hierarchy 65.8 s, run 96.3 s* (`k128c_bc5cdr_ctd.log`).
  Root matcher on MPS (`11-45-47`, `mps_bc5cdr_ctd.log`): acc@1 0.8551, MRR 0.8972, hybrid 0.8873 (float32 order
  noise); root node 32.0 -> 20.1 s, hierarchy 54.6 s, run 85.5 s*.
  No per-node `gc.collect()` (`12-21-30`, `nogc_bc5cdr_ctd.log`): dev metrics identical to `11-45-47`; leaf layer
  34.9 -> 11.3 s, hierarchy 54.6 -> 31.1 s, run 61.6 s*. Eval (`-path_score`, beam 10) 42 -> 9 s after batching
  the leaf cosine (`g1a_eval.log`) and the routing (`g1b_eval.log`), output identical.

## 2026-10-08: BC5CDR-disease dev, knn fusion (`-knn_beta`)

Tree `12-21-30`, dev, `-beam_size 10 -topk 0 -alpha 0 -path_score`. Each candidate's tree score x exp(beta * knn),
knn = max cosine of the query's SapBERT block to the label's training rows (train + CTD), no retraining. Screen:
`scripts/diagnose_unseen.py` (`outputs/logs/unseen_diag2.log`); beta 10 run: `outputs/logs/knn10_dev_eval.log`.

| beta | acc@1 | seen, 1 (n=3140) | seen, >1 (n=212) | unseen (n=953) | hybrid |
|---|---|---|---|---|---|
| 0 (tree) | 0.8551 | 0.9404 | 0.7594 | 0.5950 | 0.8873 |
| 5 | 0.9015 | 0.9761 | 0.8113 | 0.6758 | 0.9052 |
| **10** | **0.9041** | 0.9783 | **0.8113** | **0.6800** | **0.9062** |
| 20 | 0.9003 | **0.9787** | 0.8113 | 0.6621 | 0.9022 |
| 40 | 0.8976 | 0.9783 | 0.8113 | 0.6506 | 0.8997 |
| knn only (in candidates) | 0.8846 | 0.9783 | 0.5896 | 0.6411 | 0.8976 |

- Beta 10: MRR 0.8972 -> 0.9335, R@5 0.9473 -> 0.9668, R@20 0.9672 -> 0.9833 (R@cand unchanged 0.9914).
- Why: on unseen strings the tree had gold at rank 2-5 for 0.207 of rows (routing loss 0.038); SapBERT 1-NN over
  train rows alone is 0.638 vs tree 0.595, and they err on different rows (oracle 0.738). The tree prefers more
  specific CTD labels ("obese" -> obesity, morbid); the nearest synonym corrects many of them.
- Unseen-row flat 1-NN by feature block: all 0.600, SapBERT 0.638, char 0.517, context 0.025.

## 2026-10-07: BC5CDR-disease test, selected configuration

Trees `13-47-38` (depth 2, 16 leaves) and `12-21-30` (depth 2, 128 leaves, current config; also with `-knn_beta 10`), both train + CTD, `datasets/BC5CDR/disease/test.pubtator` (4410 rows, 4399 in vocab, 640 gold
labels). XMR4EL eval `-beam_size 10 -topk 0 -alpha 0 -path_score`; PECOS as in the dev section on export
`outputs/pecos/bc5cdr_dict_test` (train 34.4 s). Dictionary acc@1 0.770 (seen 1-label 0.985).

| system | beam | acc@1 | MRR | R@5 | R@10 | R@20 | seen, 1 (n=3287) | seen, >1 (n=174) | unseen (n=938) | hybrid | cov x acc | cov x hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XMR4EL, 16 leaves (`13-47-38`) | 10 | 0.860 | 0.898 | 0.945 | 0.963 | 0.974 | 0.930 | 0.730 | 0.641 | 0.906 | 0.858 | 0.904 |
| XMR4EL, 128 leaves (`12-21-30`) | 10 | 0.873 | 0.909 | 0.949 | 0.961 | 0.965 | 0.947 | 0.672 | 0.653 | 0.909 | 0.871 | 0.906 |
| XMR4EL, 128 leaves + knn beta 10 (`12-21-30`) | 10 | 0.916 | 0.940 | **0.969** | **0.977** | **0.980** | **0.985** | 0.730 | 0.712 | 0.921 | 0.914 | 0.919 |
| XMR4EL, same config, new tree (`10-18-51`) | 10 | 0.918 | 0.941 | 0.968 | 0.974 | 0.978 | **0.985** | 0.718 | 0.719 | **0.923** | 0.915 | **0.920** |
| `10-18-51` + reranker, not seen 1, w 0.25 | 10 | **0.923** | **0.944** | - | - | - | **0.985** | **0.816** | **0.725** | - | **0.921** | - |
| PECOS, our features | 2 | 0.852 | 0.876 | 0.905 | 0.908 | 0.909 | 0.925 | 0.747 | 0.618 | 0.901 | 0.850 | 0.899 |
| PECOS, our features | 10 | 0.862 | 0.897 | 0.939 | 0.948 | 0.954 | 0.933 | 0.747 | 0.634 | 0.904 | 0.860 | 0.902 |

XMR4EL recall@cand 0.991 at 16 leaves (1000 cand/query), 0.989 at 128 (917 cand/query). Eval 54 s at 16 leaves
(before eval batching), 15 s at 128 (`outputs/logs/test_12-21-30_eval.log`); PECOS predict 2.3 s.

- At 16 leaves XMR4EL ties PECOS on test (0.860 vs 0.862 at beam 10; hybrid 0.906 vs 0.904), as on dev.
- At 128 leaves it leads PECOS by 1.1 pt acc@1 (0.873 vs 0.862), hybrid 0.909 vs 0.904. Versus 16 leaves: seen
  1-label +0.017 and unseen +0.012, but seen >1-label -0.058 (n=174) and R@20 -0.009 (cross-leaf scoring, as on dev).
- `-knn_beta 10` (beta chosen on dev, test run once, `outputs/logs/knn10_test_eval.log`): acc@1 0.873 -> 0.916, unseen
  0.653 -> 0.712, hybrid 0.909 -> 0.921; the tree alone (0.916) is now above the old hybrid. Dev section 2026-10-08 below.
- Reranker (2026-10-09, `outputs/logs/r3_test_eval.log`; scope and w chosen on dev from 3 x 7 configs, test run
  once): cross-encoder (SapBERT init) over the tree's top 10, final = log(tree score) + 0.25 x logit, applied only to
  rows whose mention string is not seen in train (+ CTD) with exactly one label; recall columns are not reported for
  the reranked row. acc@1 0.918 -> 0.923 (+23 rows): seen >1 +17, unseen +6, seen 1 untouched. On dev the same
  choice gave +38 (unseen +26); on test the gain is close to the tree's own hybrid (0.9229 vs 0.9227). Ungated
  (all rows, w 0.25) test 0.920 with seen 1 -13 rows. `10-18-51` vs `12-21-30`: same code and config, new server
  (fresh CUDA encoding).
- Hybrid 0.906 vs literature 93.2 (BioSyn) / 93.5 (SapBERT). The protocols are not matched: this trains on train only
  (+ CTD), drops `-1` ids and composites without column 7, and excludes 11 out-of-vocabulary rows.

## Commands

```bash
# train (500-label diagnosis scale)
.venv/bin/python scripts/train.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config configs/xmr4el_base_config.json -ds_len 500

# eval (all in-vocabulary dev rows; base-config trees: -alpha 0 -path_score = leaf matcher x path probability)
.venv/bin/python scripts/evaluate.py -xmodel_path outputs/saved_trees/<run> \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -beam_size 2 -topk 0 -alpha 0 -path_score \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
# -alpha 1 without -path_score = pure cosine to leaf z (Session 8-9 tree rows and the cosine rows after);
# alpha in (0, 1) mixes in the leaf matcher's cluster probability. -scorer ranker uses the trained rankers
# (Session 7 rows before the cosine rows)

# routing / flat / dictionary diagnostic on the same rows as eval
.venv/bin/python scripts/diagnose_routing.py -xmodel_path outputs/saved_trees/<run> \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -max_rows 100000

# flat feature screen, no training (vocabulary/rows of the given tree)
.venv/bin/python scripts/experiments/screen_features.py -xmodel_path outputs/saved_trees/<run> \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt -features <list>
```

Synthetic checks: see `docs/pipeline.md`.
