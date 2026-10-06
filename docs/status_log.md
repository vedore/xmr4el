# Status log

Completed steps and history moved out of `STATUS.md`. Not a work order; results live in
`docs/results.md`.


## Session 12 (2026-10-06)

**Session 12 (2026-10-06): nothing in flight.** Joint retrain `xmodel_1000_flag6_sapbert_abbrev_joint`: acc@1 0.8436 /
MRR 0.8827 (`-alpha 0 -path_score`, beam 2), cosine 0.803 / 0.845; same as the liblinear tree (0.843 / 0.884). Train
375.7 s (6m34 real, 158 min CPU, Linux) vs PECOS 12.4 s; stage breakdown not yet seen (suspect SapBERT on CPU or BLAS
thread oversubscription). Recorded in `docs/results.md`. External-review bugs fixed (uncommitted; none changes recorded
numbers): label matrix in row order (`generate_label_matrix`); leaves too small to cluster keep identity C instead of
dropping the last layer (`MLModel.train`, `Clustering.load` without model); `XModel.predict` beam default 5; predict
CSR width = root's label count. Regression in `test/xmr4el/test_no_rankers.py`.
Second review round (also uncommitted, prediction-neutral for depth-2 trees): transformer batches kept in memory (no
`batch_dir`, nothing deleted; `test_transformer_batch_dir.py` removed); OOM keeps the reduced batch size; leaves are no
longer clustered at all; `cut_half_cluster` now really halves (it was a no-op; matters only at depth >= 3); an
internal node that cannot split raises `ValueError` instead of truncating the tree; `XModel.predict`'s global cosine
rerank branch deleted (unused; topk_mode "global" goes to the hierarchy, whose final_path matches its scores).

Repository-structure review done; restructure plan in `STATUS.md` § "Restructure plan".

## Session 11 (2026-10-02)

**Next after the retrain:** user still does not want the full-label run; candidates: unseen strings (abbreviations not
defined in the doc, generic labels), candidate reranker on top of matcher x path (must beat 0.843 at 1000), encoder swap
(flat screen first). Session-11 history below.

**Plan (user decision 2026-10-02, session 10 end):** (1) abbreviation expansion, then (2) a candidate reranker.
Both are tested flat first (`screen_features.py` on tree `xmodel_1000_flag6_sapbert_sgd`, 1000 labels), so no training run. Baselines on those
11661 dev rows: flat flag-6 features 0.800 / MRR 0.852 / hybrid 0.827; tree 0.797 / 0.844 / hybrid 0.824. Seen 1 label 0.951, seen >1
label 0.844, unseen strings 0.442 (tree).

(1) Abbreviation expansion. Of 2848 unseen dev rows, 524 look like abbreviations; 377 have "(ABBR)" in the same document.
- Detect "long form (SF)" pairs per document (Schwartz-Hearst style: the short form's characters must appear in order in
  the preceding words). Where: `Preprocessor.load_pubtator_file` has the document text and spans. Screen first in
  `screen_features.py` (its loader at `main`/`load`), then move it into the shared loader.
- Apply the same expansion to train and dev. It also changes the PIFA centroids and the dictionary key; decide whether the
  dictionary/hybrid uses the raw or the expanded string, and report both.
- Variants: replace the mention with the long form, or append it ("SSI surgical site infection"). Report the new acc@1 per
  mention-string group and on the abbreviation rows. Check: number of rows changed in train/dev, plus 30 sampled pairs for
  precision.

(2) Candidate reranker (replaces the hard hybrid rule; the old per-label rankers stay off).
- Candidates: top-N (20-50) by cosine. Features per (mention, label): cosine of each block (SapBERT, char, context) and the
  combined one, candidate rank, log label train frequency, string-label train count/share, string seen, number of labels
  for the string.
- Leakage: k-fold over train **documents**. Dictionary, frequency features **and the PIFA centroids** come from the other
  folds only; otherwise a train row's own embedding sits inside its gold centroid.
- Model: sklearn LogisticRegression or HistGradientBoosting (pointwise) first; lightgbm is installed (lambdarank) if
  needed. No new deps. Target: beat hybrid 0.827 flat. Then apply it to the tree's `score_csr` in eval.

**Done (session 11): (1) abbreviation expansion.** Flat screen (results Session 11): append ("SF long form") 0.810 / MRR
0.861 / hybrid 0.835 vs 0.800 / 0.852 / 0.827; abbrev rows 0.386 -> 0.556. Moved into the shared loader:
`Preprocessor.best_long_form` / `abbreviations`, `load_pubtator_file(..., abbrev="append"|"replace"|None)`; XModel key
`abbrev_expansion` (None for old trees via getattr); train, eval (test + dictionary train side) and diagnose pass it.
Config: now `.models/xmr4el_base_config.json` (flag-6 SapBERT + `"abbrev_expansion": "append"`). Loader output on dev
matches the screen's expansion exactly (3895 rows, 0 diffs). Regression in `test_text_encoder.py`. Uncommitted. No tree yet.
`screen_features.py -abbrev append` gives the new flat baseline for step (2).

**Done (session 11, user request): flags reduced to 1 and 6.** `TextEncoder._encode(texts, fit)` is the one path for
train/query: flag 1 = TF-IDF (-> dimension model) of the whole text; flag 6 = [transformer(mention) | char TF-IDF -> SVD
(mention) | TF-IDF -> SVD(context)], each block L2-normalised. Flags 2-5 raise. Transformer wrapper: `MODEL_NAMES`
(`sapbert`, `sentencetbiobert`, `biobert`) or any checkpoint via `kwargs.model_name` (+ optional `pooling`,
`max_seq_length`); metaclass registry and per-model classes removed. `xmodel_1000_flag6_sapbert_sgd` encoder output bit-identical (64 dev rows,
max diff 0.0). Configs: `xmr4el_base_config.json` = flag 6 + SapBERT + abbrev append; `xmr4el_flag1_config.json` = old base
TF-IDF with flag 1 (TSV); `xmr4el_flag6_sapbert_config.json` kept (recipe of `xmodel_1000_flag6_sapbert_sgd`); flag-5 configs and
`run_ablation.sh` removed. XModel default `emb_flag` 6; TSV training requires flag 1. Trees with flags 1-5 no longer predict.
Pending user: delete unused trees (all but `xmodel_1000_flag6_sapbert_sgd`; the auto-mode classifier blocked `rm`). Uncommitted.

**Done (session 11): PECOS vs XMR4EL, 500 labels** (results "Session 11: XMR4EL vs PECOS"). Tree
`xmodel_500_flag6_sapbert_abbrev_sgd` (base config, 500 labels): 0.817 / MRR 0.867 / hybrid 0.851. PECOS XR-Linear on the tree's own
features and rows: 0.861 / 0.891 (beam 2) / hybrid 0.865; seen ambiguous 0.839 -> 0.933, unseen 0.521 -> 0.571. Features are
not the gap; the leaf label scorer is (cosine to PIFA centroid vs PECOS's per-label linear classifier with in-cluster
negatives). Tools: `pecos.dockerfile`, `test/xmr4el/pecos_compare.py` (export/score), `test/xmr4el/pecos_run.py`.

**`-alpha 0` results:** beam 2 0.394 (R@5 0.917); beam 1 0.794 (cap 0.898, i.e. 0.88 within the leaf vs cosine 0.86,
PECOS ~0.90). Cause confirmed: leaf matcher probs are per-label sigmoids only comparable within a leaf (multilabel OneVsRest, no
renormalisation) and leaves are merged by max with no path factor. **Fix implemented (uncommitted):** `path_score` predict option (HierarchicaMLModel -> XModel -> eval
`-path_score`): leaf score x exp(path_logscore) = P(leaf) * P(label | leaf), as XR-Linear. Off by default (recorded rows
unchanged). Regression in `test_no_rankers.py`. Result: 0.801 / MRR 0.858 / hybrid 0.856 (merge fixed; seen 1 0.935, seen >1 0.812, unseen 0.540) vs cosine 0.817,
PECOS 0.861. Remaining gap = leaf classifier. Screen `screen_leaf_scorer.py` (oracle routing, within-leaf acc@1): cosine 0.866,
logreg 0.891, logreg_bal 0.920, svm 0.921 (results Session 11). **Done (uncommitted):** base config matcher (all layers)
= `sklearnlogisticregression` liblinear L2 C 1 balanced n_jobs 1; leaf `early_stopping` override now SGD-only
(`MLModel.train`); matcher selfcheck reads the SGD config from `xmr4el_flag6_sapbert_config.json`.
**Next (user):** retrain `-ds_len 500` with the base config, eval `-beam_size 2 -alpha 0 -path_score` (+ `-alpha 1` for the
cosine row), record train secs. Trained: `xmodel_500_flag6_sapbert_abbrev_logreg` (logreg matcher, 500 labels;
train secs not yet reported); eval pending. Done: `penalty`/`l1_ratio`/`n_jobs` dropped from the logreg wrapper defaults and
base config (sklearn 1.9.1 deprecations); verified warning-free and coef-identical to explicit `penalty="l2"`.
**Result (results "liblinear leaf matcher + path score"):** 500 labels 0.862 / MRR 0.896 = PECOS 0.861; 1000 labels
0.843 / 0.884 vs previous 1000 tree 0.797. Eval default for these trees: `-beam_size 2 -alpha 0 -path_score`. Next: full
labels (no `-ds_len`) with the base config; optional PECOS at 1000/full. Weak group: unseen strings (0.53-0.58).
**Speed (user priority, session 11):** PECOS at 1000 labels: 0.838 (tree 0.843). Profile (500 labels): SapBERT 47%,
OneVsRest liblinear wait 34%. Done (uncommitted): dedup of texts before embedding + MPS device in `transformers.py`
(regression in `test_text_encoder.py`). Leaf screen at 1000: liblinear 238 s, tol 1e-2 197 s, svd 256
143 s (-0.0055, rejected). Done (uncommitted): `JointOvRLogistic` / config type `jointlogisticregression` in
`classifier_model.py` (liblinear's OvR L2 logistic objective for all labels of a node in one L-BFGS problem): same
0.8974 in 19 s; test `test/xmr4el/test_joint_logistic.py`. Base config matcher switched to it. Next: one timed
`-ds_len 1000` retrain, eval `-beam_size 2 -alpha 0 -path_score`; expect 0.843 (tree) within float noise. User does not want the full run yet.
Trained (1000 labels, base config): `xmodel_1000_flag6_sapbert_abbrev_logreg` (renamed from `xmodel_2026-10-02_16-10-54`); eval pending.
Training queue after that eval: (a) `-ds_len 1000` base config vs `xmodel_1000_flag6_sapbert_sgd` (0.797 cosine); (b) full labels (no
`-ds_len`): liblinear removes the old leaf-cost blocker, and under `-alpha 0` the leaf's matcher top-100 cut ranks by the
same score, so it no longer drops labels the scorer would rank first; (c) encoder swap only after a flat screen. Target PECOS 0.861; expected ~0.92 x routing recall (~0.95) ~ 0.87.
This replaces step (2): a reranker must now beat 0.861, not hybrid 0.835.
Training time (user): tree 3000 s vs PECOS 7.3 s (PECOS excludes featurization, ~2-4 min of ours). Measured on one
83-label leaf of the exported X: our leaf matcher (SGD log_loss, L1, `early_stopping` forced off at the leaf, tol 1e-4)
takes 25 s/label, 168-823 epochs -> ~210 core-min for 500 labels = ~30 min on 8 cores: the bulk of the 3000 s. Same
data: SGD with early stopping 0.26 s, liblinear squared hinge (sklearn LinearSVC, PECOS-like) 0.15 s/label. A liblinear
leaf classifier is the candidate fix for both speed and the 0.861 gap; decide after `-alpha 0`.

(2) candidate reranker: on hold until the `-alpha 0` result (see above).
Code not frozen: no user run in flight.

## Sessions 8-10 (2026-10-01)

State (flag 4, 500 labels, tree `xmodel_2026-10-01_11-20-27`): hierarchy + cosine leaf scorer
acc@1 0.764, MRR 0.822 = flat nearest-label 0.767 > dictionary 0.717; hybrid (dict if string
seen, else tree) 0.805. recall@cand 0.949 is the cap. 72% of tree errors are unseen mention
strings (acc@1 0.343). Rankers are out of prediction (`-scorer cosine` default); user redesigns
them later.

Next (user decision 2026-10-01): **explore how mentions and labels are vectorized and used**
before scaling labels. Scaling waits: any tree trained now is obsolete after a feature change, and a
user run in flight freezes code.

Featurization trace (session 8, from code):
- Input row = `"{mention} [SEP] {title abstract}"` (`Preprocessor.load_pubtator_file`); PubTator
  offsets are dropped, so the context is the whole document, identical for every mention in it.
- Flag 4: mention -> `pritamdeka/S-BioBert-snli-multinli-stsb` (sentence-similarity model, not
  trained for synonymy), raw `normalize_embeddings=False`; context -> word TF-IDF 30k -> SVD 1500.
  hstack, one L2 norm: mention holds 0.998 of the norm, so X ~ the mention embedding alone.
- Z = PIFA: normalised sum of the label's train rows = centroid of its train mention embeddings
  (no label names; KB out of scope). Leaf cosine scorer = cos(X_aug, Z_aug); the 3 matcher extras
  scale a query row uniformly, so within-query order = flat cosine restricted to candidates.
- No feature sees the mention's characters: unseen surface variants depend entirely on S-BioBert.

Screen done (`docs/results.md` Session 8, rounds 1-2): flat `sbiobert` + char TF-IDF wins (0.784,
unseen 0.448); its tree-feasible form `sbiobert` + char-SVD 768 gives 0.775 / unseen 0.427 /
hybrid 0.826 vs current 0.766 / 0.340 / 0.804. `max` (1-NN) scoring and word TF-IDF rejected.

Implemented (uncommitted): `emb_flag` 5 in `TextEncoder` = [transformer(mention) | SVD(TF-IDF(mention))],
each block L2-normalised before the concat; config `.models/xmr4el_flag5_config.json` (char_wb 2-4,
sublinear, max_df 1.0, SVD 768, else base). Tfidf wrapper casts JSON `ngram_range` lists to tuples.
Regression `test/xmr4el/test_text_encoder.py`. Train guard and `diagnose_routing.py` know flag 5.

SapBERT support (user decision 2026-10-01: compare SapBERT vs S-BioBert; train with the winner):
transformer type `sapbert` (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`), loaded by
`sentence_model` in `transformers.py` with [CLS] pooling and max 25 tokens (`CLS_POOLED`); every
other model loads as before (parity checked, max abs diff 0.0). Config
`.models/xmr4el_flag5_sapbert_config.json` = flag-5 config with `sapbert`. `screen_features.py`
encodes through the same loader, bypassing `./batch_dir`.

Screen round 3 recorded (Session 8): SapBERT wins. `sapbert` + charsvd / centroid 0.793, unseen
0.466, hybrid 0.836 vs `sbiobert` + charsvd 0.775 / 0.427 / 0.826 (current tree features 0.766).

Rankers off (user decision 2026-10-01): config key `train_rankers` (XModel -> HierarchicaMLModel ->
MLModel), `false` in all three `.models` configs. Prediction-neutral under `-scorer cosine`: root
`fused_scores` are matcher-only and the leaf's (only ranker consumer) feed discarded children;
`test/xmr4el/test_no_rankers.py` trains a synthetic tree both ways and asserts identical cosine
scores. `predict` treats a missing `ranker_model` as all-cosine. Old trees keep their rankers.

Done: flag-5 SapBERT tree `xmodel_2026-10-01_15-10-04` (results Session 8): acc@1 0.790, MRR 0.844,
unseen 0.464, hybrid 0.836 vs `11-20-27` 0.764 / 0.822 / 0.343 / 0.805; flat ceiling 0.793.
Retained: new baseline is `15-10-04` + `xmr4el_flag5_sapbert_config.json`. Weak spot: seen ambiguous
strings 0.804 (dict 0.928).

User plan (2026-10-01): context screen (done) -> train 1000 labels -> try full.

Context screen recorded (results Session 9): + context TF-IDF adds +0.010-0.012 flat (0.793 -> 0.805),
seen ambiguous strings 0.804 -> 0.835; hybrid only +0.003. 10-word window = whole document; SVD 768
loses nothing. Chosen: equal block weights (untuned), window 10.

Implemented (uncommitted): `emb_flag` 6 = flag 5 + third block SVD(TF-IDF(context)), each block
L2-normalised; persisted as `context_vectorizer_model` / `context_dimension_model`. XModel keys
`context_vectorizer_config`, `context_dimension_config`, `context_window`; train/eval/diagnose load
PubTator with `window=model.context_window` (context = +-N words, mention excluded; None = whole
document, old behaviour). Config `.models/xmr4el_flag6_sapbert_config.json`. Loader also returns
`spans`. Regression: `test_text_encoder.py` (flag 6 blocks, empty window, save/load, loader window).

Flag-6 1000-label tree `xmodel_1000_flag6_sapbert_sgd` recorded (results Session 9): tree 0.797 / MRR 0.844 /
hybrid 0.824 vs flat ceiling 0.800 / 0.852 / 0.827; gap 0.003, same as 500. Context +0.019 flat at 1000.
Eval must use `-alpha 1` (alpha 0.5 gave 0.759: the matcher cluster probability gets mixed in).

Full-label plan (session 10, from code): train has 18520 labels, 122241 rows (`-ds_len` = labels; omit = all).
Same flag-6 config, depth 2, K 6 -> 6 leaves of ~3.1k labels each (balanced k-means). Cost and risk:
- Leaf matcher = one-vs-rest SGD over every leaf label (`MatcherTrainer`, identity C): ~18.5k binary fits on
  ~20k rows each, about 60x the 1000-label leaf work. Dominant cost; memory fine (X ~1-2 GB, leaf
  predict_proba ~0.5 GB dense).
- New filter: `_predict_one_leaf` keeps the leaf matcher's top 100 labels (hard-coded `beam_size=100`) before cosine
  scoring. At 1000 labels (167/leaf) it lost 0.003 (recall@cand 0.943 vs root top-2 0.946). At ~3.1k/leaf it is
  a real cut: measure as root top-2 (diagnose) minus recall@cand (eval).
- Not chosen: depth 3 (smaller leaves). That would be a second factor; decide after the depth-2 result.

Postponed (user cannot run it now): full run (flag 6, no `-ds_len`), then eval `-alpha 1 -train_path`, diagnose, screen
`sapbert+charsvd+ctxwinsvd`. **Freeze code.** On output: record the full row (tree vs flat, leaf-cut loss);
large leaf-cut loss -> leaf top-k / candidate selection by cosine is the next lever.

Error profile, 1000 labels (session 10; tree errors from results Session 9, rest from corpus counts): 2370 errors =
unseen strings ~1589 (67%), seen ambiguous ~513 (22%), seen single-label ~269 (11%); gold missing from candidates in 668 rows.
Of 2848 unseen dev rows: 524 look like abbreviations (377 have "(ABBR)" in the same doc); 1038 have a top-20 frequent
(generic) gold label such as "findings" or "disease". The dictionary beats the tree on seen strings. Candidate levers, in order:
abbreviation expansion (flat screen) -> candidate reranker (per-block cosines, label frequency, string-label count;
train it with k-fold so dictionary features are not leaked) -> SapBERT fine-tuning. Hierarchy costs only 0.003: not a lever.

Eval output (session 10): `test_evaluate_pipeline.py` prints one compact report; library prints/logs/progress hidden
unless `-verbose`. `-alpha` default is now 1 (warns when alpha != 1 under cosine). The ranker split prints only with
`-scorer ranker`, the defect-#6 tie check only when scores collapse, and the random-ordering line is gone. Metric values unchanged.

Bug fixed (session 10): `per_leaf` `-topk k>0` kept each leaf's first k labels in label-index order, not score order
(`_maybe_leaf_topk`, `xmr4el/xmr/base.py`); `-topk 1` gave acc@1 0.006 on `xmodel_1000_flag6_sapbert_sgd`. `-topk 0` (every recorded row)
was unaffected. Regression in `test/xmr4el/test_no_rankers.py`. Confirmed: `-topk 1` now gives acc@1 0.7967 (= topk 0).

Then scale (`-ds_len` 1000 -> full) with the chosen features; see "Scale and external baseline".

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
