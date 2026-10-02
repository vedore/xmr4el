# Status and next steps

History and completed steps: `docs/status_log.md`. Valid results: `docs/results.md`.

## Resume here

Last updated 2026-10-02 (session 10). A new session starts from this block; update it at the
end of every step and before the user resets the chat.

**NEXT (user decision 2026-10-02, session 10 end):** work on (1) abbreviation expansion, then (2) a candidate reranker.
Both are tested flat first (`screen_features.py` on tree `17-48-39`, 1000 labels), so no training run. Baselines on those
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

Code not frozen: no user run in flight. Session 10 changes are uncommitted (STATUS, results, eval report, topk fix).

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

Flag-6 1000-label tree `xmodel_2026-10-01_17-48-39` recorded (results Session 9): tree 0.797 / MRR 0.844 /
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
(`_maybe_leaf_topk`, `xmr4el/xmr/base.py`); `-topk 1` gave acc@1 0.006 on `17-48-39`. `-topk 0` (every recorded row)
was unaffected. Regression in `test/xmr4el/test_no_rankers.py`. Confirmed: `-topk 1` now gives acc@1 0.7967 (= topk 0).

Then scale (`-ds_len` 1000 -> full) with the chosen features; see "Scale and external baseline".

Commands: `docs/results.md` § Commands.

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
| Features | Flag 4: transformer mention + reduced TF-IDF context, then row normalization. Flag 5: transformer + reduced char TF-IDF, both on the mention, per-block L2. Flag 6: flag 5 + reduced word TF-IDF of a context window | `TextEncoder.encode/predict`, base / flag5 config |
| Clustering | Seeded balanced k-means; undersized clusters reassigned instead of dropping labels | `xmr4el/models/cluster_wrapper/clustering_model.py`, `xmr4el/clustering/train.py` |
| Ranker training | Off by default (`train_rankers: false`). When on: epochs reuse models and vary sampling seeds; `E_warm=1`, `log_loss`, `neg_mult=5` | `xmr4el/ranker/train.py`, `.models/xmr4el_base_config.json` |
| Leaf matching | Newly trained leaf matchers are label-level; leaf early stopping is disabled for singleton positives | `MLModel.train`, `xmr4el/xmr/base.py` |
| Leaf scoring | Cosine by default (`cosine_scorer`, eval `-scorer cosine`); `-scorer ranker` uses trained rankers, falling back to cosine when missing/raising | `MLModel.predict`, `xmr4el/xmr/base.py` |

Implemented does not mean validated on dev. Saved trees retain their trained classifiers and leaf
structure; loading them with current code does not apply training changes retroactively.

Synthetic checks passed during the latest review:

```bash
.venv/bin/python test/xmr4el/test_data_loading.py
.venv/bin/python test/xmr4el/test_evaluate_pipeline.py -selfcheck
.venv/bin/python test/xmr4el/diagnose_routing.py -selfcheck
.venv/bin/python test/xmr4el/test_prepare_layer.py
.venv/bin/python test/xmr4el/screen_features.py -selfcheck
.venv/bin/python test/xmr4el/test_text_encoder.py
.venv/bin/python test/xmr4el/test_no_rankers.py
```

`test_data_loading.py` covers the mapping through the real `XModel.train/save/load` paths
(encoder and hierarchy stubbed): unsorted labels plus an empty group, legacy first-seen metadata,
and a legacy empty group, which load refuses. It fails when any of the mapping fixes is removed.

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

## Pipeline observations

Observations and hypotheses, not a mandatory sequence of fixes.

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

## Scale and external baseline

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
