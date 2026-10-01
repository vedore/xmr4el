# Status and next steps

History and completed steps: `docs/status_log.md`. Valid results: `docs/results.md`.

## Resume here

Last updated 2026-10-01 (session 8). A new session starts from this block; update it at the
end of every step and before the user resets the chat.

Code not frozen (no user run in flight). Session 7 work committed (`f24b35c`, `31d5486`); session 8 uncommitted.

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

Next: user trains `.models/xmr4el_flag5_sapbert_config.json` (500 labels) and evaluates (cosine,
alpha 1, breakdown). **Freeze code while it runs.** Expect tree ~ flat 0.793. Record against
`11-20-27` as a bundle (encoder S-BioBert -> SapBERT + flag 4 -> 5), retain/revert, then commit.
Not screened: local context window (abbreviations; ambiguous seen strings stuck at ~0.80; needs
PubTator offsets).

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
| Features | Flag 4: transformer mention + reduced TF-IDF context, then row normalization. Flag 5: transformer + reduced char TF-IDF, both on the mention, per-block L2 | `TextEncoder.encode/predict`, base / flag5 config |
| Clustering | Seeded balanced k-means; undersized clusters reassigned instead of dropping labels | `xmr4el/models/cluster_wrapper/clustering_model.py`, `xmr4el/clustering/train.py` |
| Ranker training | Epochs reuse models and vary sampling seeds; config uses `E_warm=1`, `log_loss`, `neg_mult=5` | `xmr4el/ranker/train.py`, `.models/xmr4el_base_config.json` |
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
