# Pipeline reference

What is implemented, synthetic checks, pipeline observations and the external-baseline notes.
Moved out of `STATUS.md` (2026-10-06).

## What is implemented

| Area | Current behavior | Evidence |
|---|---|---|
| Model loading | Numeric `ml_<n>` order; guard checks parent/child label sets | `HierarchicalMLModel.load`, `xmr4el/hierarchy/tree.py` |
| Label mapping | Training keeps `MultiLabelBinarizer.classes_` (empty label groups have no column); load asserts label count equals `Z` rows | `XModel._fit/load`, `xmr4el/xmodel.py` |
| Evaluation | Hierarchy scores via `per_leaf`; acc@1, MRR, recall@k, candidate recall, vocabulary coverage | `scripts/evaluate.py` |
| Features | Flag 1: TF-IDF (-> dimension model) of the whole text (TSV). Flag 6: [transformer(mention) \| char TF-IDF -> SVD(mention) \| TF-IDF -> SVD(context window)], per-block L2; optional `abbrev_expansion`. Flags 2-5 raise | `TextEncoder._encode`, base / flag1 config |
| Clustering | Seeded balanced spherical k-means (numpy): recursive balanced 2-means, then joint balanced refinement over all k clusters; undersized clusters reassigned instead of dropping labels | `xmr4el/hierarchy/clusterers.py` |
| Ranker training | Off by default (`train_rankers: false`). When on: epochs reuse models and vary sampling seeds; `E_warm=1`, `log_loss`, `neg_mult=5` | `xmr4el/learning/ranker.py`, `configs/xmr4el_base_config.json` |
| Leaf matching | Label-level leaf matchers; base config matcher `jointlogisticregression` (`JointOvRLogistic`); leaf `early_stopping` override is SGD-only | `MLModel.train`, `xmr4el/hierarchy/node.py`, `xmr4el/learning/classifiers.py` |
| Leaf scoring | `evaluate.py` defaults to `-scorer cosine`; `XModel.predict(scorer=None)` uses trained rankers when present, else cosine. `-scorer ranker` uses trained rankers, falling back to cosine when missing/raising and logging failures | `MLModel.predict`, `xmr4el/learning/scoring.py` |
| Path score | `-path_score`: leaf score x exp(path_logscore); current trees eval with `-beam_size 2 -alpha 0 -path_score` | `HierarchicalMLModel.predict`, `XModel.predict` |
| Knn fusion | `-knn_beta b`: each candidate x exp(b * max cosine of the query's mention block to the label's training rows); BC5CDR uses 10, off by default | `XModel.predict`, `scoring.label_max_cos` |

Implemented does not mean validated on dev. Saved trees retain their trained classifiers and leaf
structure; loading them with current code does not apply training changes retroactively.

Synthetic checks (all pass, 2026-10-07):

```bash
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/evaluate.py -selfcheck
.venv/bin/python scripts/diagnose_routing.py -selfcheck
.venv/bin/python scripts/diagnose_unseen.py -selfcheck
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
```

`tests/integration/test_label_mapping.py` covers the mapping through the real `XModel.train/save/load` paths
(encoder and hierarchy stubbed): unsorted labels plus an empty group, and a saved label list longer
than `Z`, which load refuses. It fails when any of the mapping fixes is removed.

## Pipeline observations

Observations and hypotheses, not a mandatory sequence of fixes.

| Pipeline stage | Confirmed behavior | Follow-up if implicated |
|---|---|---|
| Data | `-ds_len` keeps the first N label groups; the historical 500-label vocabulary covered about 18.4% of dev mentions | Keep vocabulary fixed for debugging; use a seeded, recorded sample or full vocabulary for broader claims |
| Features | Flag 6 L2-normalises each block before the concat, so the three blocks carry equal, untuned weight | If block diagnostics suggest a problem, compare one weighting change with a retrained control |
| Label embeddings / hierarchy | PIFA uses full mention+context features | If flat retrieval works but centroid routing fails, isolate a label-representation or clustering change; PIFA itself is not a bug |
| Matcher | Child models train on in-cluster rows but receive beam-routed queries | If root routing works and child rejection fails, measure out-of-cluster errors before adding routed negatives |
| Ranker | Each per-label linear model sees a constant `Z_label` block, contributing an intercept-like term | Measure scoring contribution before redesigning; constant features are not evidence of failure |
| Traversal / scoring | Ancestor `path_logscore` prunes the beam; it enters final leaf scores only with `-path_score` (leaf score x exp(path_logscore)), off by default | Without `-path_score`, leaf scores are not comparable across leaves |
| Training computation | Internal nodes set `fused_scores` from matcher cluster scores in `MLModel.train`, consumed by `prepare_layer`; leaves prepare no children (`tests/hierarchy/test_no_rankers.py`) | Matcher/ranker fusion of these scores is future work (`STATUS.md`) |

`-path_score` is a fixed route-score rule. A learned weight needs separate justification and dev
tuning. Neither route weighting nor `log_loss` alone establishes calibration across leaves.

## Scale and external baseline

- Keep the per-label ranker as a research hypothesis and measure its contribution within XMR4EL.
- Then run a PECOS baseline on the same split, label mapping and instance features. Record hierarchy,
  negative sampling, scoring, search budget, runtime and memory differences. Similar parameter names
  do not establish equivalent algorithms.
- Custom layer feature augmentation remains in `prepare_layer`; the proposed scoring/matcher changes
  do not make XMR4EL differ from PECOS only by its ranker. Treat this as a system comparison unless
  other differences are explicitly controlled.
- PECOS runs in Docker (`pecos.dockerfile`, linux/amd64; libpecos has no macOS wheel) via
  `scripts/baselines/pecos_run.py`; results in `docs/results.md` (Session 11, BC5CDR). Use
  `-threshold 0`: PECOS's default weight pruning (0.1) cost it 0.075 acc@1 on BC5CDR dev.
- MedMentions st21pv is the main dataset; BC5CDR is also evaluated (2026-10-06). Report actual training-label count and dev/test coverage;
  the dataset's advertised label count is not automatically the model vocabulary. Reserve test
  evaluation for the selected configuration. Consider MedMentions/full only if scaling is needed.

No broad rewrite, learned score calibration, or additional encoder dependency is scheduled.
