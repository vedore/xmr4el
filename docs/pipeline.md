# Pipeline reference

What is implemented, synthetic checks, pipeline observations and the external-baseline notes.
Moved out of `STATUS.md` (2026-10-06).

## What is implemented

| Area | Current behavior | Evidence |
|---|---|---|
| Model loading | Numeric `ml_<n>` order; guard checks parent/child label sets | `HierarchicalMLModel.load`, `xmr4el/hierarchy/tree.py` |
| Label mapping | Training keeps `MultiLabelBinarizer.classes_` (empty label groups have no column); load sorts legacy first-seen lists and asserts label count equals `Z` rows | `XModel._fit/load`, `xmr4el/xmodel.py` |
| Evaluation | Hierarchy scores via `per_leaf`; acc@1, MRR, recall@k, candidate recall, vocabulary coverage | `scripts/evaluate.py` |
| Features | Flag 1: TF-IDF (-> dimension model) of the whole text (TSV). Flag 6: [transformer(mention) \| char TF-IDF -> SVD(mention) \| TF-IDF -> SVD(context window)], per-block L2; optional `abbrev_expansion`. Flags 2-5 raise | `TextEncoder._encode`, base / flag1 config |
| Clustering | Seeded balanced k-means; undersized clusters reassigned instead of dropping labels | `xmr4el/hierarchy/clusterers.py` |
| Ranker training | Off by default (`train_rankers: false`). When on: epochs reuse models and vary sampling seeds; `E_warm=1`, `log_loss`, `neg_mult=5` | `xmr4el/learning/ranker.py`, `configs/xmr4el_base_config.json` |
| Leaf matching | Label-level leaf matchers; base config matcher `jointlogisticregression` (`JointOvRLogistic`); leaf `early_stopping` override is SGD-only | `MLModel.train`, `xmr4el/hierarchy/node.py`, `xmr4el/learning/classifiers.py` |
| Leaf scoring | Eval defaults to cosine (`predict(scorer="cosine")`, eval `-scorer cosine`); `-scorer ranker` uses trained rankers, falling back to cosine when missing/raising and logging failures | `MLModel.predict`, `xmr4el/learning/scoring.py` |
| Path score | `-path_score`: leaf score x exp(path_logscore); current trees eval with `-beam_size 2 -alpha 0 -path_score` | `HierarchicalMLModel.predict`, `XModel.predict` |

Implemented does not mean validated on dev. Saved trees retain their trained classifiers and leaf
structure; loading them with current code does not apply training changes retroactively.

Synthetic checks passed during the latest review:

```bash
.venv/bin/python -m pytest tests/
.venv/bin/python scripts/evaluate.py -selfcheck
.venv/bin/python scripts/diagnose_routing.py -selfcheck
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
```

`tests/integration/test_label_mapping.py` covers the mapping through the real `XModel.train/save/load` paths
(encoder and hierarchy stubbed): unsorted labels plus an empty group, legacy first-seen metadata,
and a legacy empty group, which load refuses. It fails when any of the mapping fixes is removed.

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
