# Pipeline reference

What is implemented, synthetic checks, pipeline observations and the external-baseline notes.
Moved out of `STATUS.md` (2026-10-06).

## What is implemented

| Area | Current behavior | Evidence |
|---|---|---|
| Model loading | Numeric `ml_<n>` order; guard checks parent/child label sets | `HierarchicalMLModel.load`, `xmr4el/hierarchy/tree.py` |
| Label mapping | Training keeps `MultiLabelBinarizer.classes_` (empty label groups have no column); load raises `ValueError` unless label count equals `Z` rows | `XModel._fit/load`, `xmr4el/xmodel.py` |
| Evaluation | Hierarchy scores via `XModel.predict` (beam search, optional knn); acc@1, MRR, recall@k, candidate recall, vocabulary coverage | `scripts/evaluate.py` |
| Features | `"tfidf"`: TF-IDF (-> dimension model) of the whole text (TSV). `"sapbert_char_context"`: [transformer(mention) \| char TF-IDF -> SVD(mention) \| TF-IDF -> SVD(context window)], per-block L2; optional `abbrev_expansion`. Other names raise | `TextEncoder._encode`, `features` in the config |
| Clustering | Seeded balanced spherical k-means (numpy): recursive balanced 2-means, then joint balanced refinement over all k clusters; undersized clusters reassigned instead of dropping labels | `xmr4el/hierarchy/clusterers.py` |
| Leaf matching | Label-level leaf matchers; base config matcher `jointlogisticregression` (`JointOvRLogistic`) | `MLModel.train`, `xmr4el/hierarchy/node.py`, `xmr4el/learning/classifiers.py` |
| Leaf scoring | Each visited leaf contributes its 100 best labels by leaf matcher probability (`LEAF_CANDIDATES`), scored matcher probability x exp(path_logscore) (XR-Linear) | `HierarchicalMLModel.predict` |
| Knn fusion | `-knn_beta b`: each candidate x exp(b * max cosine of the query's mention block to the label's training rows), computed for the candidate pairs only; BC5CDR uses 10, off by default | `XModel.predict`, `scoring.label_max_cos` |

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
| Features | `"sapbert_char_context"` L2-normalises each block before the concat, so the three blocks carry equal, untuned weight | If block diagnostics suggest a problem, compare one weighting change with a retrained control |
| Label embeddings / hierarchy | PIFA uses full mention+context features | If flat retrieval works but centroid routing fails, isolate a label-representation or clustering change; PIFA itself is not a bug |
| Matcher | Child models train on in-cluster rows but receive beam-routed queries | If root routing works and child rejection fails, measure out-of-cluster errors before adding routed negatives |
| Traversal / scoring | Ancestor `path_logscore` prunes the beam and multiplies every leaf score (exp(path_logscore)) | Leaf matcher probabilities alone are not comparable across leaves |
| Training computation | Internal nodes set `fused_scores` from matcher cluster scores in `MLModel.train`, consumed by `prepare_layer`; leaves prepare no children (`tests/hierarchy/test_tree.py`) | Fusing other scores into these is future work (`STATUS.md`) |

The path score is a fixed route-score rule. A learned weight needs separate justification and dev
tuning. Neither route weighting nor `log_loss` alone establishes calibration across leaves.

## Scale and external baseline

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
