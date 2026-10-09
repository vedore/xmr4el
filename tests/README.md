# Tests

Run from the repository root with Python 3.12+ and the development dependencies:

```bash
uv sync --locked --group dev
.venv/bin/python -m pytest tests/ -q
```

Tests use synthetic inputs, temporary files, and mocked transformer encoders.
They do not require corpus datasets or pretrained-model downloads. Some checks train
small classifiers/hierarchies, so the project ML dependencies are still required.

## Coverage by folder

| Path | Checks |
| --- | --- |
| `data/` | PubTator/grouped TSV readers, label alignment, training CLI inputs, BC5CDR `-1`/composite ids, split CLI output, CTD dictionary conversion. |
| `features/` | Text encoding, feature-block normalization, embedding row order, CUDA/MPS out-of-memory recovery. |
| `hierarchy/` | Cluster reassignment, balanced k-means, layer preparation, tree training. |
| `learning/` | Joint logistic classifier, mention-kNN label scores (`label_max_cos`). |
| `integration/` | Config preservation, import behavior, label mapping, pipeline persistence. |
| `test_eval.py` | Ranking metric self-check and known metric values. |
| `test_run_experiment.py` | Tracked run wrapper on a synthetic spec: run dir files (status, provenance hashes, git untracked), metrics = `evaluate_tree`, eval-only reuse, spec errors (names, search values, missing inputs, bad model config), no overwrite, empty eval fails the run, log handler removed. |
| `test_eval_quiet.py` | Evaluation quiet/verbose settings in fresh Python processes. |
| `test_logging.py` | Logging levels, application-owned handlers, and label-truncation warnings. |
| `test_audit_regressions.py` | Bug-audit regressions: knn before topk, candidate knn, split CLI, save/load errors, predict config checks. |

## Run a subset

```bash
.venv/bin/python -m pytest tests/data/ -q
.venv/bin/python -m pytest tests/features/ -q
.venv/bin/python -m pytest tests/hierarchy/ -q
.venv/bin/python -m pytest tests/learning/ -q
.venv/bin/python -m pytest tests/integration/ -q
.venv/bin/python -m pytest tests/test_eval.py tests/test_eval_quiet.py -q
```

Run one regression check by its pytest node ID:

```bash
.venv/bin/python -m pytest tests/features/test_transformers.py::test_mps_oom -q
.venv/bin/python -m pytest tests/integration/test_label_mapping.py::test_label_mapping -q
```

Inspect collection without executing tests, or stop at the first failure:

```bash
.venv/bin/python -m pytest tests/ --collect-only -q
.venv/bin/python -m pytest tests/ -x -q
```

Use `-s` to show captured prints or `-vv` for individual test names.
For script self-checks and training/evaluation commands, see [scripts](../scripts/README.md).
