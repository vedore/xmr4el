# Tests

Run from the repository root with Python 3.12+ and the development dependencies:

```bash
uv sync --locked --group dev
source .venv/bin/activate
python -m pytest tests/ -q
```

Tests use synthetic inputs, temporary files, and mocked transformer encoders.
They do not require corpus datasets or pretrained-model downloads. Some checks train
small classifiers/hierarchies, so the project ML dependencies are still required.

Current checkout: collection fails because `xmr4el/xmodel.py` and `xmr4el/eval.py`
are missing. Restore those modules to run the complete suite. Legacy-import checks
also require the compatibility modules referenced by `integration/test_legacy_imports.py`.

## Coverage by folder

| Path | Checks |
| --- | --- |
| `data/` | PubTator/grouped TSV readers, label alignment, training CLI inputs, split CLI output. |
| `features/` | Text encoding, feature-block normalization, embedding row order, CUDA/MPS out-of-memory recovery. |
| `hierarchy/` | Cluster reassignment, layer preparation, training without rankers, hierarchy persistence. |
| `learning/` | Joint logistic classifier, single-positive matcher handling, ranker curriculum, score fusion. |
| `integration/` | Config preservation, import behavior, legacy pickle imports, label mapping, pipeline persistence. |
| `test_eval.py` | Ranking metric self-check and known metric values. |
| `test_eval_quiet.py` | Evaluation quiet/verbose settings in fresh Python processes. |

## Run a subset

```bash
python -m pytest tests/data/ -q
python -m pytest tests/features/ -q
python -m pytest tests/hierarchy/ -q
python -m pytest tests/learning/ -q
python -m pytest tests/integration/ -q
python -m pytest tests/test_eval.py tests/test_eval_quiet.py -q
```

Run one regression check by its pytest node ID:

```bash
python -m pytest tests/features/test_transformers.py::test_mps_oom -q
python -m pytest tests/integration/test_label_mapping.py::test_label_mapping -q
```

Inspect collection without executing tests, or stop at the first failure:

```bash
python -m pytest tests/ --collect-only -q
python -m pytest tests/ -x -q
```

Use `-s` to show captured prints or `-vv` for individual test names.
For script self-checks and training/evaluation commands, see [scripts](../scripts/README.md).
