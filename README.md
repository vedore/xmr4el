# XMR4EL

Extreme multi-label ranking (XMR) for biomedical entity linking. XMR4EL links a mention, given with its
document context, to a concept ID from a large label space (MeSH, OMIM, UMLS CUIs, ...). It clusters the
labels into a tree, trains a matcher per tree node, and searches the tree with a beam at prediction time.

It is a research codebase: training and evaluation run from scripts on local files. It does not detect
mentions; prediction takes mentions you supply.

## How it works

1. **Read** PubTator annotations (or grouped TSV texts). Each annotation becomes `mention [SEP] context`
   with its concept ID as the label.
2. **Encode** each row. The default (`"sapbert_char_context"`) joins three L2-normalised blocks:
   SapBERT([CLS]) of the mention, character TF-IDF -> SVD of the mention, and TF-IDF -> SVD of the
   context window.
3. **Embed labels** with PIFA: the normalised sum of each label's training rows.
4. **Cluster** the label embeddings into a hierarchy with balanced k-means.
5. **Train** one-vs-rest L2 logistic matchers: per internal node over its child clusters, per leaf over
   its labels.
6. **Predict** by beam search from the root; each visited leaf contributes its best labels, scored by leaf
   probability x routing-path probability, optionally reweighted by mention similarity to the label's
   training rows (`knn_beta`).

## Installation

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) (macOS: `brew install uv`):

```bash
git clone https://github.com/lasigeBioTM/XMR4EL.git xmr4el && cd xmr4el
uv sync --locked
```

`uv.lock` pins every dependency. On Linux, torch comes from the PyTorch CUDA 12.6 index (the PyPI Linux
build needs CUDA 13, NVIDIA driver >= 580). Encoding and matcher training use CUDA when available, then Apple
MPS, otherwise CPU. The first transformer use downloads the model from the Hugging Face Hub.

Commands below use `.venv/bin/python`; inside the Docker image use `python3`.

## Quick start

Train on a PubTator corpus, then evaluate the saved tree on a held-out split:

```bash
.venv/bin/python scripts/train.py \
  -train_path datasets/ncbi_disease/train.pubtator \
  -model_config configs/xmr4el_bc5cdr_config.json && \
.venv/bin/python scripts/evaluate.py \
  -xmodel_path outputs/saved_trees/<run> \
  -test_path datasets/ncbi_disease/dev.pubtator
```

`-ds_len N` trains on the first `N` label groups only (a fast smoke run); it counts labels, not documents
or mentions. Trees are saved to `outputs/saved_trees/xmodel_<timestamp>/`.

A tiny dataset for trying the TUI end to end (40 / 10 / 10 NCBI disease documents, 95 training labels) uses
`configs/xmr4el_tiny_config.json` (4 clusters, 64 SVD components). Build it once from the local NCBI files:

```bash
mkdir -p datasets/ncbi_tiny && for s in train:40 dev:10 test:10; do awk -v n=${s#*:} 'BEGIN{RS="";ORS="\n\n"} NR<=n' datasets/ncbi_disease/${s%%:*}.pubtator > datasets/ncbi_tiny/${s%%:*}.pubtator; done
```

Predict from Python:

```python
from xmr4el.xmodel import XModel

model = XModel.load("outputs/saved_trees/<run>")
scores = model.predict(["aspirin [SEP] Aspirin reduced the risk of stroke."])
best = model.initial_labels[scores[0].toarray().argmax()]
```

`scores` is a sparse (queries x labels) matrix; column `j` is `model.initial_labels[j]` (sorted label IDs).
Search settings default to the tree's `predict_config`; `beam_size`, `topk` and `knn_beta` keyword
arguments override them.

## Input formats

Datasets are not part of the repository: put them under `datasets/` or `data/`.

### PubTator

```
12345|t|Aspirin and stroke
12345|a|Aspirin reduced the risk of stroke.
12345	0	7	Aspirin	Chemical	D001241
12345	12	18	stroke	Disease	D020521
```

Title and abstract lines, then one tab-separated annotation per line: document ID, start, end, mention,
type, concept ID. Offsets index `title + " " + abstract`. The context is that text, or the
`context_window` words on each side of the mention (10 in both provided configs).

- ID `-1` is skipped. A composite ID `D1|D2` becomes one example per ID, with texts from column 7; a
  composite without column 7 is skipped (BC5CDR conventions).
- An annotation without a concept ID is skipped with a warning (`file:line`). Relation lines are ignored.
- A mention that differs from the text at its offsets is kept, with a warning.
- An abstract or annotation that does not belong to the current document, or offsets outside it, stop
  the load with `file:line`.

### Grouped TSV

Rows are `group_id<TAB>text`; a separate file has one label ID per line, aligned with the sorted group IDs.
Plain text has no `[SEP]`, so it needs `"features": "tfidf"`:

```bash
.venv/bin/python -c "import json; c = json.load(open('configs/xmr4el_base_config.json')); c['features'] = 'tfidf'; json.dump(c, open('configs/local_tfidf_config.json', 'w'), indent=4)" && \
.venv/bin/python scripts/train.py \
  -train_path data/train/chemical/train_Chemical.txt \
  -labels_path data/train/chemical/labels.txt \
  -model_config configs/local_tfidf_config.json
```

The loader rejects too few labels and drops labels beyond the number of groups: check that the two files
are aligned.

Conversion tools: `scripts/split_pubtator.py` (train/dev/test splits), `scripts/to_pubtator.py` (filter
PubTator, convert BioC XML), `scripts/dict_to_pubtator.py` (CTD dictionaries as PubTator training rows).
Dataset recipes are in `scripts/README.md`.

## Evaluation

```bash
.venv/bin/python scripts/evaluate.py -xmodel_path outputs/saved_trees/<run> -test_path <file.pubtator>
```

Reports acc@1, MRR, recall@k, candidate recall (`R@cand`) and vocabulary coverage. Metrics are computed on
mentions whose gold label has training text (coverage is reported); `acc@1_all` counts the others as misses.
`-beam_size`, `-topk` and `-knn_beta` override the saved search settings. `topk 0` keeps every candidate
(at most 100 per visited leaf); with `topk > 0`, `R@cand` also counts ranking losses.

`scripts/run_experiment.py -spec configs/experiments/<name>.json` trains and evaluates one spec and records
the run (spec, metrics, git state, input hashes, package versions) under `outputs/runs/`. Recorded results
are in `docs/results.md`. Each `metrics_<split>.json` also records `eval_protocol` (metric definitions),
`population_sha256` (evaluated rows) and, with an eval `train_path`, `strings_sha256` (seen/unseen groups and
dictionary answers); the TUI compares evaluations only through these. An input file that changes during a run
fails it before metrics are written.

Browse saved trees, their evaluations and active runs in the terminal (optional `tui` extra; reads
`outputs/runs/` and `outputs/saved_trees/` only, never loads a tree; writes nothing except when it cancels a run):

```bash
uv sync --locked --extra tui
.venv/bin/python -m xmr4el_tui            # or xmr4el-tui; --root <repo> when started elsewhere
```

Dashboard: Explorer (Models > dataset > tree > evaluation, then runs without a saved tree), Description of
the selected item, Active runs. `enter` describes, `]` / `[` (or `ctrl+tab`) cycle panels, `ctrl+b` hides the
explorer, `<` / `>` resize it, `ctrl+r` rereads every run file, `?` lists the keys, `ctrl+c` twice quits.
Mouse drag selects text and copies it. It rescans every 5 s. `View Logs` in the Description opens a run's log.

Run Monitor (`enter` on an active run, `right` in the Description): Run overview, Pipeline (stages, parts,
hierarchy layers), Resources of the selected stage, Live logs (`/` search, `n` / `N`, `l` level, `f`
fullscreen). `c` follows the current stage again, `esc` returns. Cancel Run (`s`) sends SIGTERM after a
confirmation, only to a run this TUI (or an earlier one) launched; runs started from a shell are monitored only.
Results (`r`, or `left` in the Description, on an evaluation with metrics) compares it, as the baseline, with up to
three other evaluations of the same test file (VERIFIED / CAUTION / INCOMPATIBLE, deltas only where the
recorded evidence allows them). New Experiment (`n`) starts with Dataset Setup: datasets recognized from
`configs/experiments/*.json` and `datasets/`, or registered (saved in `outputs/tui/datasets.json`); `enter`
configures one, the training file and Development / Test files can be changed with a file picker, `ctrl+s` confirms
the dataset for the experiment (launching from New Experiment is planned).

## Configuration

Model configs are JSON: `configs/xmr4el_base_config.json` (MedMentions) and
`configs/xmr4el_bc5cdr_config.json` (BC5CDR, also used for NCBI-disease, BioRED and NLM-Chem).

| Key | Meaning |
|---|---|
| `features` | `"sapbert_char_context"` (PubTator) or `"tfidf"` (plain text) |
| `transformer_config` | `type` `sapbert`, `sentencetbiobert` or `biobert`, or `kwargs.model_name` (HF ID or local directory); optional `pooling` (`cls` / `mean`), `max_seq_length`, `revision` |
| `abbrev_expansion` | `"append"` / `"replace"`: expand in-document abbreviations in mentions |
| `context_window` | words on each side of the mention used as context |
| `vectorizer_config`, `context_vectorizer_config` | TF-IDF of the mention / context |
| `dimension_config`, `context_dimension_config` | SVD of those blocks (768 columns each in both configs) |
| `clustering_config` | `balancedkmeans`: `n_clusters` per node, `iter_limit`, `seed` |
| `depth`, `min_leaf_size` | fixed tree depth: every internal node is split; clusters below `min_leaf_size` labels are merged into the nearest larger one, and training fails if a node cannot form two clusters of that size (small datasets: lower `depth`, `min_leaf_size` or `n_clusters`) |
| `matcher_config` | `jointlogisticregression`: `C`, `class_weight`, `tol`, `max_iter` |
| `predict_config` | default search: `beam_size`, `topk`, `knn_beta`; saved with the tree |

Training resolves the transformer `revision` (default `main`) to its commit hash, offline to the locally
cached one; the saved tree keeps the hash, so queries are encoded by the trained checkpoint. Embeddings are
cached in `outputs/cache/transformers/` keyed by model, revision, pooling, length, dtype and texts;
`XMR4EL_TRANSFORMER_CACHE=0` turns the cache off.

## Docker

For a remote server with an NVIDIA GPU (NVIDIA Container Toolkit), build the dependency image once:

```bash
docker build -f xmr4el.dockerfile -t xmr4el .
```

It installs the locked dependencies in `/opt/venv`; the code is mounted from the checkout at `/app`.
Rebuild only after `pyproject.toml`, `uv.lock` or the Dockerfile changes. Run training detached (drop
`--gpus all` for CPU):

```bash
mkdir -p "$HOME/.cache/huggingface" && \
docker run -d --name xmr4el-train --init --gpus all --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface \
  xmr4el python3 scripts/train.py -train_path <file.pubtator> -model_config configs/xmr4el_bc5cdr_config.json && \
docker logs -f xmr4el-train
```

The Hugging Face mount keeps downloaded models between runs; `--user` keeps outputs owned by you.
`docker wait xmr4el-train` prints the exit code (0 = success); `docker rm xmr4el-train` before the next run.
For an interactive shell, replace `-d` with `-it` and the command with `/bin/bash`.

## Limitations

- **Mentions must be given.** There is no mention detection.
- **Labels need training text.** A label without a training row has no PIFA embedding and is never predicted;
  add dictionary names (`scripts/dict_to_pubtator.py`) to cover more labels.
- **Dense memory.** Features, PIFA label embeddings and matcher weights are dense float32. With the provided
  configs the features have at most 2304 columns (SVD). A TF-IDF block without SVD (its `n_components` above
  the vocabulary size) keeps the features, matchers and kNN index sparse, but the label embeddings stay labels x features and each node's weights features x
  children, so reduce the feature dimension for large vocabularies.
- **GPU runs are not bit-reproducible.** Seeds are fixed, but retraining on GPU moves a few predictions.
- **Old saved trees are not migrated.** A tree saved by an older layout does not load; retrain it.

## Security

Saved trees are Python pickles: `XModel.load` can run arbitrary code from the files it loads. Load only trees
you created or trust. XMR4EL itself never calls `torch.load`, `torch.jit` or `torch.export`; transformer
weights load through Hugging Face `transformers`. Model downloads come from the Hugging Face Hub; pin
`revision` in the config to fix the checkpoint.

## Development

Checks on synthetic data (no corpus needed):

```bash
.venv/bin/python -m pytest -q && \
uvx ruff check xmr4el scripts tests && \
.venv/bin/python scripts/evaluate.py -selfcheck && \
.venv/bin/python scripts/diagnose_routing.py -selfcheck && \
.venv/bin/python scripts/diagnose_unseen.py -selfcheck && \
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
```

CI (`.github/workflows/ci.yml`) runs ruff, pytest and an installed-wheel smoke test on CPU.

Layout:

- `xmr4el/data/`: PubTator / TSV readers, splits and export.
- `xmr4el/features/`: text encoder, TF-IDF, SVD, transformer embeddings, PIFA.
- `xmr4el/hierarchy/`: clustering, tree training, beam search, persistence.
- `xmr4el/learning/`: one-vs-rest logistic matchers, mention-kNN scoring.
- `xmr4el/xmodel.py`: public API (`XModel.train`, `predict`, `save`, `load`); `xmr4el/eval.py`: metrics.
- `scripts/`: train, evaluate, experiment runner, data conversion, diagnostics, baselines (`scripts/README.md`).
- `configs/`: model configs and experiment specs. [Documentation](docs/README.md): architecture,
  pipeline notes, results, interface design, and historical audits.

The archived [architecture assessment](docs/archive/architecture.md) records the 2026-10-23 scope, cleanup
decisions and remaining risks. TF-IDF, SVD, balanced k-means and joint
logistic regression use explicit adapters; their config `type` names are validated. There is no backend
plugin registry. This cleanup changes internal saved-model layouts: retrain previous trees.

## License

MIT; see `LICENSE`.
