# XMR4EL

Extreme multi-label ranking for entity linking over large label spaces.
The model trains on texts grouped by label and predicts ranked label IDs.

## Pipeline

1. Load local PubTator annotations or grouped TSV texts and a label file.
2. Encode texts with TF-IDF, transformer embeddings, or both.
3. Build PIFA label embeddings from training features.
4. Cluster labels into a hierarchy.
5. Train cluster matchers and label-level leaf matchers with per-label rankers.
6. Traverse the hierarchy and return ranked label scores.

## Setup

Python 3.12 or later is required. On macOS, install `uv` and the LightGBM
OpenMP runtime with `brew install uv libomp`, then run:

```bash
uv sync --locked
source .venv/bin/activate
```

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.
Transformer encoding uses CUDA when available, otherwise CPU. Its first use
may download the configured pretrained model.

### Run on another server with Docker

Clone this repository on the server, enter its directory, and build the dependency image:

```bash
docker build -f xmr4el.dockerfile -t xmr4el .
```

The image installs Python 3.12 and dependencies from `uv.lock` in `/opt/venv`.
Source code comes from the server checkout mounted at `/app`; the host `.venv` is unused.
The Linux x86-64 lock includes large CUDA packages, so the first build can take time and disk
space even when running on CPU.

Copy your datasets into the checkout's `datasets/` or `data/` directory on the server;
Git does not include them. Only the base config is tracked, so copy any local experiment
config separately. First check the mounted code:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  xmr4el python test/xmr4el/test_data_loading.py
```

For GPU access, the server needs an NVIDIA GPU and NVIDIA Container Toolkit configured.
The commands below use `--gpus all`; omit it for CPU-only runs.

#### Interactive container

Open a reusable interactive container:

```bash
mkdir -p "$HOME/.cache/huggingface"
docker run -it --name xmr4el-shell --init --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  xmr4el /bin/bash
```

Inside the container, start training:

```bash
python test/xmr4el/test_train_pipeline.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config .models/xmr4el_base_config.json -ds_len 500
```

After `exit`, reopen the same container with `docker start -ai xmr4el-shell`.

#### Detached training

Start training detached so disconnecting SSH does not stop it:

```bash
mkdir -p "$HOME/.cache/huggingface"
docker run -d --name xmr4el-train --init --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  xmr4el python test/xmr4el/test_train_pipeline.py \
    -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
    -model_config .models/xmr4el_base_config.json -ds_len 500

docker logs -f xmr4el-train
```

Saved trees appear on the server under `test/test_data/saved_trees/`. The cache mount preserves
downloaded transformer models; the user mapping keeps output files owned by your server user.
No Docker CPU limit is set.

After the run finishes, update code and start a fresh container:

```bash
docker wait xmr4el-train  # prints the training exit code; 0 means success
docker rm xmr4el-train
git pull --ff-only
```

Repeat the training command. Code-only updates need no rebuild; rebuild the image if
`pyproject.toml`, `uv.lock`, or the Dockerfile changes. Pull updates between runs so a running
experiment uses a consistent checkout. Evaluation uses the same mount and image with
`python test/xmr4el/test_evaluate_pipeline.py` and the arguments below.

## Local inputs

Keep input corpora and label files under `data/` or `datasets/`.

### PubTator

The loader reads title, abstract, and annotation lines. Each annotation becomes
`mention [SEP] title + abstract` paired with the label ID in column six.
Training groups these examples by label. Prediction uses supplied mentions;
mention detection is not implemented.

```bash
python test/xmr4el/test_train_pipeline.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config .models/xmr4el_base_config.json \
  -ds_len 500
```

`-ds_len` limits the number of label groups, not documents or mentions.
Saved models go to `test/test_data/saved_trees/xmodel_<timestamp>/`.

### Grouped TSV

Training rows are `group_id<TAB>text`. A separate file lists one label ID per
line, aligned with the sorted group IDs. Use a copy of the base configuration
with `emb_flag` `1` for plain text without `[SEP]` (`.models/xmr4el_flag1_config.json`).

```bash
python test/xmr4el/test_train_pipeline.py \
  -train_path data/train/chemical/train_Chemical.txt \
  -labels_path data/train/chemical/labels.txt \
  -model_config .models/xmr4el_flag1_config.json \
  -ds_len 500
```

The loader rejects too few labels and truncates excess labels to the number of
groups; check that the files are aligned before training.

## Evaluation

```bash
python test/xmr4el/test_evaluate_pipeline.py \
  -xmodel_path test/test_data/saved_trees/<run> \
  -test_path datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt \
  -beam_size 5 -topk 20
```

Evaluation reports acc@1, MRR, recall@k, candidate recall, and vocabulary coverage.
It excludes mentions whose gold labels are absent from training, so metrics are
conditional on vocabulary coverage. `topk` is per leaf; `0` removes that final
cut but the leaf matcher still has an internal 100-candidate limit.

For prediction from Python:

```python
from xmr4el.xmr.model import XModel

model = XModel.load("test/test_data/saved_trees/<run>")
routes, scores = model.predict(
    ["mention [SEP] context"], beam_size=5, topk=20, topk_mode="per_leaf"
)
```

The `per_leaf` mode returns hierarchy scores. The alternate `global` path in
`XModel.predict` reranks retrieved labels with cosine similarity.

## Configuration

`.models/xmr4el_base_config.json` selects component types and parameters.
`emb_flag` controls features:

- `1`: TF-IDF (-> dimension model) of the whole input (`.models/xmr4el_flag1_config.json`).
- `6`: input `mention [SEP] context`; three blocks, each L2-normalised before the concat:
  transformer(mention), char TF-IDF -> SVD(mention), TF-IDF -> SVD(context window).

Transformer: `transformer_config.type` `sapbert`, `sentencetbiobert` or `biobert`, or any
checkpoint with `"kwargs": {"model_name": "<HF id>"}` (optional `pooling`, e.g. `"cls"`, and
`max_seq_length`). `abbrev_expansion` `"append"` expands in-document abbreviations in PubTator input.

The default uses flag 6 with SapBERT, abbreviation expansion, balanced k-means, and one-vs-rest
L2 logistic-regression matchers (`jointlogisticregression`: liblinear's objective, all labels of a node solved at once). Evaluate such trees with `-alpha 0 -path_score`
(leaf matcher probability x routing path probability).
Use the available sklearn, FAISS, and PyTorch clustering/classifier wrappers
through the existing configuration registries.

## Layout

- `xmr4el/featurization/`: local readers, text encoders, and label embeddings.
- `xmr4el/clustering/`: label hierarchy construction.
- `xmr4el/matcher/`: cluster and leaf-label classifiers.
- `xmr4el/ranker/`: per-label scoring and negative sampling.
- `xmr4el/models/`: component wrappers and registries.
- `xmr4el/xmr/`: training, traversal, persistence, and the `XModel` API.
- `xmr4el/utils/`: temporary array storage and local PubTator splitting.
- `test/`: train/evaluate scripts and small regression checks.
- `STATUS.md`: current state, next step. `docs/pipeline.md`: implemented behavior, observations. `docs/status_log.md`: history. `docs/results.md`: results.
- `data/`, `datasets/`: local inputs, excluded from version control.

## Checks

These checks use synthetic inputs and do not run corpus training:

```bash
python test/xmr4el/test_data_loading.py
python -m xmr4el.clustering.train
python -m xmr4el.matcher.train
python -m xmr4el.ranker.train
python test/xmr4el/test_evaluate_pipeline.py -selfcheck
python test/xmr4el/diagnose_routing.py -selfcheck
```

The user runs full training and evaluation. Keep experiment configurations and
splits fixed when comparing changes; see `docs/results.md` for invalidated and
post-fix runs.
