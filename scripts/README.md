# Scripts

Run commands from the repository root with Python 3.12+ and the project environment
(`.venv/bin/python` locally; `python3` inside the Docker containers):

```bash
uv sync --locked
```

See the [project README](../README.md) for Docker setup and model configuration.
Inputs live under `data/` or `datasets/`; saved trees live under `outputs/saved_trees/`.
Replace the example corpus paths and `<run>` with your local files and saved tree directory.

## Available scripts

| Script | Purpose |
| --- | --- |
| `split_pubtator.py` | Preprocessing: split a PubTator corpus by PMID. |
| `dict_to_pubtator.py` | Preprocessing: CTD vocabulary (MEDIC/chemicals) -> PubTator pseudo-documents. |
| `train.py` | Group texts by label, encode features, build label embeddings and a hierarchy, then train matchers and save the model. |
| `evaluate.py` | Traverse a saved tree and report ranking metrics and candidate recall. |
| `diagnose_routing.py` | Compare root matcher, cosine routing, flat retrieval, and a mention dictionary on train/dev inputs. |
| `diagnose_unseen.py` | Unseen-string rows: gold-rank buckets, flat 1-NN per feature block, knn fusion screen over beta, error TSV. |
| `experiments/screen_features.py` | Compare feature variants with flat label retrieval, without training a tree. |
| `experiments/screen_leaf_scorer.py` | Fit and compare leaf scorers with oracle routing using exported tree features. |
| `experiments/beam_sweep.py` | Repeat evaluation for beam sizes in steps of five. |
| `baselines/pecos_compare.py` | Export aligned training/dev features and score PECOS predictions. |
| `baselines/pecos_run.py` | Train PECOS XR-Linear on exported inputs and save predictions. |

## Split PubTator inputs

```bash
.venv/bin/python scripts/split_pubtator.py \
  --input datasets/corpus_pubtator.txt \
  --outdir datasets/splits \
  --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1 \
  --emit_jsonl
```

Writes `corpus_pubtator_train.txt`, `corpus_pubtator_dev.txt`, and
`corpus_pubtator_test.txt`; `--emit_jsonl` also writes `train.jsonl`, `dev.jsonl`, and
`test.jsonl`. To use official splits, provide all three PMID-list options:
`--train_pmids`, `--dev_pmids`, and `--test_pmids`.

## Convert a CTD vocabulary

```bash
.venv/bin/python scripts/dict_to_pubtator.py \
  -ctd_path CTD_diseases.tsv.gz -out datasets/CTD/ctd_disease.pubtator -type Disease
```

Each name/synonym becomes a one-mention PubTator pseudo-document; only `MESH:` ids are kept
(prefix stripped). Append the output to a PubTator training file to train on dictionary names.

## Train

PubTator annotations are grouped by label; each text combines the mention and context.

```bash
.venv/bin/python scripts/train.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config configs/xmr4el_base_config.json \
  -ds_len 500
```

`-ds_len` takes the first N label groups in input order, not a random sample.
The default feature configuration uses transformer embeddings and may download a model.
Saved trees are written to `outputs/saved_trees/xmodel_<timestamp>/`.

Logging defaults to timestamped INFO summaries: input/config paths, selected groups,
feature and label-matrix shapes, transformer model/device, hierarchy layers/nodes,
elapsed seconds, and the saved model path. `-verbose` adds DEBUG batch, per-label,
and cluster-size diagnostics; `-quiet` keeps warnings and errors. K-means progress
bars default to DEBUG unless `tqdm_flag` is explicitly configured.
Logs contain counts and settings, not training texts or embedding arrays.
The saved tree retains its model configuration; evaluation prints metrics separately.

Capture a run using the existing shell tools:

```bash
mkdir -p outputs/logs
.venv/bin/python scripts/train.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config configs/xmr4el_base_config.json 2>&1 | tee outputs/logs/train.log
```

For Python callers, configure a handler with `logging.basicConfig(level=logging.INFO)`.
The library uses a `NullHandler` and does not configure the application's console.
`xmr4el.set_verbosity(0/1/2)` selects WARNING/INFO/DEBUG.
This follows [Python's logging guidance](https://docs.python.org/3/howto/logging.html)
and the experiment record of [parameters, metrics, and artifacts](https://mlflow.org/docs/latest/tracking/),
using the existing logs, evaluation report, and saved tree without an extra tracking dependency.

For grouped TSV input (`group_id<TAB>text`), supply one label per line aligned
with sorted group IDs and a config with `"features": "tfidf"` (README § Grouped TSV):

```bash
.venv/bin/python scripts/train.py \
  -train_path data/train.tsv -labels_path data/labels.txt \
  -model_config configs/local_tfidf_config.json
```

## Evaluate and diagnose a saved tree

Define reusable paths for the remaining examples:

```bash
MODEL="outputs/saved_trees/<run>"
TRAIN=datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
DEV=datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt
```

Leaf score = leaf matcher probability x routing path probability (`HierarchicalMLModel.predict`);
search settings default to the tree's `predict_config` (`-beam_size`, `-topk`, `-knn_beta` override):

```bash
.venv/bin/python scripts/evaluate.py \
  -xmodel_path "$MODEL" -test_path "$DEV" \
  -train_path "$TRAIN"
```

Reports acc@1, MRR, recall@k, candidate recall, and vocabulary coverage.
Only mentions with a gold label in the training vocabulary enter ranking metrics.
`-train_path` adds the seen/unseen mention-string breakdown; `-verbose` shows library diagnostics.
Warnings (including OOM recovery) remain visible in quiet evaluation.
`-topk k` keeps each query's `k` best labels (default 0 = all candidates); `-knn_beta` (default 0) is the
mention-kNN fusion.

Diagnose routing before retraining; compare routing accuracy with the printed chance line:

```bash
.venv/bin/python scripts/diagnose_routing.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -max_rows 5000
```

## Experiments

Compare flat retrieval features and centroid versus maximum training-row cosine scoring:

```bash
.venv/bin/python scripts/experiments/screen_features.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -features 'sapbert+charsvd,sapbert+charsvd+ctxwin*0.5' \
  -scorers centroid,max -abbrev none,append -show_errors 10
```

For leaf scoring, first export features with the PECOS baseline export command below
(`-data` defaults to `outputs/pecos/<tree name>`):

```bash
.venv/bin/python scripts/experiments/screen_leaf_scorer.py \
  -xmodel_path "$MODEL" -data "$EXPORT" \
  -scorers cosine,logreg,logreg_bal,svm
```

This measures within-leaf accuracy with perfect routing, not end-to-end accuracy.

Sweep beams 5, 10, 15, 20, and 25 with the 20 best labels per query:

```bash
.venv/bin/python scripts/experiments/beam_sweep.py "$MODEL" "$DEV" 5 25 20
```

The sweep uses the tree's `predict_config` for `-knn_beta`.

## PECOS baseline

Export features using the exact PubTator training file used for the saved tree:

```bash
EXPORT=outputs/pecos/comparison
.venv/bin/python scripts/baselines/pecos_compare.py export \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" -out "$EXPORT"
```

Export writes sparse training/dev features, training labels, dev gold indices, and texts.
`pecos_run.py` additionally requires PECOS, which is not in the project dependencies;
run it in a PECOS-enabled Linux/Docker environment with access to the export directory.

```bash
python3 scripts/baselines/pecos_run.py \
  -data "$EXPORT" -features ours,tfidf -beams 2,10 -topk 100 -threshold 0
.venv/bin/python scripts/baselines/pecos_compare.py score -out "$EXPORT"
```

`ours` uses the tree's exported features; `tfidf` trains PECOS's own TF-IDF on the
same texts. Prediction files are named `pred_<features>_b<beam>.npz`.
`-threshold` is XR-Linear weight pruning (default 0.1, PECOS's default); on BC5CDR dev the
default alone cost PECOS 0.075 acc@1 (`docs/results.md`), so comparisons use `0`.
`-nr_splits` (default 8) and `-max_leaf_size` (default 100) set PECOS's hierarchy.

## Checks without corpus training

```bash
.venv/bin/python scripts/evaluate.py -selfcheck
.venv/bin/python scripts/diagnose_routing.py -selfcheck
.venv/bin/python scripts/experiments/screen_features.py -selfcheck
.venv/bin/python scripts/experiments/screen_leaf_scorer.py -selfcheck
.venv/bin/python scripts/baselines/pecos_compare.py selfcheck
```

For CLI options, use `.venv/bin/python scripts/<script>.py --help`; the beam sweep uses the
five positional arguments shown above. See [tests](../tests/README.md) for pytest commands.
