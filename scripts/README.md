# Scripts

Run commands from the repository root with Python 3.12+ and the project environment:

```bash
uv sync --locked
source .venv/bin/activate
```

See the [project README](../README.md) for Docker setup and model configuration.
Inputs live under `data/` or `datasets/`; saved trees live under `outputs/saved_trees/`.
Replace the example corpus paths and `<run>` with your local files and saved tree directory.

Current checkout: `xmr4el/xmodel.py` and `xmr4el/eval.py` are missing. Restore those
modules before running training, evaluation, diagnostics, screens, or baseline export/scoring.
The split CLI does not depend on them.

## Available scripts

| Script | Purpose |
| --- | --- |
| `split_pubtator.py` | Preprocessing: split a PubTator corpus by PMID. |
| `train.py` | Group texts by label, encode features, build label embeddings and a hierarchy, then train matchers/rankers and save the model. |
| `evaluate.py` | Traverse a saved tree and report ranking metrics and candidate recall. |
| `diagnose_routing.py` | Compare root matcher, cosine routing, flat retrieval, and a mention dictionary on train/dev inputs. |
| `experiments/screen_features.py` | Compare feature variants with flat label retrieval, without training a tree. |
| `experiments/screen_leaf_scorer.py` | Fit and compare leaf scorers with oracle routing using exported tree features. |
| `experiments/beam_sweep.py` | Repeat evaluation for beam sizes in steps of five. |
| `baselines/pecos_compare.py` | Export aligned training/dev features and score PECOS predictions. |
| `baselines/pecos_run.py` | Train PECOS XR-Linear on exported inputs and save predictions. |

## Split PubTator inputs

```bash
python scripts/split_pubtator.py \
  --input datasets/corpus_pubtator.txt \
  --outdir datasets/splits \
  --train_ratio 0.8 --dev_ratio 0.1 --test_ratio 0.1 \
  --emit_jsonl
```

Writes `corpus_pubtator_train.txt`, `corpus_pubtator_dev.txt`, and
`corpus_pubtator_test.txt`; `--emit_jsonl` also writes `train.jsonl`, `dev.jsonl`, and
`test.jsonl`. To use official splits, provide all three PMID-list options:
`--train_pmids`, `--dev_pmids`, and `--test_pmids`.

## Train

PubTator annotations are grouped by label; each text combines the mention and context.

```bash
python scripts/train.py \
  -train_path datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt \
  -model_config configs/xmr4el_base_config.json \
  -ds_len 500
```

`-ds_len` takes the first N label groups in input order, not a random sample.
The default feature configuration uses transformer embeddings and may download a model.
Saved trees are written to `outputs/saved_trees/xmodel_<timestamp>/`.

For grouped TSV input (`group_id<TAB>text`), supply one label per line aligned
with sorted group IDs and use `emb_flag` 1:

```bash
python scripts/train.py \
  -train_path data/train.tsv -labels_path data/labels.txt \
  -model_config configs/xmr4el_flag1_config.json
```

## Evaluate and diagnose a saved tree

Define reusable paths for the remaining examples:

```bash
MODEL="outputs/saved_trees/<run>"
TRAIN=datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
DEV=datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt
```

For the base configuration's logistic matchers, use matcher probabilities with
routing path probabilities:

```bash
python scripts/evaluate.py \
  -xmodel_path "$MODEL" -test_path "$DEV" \
  -beam_size 5 -topk 20 -alpha 0 -path_score \
  -train_path "$TRAIN"
```

Reports acc@1, MRR, recall@k, candidate recall, and vocabulary coverage.
Only mentions with a gold label in the training vocabulary enter ranking metrics.
`-train_path` adds the seen/unseen mention-string breakdown; `-verbose` shows library output.
Use `-alpha 1 -scorer cosine` for cosine leaf scores or `-alpha 1 -scorer ranker`
to evaluate trained rankers. `-topk` is per leaf; `0` disables that final cut.

Diagnose routing before retraining; compare routing accuracy with the printed chance line:

```bash
python scripts/diagnose_routing.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -max_rows 5000
```

## Experiments

Compare flat retrieval features and centroid versus maximum training-row cosine scoring:

```bash
python scripts/experiments/screen_features.py \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" \
  -features 'sapbert+charsvd,sapbert+charsvd+ctxwin*0.5' \
  -scorers centroid,max -abbrev none,append -show_errors 10
```

For leaf scoring, first export features using the baseline export command below:

```bash
python scripts/experiments/screen_leaf_scorer.py \
  -xmodel_path "$MODEL" -data "$EXPORT" \
  -scorers cosine,logreg,logreg_bal,svm
```

This measures within-leaf accuracy with perfect routing, not end-to-end accuracy.

Sweep beams 5, 10, 15, 20, and 25 with 20 results per leaf:

```bash
python scripts/experiments/beam_sweep.py "$MODEL" "$DEV" 5 25 20
```

The sweep uses `evaluate.py` defaults (`-alpha 1`, cosine scoring, no path score);
it does not accept scorer/fusion overrides.

## PECOS baseline

Export features using the exact PubTator training file used for the saved tree:

```bash
EXPORT=outputs/pecos/comparison
python scripts/baselines/pecos_compare.py export \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" -out "$EXPORT"
```

Export writes sparse training/dev features, training labels, dev gold indices, and texts.
`pecos_run.py` additionally requires PECOS, which is not in the project dependencies;
run it in a PECOS-enabled Linux/Docker environment with access to the export directory.

```bash
python scripts/baselines/pecos_run.py \
  -data "$EXPORT" -features ours,tfidf -beams 2,10 -topk 100
python scripts/baselines/pecos_compare.py score -out "$EXPORT"
```

`ours` uses the tree's exported features; `tfidf` trains PECOS's own TF-IDF on the
same texts. Prediction files are named `pred_<features>_b<beam>.npz`.

## Checks without corpus training

```bash
python scripts/evaluate.py -selfcheck
python scripts/diagnose_routing.py -selfcheck
python scripts/experiments/screen_features.py -selfcheck
python scripts/experiments/screen_leaf_scorer.py -selfcheck
python scripts/baselines/pecos_compare.py selfcheck
```

For CLI options, use `python scripts/<script>.py --help`; the beam sweep uses the
five positional arguments shown above. See [tests](../tests/README.md) for pytest commands.
