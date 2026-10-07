# Full-vocabulary XMR4EL / PECOS XR-Linear comparison

Starting configuration, not a full-scale validated result. Train on every label observed in
MedMentions **st21pv/train**; neither model learns labels absent from that training split.
Assumes Linux x86-64, NVIDIA CUDA access through Docker, and the dataset in this checkout.

`configs/xmr4el_full_cuda_config.json` retains the current features and L2 logistic matchers.
Changes from the base: SapBERT batch 256, GPU balanced clustering, eight branches at each internal
node, no branch halving, and four layers (three cluster layers plus label leaves). With balanced
splits this gives up to 512 leaves, rather than six large leaves. Actual sizes are logged;
`max_leaf_size` does not enforce a leaf-size cap in XMR4EL. Rankers remain disabled.

CUDA is used for SapBERT and XMR4EL clustering. SVD and logistic matchers use CPU/RAM;
PECOS XR-Linear uses CPU. The feature matrices also occupy system RAM, so CUDA alone does not
guarantee the full run fits. Batch 256 is a starting value, not a VRAM guarantee; lower it if needed.

Both systems receive the same training rows, label columns, SapBERT/character/context features,
and evaluation rows. Their clustering, training objectives and search work differ: equal beam sizes
do not mean equal compute. Record accuracy, MRR, recall@k, seen/unseen accuracy, vocabulary coverage,
candidate counts, encoding time, classifier training time, prediction time and peak RAM separately.
PECOS's reported training time excludes the shared feature encoding; XMR4EL's CLI total includes it.

## Build and enter the CUDA container

Run from the repository root on the CUDA machine:

```bash
docker build -f xmr4el.dockerfile -t xmr4el .
docker build -f pecos.dockerfile -t xmr4el-pecos .
mkdir -p outputs/hf-cache
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  -e HF_HOME=/app/outputs/hf-cache \
  xmr4el /bin/bash
```

## Train XMR4EL and compare on dev

Inside that container:

```bash
set -euo pipefail
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

TRAIN=datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
DEV=datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt
EXPORT=outputs/pecos/full_dev

# Omit -ds_len to keep every training label (CLI default: 10,000,000).
python3 scripts/train.py -train_path "$TRAIN" \
  -model_config configs/xmr4el_full_cuda_config.json

# Use the tree just trained; do not run another training concurrently.
MODEL=$(python3 -c 'from pathlib import Path; print(max(Path("outputs/saved_trees").glob("xmodel_*"), key=lambda p: p.stat().st_mtime))')
printf 'Model: %s\n' "$MODEL"

for BEAM in 2 10; do
  python3 scripts/evaluate.py -xmodel_path "$MODEL" \
    -test_path "$DEV" -train_path "$TRAIN" \
    -beam_size "$BEAM" -topk 0 -alpha 0 -path_score
done

python3 scripts/baselines/pecos_compare.py export \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$DEV" -out "$EXPORT"
exit
```

Back on the host, train PECOS once on the exported features and score both beams:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" -w /app \
  xmr4el-pecos python3 scripts/baselines/pecos_run.py \
  -data outputs/pecos/full_dev -features ours \
  -nr_splits 8 -max_leaf_size 100 -beams 2,10 -topk 100 -threshold 0

docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  xmr4el python3 scripts/baselines/pecos_compare.py score -out outputs/pecos/full_dev
```

`-threshold 0` disables XR-Linear weight pruning (script default 0.1, PECOS's default); on BC5CDR dev
the default alone cost PECOS 0.075 acc@1 (`docs/results.md`).
PECOS is saved under `outputs/pecos/full_dev/model_ours`. Its hierarchy automatically grows
with the label count; XMR4EL uses the configured fixed depth. Compare the same-features `ours`
baseline first. `-features ours,tfidf` also runs PECOS's own TF-IDF as a separate system baseline.

## Final test without retraining

Select the beam on dev, then freeze the configuration. Re-enter the CUDA container using the
command above. Replace the model path and beam below with the chosen values:

```bash
MODEL=outputs/saved_trees/xmodel_REPLACE_WITH_YOUR_RUN
BEAM=2
TRAIN=datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
TEST=datasets/MedMentions/st21pv/data/corpus_pubtator_test.txt
python3 scripts/evaluate.py -xmodel_path "$MODEL" \
  -test_path "$TEST" -train_path "$TRAIN" \
  -beam_size "$BEAM" -topk 0 -alpha 0 -path_score
python3 scripts/baselines/pecos_compare.py export \
  -xmodel_path "$MODEL" -train_path "$TRAIN" -test_path "$TEST" \
  -out outputs/pecos/full_test
exit
```

On the host, set the selected beam and predict with the saved PECOS model:

```bash
BEAM=2
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" -w /app -e BEAM="$BEAM" \
  xmr4el-pecos python3 -c '
import os
from scipy.sparse import load_npz, save_npz
from pecos.xmc.xlinear.model import XLinearModel
beam = int(os.environ["BEAM"])
model = XLinearModel.load("outputs/pecos/full_dev/model_ours")
pred = model.predict(load_npz("outputs/pecos/full_test/X_dev.npz"), beam_size=beam, only_topk=100)
save_npz(f"outputs/pecos/full_test/pred_ours_b{beam}.npz", pred)
'
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/app" \
  xmr4el python3 scripts/baselines/pecos_compare.py score -out outputs/pecos/full_test
```

Export filenames still say `dev` even when `-test_path` points at test. Metrics exclude gold labels
absent from training; report kept/total rows and coverage times acc@1 alongside conditional accuracy.
The unseen-string group uses lower-cased mention keys after abbreviation expansion, within the
training vocabulary. Retain the same keys for both systems.

## Offline checks

```bash
.venv/bin/python -m json.tool configs/xmr4el_full_cuda_config.json >/dev/null
JOBLIB_MULTIPROCESSING=0 .venv/bin/python -m pytest \
  tests/integration/test_config_and_imports.py \
  tests/integration/test_pipeline_persistence.py \
  tests/features/ tests/learning/ tests/hierarchy/ tests/test_eval.py -q
.venv/bin/python scripts/baselines/pecos_compare.py selfcheck
```

These checks do not validate CUDA hardware or full-corpus memory use. Use a freshly trained tree
for this comparison.
