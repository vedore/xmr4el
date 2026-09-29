#!/usr/bin/env bash
# Seeded ablation matrix on st21pv: features (emb_flag 1 vs 4) x label space (500 vs 1000).
# All code fixes on; clustering is seeded, so reruns are identical and deltas are attributable.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
TRAIN=datasets/MedMentions/st21pv/data/corpus_pubtator_train.txt
DEV=datasets/MedMentions/st21pv/data/corpus_pubtator_dev.txt
OUT=test/test_data/ablation
mkdir -p "$OUT"

for spec in "1 500" "4 500" "1 1000" "4 1000"; do
  set -- $spec; FLAG=$1; DSLEN=$2
  TAG="flag${FLAG}_ds${DSLEN}"
  CFG="$OUT/config_${TAG}.json"
  $PY -c "import json; c=json.load(open('.models/xmr4el_base_config.json')); c['emb_flag']=$FLAG; json.dump(c, open('$CFG','w'), indent=4)"

  echo "=== $TAG :: train ==="
  $PY test/xmr4el/test_train_pipeline.py -train_path "$TRAIN" -model_config "$CFG" -ds_len "$DSLEN" 2>&1 | tee "$OUT/train_$TAG.log"

  TREE=$(ls -td test/test_data/saved_trees/*/ | head -1)
  echo "=== $TAG :: eval ($TREE) ==="
  $PY test/xmr4el/test_evaluate_pipeline.py -xmodel_path "$TREE" -test_path "$DEV" -beam_size 5 -topk 20 2>&1 | tee "$OUT/eval_$TAG.log"
done

echo; echo "=== summary ==="
grep -H -E "In-vocabulary|^acc@1|^MRR|^recall@(5|20|100|candidates)" "$OUT"/eval_*.log
grep -h "reassigned" "$OUT"/train_*.log | sort | uniq -c
