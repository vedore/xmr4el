#!/usr/bin/env bash
# 1-NN baselines only (scripts/baselines/nn_baselines.py), eval-only on each G1 set's latest plain run tree
# (host shell, xmr4el image), dev + test -> outputs/nn/<set>_<split>_nn.{json,log}. Bundle outputs/nn.tar.gz.
# Usage: bash scripts/run_nn.sh [set ...]
set -euo pipefail
mkdir -p outputs/nn "$HOME/.cache/huggingface"
XMR=(docker run --rm --init --gpus all --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app
     --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface xmr4el python3)
field() { python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$@"; }
for s in ${*:-ncbi_disease biored_disease bc5cdr_chemical biored_chemical nlm_chem}; do
  run=$(ls -d outputs/runs/*_"$s" | grep -E "/[0-9-]{10}_[0-9-]{8}_$s\$" | sort | tail -1)  # not i1_/lc_ eval runs
  tree=$(field "$run/tree.json" 's["xmodel_path"]')
  for split in dev test; do
    spec=configs/experiments/$s.json; [ "$split" = test ] && spec=configs/experiments/${s}_test.json
    "${XMR[@]}" scripts/baselines/nn_baselines.py -xmodel_path "$tree" \
      -test_path "$(field "$spec" 's["evals"][0]["test_path"]')" -train_path "$(field "$spec" 's["evals"][0]["train_path"]')" \
      -out outputs/nn/${s}_${split}_nn.json 2>&1 | tee outputs/nn/${s}_${split}_nn.log
  done
done
tar czf outputs/nn.tar.gz outputs/nn
echo "done: outputs/nn.tar.gz"
