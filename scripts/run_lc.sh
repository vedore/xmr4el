#!/usr/bin/env bash
# leaf_candidates > 100 at beam 10, eval-only on the latest run_g1 BC5CDR-chemical tree (host shell, xmr4el image):
#   accuracy  dev at leaf_candidates 200 / 500 -> run dir (100 = the I1 `dev` row, same tree)
#   latency   tree search ms/query at 100 / 200 / 500 -> outputs/lc/bc5cdr_chemical.json (vs the I3 rerun)
# Bundle outputs/lc.tar.gz. Usage: bash scripts/run_lc.sh
set -euo pipefail
mkdir -p outputs/lc "$HOME/.cache/huggingface"
XMR=(docker run --rm --init --gpus all --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app
     --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface xmr4el python3)
TREE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["xmodel_path"])' \
  "$(ls -d outputs/runs/*_bc5cdr_chemical | sort | tail -1)/tree.json")
"${XMR[@]}" scripts/run_experiment.py -spec configs/experiments/lc_bc5cdr_chemical.json -xmodel_path "$TREE"
"${XMR[@]}" scripts/bench_latency.py -xmodel_path "$TREE" -test_path datasets/BC5CDR/chemical/dev.pubtator \
  -leaf_candidates 100 200 500 -out outputs/lc/bc5cdr_chemical.json 2>&1 | tee outputs/lc/bc5cdr_chemical.log
tar czf outputs/lc.tar.gz "$(ls -d outputs/runs/*_lc_bc5cdr_chemical | sort | tail -1)" outputs/lc
echo "done: outputs/lc.tar.gz"
