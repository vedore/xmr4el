#!/usr/bin/env bash
# Context-shift routing diagnostic (scripts/diagnose_context.py), eval-only on each set's latest plain run tree
# (host shell, xmr4el image): BC5CDR-chemical, NLM-Chem, NCBI-disease (control) dev -> outputs/ctx/<set>.{json,log}.
# Bundle outputs/ctx.tar.gz. Usage: bash scripts/run_ctx.sh
set -euo pipefail
mkdir -p outputs/ctx "$HOME/.cache/huggingface"
XMR=(docker run --rm --init --gpus all --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app
     --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface xmr4el python3)
field() { python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$@"; }
for s in bc5cdr_chemical nlm_chem ncbi_disease; do
  run=$(ls -d outputs/runs/*_"$s" | grep -E "/[0-9-]{10}_[0-9-]{8}_$s\$" | sort | tail -1)  # not i1_/lc_ eval runs
  "${XMR[@]}" scripts/diagnose_context.py -xmodel_path "$(field "$run/tree.json" 's["xmodel_path"]')" \
    -test_path "$(field "configs/experiments/$s.json" 's["evals"][0]["test_path"]')" \
    -train_path "$(field "configs/experiments/$s.json" 's["train_path"]')" -out outputs/ctx/$s.json 2>&1 | tee outputs/ctx/$s.log
done
tar czf outputs/ctx.tar.gz outputs/ctx
echo "done: outputs/ctx.tar.gz"
