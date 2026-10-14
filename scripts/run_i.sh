#!/usr/bin/env bash
# Plan I, eval-only on each set's latest run_g1 tree (host shell, xmr4el image), after run_g.sh:
#   I1 + I2  chemical sets dev with knn on/off (routed = gold among the candidates, per string group);
#            BC5CDR-chemical beam 20/50/128 (128 = every root cluster)  -> run dirs
#   I3       latency / memory: tree vs SapBERT 1-NN vs TF-IDF 1-NN at 13k / 18.5k / 180k labels -> outputs/i3/<set>.json
# Bundles both into outputs/i.tar.gz. Usage: bash scripts/run_i.sh [i3]  (i3: I3 only, bundle outputs/i3.tar.gz)
set -euo pipefail
mkdir -p outputs/i3 "$HOME/.cache/huggingface"
XMR=(docker run --rm --init --gpus all --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app
     --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface xmr4el python3)
tree() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["xmodel_path"])' \
  "$(ls -d outputs/runs/*_"$1" | sort | tail -1)/tree.json"; }
dev() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["evals"][0]["test_path"])' "configs/experiments/$1.json"; }
RUNS=()
[ "${1:-}" = i3 ] || for s in bc5cdr_chemical biored_chemical nlm_chem; do
  "${XMR[@]}" scripts/run_experiment.py -spec configs/experiments/i1_$s.json -xmodel_path "$(tree "$s")"
  RUNS+=("$(ls -d outputs/runs/*_i1_"$s" | sort | tail -1)")
done
for s in ncbi_disease medmentions_st21pv bc5cdr_chemical nlm_chem; do
  "${XMR[@]}" scripts/bench_latency.py -xmodel_path "$(tree "$s")" -test_path "$(dev "$s")" -out outputs/i3/$s.json \
    2>&1 | tee outputs/i3/$s.log
done
OUT=outputs/i${1:+3}.tar.gz
tar czf "$OUT" ${RUNS[@]+"${RUNS[@]}"} outputs/i3
echo "done: $OUT"
