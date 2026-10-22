#!/usr/bin/env bash
# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
# Equivalence test XMR4EL tree vs PECOS XR-Linear on NCBI-disease dev, one difference removed per step:
#   A  G1 tree 15-32-01 (balanced weights, routing features): PECOS own hierarchy / the tree's chain / + logistic + sigmoid
#   C  new tree, class_weight null (= PECOS's objective): PECOS chain + logistic + sigmoid
#   D  C + routing_features false (leaves see the root's X, as PECOS): same PECOS, plus eps 0.001 (solver tolerance)
# XMR4EL side: run_experiment.py dev (knn 10) + dev_knn0; pecos_compare.py agree compares row by row with knn off.
# Host shell, both images (as run_g2_pecos.sh). Writes outputs/eq/ + run dirs, bundled into outputs/eq.tar.gz.
set -euo pipefail
OUT=outputs/eq; EXP=outputs/pecos/eq; mkdir -p "$OUT" "$EXP" "$HOME/.cache/huggingface"
RUN=(docker run --rm --init --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app)
XMR=("${RUN[@]}" --gpus all --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface"
     -e HF_HOME=/cache/huggingface xmr4el python3)
PEC=("${RUN[@]}" xmr4el-pecos python3 scripts/baselines/pecos_run.py -features ours -beams 10 -topk 100 -threshold 0)
TRAIN=datasets/ncbi_disease/train_plus_ctd.pubtator; DEV=datasets/ncbi_disease/dev.pubtator
RUNS=()

declare -A TREE=([a]=outputs/saved_trees/xmodel_2026-10-12_15-32-01)
"${XMR[@]}" scripts/run_experiment.py -spec configs/experiments/ncbi_eq_a.json -xmodel_path "${TREE[a]}" \
  2>&1 | tee "$OUT/a_xmr.log"
RUNS+=("$(ls -td outputs/runs/*_ncbi_eq_a | head -1)")
for k in c d; do
  "${XMR[@]}" scripts/run_experiment.py -spec configs/experiments/ncbi_eq_$k.json 2>&1 | tee "$OUT/${k}_xmr.log"
  run=$(ls -td outputs/runs/*_ncbi_eq_$k | head -1); RUNS+=("$run")
  TREE[$k]=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["xmodel_path"])' "$run/tree.json")
done

for k in a c d; do
  "${XMR[@]}" scripts/baselines/pecos_compare.py export -xmodel_path "${TREE[$k]}" -train_path "$TRAIN" \
    -test_path "$DEV" -out "$EXP/$k" 2>&1 | tee "$OUT/${k}_export.log"
done
"${PEC[@]}" -data "$EXP/a" -nr_splits 16 -max_leaf_size 100 -tag _own 2>&1 | tee "$OUT/a_pecos_own.log"
"${PEC[@]}" -data "$EXP/a" -chain -tag _chain 2>&1 | tee "$OUT/a_pecos_chain.log"
for k in a c d; do
  "${PEC[@]}" -data "$EXP/$k" -chain -solver L2R_LR_DUAL -post sigmoid -tag _chain_lr 2>&1 | tee "$OUT/${k}_pecos_chain_lr.log"
done
"${PEC[@]}" -data "$EXP/d" -chain -solver L2R_LR_DUAL -post sigmoid -eps 0.001 -tag _chain_lr_tight \
  2>&1 | tee "$OUT/d_pecos_chain_lr_tight.log"
for k in a c d; do
  "${XMR[@]}" scripts/baselines/pecos_compare.py score -out "$EXP/$k" 2>&1 | tee "$OUT/${k}_score.log"
  "${XMR[@]}" scripts/baselines/pecos_compare.py agree -out "$EXP/$k" -beam_size 10 2>&1 | tee "$OUT/${k}_agree.log"
done
tar czf outputs/eq.tar.gz "$OUT" "${RUNS[@]}"
echo "done: outputs/eq.tar.gz"
