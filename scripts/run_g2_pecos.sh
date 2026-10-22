#!/usr/bin/env bash
# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
# Plan G baseline 2: PECOS XR-Linear on each G1 tree's own features (same rows, labels, vocabulary as XMR4EL).
# Runs on the host (two images): xmr4el exports dev + test and scores, xmr4el-pecos trains once on the dev export
# (threshold 0, as the BC5CDR-disease rows) and predicts test with that model. Stops at the first failure.
# Writes outputs/g2/ (<set>_<split>_{export,pecos,score}.log, peak_rss.tsv) and bundles it into outputs/g2.tar.gz
# (exports and PECOS models stay in outputs/pecos/g2/). Usage: bash scripts/run_g2_pecos.sh [set ...]
set -euo pipefail
SETS=${*:-ncbi_disease biored_disease bc5cdr_chemical biored_chemical nlm_chem}
declare -A TREE=([bc5cdr]=2026-10-09_14-12-26)  # F3 BC5CDR-disease; run_g1 sets use their latest run's tree
OUT=outputs/g2; EXP=outputs/pecos/g2; mkdir -p "$OUT" "$EXP" "$HOME/.cache/huggingface"
RUN=(docker run --rm --init --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app)
XMR=("${RUN[@]}" --gpus all --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface"
     -e HF_HOME=/cache/huggingface xmr4el)
PEC=("${RUN[@]}" xmr4el-pecos)
# In-container wrapper: run the command, append "tag<TAB>peak RSS GB" (Linux ru_maxrss is KB) to peak_rss.tsv.
PEAK='import resource, subprocess, sys
r = subprocess.run(sys.argv[3:]).returncode
gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
open(sys.argv[1], "a").write(f"{sys.argv[2]}\t{gb:.1f}\n"); sys.exit(r)'
path() { python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["evals"][0][sys.argv[2]] if "evals" in d else d[sys.argv[2]])' "$1" "$2"; }

for s in $SETS; do
  if [ -n "${TREE[$s]:-}" ]; then tree=outputs/saved_trees/xmodel_${TREE[$s]}
  else tree=$(path "$(ls -d outputs/runs/*_"$s" | sort | tail -1)/tree.json" xmodel_path); fi  # set not in TREE: latest run_g1 tree
  for split in dev test; do
    spec=configs/experiments/$s.json; [ "$split" = test ] && spec=configs/experiments/${s}_test.json
    rm -rf "$EXP/${s}_$split"  # export refuses a non-empty dir; a rerun replaces the old export
    "${XMR[@]}" python3 -c "$PEAK" "$OUT/peak_rss.tsv" "$s export $split" python3 scripts/baselines/pecos_compare.py \
      export -xmodel_path "$tree" -train_path "$(path "$spec" train_path)" -test_path "$(path "$spec" test_path)" \
      -out "$EXP/${s}_$split" 2>&1 | tee "$OUT/${s}_${split}_export.log"
  done
  rm "$EXP/${s}_test/X_trn.npz" "$EXP/${s}_test/Y_trn.npz"  # test is predicted with the dev model: no copy needed
  "${PEC[@]}" python3 -c "$PEAK" "$OUT/peak_rss.tsv" "$s pecos train+dev" python3 scripts/baselines/pecos_run.py \
    -data "$EXP/${s}_dev" -features ours -nr_splits 16 -max_leaf_size 100 -beams 10 -topk 100 -threshold 0 \
    2>&1 | tee "$OUT/${s}_dev_pecos.log"
  "${PEC[@]}" python3 -c "$PEAK" "$OUT/peak_rss.tsv" "$s pecos test" python3 scripts/baselines/pecos_run.py \
    -data "$EXP/${s}_test" -model "$EXP/${s}_dev/model_ours" -features ours -beams 10 -topk 100 \
    2>&1 | tee "$OUT/${s}_test_pecos.log"
  for split in dev test; do
    "${XMR[@]}" python3 scripts/baselines/pecos_compare.py score -out "$EXP/${s}_$split" 2>&1 | tee "$OUT/${s}_${split}_score.log"
  done
done
tar czf outputs/g2.tar.gz "$OUT"
echo "done: outputs/g2.tar.gz"
