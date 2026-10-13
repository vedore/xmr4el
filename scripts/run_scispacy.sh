#!/usr/bin/env bash
# Plan G baseline 5: scispaCy candidate generator (umls KB) on run_g2_pecos.sh's exports (same rows and labels).
# Host shell, after run_g2_pecos.sh for the set. Writes outputs/g2/<set>_<split>_scispacy.log, re-scores the
# export (score log then has PECOS + scispaCy) and re-bundles outputs/g2.tar.gz. Usage: bash scripts/run_scispacy.sh set...
set -euo pipefail
OUT=outputs/g2; EXP=outputs/pecos/g2; mkdir -p "$HOME/.cache/scispacy"
RUN=(docker run --rm --init --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app)
for s in "$@"; do
  for split in dev test; do
    "${RUN[@]}" --mount "type=bind,src=$HOME/.cache/scispacy,dst=/cache/scispacy" xmr4el-scispacy \
      python3 scripts/baselines/scispacy_run.py -data "$EXP/${s}_$split" -kb umls 2>&1 | tee "$OUT/${s}_${split}_scispacy.log"
    "${RUN[@]}" --gpus all xmr4el python3 scripts/baselines/pecos_compare.py score -out "$EXP/${s}_$split" \
      2>&1 | tee "$OUT/${s}_${split}_score.log"
  done
done
tar czf outputs/g2.tar.gz "$OUT"
echo "done: outputs/g2.tar.gz"
