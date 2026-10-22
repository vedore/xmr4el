#!/usr/bin/env bash
# LOCAL TEST ARTIFACT (user's baseline comparison): kept on `repeat`, not merged into `main`.
# Plan G end to end, host shell: run_g1.sh (xmr4el container: train + dev, test, 1-NN) -> run_g2_pecos.sh (PECOS)
# -> run_scispacy.sh (MedMentions only). Default = the five G1 sets (retrained: new trees replace the G1 rows) +
# MedMentions st21pv. Bundles: outputs/g1.tar.gz, outputs/g2.tar.gz. Usage: bash scripts/run_g.sh [set ...]
set -euo pipefail
SETS=${*:-ncbi_disease biored_disease bc5cdr_chemical biored_chemical nlm_chem medmentions_st21pv}
MM=$([[ " $SETS " == *" medmentions_st21pv "* ]] && echo 1 || true)
[ -n "$MM" ] && docker build --platform linux/amd64 -f scispacy.dockerfile -t xmr4el-scispacy .  # fail before training
mkdir -p "$HOME/.cache/huggingface"
docker run --rm --init --gpus all --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" -w /app \
  --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" -e HF_HOME=/cache/huggingface \
  xmr4el bash scripts/run_g1.sh $SETS
bash scripts/run_g2_pecos.sh $SETS
[ -n "$MM" ] && bash scripts/run_scispacy.sh medmentions_st21pv
echo "done: outputs/g1.tar.gz outputs/g2.tar.gz"
