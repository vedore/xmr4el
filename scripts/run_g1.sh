#!/usr/bin/env bash
# Plan G1: per set, train + dev eval (run_experiment.py), test eval on the same tree, SapBERT / TF-IDF 1-NN
# baselines on dev and test. Stops at the first failure (chemical sets last: 180k-label scale).
# Writes outputs/g1/ (logs, <set>_<split>_nn.json, peak_rss.tsv = peak RAM per step) and bundles it with this
# script's run dirs into outputs/g1.tar.gz (no trees). Usage: bash scripts/run_g1.sh [set ...]
set -euo pipefail
PY=${PY:-python3}
SETS=${*:-ncbi_disease biored_disease bc5cdr_chemical biored_chemical nlm_chem}
OUT=outputs/g1; mkdir -p "$OUT"
RUNS=()

# peak <tag> <cmd...>: run cmd, append "tag<TAB>peak RSS GB" (Linux ru_maxrss is KB) to peak_rss.tsv.
peak() {
  "$PY" -c 'import resource, subprocess, sys
r = subprocess.run(sys.argv[3:]).returncode
gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
open(sys.argv[1], "a").write(f"{sys.argv[2]}\t{gb:.1f}\n"); sys.exit(r)' "$OUT/peak_rss.tsv" "$@"
}
# paths <spec> <key>: the first eval's path
path() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["evals"][0][sys.argv[2]])' "$1" "$2"; }

for s in $SETS; do
  peak "$s train+dev" "$PY" scripts/run_experiment.py -spec configs/experiments/$s.json 2>&1 | tee "$OUT/${s}_dev.log"
  run=$(ls -td outputs/runs/*_"$s" | head -1); RUNS+=("$run")
  tree=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["xmodel_path"])' "$run/tree.json")
  peak "$s test" "$PY" scripts/run_experiment.py -spec configs/experiments/${s}_test.json -xmodel_path "$tree" \
    2>&1 | tee "$OUT/${s}_test.log"
  RUNS+=("$(ls -td outputs/runs/*_"$s"_test | head -1)")
  for spec in configs/experiments/$s.json configs/experiments/${s}_test.json; do
    split=$([ "$spec" = configs/experiments/$s.json ] && echo dev || echo test)
    peak "$s nn $split" "$PY" scripts/baselines/nn_baselines.py -xmodel_path "$tree" \
      -test_path "$(path "$spec" test_path)" -train_path "$(path "$spec" train_path)" \
      -out "$OUT/${s}_${split}_nn.json" 2>&1 | tee "$OUT/${s}_${split}_nn.log"
  done
done
tar czf outputs/g1.tar.gz "$OUT" "${RUNS[@]}"
echo "done: outputs/g1.tar.gz"
