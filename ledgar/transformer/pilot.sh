#!/bin/bash
# Pilot runs: choose the learning rate and maximum input length on a small
# budget before committing ~3 CPU-hours to the full fine-tuning run.
# 6,000 training provisions, 3 epochs, scored on 2,000 validation provisions.
# The test split is never read (--no-test).
#
# Run from the ledgar/ directory:  bash transformer/pilot.sh
set -euo pipefail
export HF_HUB_DISABLE_XET=1          # plain HTTPS downloads (xet storage host is not reachable here)
PY=${PY:-.venv/bin/python}

run() {  # run <name> <lr> <max_length>
  local out=models/pilot/$1
  [ -f "$out/summary.json" ] && return 0
  "$PY" transformer/finetune.py --data data/raw --out "$out" --lr "$2" --max-length "$3" \
    --train-subset 6000 --val-subset 2000 --epochs 3 --no-test --log-every 50
  rm -f "$out/checkpoint.pt" "$out/best_model.pt"
}

run lr5e-5_len128 5e-5 128
run lr1e-4_len128 1e-4 128
run lr2e-4_len128 2e-4 128
run lr2e-4_len256 2e-4 256
run lr4e-4_len128 4e-4 128
