#!/bin/bash
# Full fine-tuning run with the settings chosen in the pilot
# (results/14_pilot_runs.csv): learning rate 2e-4, 256 word pieces, 3 epochs.
# About 3.5 hours on 4 CPU cores. The run checkpoints after every epoch and
# resumes from the last checkpoint if it is restarted.
#
# Run from the ledgar/ directory:  bash transformer/train_full.sh
set -euo pipefail
export HF_HUB_DISABLE_XET=1
PY=${PY:-.venv/bin/python}
OUT=models/legal-bert-small
mkdir -p "$OUT"
until [ -f "$OUT/summary.json" ]; do
  "$PY" transformer/finetune.py --data data/raw --out "$OUT" \
    --lr 2e-4 --max-length 256 --epochs 3 --batch-size 32 --log-every 100 >> "$OUT/train.log" 2>&1 || sleep 5
done
