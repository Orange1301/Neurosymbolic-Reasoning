#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

TRAIN_FILE="${TRAIN_FILE:-dataset/filtered/train/folio_train.json}"
VALID_FILE="${VALID_FILE:-dataset/filtered/valid/folio_valid.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/outputs/basic2_mt5_base_folio}"
MODEL_NAME="${MODEL_NAME:-google/mt5-base}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

python - <<'PY'
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: Basic 2 real training requires a Kaggle GPU. Enable GPU accelerator.")
if torch.cuda.device_count() == 1:
    print("WARNING: only 1 GPU detected. T4 x2 training expects 2 GPUs.")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

accelerate launch --num_processes "$NUM_PROCESSES" scripts/train_basic2_mt5_folio.py \
  --profile base \
  --train-file "$TRAIN_FILE" \
  --valid-file "$VALID_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --max-source-length 768 \
  --max-target-length 768 \
  --num-train-epochs 20 \
  --learning-rate 5e-5 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 5 \
  --save-total-limit 3 \
  --fp16 \
  --gradient-checkpointing \
  --optim adafactor \
  --predict-with-generate \
  --generation-max-length 768 \
  --seed 42
