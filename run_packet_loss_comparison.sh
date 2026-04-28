#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
BASELINE_CKPT="${BASELINE_CKPT:-./models_temp/abl_1_baseline.pth}"
FULL_CKPT="${FULL_CKPT:-./models_temp/abl_6_refined_adaptive.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/$(date +%F_%H-%M-%S)_packet_loss_comparison}"

python "${SCRIPT_DIR}/tools/run_paper_experiments.py" \
  --experiment packet_loss \
  --dataset_path "${DATASET_PATH}" \
  --baseline_ckpt "${BASELINE_CKPT}" \
  --full_ckpt "${FULL_CKPT}" \
  --methods "baseline,jpeg,h264,h265,av1,full" \
  --packet_loss_rates "0,0.1,0.2,0.3,0.4" \
  --test_snr_db 20 \
  --batch_size 1 \
  --num_workers 0 \
  --output_dir "${OUTPUT_DIR}"
