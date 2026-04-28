#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
BASELINE_CKPT="${BASELINE_CKPT:-./models_temp/abl_1_baseline.pth}"
FULL_CKPT="${FULL_CKPT:-./models_temp/abl_6_refined_adaptive.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/$(date +%F_%H-%M-%S)_codec_snr_comparison}"
H264_ENCODER="${H264_ENCODER:-libx264}"
H265_ENCODER="${H265_ENCODER:-libx265}"
AV1_ENCODER="${AV1_ENCODER:-libaom-av1}"

python "${SCRIPT_DIR}/tools/run_paper_experiments.py" \
  --experiment snr \
  --dataset_path "${DATASET_PATH}" \
  --baseline_ckpt "${BASELINE_CKPT}" \
  --full_ckpt "${FULL_CKPT}" \
  --methods "jpeg,h264,h265,av1" \
  --snr_values=-5,0,5,10,15,20 \
  --snr_packet_loss_rate 0.0 \
  --h264_encoder "${H264_ENCODER}" \
  --h265_encoder "${H265_ENCODER}" \
  --av1_encoder "${AV1_ENCODER}" \
  --batch_size 1 \
  --num_workers 4 \
  --output_dir "${OUTPUT_DIR}"
