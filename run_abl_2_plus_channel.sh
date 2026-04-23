#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-../Wildtrack_dataset}"
MODEL_PATH="${MODEL_PATH:-./models_temp/MultiviewDetector.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"
SNR_SWEEP="${SNR_SWEEP:--5,0,5,10,15,20}"

# +Channel: channel corruption only, disable CSI and cross-view
python main_coding_and_inference.py \
  --dataset_path "${DATASET_PATH}" \
  --model_path "${MODEL_PATH}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --tau_1 "${TAU1}" \
  --tau_2 "${TAU2}" \
  --method proposed_jscc \
  --ablate_no_jscc \
  --ablate_no_csi \
  --ablate_no_cross_view \
  --snr_sweep "${SNR_SWEEP}" \
  --exp_name abl_2_plus_channel
