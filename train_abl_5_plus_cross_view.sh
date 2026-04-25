#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-./models_temp/MultiviewDetector.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"

"${PYTHON_BIN}" main_coding_and_inference.py --dataset_path "${DATASET_PATH}" --model_path "${INIT_MODEL_PATH}" --epochs "${EPOCHS}" --train_epochs "${EPOCHS}" --batch_size "${BATCH}" --tau_1 "${TAU1}" --tau_2 "${TAU2}" --method proposed_jscc --rate_aware_training --keep_latest_token --target_comm_kb 28.5 --min_keep_per_camera 1 --frame_dropout_noise_std 0.03 --lambda_consistency 0.03 --rate_view_dropout_prob 0.2 --early_stop_patience 8 --early_stop_min_delta 0.03 --early_stop_min_epochs 14 --save_prefix abl_5_plus_cross_view --exp_name train_abl_5_plus_cross_view
