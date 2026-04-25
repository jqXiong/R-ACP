#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
MODEL_PATH="${MODEL_PATH:-./models_temp/abl_4_plus_csi.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"
SNR_SWEEP="${SNR_SWEEP:--5,0,5,10,15,20}"

"${PYTHON_BIN}" main_coding_and_inference.py --dataset_path "${DATASET_PATH}" --model_path "${MODEL_PATH}" --epochs "${EPOCHS}" --batch_size "${BATCH}" --tau_1 "${TAU1}" --tau_2 "${TAU2}" --method proposed_jscc --rate_aware_training --keep_latest_token --target_comm_kb 28.5 --min_keep_per_camera 1 --frame_dropout_noise_std 0.03 --lambda_consistency 0.03 --rate_view_dropout_prob 0.2 --ablate_no_cross_view --jscc_csi_gain_scale 0.35 --jscc_importance_gain_scale 0.8 --jscc_low_snr_disable_csi_threshold 5 --snr_sweep="${SNR_SWEEP}" --snr_sweep_resume --test_only --save_prefix abl_4_plus_csi --exp_name abl_4_plus_csi
