#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
MODEL_PATH="${MODEL_PATH:-./models_temp/abl_4_refined_prune_masked.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"
SNR_SWEEP="${SNR_SWEEP:--5,0,5,10,15,20}"

"${PYTHON_BIN}" main_coding_and_inference.py --dataset_path "${DATASET_PATH}" --model_path "${MODEL_PATH}" --epochs "${EPOCHS}" --batch_size "${BATCH}" --tau_1 "${TAU1}" --tau_2 "${TAU2}" --method baseline_refined --refine_keep_cameras 6 --refine_score_mode current --refine_weighted_entropy --snr_sweep="${SNR_SWEEP}" --snr_sweep_resume --test_only --save_prefix abl_4_refined_prune_masked --exp_name abl_4_refined_prune_masked
