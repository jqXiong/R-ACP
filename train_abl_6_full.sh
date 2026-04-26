#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-./models_temp/MultiviewDetector.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"

"${PYTHON_BIN}" main_coding_and_inference.py --dataset_path "${DATASET_PATH}" --model_path "${INIT_MODEL_PATH}" --epochs "${EPOCHS}" --train_epochs "${EPOCHS}" --batch_size "${BATCH}" --tau_1 "${TAU1}" --tau_2 "${TAU2}" --method baseline_refined --refine_keep_cameras 6 --refine_weighted_entropy --refine_score_mode current_temporal_uncertainty --refine_score_noise_std 0.02 --refine_soft_weighting --save_prefix abl_6_refined_full --exp_name train_abl_6_refined_full
