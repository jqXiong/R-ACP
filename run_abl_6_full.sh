#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
MODEL_PATH="${MODEL_PATH:-./models_temp/abl_6_refined_adaptive.pth}"
EPOCHS="${EPOCHS:-30}"
TAU1="${TAU1:-2}"
TAU2="${TAU2:-2}"
BATCH="${BATCH:-1}"
SNR_SWEEP="${SNR_SWEEP:--5,0,5,10,15,20}"

"${PYTHON_BIN}" main_coding_and_inference.py --dataset_path "${DATASET_PATH}" --model_path "${MODEL_PATH}" --epochs "${EPOCHS}" --batch_size "${BATCH}" --tau_1 "${TAU1}" --tau_2 "${TAU2}" --method baseline_refined --refine_keep_cameras 6 --refine_weighted_entropy --refine_score_mode current_temporal --refine_adaptive_keep_margin 0.03 --refine_scale_min 0.84 --refine_scale_max 0.99 --refine_apply_entropy_scale --refine_channel_keep_ratio 0.875 --refine_channel_min_keep 6 --refine_channel_drop_floor 0.35 --refine_snr_aware --refine_snr_reference_db 10 --refine_low_snr_camera_bonus 1 --refine_low_snr_channel_bonus 0.125 --refine_low_snr_drop_floor 0.55 --refine_low_snr_scale_boost 0.05 --refine_use_cross_view_fusion --refine_use_bev_attention --refine_strong_head_weight 0.65 --snr_sweep="${SNR_SWEEP}" --snr_sweep_resume --test_only --save_prefix abl_6_refined_adaptive --exp_name abl_6_refined_adaptive
