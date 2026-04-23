#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-./Data/Wildtrack}"
MODEL_PATH="${MODEL_PATH:-./models_temp/MultiviewDetector.pth}"

echo "[check] python: $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

echo "[check] dataset path: ${DATASET_PATH}"
[ -d "${DATASET_PATH}" ] || { echo "[error] dataset dir not found: ${DATASET_PATH}"; exit 1; }
[ -f "${DATASET_PATH}/calibrations/extrinsic/extr_CVLab1.xml" ] || { echo "[error] missing file: ${DATASET_PATH}/calibrations/extrinsic/extr_CVLab1.xml"; exit 1; }

echo "[check] model path: ${MODEL_PATH}"
[ -f "${MODEL_PATH}" ] || { echo "[error] model file not found: ${MODEL_PATH}"; exit 1; }

echo "[check] py syntax"
"${PYTHON_BIN}" -m py_compile main_coding_and_inference.py multiview_detector/models/persp_trans_detector.py collect_snr_sweep.py

echo "[ok] precheck passed"
