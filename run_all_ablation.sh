#!/usr/bin/env bash
set -euo pipefail

bash ./run_abl_1_baseline.sh
bash ./run_abl_2_plus_channel.sh
bash ./run_abl_3_plus_jscc.sh
bash ./run_abl_4_plus_csi.sh
bash ./run_abl_5_plus_cross_view.sh
bash ./run_abl_6_full.sh

python collect_snr_sweep.py --logs_dir logs --output_csv logs/snr_sweep_summary.csv
