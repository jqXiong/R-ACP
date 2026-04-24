#!/usr/bin/env bash
set -euo pipefail

bash ./train_abl_1_baseline.sh
bash ./train_abl_2_plus_channel.sh
bash ./train_abl_3_plus_jscc.sh
bash ./train_abl_4_plus_csi.sh
bash ./train_abl_5_plus_cross_view.sh
bash ./train_abl_6_full.sh

echo "Training done. Checkpoints saved under models_temp/abl_*.pth"
