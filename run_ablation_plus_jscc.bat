@echo off
setlocal

REM ====== 请按需修改 ======
set DATASET_PATH=../Wildtrack_dataset
set MODEL_PATH=./models_temp/MultiviewDetector.pth
set EPOCHS=30
set TAU1=2
set TAU2=2
set BATCH=1
set SNR_SWEEP=-5,0,5,10,15,20

python main_coding_and_inference.py ^
  --dataset_path "%DATASET_PATH%" ^
  --model_path "%MODEL_PATH%" ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH% ^
  --tau_1 %TAU1% ^
  --tau_2 %TAU2% ^
  --method proposed_jscc ^
  --ablate_no_cross_view ^
  --snr_sweep "%SNR_SWEEP%" ^
  --exp_name abl_plus_jscc

endlocal
