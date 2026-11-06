#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
dataset_name=ETTm2

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.002005829768421323 \
  --depth 3 \
  --filter_len 265 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.3177680911544014 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.0051740156176361875 \
  --depth 4 \
  --filter_len 48 \
  --period_len 24 \
  --use_revIN True \
  --sampling_fraction 0.6293876689306426 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.006424805355398046 \
  --depth 1 \
  --filter_len 200 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.22355146404384763 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.01961054120571662 \
  --depth 1 \
  --filter_len 420 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.5901891998364162 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
