#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank
dataset_name=ETTm1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_720_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.012146354312119651 \
  --depth 64 \
  --filter_len 570 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0 \
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
  --data_path ETTm1.csv \
  --model_id ETTm1_720_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.007453496404691516 \
  --depth 74 \
  --filter_len 575 \
  --period_len 12 \
  --use_revIN True \
  --sampling_fraction 0 \
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
  --data_path ETTm1.csv \
  --model_id ETTm1_720_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.009582483583413442 \
  --depth 24 \
  --filter_len 658 \
  --period_len 24 \
  --use_revIN True \
  --sampling_fraction 0 \
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
  --data_path ETTm1.csv \
  --model_id ETTm1_720_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.004233983247896967 \
  --depth 24 \
  --filter_len 659 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
