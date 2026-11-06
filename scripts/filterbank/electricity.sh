#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank
dataset_name=electricity

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_720_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.009770029170952951 \
  --depth 79 \
  --filter_len 695 \
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
  >logs/${model_name}_${dataset_name}_720_96.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_720_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.00781261453813224 \
  --depth 64 \
  --filter_len 716 \
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
  >logs/${model_name}_${dataset_name}_720_192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_720_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.008139464800989666 \
  --depth 59 \
  --filter_len 596 \
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
  >logs/${model_name}_${dataset_name}_720_336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.010899678428423343 \
  --depth 63 \
  --filter_len 603 \
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
  >logs/${model_name}_${dataset_name}_720_720.log
