#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
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
  --learning_rate 0.0016709830407849477 \
  --depth 5 \
  --filter_len 694 \
  --period_len 24 \
  --use_revIN True \
  --sampling_fraction 0.30051104468314777 \
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
  --learning_rate 0.002133557420957764 \
  --depth 5 \
  --filter_len 504 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.26076558277777334 \
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
  --learning_rate 0.003472320681247071 \
  --depth 3 \
  --filter_len 580 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.20086314296368252 \
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
  --learning_rate 0.003634257141514806 \
  --depth 3 \
  --filter_len 533 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.4490637650531487 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
