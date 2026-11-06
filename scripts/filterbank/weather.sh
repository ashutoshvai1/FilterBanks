#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank
dataset_name=weather

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_720_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.008304499553298022 \
  --depth 19 \
  --filter_len 708 \
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
  --data_path weather.csv \
  --model_id weather_720_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.029021463705562647 \
  --depth 29 \
  --filter_len 705 \
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
  --data_path weather.csv \
  --model_id weather_720_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.005812840107153695 \
  --depth 32 \
  --filter_len 657 \
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
  --data_path weather.csv \
  --model_id weather_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.004387322123363646 \
  --depth 79 \
  --filter_len 676 \
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
