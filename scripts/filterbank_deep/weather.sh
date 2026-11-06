#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
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
  --learning_rate 0.0011618274066971853 \
  --depth 4 \
  --filter_len 318 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.10797439005482197 \
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
  --learning_rate 0.001149343736161983 \
  --depth 1 \
  --filter_len 147 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.326825986166277 \
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
  --learning_rate 0.0034164168201304902 \
  --depth 1 \
  --filter_len 656 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.7892185523429708 \
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
  --learning_rate 0.005397088819549739 \
  --depth 3 \
  --filter_len 719 \
  --period_len 24 \
  --use_revIN True \
  --sampling_fraction 0.03275861777837629 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
