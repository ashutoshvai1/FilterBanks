#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank
dataset_name=traffic

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_720_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.011344827200981745 \
  --depth 80 \
  --filter_len 662 \
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
  --data_path traffic.csv \
  --model_id traffic_720_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.009438534795131712 \
  --depth 78 \
  --filter_len 466 \
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
  >logs/${model_name}_${dataset_name}_720_192.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_720_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.01255135409142453 \
  --depth 77 \
  --filter_len 570 \
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
  --data_path traffic.csv \
  --model_id traffic_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.009313699085456803 \
  --depth 60 \
  --filter_len 713 \
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
  >logs/${model_name}_${dataset_name}_720_720.log
