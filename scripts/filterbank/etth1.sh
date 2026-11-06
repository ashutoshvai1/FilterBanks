#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank
dataset_name=ETTh1

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.005301658434104317 \
  --depth 17 \
  --filter_len 522 \
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
  --data_path ETTh1.csv \
  --model_id ETTh1_720_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.01603345849489098 \
  --depth 27 \
  --filter_len 426 \
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
  --data_path ETTh1.csv \
  --model_id ETTh1_720_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.021457350313960902 \
  --depth 55 \
  --filter_len 343 \
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
  >logs/${model_name}_${dataset_name}_720_336.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_720_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.007913492667390764 \
  --depth 36 \
  --filter_len 365 \
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
