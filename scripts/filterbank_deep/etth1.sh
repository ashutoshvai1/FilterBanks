#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
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
  --learning_rate 0.011081784996033099 \
  --depth 4 \
  --filter_len 420 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.3911783887889907 \
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
  --learning_rate 0.0024861182735635397 \
  --depth 2 \
  --filter_len 373 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.1506494024466417 \
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
  --learning_rate 0.010476125910436326 \
  --depth 6 \
  --filter_len 198 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.2652553754898248 \
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
  --learning_rate 0.0037329189277165793 \
  --depth 10 \
  --filter_len 342 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.9743636180620836 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
