#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
dataset_name=ETTh2

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_720_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.010765505856731567 \
  --depth 8 \
  --filter_len 83 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.3022573182287385 \
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
  --data_path ETTh2.csv \
  --model_id ETTh2_720_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.02637300858162085 \
  --depth 10 \
  --filter_len 119 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.35642582277345225 \
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
  --data_path ETTh2.csv \
  --model_id ETTh2_720_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.005173879667262885 \
  --depth 6 \
  --filter_len 281 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.3655778822310173 \
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
  --data_path ETTh2.csv \
  --model_id ETTh2_720_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 \
  --learning_rate 0.0024568475955245497 \
  --depth 1 \
  --filter_len 28 \
  --period_len 12 \
  --use_revIN True \
  --sampling_fraction 0.4281574472933019 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
