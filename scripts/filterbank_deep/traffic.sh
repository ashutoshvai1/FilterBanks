#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear_Filterbank_deep
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
  --learning_rate 0.0024906031338176393 \
  --depth 10 \
  --filter_len 23 \
  --period_len 4 \
  --use_revIN True \
  --sampling_fraction 0.9763874926964322 \
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
  --learning_rate 0.0017711197924229676 \
  --depth 9 \
  --filter_len 250 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.7309822503501456 \
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
  --learning_rate 0.004334717150792044 \
  --depth 5 \
  --filter_len 244 \
  --period_len 48 \
  --use_revIN True \
  --sampling_fraction 0.7301363486415775 \
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
  --learning_rate 0.002378467974033607 \
  --depth 8 \
  --filter_len 470 \
  --period_len 12 \
  --use_revIN True \
  --sampling_fraction 0.4284027002710315 \
  --n_blocks 1 \
  --dropout 0 \
  --hidden_size 512 \
  --use_layer_norm False \
  --lradj type3 \
  --train_epochs 100 \
  --train_mode H \
  --noise_variance 0 \
  >logs/${model_name}_${dataset_name}_720_720.log
