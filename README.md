# Filter Banks: from Low-Rank Representations to Deep Models for Efficient Time Series Forecasting

This repository contains the code needed to reproduce the main results of "Filter Banks: from Low-Rank Representations to Deep Models for Efficient Time Series Forecasting" (NeurIPS 2026).

## Setup

Install the requirements:
```bash
pip install -r requirements.txt
```
Download the datasets from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) folder provided by Autoformer and place the CSV files in `./dataset/` (for example `./dataset/ETTh1.csv`, `./dataset/weather.csv`).

All commands below are run from the repository root.

## RRR analysis (Section 3.1.1, Figure 1)

`rrr_rank_analysis.py` fits full-rank OLS with an intercept on the training split and evaluates reduced-rank regression (RRR) at every rank from 1 up to the full rank `H`. It reports train, validation and test MSE. The solution is closed form and runs on CPU, so each dataset/horizon pair takes about a minute.
```bash
# Single dataset and horizon
python rrr_rank_analysis.py --dataset ETTh1 --pred_len 96

# All datasets and horizons, then the Figure 1b table and the Figure 1a curves
bash scripts/rrr/run_all.sh
```
Per-run outputs go to `output/rrr_analysis_v2/{dataset}_720_{horizon}_{timestamp}/metrics/`:

- `rank_analysis.csv`: RRR MSE per rank.
- `ols_metrics.json`: full-rank OLS MSE.
- `eigenvalues.csv`

`analysis/compile_rrr_analysis.py` selects the rank `r*` by early stopping on validation MSE (threshold `1e-4`, patience 3). It reports the compression `L*H / (r*(L+H))` and the test-MSE improvement over OLS in `output/rrr_summary.{csv,tex}`. `analysis/plot_rank_train_val_test.py` writes the ETTh1 and Weather H=96 curves to `output/rrr_figures/`.

## Filterbank (FB-LR) and Filterbank-deep (FB-deep), Table 2

The model implementations are in `./models`:

- `DLinear_Filterbank.py`: FB-LR.
- `DLinear_Filterbank_deep.py`: FB-deep, used with `--n_blocks 1`.

The scripts in `./scripts/` reproduce Table 2 with the tuned hyperparameters. Each script trains with a look-back window of 720 for the prediction horizons {96, 192, 336, 720}:
```bash
bash scripts/filterbank/etth1.sh        # FB-LR
bash scripts/filterbank_deep/etth1.sh   # FB-deep
```
Scripts are provided for ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity and Traffic. Training logs, including the final test MSE/MAE, go to `logs/{model}_{dataset}_720_{horizon}.log`.

The hyperparameters came from Optuna searches run with `optuna_objective.py`. The selected configurations are listed in `compiled_results/filterbank_exp/results.csv` and `compiled_results/filterbank_deep_exp/results.csv`.

## Inference cost (Table 3)

To measure the inference-cost metrics on your own hardware, run (requires a CUDA GPU):
```bash
python benchmark_compute.py
```
It writes median inference time (ms), MACs and parameter counts to `model_performance_metrics.csv`.

`DLinear_Filterbank_matrix.py` is an equivalent single-matrix form of FB-LR used for benchmarking. `debug/test_matrix_equivalence.py` checks the equivalence.

## Acknowledgements

This code builds on [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) (Apache 2.0).
