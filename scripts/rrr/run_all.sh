#!/bin/bash
# RRR rank analysis (paper Sec. 3.1.1, Fig. 1): 5 datasets x 4 horizons, L=720.
# Closed form, CPU only; each run takes about a minute. Run from the repository root.
set -e
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 weather; do
  python -u rrr_rank_analysis.py --dataset $dataset --seq_len 720 --all_horizons \
    --output_root output/rrr_analysis_v2
done
# Fig. 1b table (r*, compression, improvement over OLS) and Fig. 1a curves
python analysis/compile_rrr_analysis.py --base_dir output/rrr_analysis_v2 --out output/rrr_summary
python analysis/plot_rank_train_val_test.py --runs_dir output/rrr_analysis_v2 --out_dir output/rrr_figures
