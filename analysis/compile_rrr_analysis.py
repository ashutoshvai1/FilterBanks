"""Compile the RRR-vs-OLS table (paper Fig. 1b) from rrr_rank_analysis.py outputs.

Reads, for each dataset/horizon, the latest run in --base_dir:
  metrics/rank_analysis.csv  (RRR with intercept, ranks 1..H)
  metrics/ols_metrics.json   (full-rank OLS with intercept)
A missing OLS file is an error (no fallback approximation).

r* = early-stopping rule on validation MSE: keep increasing rank while validation
MSE improves by more than --threshold (default 1e-4); stop after --patience (3)
consecutive ranks without such an improvement. The validation argmin rank is also
reported for reference.

Compression = parameters of full OLS / parameters of rank-r* RRR, as defined in the
paper (Sec. 3.1, intercepts not counted):  L*H / (r*(L + H)).

Usage: python analysis/compile_rrr_analysis.py [--base_dir output/rrr_analysis_v2]
"""
import argparse
import glob
import json
import os

import pandas as pd

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather']
HORIZONS = [96, 192, 336, 720]


def find_optimal_rank_from_validation(val_mse_list, ranks, threshold=1e-4, patience=3):
    """Early-stopping rank selection (see module docstring). Returns (rank, val_mse)."""
    if len(val_mse_list) < 2:
        return ranks[0], val_mse_list[0]
    best_val_mse, patience_counter, optimal_idx = float('inf'), 0, 0
    for i, val_mse in enumerate(val_mse_list):
        if best_val_mse - val_mse > threshold:
            best_val_mse, optimal_idx, patience_counter = val_mse, i, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return ranks[optimal_idx], val_mse_list[optimal_idx]


def parse_run(run_dir, threshold, patience):
    df = pd.read_csv(os.path.join(run_dir, 'metrics', 'rank_analysis.csv'))
    ols_path = os.path.join(run_dir, 'metrics', 'ols_metrics.json')
    if not os.path.exists(ols_path):
        raise FileNotFoundError(f"Missing {ols_path}: refusing to approximate the OLS baseline")
    with open(ols_path) as f:
        ols = json.load(f)

    L, H = ols['seq_len'], ols['pred_len']
    if df['rank'].max() != H:
        raise ValueError(f"{run_dir}: ranks go to {df['rank'].max()}, expected full rank {H}")

    ranks, val = df['rank'].tolist(), df['rrr_val_mse'].tolist()
    r_star, val_star = find_optimal_rank_from_validation(val, ranks, threshold, patience)
    row = df.loc[df['rank'] == r_star].iloc[0]
    r_argmin = int(df.loc[df['rrr_val_mse'].idxmin(), 'rank'])

    rrr_test, ols_test = row['rrr_test_mse'], ols['ols_test_mse']
    return {
        'dataset': ols['dataset'], 'horizon': H, 'optimal_rank': int(r_star), 'full_rank': min(L, H),
        'compression_ratio': (L * H) / (r_star * (L + H)),
        'validation_mse': val_star, 'rrr_test_mse': rrr_test, 'ols_test_mse': ols_test,
        'improvement_pct': (ols_test - rrr_test) / ols_test * 100,
        'argmin_rank': r_argmin,
        'argmin_test_mse': float(df.loc[df['rank'] == r_argmin, 'rrr_test_mse'].iloc[0]),
        'ols_val_mse': ols['ols_val_mse'], 'run_dir': run_dir,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base_dir', default='output/rrr_analysis_v2')
    p.add_argument('--threshold', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--out', default='analysis/rrr_analysis_summary_v2')
    args = p.parse_args()

    results, missing = [], []
    for ds in DATASETS:
        for h in HORIZONS:
            runs = sorted(glob.glob(os.path.join(args.base_dir, f"{ds}_720_{h}_*")))
            runs = [r for r in runs if os.path.exists(os.path.join(r, 'metrics', 'rank_analysis.csv'))]
            if not runs:
                missing.append(f"{ds}/{h}")
                continue
            results.append(parse_run(runs[-1], args.threshold, args.patience))
    if missing:
        raise SystemExit(f"Missing runs: {', '.join(missing)}")

    df = pd.DataFrame(results)
    full = df.copy()
    full.to_csv(args.out + '_full.csv', index=False, float_format='%.6f')

    print(f"r* rule: threshold={args.threshold}, patience={args.patience}")
    print(f"Average compression: {df['compression_ratio'].mean():.2f}x | "
          f"Average improvement: {df['improvement_pct'].mean():.2f}%")

    table = df[['dataset', 'horizon', 'optimal_rank', 'full_rank', 'compression_ratio',
                'validation_mse', 'rrr_test_mse', 'ols_test_mse', 'improvement_pct']].copy()
    table['compression_ratio'] = table['compression_ratio'].map(lambda x: f"{x:.2f}×")
    for c in ('validation_mse', 'rrr_test_mse', 'ols_test_mse'):
        table[c] = table[c].map(lambda x: f"{x:.3f}")
    table['improvement_pct'] = table['improvement_pct'].map(lambda x: f"{round(x, 2) + 0.0:.2f}")
    table.columns = ['Dataset', 'Horizon', 'Optimal rank (r*)', 'Full rank', 'Compression',
                     'Validation MSE', 'RRR test MSE', 'OLS test MSE', 'Improvement(%)']
    table.to_csv(args.out + '.csv', index=False)
    table.to_latex(args.out + '.tex', index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {args.out}.csv/.tex and {args.out}_full.csv")


if __name__ == "__main__":
    main()
