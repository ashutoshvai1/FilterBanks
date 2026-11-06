"""
RRR rank analysis (paper Sec. 3.1.1, Fig. 1).

For one dataset / horizon:
  1. Fit full-rank OLS with intercept on the training split (pooled channels, no
     instance norm): y_hat = W x + b. Identical to models/OLS.py with
     instance_norm=False (sklearn Ridge, alpha=1e-6, fit_intercept=True).
  2. Reduced-rank regression with intercept (centered formulation):
         y_hat_r = y_mean + Q_r Q_r' W (x - x_mean)
     where Q are eigenvectors of the covariance of centered OLS fitted values.
     The intercept is not rank-constrained; rank = pred_len recovers OLS exactly.
  3. Exact train/val/test MSE for every rank 0..pred_len (no cap) in one pass.

Outputs (output/rrr_analysis_v2/{dataset}_{seq_len}_{pred_len}_{timestamp}/):
  metrics/rank_analysis.csv   rank, rrr_train_mse, rrr_val_mse, rrr_test_mse (rank 1..pred_len)
  metrics/ols_metrics.json    full-rank OLS (with bias) MSEs, rank-0 MSEs, run config
  metrics/eigenvalues.csv     spectrum of the centered fitted-value covariance
  logs/rrr_analysis_log.txt

Usage:
  python rrr_rank_analysis.py --dataset ETTh1 --pred_len 96
  python rrr_rank_analysis.py --dataset ETTh1 --pred_len 96 --check_sklearn
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.append('.')

from data_provider.data_factory import data_provider
from utils.analysis_utils import rrr_rank_sweep_with_intercept

ETT_DATASETS = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}
FEATURE_COUNTS = {
    'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7, 'weather': 21, 'exchange_rate': 8,
    'traffic': 862, 'electricity': 321, 'national_illness': 7,
}


class SimpleArgs:
    """Minimal args for data_provider."""
    def __init__(self, dataset='ETTh1', seq_len=720, pred_len=96, features='M', batch_size=32):
        # ETT datasets must use their dedicated loaders (standard 12/4/4-month splits);
        # everything else uses Dataset_Custom (70/10/20 split).
        self.data = dataset if dataset in ETT_DATASETS else 'custom'
        self.data_path = f'{dataset}.csv'
        self.seq_len = seq_len
        self.label_len = seq_len // 2
        self.pred_len = pred_len
        self.root_path = './dataset/'
        self.features = features
        self.target = 'OT'
        self.embed = 'fixed'
        self.freq = 'h'
        self.batch_size = batch_size
        self.num_workers = 0
        self.enc_in = FEATURE_COUNTS.get(dataset, 7)


class TeeOutput:
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_splits(dataset, seq_len, pred_len):
    args = SimpleArgs(dataset, seq_len, pred_len)
    splits = {}
    for flag in ('train', 'val', 'test'):
        ds, _ = data_provider(args, flag)
        splits[flag] = ds.data_x
    print(f"Loader: {type(ds).__name__} (args.data={args.data})")
    for k, v in splits.items():
        print(f"  {k}: data_x shape {v.shape}")
    return splits


def check_against_sklearn(train_x, seq_len, pred_len, info):
    """Compare our closed-form OLS to models/OLS.py (sklearn Ridge, svd solver)."""
    from models.OLS import OLS
    t0 = time.time()
    ols = OLS(dataset_train=train_x, context_length=seq_len, horizon=pred_len,
              instance_norm=False, individual=False, verbose=False)
    dW = np.abs(ols.weight_matrix - info['W']).max()
    db = np.abs(ols.bias - info['b']).max()
    print(f"sklearn check ({time.time() - t0:.1f}s): max|dW|={dW:.3e}, max|db|={db:.3e}, "
          f"max|W|={np.abs(info['W']).max():.3e}")
    return {'max_abs_dW': float(dW), 'max_abs_db': float(db)}


def run_rrr_analysis(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, f"{args.dataset}_{args.seq_len}_{args.pred_len}_{timestamp}")
    for sub in ('metrics', 'logs'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    tee = TeeOutput(os.path.join(output_dir, 'logs', 'rrr_analysis_log.txt'))
    original_stdout, sys.stdout = sys.stdout, tee
    try:
        t0 = time.time()
        print(f"Output directory: {output_dir}")
        print(f"Dataset={args.dataset} seq_len={args.seq_len} pred_len={args.pred_len} "
              f"alpha={args.alpha} (OLS with intercept, no instance norm, pooled channels)")

        splits = load_splits(args.dataset, args.seq_len, args.pred_len)
        df, info = rrr_rank_sweep_with_intercept(
            splits['train'], splits['val'], splits['test'], args.seq_len, args.pred_len,
            alpha=args.alpha, chunk_windows=args.chunk_windows)
        print(f"Sweep done in {time.time() - t0:.1f}s; rows per split: {info['n_rows']}")
        print(f"Intercept: max|b|={np.abs(info['b']).max():.4e}")

        full = df.iloc[-1]
        ols = {k: float(info['ols_mse_direct'][k]) for k in ('train', 'val', 'test')}
        print(f"Full OLS (with bias) - Train: {ols['train']:.6f}, Val: {ols['val']:.6f}, Test: {ols['test']:.6f}")
        best = df.iloc[1:].loc[df['rrr_val_mse'].iloc[1:].idxmin()]
        print(f"RRR val-argmin rank {int(best['rank'])}: Train {best['rrr_train_mse']:.6f}, "
              f"Val {best['rrr_val_mse']:.6f}, Test {best['rrr_test_mse']:.6f}")

        metrics = {
            'dataset': args.dataset, 'seq_len': args.seq_len, 'pred_len': args.pred_len,
            'alpha': args.alpha, 'instance_norm': False, 'intercept': True, 'individual': False,
            'rrr_formulation': 'centered (intercept not rank-constrained)',
            'loader': 'ETT' if args.dataset in ETT_DATASETS else 'custom',
            'ols_train_mse': ols['train'], 'ols_val_mse': ols['val'], 'ols_test_mse': ols['test'],
            'rank0_train_mse': float(df.iloc[0]['rrr_train_mse']),
            'rank0_val_mse': float(df.iloc[0]['rrr_val_mse']),
            'rank0_test_mse': float(df.iloc[0]['rrr_test_mse']),
            'full_rank_rrr_test_mse': float(full['rrr_test_mse']),
            'n_rows': info['n_rows'], 'max_abs_bias': float(np.abs(info['b']).max()),
            'runtime_s': round(time.time() - t0, 1), 'timestamp': timestamp,
        }
        if args.check_sklearn:
            metrics['sklearn_check'] = check_against_sklearn(splits['train'], args.seq_len, args.pred_len, info)

        mdir = os.path.join(output_dir, 'metrics')
        df[df['rank'] >= 1].to_csv(os.path.join(mdir, 'rank_analysis.csv'), index=False, float_format='%.8f')
        np.savetxt(os.path.join(mdir, 'eigenvalues.csv'), info['eigenvalues'], delimiter=',', header='eigenvalue')
        with open(os.path.join(mdir, 'ols_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {mdir} (total {time.time() - t0:.1f}s)")
        return output_dir
    finally:
        sys.stdout = original_stdout
        tee.close()


def main():
    p = argparse.ArgumentParser(description='RRR (with intercept) rank analysis over all ranks')
    p.add_argument('--dataset', default='ETTh1')
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--alpha', type=float, default=1e-6, help='Ridge stabiliser, matches models/OLS.py')
    p.add_argument('--chunk_windows', type=int, default=512, help='windows per block (memory control)')
    p.add_argument('--output_root', default='output/rrr_analysis_v2')
    p.add_argument('--check_sklearn', action='store_true', help='cross-check OLS against models/OLS.py')
    p.add_argument('--all_horizons', action='store_true')
    args = p.parse_args()

    horizons = [96, 192, 336, 720] if args.all_horizons else [args.pred_len]
    for h in horizons:
        args.pred_len = h
        run_rrr_analysis(args)


if __name__ == "__main__":
    main()
