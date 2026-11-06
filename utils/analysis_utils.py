"""Reduced-rank regression (RRR) with an intercept, evaluated at every rank.

Used by rrr_rank_analysis.py (paper Sec. 3.1.1, Fig. 1).
"""

import numpy as np
import pandas as pd

def _iter_window_chunks(data_x, seq_len, pred_len, chunk_windows=512):
    """Yield (X, Y) float64 blocks of pooled-channel windows.

    Windows match models/OLS.py (sliding_window_view over time, channels pooled):
    X: [n_windows_chunk * n_channels, seq_len], Y: [..., pred_len].
    """
    inst = np.lib.stride_tricks.sliding_window_view(data_x, seq_len + pred_len, axis=0)
    for i in range(0, inst.shape[0], chunk_windows):
        block = np.asarray(inst[i:i + chunk_windows], dtype=np.float64).reshape(-1, seq_len + pred_len)
        yield block[:, :seq_len], block[:, seq_len:]


def fit_ols_with_intercept(train_x, seq_len, pred_len, alpha=1e-6, chunk_windows=512):
    """OLS with intercept via accumulated sufficient statistics.

    Identical to sklearn Ridge(alpha, fit_intercept=True) as used in models/OLS.py
    (instance_norm=False, individual=False): solves (Xc'Xc + alpha I) W' = Xc'Yc on
    centered data and sets b = y_mean - W x_mean.

    Returns W [pred_len, seq_len], b [pred_len], x_mean [seq_len], y_mean [pred_len], Cxx [seq_len, seq_len].
    """
    n = 0
    sx = np.zeros(seq_len); sy = np.zeros(pred_len)
    sxx = np.zeros((seq_len, seq_len)); sxy = np.zeros((seq_len, pred_len))
    for X, Y in _iter_window_chunks(train_x, seq_len, pred_len, chunk_windows):
        n += X.shape[0]
        sx += X.sum(0); sy += Y.sum(0)
        sxx += X.T @ X; sxy += X.T @ Y
    x_mean, y_mean = sx / n, sy / n
    cxx = sxx - n * np.outer(x_mean, x_mean)
    cxy = sxy - n * np.outer(x_mean, y_mean)
    W = np.linalg.solve(cxx + alpha * np.eye(seq_len), cxy).T
    b = y_mean - W @ x_mean
    return W, b, x_mean, y_mean, cxx, n


def rrr_rank_sweep_with_intercept(train_x, val_x, test_x, seq_len, pred_len,
                                  alpha=1e-6, chunk_windows=512):
    """Exact train/val/test MSE of centered RRR at every rank r = 0..pred_len.

    Model at rank r (standard RRR with an unconstrained intercept):
        y_hat = y_mean + Q_r Q_r' W (x - x_mean)
    where Q holds the eigenvectors of W Cxx W' (= covariance of centered OLS fitted
    values on train), sorted by decreasing eigenvalue. r = pred_len recovers OLS
    with its bias exactly; r = 0 is the train-mean forecast.

    Because Q is square orthogonal, the squared error splits per eigen-direction:
        ||y - y_hat_r||^2 = sum_{k<r} (t_k - g_k)^2 + sum_{k>=r} t_k^2,
    with t = Q'(y - y_mean), g = Q'W(x - x_mean). One pass per split gives all ranks.

    Returns (results_df, info) where results_df has columns
    rank, rrr_train_mse, rrr_val_mse, rrr_test_mse for rank = 0..pred_len.
    """
    W, b, x_mean, y_mean, cxx, n_train = fit_ols_with_intercept(
        train_x, seq_len, pred_len, alpha=alpha, chunk_windows=chunk_windows)

    M = W @ cxx @ W.T
    evals, Q = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(evals)[::-1]
    evals, Q = evals[order], Q[:, order]

    A = W.T @ Q                     # [seq_len, pred_len]
    c = x_mean @ A                  # [pred_len]
    yq = y_mean @ Q                 # [pred_len]

    def _sweep(data_x):
        n = 0
        d = np.zeros(pred_len); s = np.zeros(pred_len)
        sse_ols = 0.0
        for X, Y in _iter_window_chunks(data_x, seq_len, pred_len, chunk_windows):
            n += X.shape[0]
            g = X @ A - c
            t = Y @ Q - yq
            d += ((t - g) ** 2).sum(0)
            s += (t ** 2).sum(0)
            sse_ols += ((Y - (X @ W.T + b)) ** 2).sum()   # direct OLS check
        total = s.sum()
        sse = np.concatenate([[total], np.cumsum(d) + total - np.cumsum(s)])
        return sse / (n * pred_len), sse_ols / (n * pred_len), n

    out, info = {}, {'n_rows': {}, 'ols_mse_direct': {}}
    for name, data_x in (('train', train_x), ('val', val_x), ('test', test_x)):
        mse, ols_direct, n = _sweep(data_x)
        out[name] = mse
        info['n_rows'][name] = n
        info['ols_mse_direct'][name] = ols_direct
        # rank = pred_len must reproduce OLS with bias
        if not np.isclose(mse[-1], ols_direct, rtol=1e-6, atol=1e-9):
            raise RuntimeError(f"Full-rank RRR ({mse[-1]:.8f}) != OLS ({ols_direct:.8f}) on {name}")

    results_df = pd.DataFrame({
        'rank': np.arange(pred_len + 1),
        'rrr_train_mse': out['train'],
        'rrr_val_mse': out['val'],
        'rrr_test_mse': out['test'],
    })
    info.update({'W': W, 'b': b, 'x_mean': x_mean, 'y_mean': y_mean,
                 'eigenvalues': evals, 'n_train_rows': n_train})
    return results_df, info
