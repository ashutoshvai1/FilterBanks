"""
Analysis utilities for SVD analysis.

This module contains functions for performing SVD analysis, RRR computation, 
and weight matrix comparisons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def perform_svd_analysis(weight_matrix, keep_top_k=None, clipping_ratio=0.05, output_dir="output/svd_analysis"):
    """
    Perform SVD decomposition and clipping on the weight matrix.
    If keep_top_k is specified, keeps only the top k singular values.
    Otherwise, uses clipping_ratio for thresholding.
    """
    print(f"Performing SVD analysis on weight matrix of shape {weight_matrix.shape}")
    
    # Convert to torch tensor if numpy
    if isinstance(weight_matrix, np.ndarray):
        W = torch.tensor(weight_matrix, dtype=torch.float32)
    else:
        W = weight_matrix.detach().cpu().float()
    
    # Perform SVD decomposition
    U, S, V = torch.linalg.svd(W)
    
    print(f"SVD shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
    print(f"Largest singular value: {S[0].item():.6f}")
    print(f"Smallest singular value: {S[-1].item():.6f}")
    
    # Plot singular values and save to output/svd_analysis/
    output_dir = "output/svd_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(S.numpy())
    plt.ylabel("Singular Values")
    plt.xlabel("Index")
    plt.title("Singular Values")
    # plt.yscale('log')
    plot_path = os.path.join(output_dir, "singular_values.png")
    plt.savefig(plot_path)
    print(f"Singular values plot saved to {plot_path}")
    
    # Create clipped singular values
    S_clipped = S.clone()
    
    if keep_top_k is not None:
        # Keep only the top k singular values
        keep_top_k = min(keep_top_k, len(S))  # Ensure k doesn't exceed number of singular values
        S_clipped[keep_top_k:] = 0
        num_kept = keep_top_k
        print(f"Keeping top {num_kept} out of {len(S)} singular values (filterbank depth-based)")
    else:
        # Use clipping ratio threshold
        threshold = clipping_ratio * S[0]
        mask = S_clipped < threshold
        S_clipped[mask] = 0
        num_kept = (S_clipped > 0).sum().item()
        print(f"Kept {num_kept} out of {len(S)} singular values (threshold: {threshold:.6f})")
    
    # Reconstruct the clipped weight matrix
    # For SVD of an m x n matrix, U is m x min(m,n), S is min(m,n), V is n x n
    # We need to use only the first min(m,n) rows of V for reconstruction
    min_dim = min(U.shape[0], V.shape[0])
    W_clipped = U @ torch.diag(S_clipped) @ V[:min_dim, :]
    
    # Plot comparison of singular values
    plt.subplot(1, 2, 2)
    plt.plot(S.numpy(), label='Original', alpha=0.7)
    plt.plot(S_clipped.numpy(), label=f'Clipped (top {num_kept})' if keep_top_k else 'Clipped', alpha=0.7)
    plt.ylabel("Singular Values")
    plt.xlabel("Index")
    plt.title("Original vs Clipped Singular Values")
    plt.legend()
    # plt.yscale('log')
    plt.tight_layout()
    # Save the second plot as well
    comparison_plot_path = os.path.join(output_dir, "singular_values_comparison.png")
    plt.savefig(comparison_plot_path)
    print(f"Comparison plot saved to {comparison_plot_path}")
    plt.show()
    
    # Calculate reconstruction error
    reconstruction_error = torch.norm(W - W_clipped, 'fro').item()
    relative_error = reconstruction_error / torch.norm(W, 'fro').item()
    
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    print(f"Relative reconstruction error: {relative_error:.6f}")
    print(f"Effective rank (non-zero singular values): {(S_clipped > 0).sum().item()}")
    
    return W, W_clipped, U, S, S_clipped, V


def compute_rrr_solution(ols_weight_matrix, train_data_x, seq_len, pred_len, keep_top_k=None, clipping_ratio=0.05, instance_norm=False, output_dir="output/svd_analysis"):
    """
    Compute Reduced Rank Regression (RRR) solution.
    
    RRR performs SVD on X*B̂_OLS and uses the top-k right singular vectors
    to construct B_RRR = B̂_OLS * V_r * V_r^T where V_r contains the first r 
    right singular vectors.
    
    Args:
        ols_weight_matrix: OLS solution matrix B̂_OLS
        train_data_x: Training data for constructing X matrix
        seq_len: Sequence length  
        pred_len: Prediction length
        keep_top_k: Number of singular vectors to keep (if None, uses clipping_ratio)
        clipping_ratio: Threshold for clipping small singular values
        instance_norm: Whether instance normalization was used in OLS
        
    Returns:
        B_RRR: Reduced rank regression solution
        svd_info: Dictionary with SVD decomposition information
    """
    print(f"Computing RRR solution with keep_top_k={keep_top_k}, clipping_ratio={clipping_ratio}")
    
    # Construct X matrix from training data
    instances = np.lib.stride_tricks.sliding_window_view(
        train_data_x, (seq_len + pred_len), axis=0
    )
    
    if len(instances) == 0:
        raise ValueError("No training instances available for RRR computation")
    
    X = instances[:, :, :seq_len]  # Shape: [n_instances, n_features, seq_len]
    
    if instance_norm:
        # Apply instance normalization to match OLS training
        X_norm = X.copy()
        context_means = np.mean(X_norm, axis=2, keepdims=True)
        X_norm = X_norm - context_means
        context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
        X_flat = np.concatenate((X_norm, context_stds), axis=2).reshape(-1, X_norm.shape[-1] + 1)
    else:
        X_flat = X.reshape(-1, X.shape[-1])  # Shape: [n_instances * n_features, seq_len]
    
    print(f"X_flat shape: {X_flat.shape}")
    print(f"OLS weight matrix shape: {ols_weight_matrix.shape}")
    
    # Convert OLS weight matrix to numpy if needed
    if isinstance(ols_weight_matrix, torch.Tensor):
        B_ols = ols_weight_matrix.detach().cpu().numpy()
    else:
        B_ols = ols_weight_matrix
    
    # Compute X * B̂_OLS
    XB = X_flat @ B_ols.T  # Shape: [n_instances * n_features, pred_len]
    print(f"X*B_OLS shape: {XB.shape}")
    
    # Perform SVD on X * B̂_OLS
    U, S, V = np.linalg.svd(XB, full_matrices=False)
    
    print(f"SVD shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
    print(f"Largest singular value: {S[0]:.6f}")
    print(f"Smallest singular value: {S[-1]:.6f}")
    
    # Determine rank reduction
    if keep_top_k is not None:
        # Keep only the top k singular values
        r = min(keep_top_k, len(S))
        print(f"Using top {r} singular vectors (filterbank depth-based)")
    else:
        # Use clipping ratio threshold  
        threshold = clipping_ratio * S[0]
        r = np.sum(S >= threshold)
        print(f"Using top {r} singular vectors (threshold: {threshold:.6f})")
    
    # Get the top-r right singular vectors
    V_r = V[:r, :]  # Shape: [r, pred_len]
    
    # Construct RRR solution: B_RRR = B̂_OLS * V_r^T * V_r
    # Since V_r is (r, pred_len), we need V_r^T * V_r to be (pred_len, pred_len)
    # Then B_RRR = B̂_OLS @ (V_r^T @ V_r) where B̂_OLS is (pred_len, seq_len)
    # But this doesn't work dimensionally. Let me correct this based on the RRR formula.
    
    # The correct RRR formula: B_RRR = (V_r^T @ V_r) @ B̂_OLS
    # where V_r^T @ V_r creates a projection matrix of size (pred_len, pred_len)
    projection_matrix = V_r.T @ V_r  # Shape: [pred_len, pred_len]
    B_RRR = projection_matrix @ B_ols  # Shape: [pred_len, seq_len]
    
    print(f"RRR solution shape: {B_RRR.shape}")
    print(f"Effective rank: {r}")
    
    # Calculate reconstruction error for X*B̂_OLS
    XB_reconstructed = U[:, :r] @ np.diag(S[:r]) @ V_r
    reconstruction_error = np.linalg.norm(XB - XB_reconstructed, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(XB, 'fro')
    
    print(f"XB reconstruction error: {reconstruction_error:.6f}")
    print(f"XB relative reconstruction error: {relative_error:.6f}")
    
    # Create visualization of singular values
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(S, 'b-', alpha=0.7, label='XB Singular Values')
    plt.axvline(x=r-1, color='r', linestyle='--', label=f'Rank cutoff (r={r})')
    plt.ylabel("Singular Values")
    plt.xlabel("Index") 
    plt.title("RRR: Singular Values of X*B̂_OLS")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(S, 'b-', alpha=0.7, label='Original')
    S_truncated = S.copy()
    S_truncated[r:] = 0
    plt.plot(S_truncated, 'r-', alpha=0.7, label=f'Truncated (top {r})')
    plt.ylabel("Singular Values")
    plt.xlabel("Index")
    plt.title("RRR: Original vs Truncated")
    plt.legend()
    
    plt.tight_layout()
    rrr_plot_path = os.path.join(plots_dir, "rrr_singular_values.png")
    plt.savefig(rrr_plot_path)
    print(f"RRR singular values plot saved to {rrr_plot_path}")
    plt.close()  # Close the figure to free memory
    
    svd_info = {
        'U': U,
        'S': S, 
        'V': V,
        'V_r': V_r,
        'rank': r,
        'reconstruction_error': reconstruction_error,
        'relative_error': relative_error
    }
    
    return B_RRR, svd_info


def compute_weight_differences(weights_dict, baseline_method='RRR'):
    """
    Compute Frobenius norm differences between weight matrices using RRR as baseline.
    
    Args:
        weights_dict: Dictionary with method names as keys and weight matrices as values
                     Expected keys: 'Full_OLS', 'SVD_OLS', 'RRR', 'Filterbank'
        baseline_method: Method to use as baseline for comparisons (default: 'RRR')
        
    Returns:
        differences_dict: Dictionary with comparison results
    """
    print(f"Computing weight differences using {baseline_method} as baseline...")
    
    if baseline_method not in weights_dict:
        raise ValueError(f"Baseline method '{baseline_method}' not found in weights_dict")
    
    baseline_weights = weights_dict[baseline_method]
    baseline_norm = np.linalg.norm(baseline_weights, 'fro')
    
    differences = {}
    
    for method_name, weights in weights_dict.items():
        if method_name == baseline_method:
            continue
            
        # Compute absolute Frobenius norm difference
        frobenius_diff = np.linalg.norm(weights - baseline_weights, 'fro')
        
        # Compute relative difference (normalized by baseline norm)
        relative_diff = frobenius_diff / baseline_norm if baseline_norm > 0 else 0.0
        
        comparison_name = f"{method_name}_vs_{baseline_method}"
        differences[comparison_name] = {
            'frobenius_norm_diff': frobenius_diff,
            'relative_diff': relative_diff,
            'baseline_norm': baseline_norm,
            'method_norm': np.linalg.norm(weights, 'fro')
        }
        
        print(f"{comparison_name}: Frobenius diff = {frobenius_diff:.6f}, Relative diff = {relative_diff:.6f}")
    
    return differences


def get_rank_range(max_rank, interval=10):
    """
    Generate rank values to test for rank analysis.
    
    Args:
        max_rank: Maximum rank to test (typically min(n_features*seq_len, pred_len))
        interval: Interval between rank values (default: 10)
        
    Returns:
        ranks: List of rank values to test
    """
    ranks = list(range(1, max_rank + 1, interval))
    # Always include max_rank if not already included
    if ranks[-1] != max_rank:
        ranks.append(max_rank)
    return ranks


def evaluate_svd_at_rank(ols_weight_matrix, rank, train_data_x, val_data_x, test_data_x, 
                        seq_len, pred_len, instance_norm=False):
    """
    Evaluate SVD-truncated OLS at a specific rank.
    
    Args:
        ols_weight_matrix: Original OLS weight matrix
        rank: Number of singular values to keep
        train_data_x, val_data_x, test_data_x: Raw data arrays
        seq_len, pred_len: Sequence and prediction lengths
        instance_norm: Whether to apply instance normalization
        
    Returns:
        train_mse, val_mse, test_mse: MSE values for each split
    """
    # Convert to torch tensor if numpy
    if isinstance(ols_weight_matrix, np.ndarray):
        W = torch.tensor(ols_weight_matrix, dtype=torch.float32)
    else:
        W = ols_weight_matrix.detach().cpu().float()
    
    # Perform SVD and truncate to specified rank
    U, S, V = torch.linalg.svd(W)
    
    # Truncate to rank
    rank = min(rank, len(S))
    S_truncated = S.clone()
    S_truncated[rank:] = 0
    
    # Reconstruct weight matrix
    min_dim = min(U.shape[0], V.shape[0])
    W_truncated = U @ torch.diag(S_truncated) @ V[:min_dim, :]
    
    # We need to evaluate each split separately
    def _evaluate_single_split(data_x, split_name):
        # Create a simple function to evaluate on a single dataset
        instances = np.lib.stride_tricks.sliding_window_view(
            data_x, (seq_len + pred_len), axis=0
        )
        
        if len(instances) == 0:
            return float('inf')
        
        X = instances[:, :, :seq_len]
        y_true = instances[:, :, seq_len:]
        
        if instance_norm:
            # Apply instance normalization
            X_norm = X.copy()
            context_means = np.mean(X_norm, axis=2, keepdims=True)
            X_norm = X_norm - context_means
            context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
            X_norm = np.concatenate((X_norm, context_stds), axis=2)
            
            w_matrix = W_truncated.numpy()
            
            X_flat = X_norm.reshape(-1, X_norm.shape[-1])
            y_pred_flat = X_flat @ w_matrix.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
            y_pred = y_pred + context_means
        else:
            # No instance normalization - use raw data
            X_flat = X.reshape(-1, X.shape[-1])
            w_matrix = W_truncated.numpy()
            
            y_pred_flat = X_flat @ w_matrix.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
        
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    train_mse = _evaluate_single_split(train_data_x, 'train')
    val_mse = _evaluate_single_split(val_data_x, 'val')
    test_mse = _evaluate_single_split(test_data_x, 'test')
    
    return train_mse, val_mse, test_mse


def evaluate_rrr_at_rank(ols_weight_matrix, rank, train_data_x, val_data_x, test_data_x, 
                        seq_len, pred_len, instance_norm=False):
    """
    Evaluate RRR at a specific rank.
    
    Args:
        ols_weight_matrix: Original OLS weight matrix
        rank: Number of singular vectors to keep
        train_data_x, val_data_x, test_data_x: Raw data arrays
        seq_len, pred_len: Sequence and prediction lengths
        instance_norm: Whether to apply instance normalization
        
    Returns:
        train_mse, val_mse, test_mse: MSE values for each split
    """
    # Construct X matrix from training data for RRR computation
    instances = np.lib.stride_tricks.sliding_window_view(
        train_data_x, (seq_len + pred_len), axis=0
    )
    
    if len(instances) == 0:
        return float('inf'), float('inf'), float('inf')
    
    X = instances[:, :, :seq_len]  # Shape: [n_instances, n_features, seq_len]
    
    if instance_norm:
        # Apply instance normalization to match OLS training
        X_norm = X.copy()
        context_means = np.mean(X_norm, axis=2, keepdims=True)
        X_norm = X_norm - context_means
        context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
        X_flat = np.concatenate((X_norm, context_stds), axis=2).reshape(-1, X_norm.shape[-1] + 1)
    else:
        X_flat = X.reshape(-1, X.shape[-1])  # Shape: [n_instances * n_features, seq_len]
    
    # Convert OLS weight matrix to numpy if needed
    if isinstance(ols_weight_matrix, torch.Tensor):
        B_ols = ols_weight_matrix.detach().cpu().numpy()
    else:
        B_ols = ols_weight_matrix
    
    # Compute X * B̂_OLS and perform SVD
    XB = X_flat @ B_ols.T  # Shape: [n_instances * n_features, pred_len]
    U, S, V = np.linalg.svd(XB, full_matrices=False)
    
    # Truncate to specified rank
    r = min(rank, len(S))
    V_r = V[:r, :]  # Shape: [r, pred_len]
    
    # Construct RRR solution
    projection_matrix = V_r.T @ V_r  # Shape: [pred_len, pred_len]
    B_RRR = projection_matrix @ B_ols  # Shape: [pred_len, seq_len]
    
    # Evaluate RRR solution on each split
    def _evaluate_single_split(data_x, split_name):
        instances = np.lib.stride_tricks.sliding_window_view(
            data_x, (seq_len + pred_len), axis=0
        )
        
        if len(instances) == 0:
            return float('inf')
        
        X = instances[:, :, :seq_len]
        y_true = instances[:, :, seq_len:]
        
        if instance_norm:
            # Apply instance normalization
            X_norm = X.copy()
            context_means = np.mean(X_norm, axis=2, keepdims=True)
            X_norm = X_norm - context_means
            context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
            X_norm = np.concatenate((X_norm, context_stds), axis=2)
            
            X_flat = X_norm.reshape(-1, X_norm.shape[-1])
            y_pred_flat = X_flat @ B_RRR.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
            y_pred = y_pred + context_means
        else:
            # No instance normalization - use raw data
            X_flat = X.reshape(-1, X.shape[-1])
            
            y_pred_flat = X_flat @ B_RRR.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
        
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    train_mse = _evaluate_single_split(train_data_x, 'train')
    val_mse = _evaluate_single_split(val_data_x, 'val')
    test_mse = _evaluate_single_split(test_data_x, 'test')
    
    return train_mse, val_mse, test_mse


def evaluate_svd_with_precomputed(U, S, V, rank, train_data_x, val_data_x, test_data_x, 
                                 seq_len, pred_len, instance_norm=False):
    """
    Evaluate SVD-truncated OLS at a specific rank using precomputed SVD decomposition.
    
    Args:
        U, S, V: Precomputed SVD decomposition components
        rank: Number of singular values to keep
        train_data_x, val_data_x, test_data_x: Raw data arrays
        seq_len, pred_len: Sequence and prediction lengths
        instance_norm: Whether to apply instance normalization
        
    Returns:
        train_mse, val_mse, test_mse: MSE values for each split
    """
    # Truncate to rank
    rank = min(rank, len(S))
    S_truncated = S.clone()
    S_truncated[rank:] = 0
    
    # Reconstruct weight matrix
    min_dim = min(U.shape[0], V.shape[0])
    W_truncated = U @ torch.diag(S_truncated) @ V[:min_dim, :]
    
    # Evaluate on each split using the existing evaluation logic
    def _evaluate_single_split(data_x, split_name):
        instances = np.lib.stride_tricks.sliding_window_view(
            data_x, (seq_len + pred_len), axis=0
        )
        
        if len(instances) == 0:
            return float('inf')
        
        X = instances[:, :, :seq_len]
        y_true = instances[:, :, seq_len:]
        
        if instance_norm:
            # Apply instance normalization
            X_norm = X.copy()
            context_means = np.mean(X_norm, axis=2, keepdims=True)
            X_norm = X_norm - context_means
            context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
            X_norm = np.concatenate((X_norm, context_stds), axis=2)
            
            X_flat = X_norm.reshape(-1, X_norm.shape[-1])
            W_np = W_truncated.detach().cpu().numpy()
            y_pred_flat = X_flat @ W_np.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
            y_pred = y_pred + context_means
        else:
            # No instance normalization - use raw data
            X_flat = X.reshape(-1, X.shape[-1])
            
            W_np = W_truncated.detach().cpu().numpy()
            y_pred_flat = X_flat @ W_np.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
        
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    train_mse = _evaluate_single_split(train_data_x, 'train')
    val_mse = _evaluate_single_split(val_data_x, 'val')
    test_mse = _evaluate_single_split(test_data_x, 'test')
    
    return train_mse, val_mse, test_mse


def evaluate_rrr_with_precomputed(V_rrr, B_ols, rank, train_data_x, val_data_x, test_data_x, 
                                 seq_len, pred_len, instance_norm=False):
    """
    Evaluate RRR at a specific rank using precomputed SVD decomposition.
    
    Args:
        V_rrr: Right singular vectors from SVD of X*B_hat_OLS
        B_ols: Original OLS weight matrix
        rank: Number of singular vectors to keep
        train_data_x, val_data_x, test_data_x: Raw data arrays
        seq_len, pred_len: Sequence and prediction lengths
        instance_norm: Whether to apply instance normalization
        
    Returns:
        train_mse, val_mse, test_mse: MSE values for each split
    """
    # Truncate to specified rank
    r = min(rank, V_rrr.shape[0])
    V_r = V_rrr[:r, :]  # Shape: [r, pred_len]
    
    # Construct RRR solution
    projection_matrix = V_r.T @ V_r  # Shape: [pred_len, pred_len]
    B_RRR = projection_matrix @ B_ols
    
    # Evaluate RRR solution on each split
    def _evaluate_single_split(data_x, split_name):
        instances = np.lib.stride_tricks.sliding_window_view(
            data_x, (seq_len + pred_len), axis=0
        )
        
        if len(instances) == 0:
            return float('inf')
        
        X = instances[:, :, :seq_len]
        y_true = instances[:, :, seq_len:]
        
        if instance_norm:
            # Apply instance normalization
            X_norm = X.copy()
            context_means = np.mean(X_norm, axis=2, keepdims=True)
            X_norm = X_norm - context_means
            context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
            X_norm = np.concatenate((X_norm, context_stds), axis=2)
            
            X_flat = X_norm.reshape(-1, X_norm.shape[-1])
            y_pred_flat = X_flat @ B_RRR.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
            y_pred = y_pred + context_means
        else:
            # No instance normalization - use raw data
            X_flat = X.reshape(-1, X.shape[-1])
            
            y_pred_flat = X_flat @ B_RRR.T
            y_pred = y_pred_flat.reshape(X.shape[0], X.shape[1], pred_len)
        
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    train_mse = _evaluate_single_split(train_data_x, 'train')
    val_mse = _evaluate_single_split(val_data_x, 'val')
    test_mse = _evaluate_single_split(test_data_x, 'test')
    
    return train_mse, val_mse, test_mse


def rank_analysis(ols_weight_matrix, train_data_x, val_data_x, test_data_x, 
                 seq_len, pred_len, rank_interval=10, instance_norm=False, output_dir="output/svd_analysis"):
    """
    Perform comprehensive rank analysis with validation curves.
    
    Args:
        ols_weight_matrix: Original OLS weight matrix
        train_data_x, val_data_x, test_data_x: Raw data arrays for each split
        seq_len, pred_len: Sequence and prediction lengths
        rank_interval: Interval between rank values to test (default: 10)
        instance_norm: Whether to apply instance normalization
        output_dir: Directory to save results
        
    Returns:
        results_df: DataFrame with rank analysis results
    """
    print("=" * 50)
    print("RANK ANALYSIS WITH VALIDATION CURVES")
    print("=" * 50)
    
    # Determine maximum rank to test
    if isinstance(ols_weight_matrix, torch.Tensor):
        weight_shape = ols_weight_matrix.shape
    else:
        weight_shape = ols_weight_matrix.shape
    
    # For weight matrix of shape [pred_len, seq_len], max rank is min(pred_len, seq_len)
    max_rank = min(weight_shape[0], weight_shape[1], 60)
    print(f"Weight matrix shape: {weight_shape}")
    print(f"Maximum possible rank: {max_rank}")
    
    # Generate rank values to test
    ranks_to_test = get_rank_range(max_rank, rank_interval)
    print(f"Testing ranks: {ranks_to_test}")
    
    # Pre-compute SVD decompositions once to reuse across all ranks
    print("\nPre-computing SVD decompositions...")
    
    # 1. SVD of weight matrix for SVD-truncated OLS
    if isinstance(ols_weight_matrix, np.ndarray):
        W = torch.tensor(ols_weight_matrix, dtype=torch.float32)
    else:
        W = ols_weight_matrix.detach().cpu().float()
    
    U_svd, S_svd, V_svd = torch.linalg.svd(W)
    print(f"  SVD of weight matrix completed (shape: {W.shape})")
    
    # 2. SVD of X*B_hat_OLS for RRR
    # Construct X matrix from training data for RRR computation
    instances = np.lib.stride_tricks.sliding_window_view(
        train_data_x, (seq_len + pred_len), axis=0
    )
    
    if len(instances) == 0:
        print("Error: No valid instances found in training data")
        return pd.DataFrame()
    
    X = instances[:, :, :seq_len]  # Shape: [n_instances, n_features, seq_len]
    
    if instance_norm:
        # Apply instance normalization to match OLS training
        X_norm = X.copy()
        context_means = np.mean(X_norm, axis=2, keepdims=True)
        X_norm = X_norm - context_means
        context_stds = np.sqrt(np.var(X_norm, axis=2, keepdims=True) + 1e-5)
        X_flat = np.concatenate((X_norm, context_stds), axis=2).reshape(-1, X_norm.shape[-1] + 1)
    else:
        X_flat = X.reshape(-1, X.shape[-1])  # Shape: [n_instances * n_features, seq_len]
    
    # Convert OLS weight matrix to numpy if needed
    if isinstance(ols_weight_matrix, torch.Tensor):
        B_ols = ols_weight_matrix.detach().cpu().numpy()
    else:
        B_ols = ols_weight_matrix
    
    # Compute X * B̂_OLS and perform SVD once
    XB = X_flat @ B_ols.T  # Shape: [n_instances * n_features, pred_len]
    U_rrr, S_rrr, V_rrr = np.linalg.svd(XB, full_matrices=False)
    print(f"  SVD of X*B_OLS completed (shape: {XB.shape})")
    
    # Initialize results storage
    results = {
        'rank': [],
        'rrr_train_mse': [],
        'rrr_val_mse': [],
        'rrr_test_mse': []
    }
    
    print(f"\nEvaluating {len(ranks_to_test)} different ranks...")
    
    for i, rank in enumerate(ranks_to_test):
        print(f"\nRank {rank} ({i+1}/{len(ranks_to_test)})...")
        
        
        # Evaluate RRR at this rank (reusing precomputed SVD)
        print(f"  Evaluating RRR at rank {rank}")
        rrr_train, rrr_val, rrr_test = evaluate_rrr_with_precomputed(
            V_rrr, B_ols, rank, train_data_x, val_data_x, test_data_x,
            seq_len, pred_len, instance_norm
        )
        
        # Store results
        results['rank'].append(rank)
        results['rrr_train_mse'].append(rrr_train)
        results['rrr_val_mse'].append(rrr_val)
        results['rrr_test_mse'].append(rrr_test)
        
        print(f"  RRR MSE - Train: {rrr_train:.6f}, Val: {rrr_val:.6f}, Test: {rrr_test:.6f}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(metrics_dir, 'rank_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nRank analysis results saved to {csv_path}")
    
    # Create validation curves plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Convert MSE to dB scale
    rrr_train_mse_db = 10 * np.log10(results['rrr_train_mse'])
    rrr_val_mse_db = 10 * np.log10(results['rrr_val_mse'])
    rrr_test_mse_db = 10 * np.log10(results['rrr_test_mse'])

    # RRR Validation Curves
    # plt.subplot(2, 2, 2)
    plt.plot(results['rank'], rrr_train_mse_db, 'o-', label='Train', alpha=0.7)
    plt.plot(results['rank'], rrr_val_mse_db, 's-', label='Validation', alpha=0.7)
    plt.plot(results['rank'], rrr_test_mse_db, '^-', label='Test', alpha=0.7)
    plt.xlabel('Rank')
    plt.ylabel('MSE (dB)')
    plt.title('RRR performance vs rank')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'rank_analysis_validation_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Validation curves plot saved to {plot_path}")
    plt.close()
        
    return results_df 

# ---------------------------------------------------------------------------
# Reduced-rank regression with intercept (centered formulation), all ranks.
# ---------------------------------------------------------------------------

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
