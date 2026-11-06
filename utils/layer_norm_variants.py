"""
Local implementations of RINorm and _ResidualBlock.
Extracted to avoid the heavy darts dependency and version conflicts.

Original sources:
- RINorm: darts.models.components.layer_norm_variants
- _ResidualBlock: darts.models.forecasting.tide_model
"""

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Structure:
    - Dense path: Linear -> ReLU -> Linear -> Dropout
    - Skip path: Linear
    - Output: dense_output + skip_output
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_size: Hidden layer size
        use_layer_norm: Whether to apply layer normalization (at the end)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super(_ResidualBlock, self).__init__()
        
        # Dense path: Linear -> ReLU -> Linear -> Dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim)
        
        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection applied
        """
        dense_output = self.dense(x)
        skip_output = self.skip(x)
        output = dense_output + skip_output
        
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output


class RINorm(nn.Module):
    """
    Reversible Instance Normalization for time series.
    
    Normalizes input along the time dimension (dim=1) for each channel independently.
    Stores the mean and std for later reversal via the inverse() method.
    
    Args:
        input_dim: Number of input channels/features
        eps: Small constant for numerical stability
        affine: Whether to include learnable affine parameters
    """
    
    def __init__(self, input_dim: int, eps: float = 1e-5, affine: bool = True):
        super(RINorm, self).__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(input_dim))
            self.beta = nn.Parameter(torch.zeros(input_dim))
        
        # Placeholders for storing normalization statistics
        self.mean = None
        self.std = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input and store statistics for reversal.
        
        Args:
            x: Input tensor of shape [Batch, Time, Channels] or [Batch, Time, Channels, 1]
        
        Returns:
            Normalized tensor of same shape
        """
        # Handle optional 4th dimension
        squeeze_last = False
        if x.dim() == 4 and x.size(-1) == 1:
            squeeze_last = True
            x = x.squeeze(-1)
        
        # Compute mean and std along time dimension (dim=1)
        # Keep dims for broadcasting
        self.mean = x.mean(dim=1, keepdim=True)  # [Batch, 1, Channels]
        self.std = x.std(dim=1, keepdim=True) + self.eps  # [Batch, 1, Channels]
        
        # Normalize
        x_norm = (x - self.mean) / self.std
        
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        
        if squeeze_last:
            x_norm = x_norm.unsqueeze(-1)
        
        return x_norm
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization using stored statistics.
        
        Args:
            x: Normalized tensor of shape [Batch, Time, Channels] or [Batch, Time, Channels, 1]
        
        Returns:
            Denormalized tensor of same shape
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot call inverse() before forward(). "
                             "Mean and std have not been computed.")
        
        # Handle optional 4th dimension
        squeeze_last = False
        if x.dim() == 4 and x.size(-1) == 1:
            squeeze_last = True
            x = x.squeeze(-1)
        
        # Remove affine transformation if applied
        if self.affine:
            x = (x - self.beta) / self.gamma
        
        # Denormalize
        x_denorm = x * self.std + self.mean
        
        if squeeze_last:
            x_denorm = x_denorm.unsqueeze(-1)
        
        return x_denorm

