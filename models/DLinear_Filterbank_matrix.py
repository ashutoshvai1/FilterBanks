# Matrix-based implementation of filter bank using y_hat = Gamma * F * x formulation
# F matrix contains convolution kernel weights, Gamma contains linear layer weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layer_norm_variants import RINorm


class Model(nn.Module):
    """
    Matrix-based CoDeNet Filter Bank Implementation using y_hat = Gamma * F * x
    
    For a single channel:
    - x: [seq_len] input
    - F: [depth, seq_len] matrix where each row is a convolution filter positioned for last output
    - Gamma: [pred_len, depth] matrix for linear transformation
    - y_hat = Gamma * F * x: [pred_len] predictions
    
    The same matrices F and Gamma are shared across all channels.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.depth = configs.depth  # number of filters (nf)
        self.kernel_size = configs.filter_len
        self.channels = configs.enc_in
        self.use_revIN = configs.use_revIN
        
        if self.use_revIN:
            self.rin = RINorm(input_dim=self.channels)
        
        # Matrix F: [depth, seq_len] - convolution filters positioned for last output
        # Each row represents a convolution kernel positioned to compute the last convolution output
        self.F = nn.Parameter(torch.randn(self.depth, self.seq_len))
        
        # Matrix Gamma: [pred_len, depth] - linear transformation weights  
        self.Gamma = nn.Parameter(torch.randn(self.pred_len, self.depth))
        
        # Combined bias: [pred_len] - combines Gamma * conv_bias + linear_bias
        self.bias = nn.Parameter(torch.zeros(self.pred_len))
        
        self._initialize_F()
    
    def _initialize_F(self):
        """Initialize F matrix to mimic convolution filters with front padding"""
        # This method will be called after weights are copied from original model
        pass
    
    def forward(self, x):
        """
        Forward pass using matrix formulation: y_hat = Gamma * F * x
        
        Args:
            x: [Batch, seq_len, Channel] input
            
        Returns:
            [Batch, pred_len, Channel] predictions
        """
        batch_size = x.size(0)
        
        if self.use_revIN:
            x = self.rin(x)
        
        # Apply matrix formulation for each channel
        # x: [Batch, seq_len, Channel]
        # We want to apply F * x for each channel independently
        
        # Reshape for batch matrix multiplication
        # x: [Batch, seq_len, Channel] -> [Batch, Channel, seq_len] -> [Batch * Channel, seq_len]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * self.channels, self.seq_len)
        
        # Apply F matrix: [depth, seq_len] @ [Batch * Channel, seq_len].T -> [depth, Batch * Channel]
        F_output = torch.matmul(self.F, x_reshaped.T)  # [depth, Batch * Channel]
        
        # Apply Gamma matrix: [pred_len, depth] @ [depth, Batch * Channel] -> [pred_len, Batch * Channel]
        predictions = torch.matmul(self.Gamma, F_output)  # [pred_len, Batch * Channel]
        
        # Add combined bias: [pred_len, 1] + [pred_len, Batch * Channel] -> [pred_len, Batch * Channel]
        predictions = predictions + self.bias.unsqueeze(1)
        
        # Reshape back to [Batch, pred_len, Channel]
        predictions = predictions.T.contiguous().view(batch_size, self.channels, self.pred_len)
        predictions = predictions.permute(0, 2, 1)  # [Batch, pred_len, Channel]
        
        if self.use_revIN:
            predictions = predictions[:, :, :, None]  # [Batch, pred_len, Channel, 1]
            predictions = self.rin.inverse(predictions)
            predictions = predictions[:, :, :, 0]  # [Batch, pred_len, Channel]
            
        return predictions 