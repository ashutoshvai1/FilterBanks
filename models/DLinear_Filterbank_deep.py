"""
Implementation of Filterbank deep. 
Comprises of sequence of decomposition residual bloacks. 
"""

import torch
import torch.nn as nn
from utils.layer_norm_variants import _ResidualBlock
from models.DLinear_Filterbank import filter_bank as series_decomp_S
from utils.layer_norm_variants import RINorm

class _BatchedResidualBlock(nn.Module):
    def __init__(self, residual_blocks: nn.ModuleList):
        super().__init__()
        
        self.depth = len(residual_blocks)
        
        # --- Register stacked weights and biases as parameters ---
        # This ensures they are moved to the correct device and included in model.parameters()
        
        # Dense path
        self.W_dense1 = nn.Parameter(torch.stack([l.dense[0].weight.T for l in residual_blocks]))
        self.b_dense1 = nn.Parameter(torch.stack([l.dense[0].bias for l in residual_blocks]))
        self.W_dense2 = nn.Parameter(torch.stack([l.dense[2].weight.T for l in residual_blocks]))
        self.b_dense2 = nn.Parameter(torch.stack([l.dense[2].bias for l in residual_blocks]))
        
        # Skip connection path
        self.W_skip = nn.Parameter(torch.stack([l.skip.weight.T for l in residual_blocks]))
        self.b_skip = nn.Parameter(torch.stack([l.skip.bias for l in residual_blocks]))
        
        # Store activation and dropout (they don't have parameters, so no need to register)
        self.relu = residual_blocks[0].dense[1]
        self.dropout = residual_blocks[0].dense[3]

    def forward(self, x):
        # x is expected to be of shape [Depth, Batch * Channels, Input Length]
        
        # --- Dense Path Calculation ---
        h1 = torch.bmm(x, self.W_dense1) + self.b_dense1.unsqueeze(1)
        h1_act = self.relu(h1)
        h2 = torch.bmm(h1_act, self.W_dense2) + self.b_dense2.unsqueeze(1)
        dense_output = self.dropout(h2)
        
        # --- Skip Connection Calculation ---
        skip_output = torch.bmm(x, self.W_skip) + self.b_skip.unsqueeze(1)
        
        # --- Final Combination ---
        return dense_output + skip_output

class DeepDecomposition(nn.Module):
    """
    Deep Convloution Decomposition block. Comprises of decomposition followed by single residual block for each branch.
    #TODO: Add support for individual processing. Currently shared model is learned across all channels. Only "else"
    branch is implemented.
    """
    def __init__(self, configs, input_dim, output_dim, use_layer_norm=True):
        super(DeepDecomposition, self).__init__()

        # Decompsition Kernel Size
        depth = configs.depth
        kernel_size = configs.filter_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.c_out=configs.c_out
        self.decomposition = series_decomp_S(depth, kernel_size, self.channels)
        
        # set fraction of decomposed output to use before the linear layer
        # setting to 1 means only the last value of the decomposed vectors is used
        if int(input_dim * configs.sampling_fraction) == 0:
            self.ip_len = 1
        else:
            self.ip_len = int(input_dim * configs.sampling_fraction)
        
        self.Linear = nn.ModuleList()

        hidden_size = configs.hidden_size
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        dropout = configs.dropout

        if self.individual:
            pass
        else:
            for j in range(depth):
                # iterate over each decomposed signal
                # define residual block for each branch, no layer norm in final layer
                # layer norm is applied after adding back decomposed signals
                self.Linear.append(_ResidualBlock(input_dim=self.ip_len, 
                                                    output_dim=output_dim,
                                                    hidden_size=hidden_size,
                                                    use_layer_norm=False,
                                                    dropout=dropout))

        # --- Create a single batched residual block from the list of blocks ---
        self.batched_residual_block = _BatchedResidualBlock(self.Linear)
        # We no longer need the ModuleList for the forward pass
        del self.Linear

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        # New optimized path
        decomp_output_tensor = self.decomposition(x)  # [Batch, Output length, Channel, nf]
        
        # Permute to [Batch, nf, Channel, Output length]
        decomp_output_tensor = decomp_output_tensor.permute(0, 3, 2, 1)
        
        if self.individual:
            raise NotImplementedError
        else:
            # Get input for residual blocks
            # [Batch, nf, Channel, ip_len]
            decomp_input = decomp_output_tensor[:, :, :, -self.ip_len:]
            
            # Take only the first `depth` components
            decomp_input_for_blocks = decomp_input[:, :self.batched_residual_block.depth, :, :]
            
            B, D, C, I = decomp_input_for_blocks.shape
            
            # Reshape for batched matrix multiplication: (B, D, C, I) -> (D, B*C, I)
            x_for_bmm = decomp_input_for_blocks.permute(1, 0, 2, 3).reshape(D, B * C, I)

            # --- Process through the efficient batched residual block ---
            final_output = self.batched_residual_block(x_for_bmm)
            
            # Reshape back to (B, D, C, O)
            O = final_output.shape[-1]
            output_bmm = final_output.view(D, B, C, O).permute(1, 0, 2, 3)

            # Sum over the decomposition dimension
            x = output_bmm.sum(dim=1)

        # Apply layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = x.permute(0,2,1) # to [Batch, Output length, Channel]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()        
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = configs.hidden_size
        self.channels = configs.enc_in
        self.use_revIN = configs.use_revIN
        if self.use_revIN:
            self.rin = RINorm(input_dim=self.channels)
        self.use_layer_norm = configs.use_layer_norm
        
        n_blocks = getattr(configs, 'n_blocks', 2)

        self.blocks = nn.ModuleList()
        if n_blocks == 1:
            self.blocks.append(DeepDecomposition(configs, self.seq_len, self.pred_len, use_layer_norm=False)) 
        elif n_blocks == 2:
            self.blocks.append(DeepDecomposition(configs, self.seq_len, self.hidden_size, use_layer_norm=self.use_layer_norm))
            self.blocks.append(DeepDecomposition(configs, self.hidden_size, self.pred_len, use_layer_norm=False))
        else:
            self.blocks.append(DeepDecomposition(configs, self.seq_len, self.hidden_size, use_layer_norm=self.use_layer_norm))
            for i in range(n_blocks-2):
                self.blocks.append(DeepDecomposition(configs, self.hidden_size, self.hidden_size, use_layer_norm=self.use_layer_norm))
            self.blocks.append(DeepDecomposition(configs, self.hidden_size, self.pred_len, use_layer_norm=False))
        self.blocks = nn.Sequential(*self.blocks)



    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # batch_size,_,_=x.size()
        
        # Apply reverse instance normalization
        if self.use_revIN:
            x = self.rin(x)

        x = self.blocks(x)

        if self.use_revIN:
            x = x[:,:,:,None]   # to [Batch, Output length, Channel, 1] for compatibility with revIN
            x = self.rin.inverse(x)
            x = x[:,:,:,0]     # back to [Batch, Output length, Channel]
        return x