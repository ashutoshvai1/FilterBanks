# Implementation of filter bank - multiple convolution in parallel followed by truncation and linear layer.
# The file name should contain "Linear" in it if it is to be processed the same as DLinear model
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layer_norm_variants import RINorm
# from models.DLinearConv_S import series_decomp
# from models.DLinear_CoDeNet_vanilla import conv_avg


class conv_avg(nn.Module):
    """
    Weighted moving average using convolution to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride, nf):
        super(conv_avg, self).__init__()
        self.kernel_size = kernel_size
        # self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.avg = nn.Conv1d(1, nf, kernel_size, stride=stride, padding=0)
        # at groups=in_channels, each channel is convolved with its own set of filters, which means no fusion of channels
        # at groups=1, each output channel contains the result of convolution over all input channels=>data fusion

    def forward(self, x):
        # padding on the both ends of time series
        # This padding results in the result of convolution having the same shape as input if the kernel size is odd
        # If kernel size is even, the resultant shape is +1 longer
        # Only front padding is done for codenet vanilla, while front and back padding is done in codenet-linear
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1), 1)
        # end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class filter_bank(nn.Module):
    def __init__(self, depth, kernel_size, channels):
        super(filter_bank, self).__init__()
        self.moving_avg = conv_avg(kernel_size, stride=1, nf=depth)
        # TODO: We can try experimenting with varying kernel_size at different depths
        # TODO: We can experiment with reusing the same convolution kernel at all depths

    def flatten_channels(self, x):
        # Flattens the channels into the batch dimension for weight sharing across channels
        # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1) # to [Batch, Channel, Input length]
        x = x.reshape(x.size(0)*x.size(1),1,x.size(2)) # to [Batch*Channel, 1, Input length]
        x = x.permute(0,2,1) # to [Batch*Channel, Input length, 1]
        return x
    
    def unflatten_channels(self, x, batch_size):
        # Unflatten the flattened channels back to [Batch, output length, Channel]
        # x: [Batch*Channel, Output length, 1]
        x = x.permute(0,2,1)
        x = x.reshape(batch_size,-1,x.size(2))
        x = x.permute(0,2,1)
        return x
        
    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(2)
        x = self.flatten_channels(x)
        moving_avg_output = self.moving_avg(x) # [Batch*Channel, Output length, nf]
        
        # Optimized: Instead of loop, reshape and stack all filter outputs at once
        # moving_avg_output: [Batch*Channel, Output length, nf] 
        # Reshape to [Batch, Channel, Output length, nf] then permute to [Batch, Output length, Channel, nf]
        output_len = moving_avg_output.size(1)
        nf = moving_avg_output.size(2)
        
        # Reshape from [Batch*Channel, Output length, nf] to [Batch, Channel, Output length, nf]
        moving_avg_output = moving_avg_output.view(batch_size, channels, output_len, nf)
        # Permute to [Batch, Output length, Channel, nf]
        moving_avg_output = moving_avg_output.permute(0, 2, 1, 3)
        
        return moving_avg_output

class Model(nn.Module):
    """
    Optimized CoDeNet Filter Bank Implementation
    
    OPTIMIZATION SUMMARY:
    - filter_bank.forward() now returns a single tensor [Batch, Output_len, Channel, nf] 
      instead of a list of nf tensors each of shape [Batch, Output_len, Channel]
    - Model.forward() uses pure tensor operations with torch.einsum for batched matrix 
      multiplication instead of Python loops over individual linear layers
    - All nf filter outputs are processed simultaneously using stacked weights/biases
    - Functionality is identical to the original implementation, just much faster
    
    After decomposition, samples the last ip_len values and applies linear transformation
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # set fraction of decomposed output to use before the linear layer
        # setting to 1 means only the last value of the decomposed vectors is used
        if int(self.seq_len * configs.sampling_fraction) == 0:
            self.ip_len = 1
        else:
            self.ip_len = int(self.seq_len * configs.sampling_fraction)

        # Decompsition Kernel Size
        depth = configs.depth
        kernel_size = configs.filter_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        # self.c_out=configs.c_out
        self.use_revIN = configs.use_revIN
        if self.use_revIN:
            self.rin = RINorm(input_dim=self.channels)

        self.decomposition = filter_bank(depth, kernel_size, self.channels)
        # self.weights = []
        self.weights = nn.ModuleList()

        if self.individual:
            raise NotImplementedError
            # iterate over different decomposition branches
            for i in range(depth+1):
                self.Linear.append(nn.ModuleList())

            # iterate over depth and channels
            for i in range(depth+1):
                for j in range(self.channels):
                    self.Linear[i].append(nn.Linear(self.seq_len,self.pred_len))
                    # self.Linear[j][i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            for j in range(depth):
                # self.weights.append(nn.Parameter((torch.rand(self.pred_len)-0.5)))
                self.weights.append(nn.Linear(self.ip_len, self.pred_len))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # batch_size,_,_=x.size()

        if self.use_revIN:
            x = self.rin(x)

        decomp_output_tensor = self.decomposition(x)  # [Batch, Output length, Channel, nf]
        decomp_output_tensor = decomp_output_tensor.permute(0, 3, 2, 1)  # [Batch, nf, Channel, Output length]
        
        if self.individual:
            raise NotImplementedError
        else:
            # Optimized: Process all filters at once using pure tensor operations
            # Extract last ip_len values for all filters: [Batch, nf, Channel, ip_len]
            decomp_input = decomp_output_tensor[:, :, :, -self.ip_len:]
            
            # Stack all linear layer weights and biases
            weights = torch.stack([layer.weight for layer in self.weights], dim=0)  # [nf, pred_len, ip_len]
            biases = torch.stack([layer.bias for layer in self.weights], dim=0)     # [nf, pred_len]
            
            # Perform batched matrix multiplication
            # decomp_input: [Batch, nf, Channel, ip_len]
            # weights: [nf, pred_len, ip_len]
            # We want: [Batch, nf, Channel, pred_len]
            
            # Transpose weights for correct multiplication: [nf, ip_len, pred_len]
            weights_t = weights.transpose(-2, -1)
            
            # Use einsum for efficient batched matrix multiplication
            # 'bnci,nip->bncp' means: (batch,nf,channel,ip_len) x (nf,ip_len,pred_len) -> (batch,nf,channel,pred_len)
            linear_outputs = torch.einsum('bnci,nip->bncp', decomp_input, weights_t)
            
            # Add biases: [nf, pred_len] -> [1, nf, 1, pred_len] for broadcasting
            linear_outputs = linear_outputs + biases.unsqueeze(0).unsqueeze(2)
            
            # Sum across all filters: [Batch, Channel, pred_len]
            x = linear_outputs.sum(dim=1).permute(0, 2, 1)  # [Batch, pred_len, Channel]

        if self.use_revIN:
            x = x[:,:,:,None]   # to [Batch, Output length, Channel, 1] for compatibility with revIN
            x = self.rin.inverse(x)
            x = x[:,:,:,0]     # back to [Batch, Output length, Channel]
        return x 
