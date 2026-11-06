"""
Benchmark the performance of Filterbank_matrix and Filterbank_deep. 

Note that Filterbank and Filterbank_matrix models are equivalent. Filterbank_matrix 
uses a matrix multiplication instead of a convolution and combines that bias terms of 
the Filterbank convolution and linear layers. Their equivalence can be verified by running 
debug/test_matrix_equivalence.py. It loads Filterbank model weights to Filterbank_matrix 
and verifies that their outputs are the same.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
# import numpy as np
import statistics
from ptflops import get_model_complexity_info
from thop import profile
import pandas as pd
from torch.profiler import profile
# , record_function, ProfilerActivity

import sys
sys.path.append('models/')

from DLinear_Filterbank import Model as Model_DLinear_Filterbank
from DLinear_Filterbank_matrix import Model as Model_DLinear_Filterbank_matrix
from DLinear_Filterbank_deep import Model as Model_DLinear_Filterbank_deep


#import torch_tensorrt
import pathlib



repetitions = 10000
batch_size = 1
input_tensor_global = None

def update_model_info(file_path, model_name, metric_name, metric_value):
    """
    Update or add a specific metric for a given model in the CSV file.
    
    Parameters:
    - file_path: Path to the CSV file.
    - model_name: Name of the model to update.
    - metric_name: The metric to update ('Median Inference Time (ms)', 'MACs', or 'Parameters').
    - metric_value: The new value for the specified metric.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if the model already exists
    if model_name in df['Model'].values:
        # Update the specified metric
        df.loc[df['Model'] == model_name, metric_name] = metric_value
    else:
        # Create a new entry for the model with default values for other metrics
        new_entry = {'Model': model_name, 
                     'Median Inference Time (ms)': float('nan'),
                     'MACs': float('nan'),
                     'Parameters': float('nan')}
        new_entry[metric_name] = metric_value
        new_df = pd.DataFrame([new_entry])
        df = pd.concat([df, new_df], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

# Function to convert tensor or tuple of tensors to tuple of tuples of ints
def get_tensor_sizes(input_tensor):
    if isinstance(input_tensor, torch.Tensor):
        # If input_tensor is a single tensor, convert its size directly
        return (tuple(input_tensor.shape),)  # Return as a tuple containing one tuple
    elif isinstance(input_tensor, tuple):
        # If input_tensor is a tuple of tensors, convert each size to a tuple
        return tuple(tuple(t.shape) for t in input_tensor)
    
def get_tensor_sizes_dict(tensor_dict):
    sizes = []
    for item in tensor_dict.values():
        if isinstance(item, torch.Tensor):
            sizes.append(tuple(item.size()))
        elif isinstance(item, tuple):
            sizes.append(tuple(tuple(x.size()) for x in item if isinstance(x, torch.Tensor)))
    return sizes
    
def prepare_input(resolution):
    global input_tensor_global
    tensor_length = len(input_tensor_global)
    input_as_dicts = {}
    if (tensor_length > 1):
        #x_enc, x_mark_enc, x_dec, x_mark_dec
        input_as_dicts["x_enc"] = input_tensor_global[0]
        input_as_dicts["x_mark_enc"] = input_tensor_global[1]
        input_as_dicts["x_dec"] = input_tensor_global[2]
        input_as_dicts["x_mark_dec"] = input_tensor_global[3]
    else:
        input_as_dicts["x"] = input_tensor_global
    return input_as_dicts

def prepare_input_tide(resolution):
    global input_tensor_global
    tensor_length = len(input_tensor_global)
    input_as_dicts = {}
    if type(input_tensor_global) is tuple:
        input_as_dicts["x_in"] = input_tensor_global
        return input_as_dicts
    if (tensor_length > 1):
        #x_enc, x_mark_enc, x_dec, x_mark_dec
        input_as_dicts["x_enc"] = input_tensor_global[0]
        input_as_dicts["x_mark_enc"] = input_tensor_global[1]
        input_as_dicts["x_dec"] = input_tensor_global[2]
        input_as_dicts["x_mark_dec"] = input_tensor_global[3]
    else:
        input_as_dicts["x"] = input_tensor_global
    return input_as_dicts

def get_profile(config, model):
    _input=torch.randn(config.batch_size, config.seq_len, config.enc_in, device="cuda")
    macs, params = profile(model, inputs=(_input,))
    print('THOP: FLOPs: ', macs)
    print('THOP: params: ', params)
    return macs, params

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # He initialization
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        # Xavier initialization for LSTM/GRU
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0)

seq_len=720
pred_len=720
# Configuration object
class Config:
    def __init__(self, seq_len, pred_len, individual, enc_in, cut_freq):
        self.batch_size = 1
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in # channels
        self.cut_freq = cut_freq
        # Autoformer & Informer wants these
        self.dropout = 0.5
        self.label_len = 48
        ##self.output_attention = False
        self.moving_avg = 25
        self.embed_type = 0
        self.d_model = 512
        self.embed = 'timeF'
        self.freq = 'h'
        self.dec_in = 7
        self.e_layers = 2
        ###self.factor = 1
        self.factor = 3
        self.n_heads = 8
        self.d_ff = 2048
        self.activation = 'gelu'
        self.d_layers = 1
        self.des = 'Exp'
        self.c_out = 7
        # Film wants these
        ##self.ab = 2
        ### Informer wants these
        ##self.distil = True
# Define model configurations
configs = Config(seq_len, pred_len, individual=False, enc_in=321, cut_freq=170)

# Timing stuff
starter_torch, ender_torch = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# Initialize the model
models = {}

configs.c_out = 7
configs.d_ff = 2048
configs.seq_len = 336
configs.d_ff = 256
configs.d_layers = 1
configs.d_model = 512
configs.dec_in = 7
configs.des = 'Exp'
configs.distil = True
configs.dropout = 0.05
configs.e_layers = 2
configs.embed = 'timeF'
configs.embed_type = 0
configs.enc_in = 7
configs.factor = 1
configs.features = 'M'
configs.freq = 'h'
configs.itr = 1
configs.label_len = 48
configs.moving_avg = 25
configs.n_heads = 8
configs.output_attention = False
configs.patience = 3

configs.depth = 3
configs.filter_len = 23
configs.hidden_size = 512
configs.period_len = 48
configs.use_layer_norm = False
configs.sampling_fraction = 0.33319183820223097
configs.test_params = None
configs.use_revIN = True
configs.dropout = 0
configs.noise_variance = 0
configs.lradj = 'type3'
configs.seq_len = 720
models["DLinear_Filterbank_deep"] = Model_DLinear_Filterbank_deep(configs)

configs.depth = 24
configs.filter_len = 659
configs.period_len = 4
configs.test_params = None
configs.use_revIN = True
configs.sampling_fraction = 0
configs.dropout = 0
configs.hidden_size = 512
configs.use_layer_norm = False
configs.noise_variance = 0
configs.lradj = 'type3'
configs.seq_len = 720
# models["DLinear_Filterbank"] = Model_DLinear_Filterbank(configs)
models["DLinear_Filterbank_matrix"] = Model_DLinear_Filterbank_matrix(configs)


# Define a DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Median Inference Time (ms)', 'MACs', 'Parameters'])

for name, model in models.items():
    if name in ("TiDe"):
        model.model.apply(init_weights)
        pytorch_model = model.model.eval().cuda().float()
        model = model.model.eval().cuda().float()
    else:
        model.apply(init_weights)
        # Move model and input to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        pytorch_model = model
    inference_times = []

    run_in_trt = False
    trt_compiled = False
                                           
    for i in range(repetitions):
        # Generate random input (assuming input dimensions are [batch, channels, sequence_length])

        input_tensor = torch.randn(batch_size, 720, configs.enc_in, device='cuda')
        '''
        if run_in_trt and not trt_compiled:
            if name in ("Autoformer", "FEDformer", "Film", "Informer", "Transformer"):
                onnx_program = torch.onnx.dynamo_export(model, [input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3]])
                #model = torch_tensorrt.compile(model,inputs = [input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3]],enabled_precisions = {torch.half},)
            else:
                onnx_program = torch.onnx.dynamo_export(model, input_tensor)
                #model = torch_tensorrt.compile(model,inputs = [input_tensor,],enabled_precisions = {torch.half},)
            onnx_program.save(name + ".onnx")
            trt_compiled = True
        '''
        # Timing inference
        starter_torch.record()
        if name in ("Autoformer", "FEDformer", "Film", "Informer", "Transformer", "TimeKAN"):
            output = model(input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3])
        else:
            output = model(input_tensor)
        ender_torch.record()
        # wait for gpu sync
        torch.cuda.synchronize()
        inference_time_torch = starter_torch.elapsed_time(ender_torch)
        inference_times.append(inference_time_torch)
    median_inference_time = statistics.median(inference_times[1:])
    #print(f"{name}: Inference times: {inference_times} ms")
    print(f"{name}: Median inference time: {median_inference_time} ms")

    #print(get_tensor_sizes(input_tensor))
    #prepare_input(input_tensor)
    input_tensor_global = input_tensor
    macs = 0
    params = 0
    
    print("------")
    print(name)
    
    macs, params = get_model_complexity_info(pytorch_model, (0,0,0), input_constructor=prepare_input, as_strings=True, print_per_layer_stat=True) #
    print("------")
    print(name, "MACs", macs, "Params", params)

    # Create a temporary DataFrame and use pd.concat
    temp_df = pd.DataFrame({
        'Model': [name],
        'Median Inference Time (ms)': [median_inference_time],
        'MACs': [macs],
        'Parameters': [params]
    })
    #results_df = pd.concat([results_df, temp_df], ignore_index=True)
    # Ensure model_performance_metrics.csv exists with correct columns
    csv_path = pathlib.Path(__file__).parent.resolve() / "model_performance_metrics.csv"
    if not csv_path.exists():
        pd.DataFrame({
            'Model': [],
            'Median Inference Time (ms)': [],
            'MACs': [],
            'Parameters': []
        }).to_csv(csv_path, index=False)
    update_model_info(pathlib.Path(__file__).parent.resolve() / "model_performance_metrics.csv", name, "Median Inference Time (ms)", median_inference_time)
    update_model_info(pathlib.Path(__file__).parent.resolve() / "model_performance_metrics.csv", name, "MACs", macs)
    update_model_info(pathlib.Path(__file__).parent.resolve() / "model_performance_metrics.csv", name, "Parameters", params)
    del model
    
# Save DataFrame to CSV
#results_df.to_csv('model_performance_metrics.csv', index=False)