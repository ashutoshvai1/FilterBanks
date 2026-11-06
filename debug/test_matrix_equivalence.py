import torch
import time
import numpy as np
from types import SimpleNamespace
import sys
import os

# Add models directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(current_dir), 'models')
sys.path.append(models_dir)

# Import both models
from DLinear_Filterbank import Model as OriginalModel
from DLinear_Filterbank_matrix import Model as MatrixModel

def create_test_config():
    """Create a test configuration for both models"""
    config = SimpleNamespace()
    config.seq_len = 720
    config.pred_len = 96
    config.enc_in = 7  # number of channels
    config.sampling_fraction = 0  # This gives ip_len = 1
    config.depth = 5  # Start with fewer filters for easier debugging
    config.filter_len = 25
    config.individual = False
    config.use_revIN = True
    return config

def copy_weights_to_matrix_model(original_model, matrix_model):
    """
    Copy weights from original model to matrix model
    
    Weight Mapping:
    1. F Matrix: Construct matrix that replicates front padding + convolution + last value extraction
    2. Gamma Matrix: Extract linear layer weights and arrange them
    3. Bias: Sum all individual biases from linear layers
    """
    with torch.no_grad():
        # 1. Copy F Matrix (convolution kernels with proper front padding logic)
        # Original does: front_pad -> convolution -> take last output
        # We need to replicate this exactly in matrix form
        
        conv_weights = original_model.decomposition.moving_avg.avg.weight  # [depth, 1, kernel_size]
        depth, _, kernel_size = conv_weights.shape
        seq_len = matrix_model.seq_len
        
        # Initialize F matrix to zeros
        matrix_model.F.data.fill_(0.0)
        
        # For each filter, construct the row that computes the last convolution output
        for i in range(depth):
            kernel = conv_weights[i, 0, :]  # [kernel_size]
            
            if kernel_size <= seq_len:
                # Typical case: front padding is irrelevant for last output
                # Last output uses positions [seq_len-kernel_size, seq_len-kernel_size+1, ..., seq_len-1]
                start_pos = seq_len - kernel_size
                matrix_model.F.data[i, start_pos:seq_len] = kernel
            else:
                # Edge case: kernel_size > seq_len, front padding matters
                # Last conv uses: some repeated x[0] + entire original sequence
                num_padding_used = kernel_size - seq_len
                # First num_padding_used elements of kernel multiply repeated x[0]
                matrix_model.F.data[i, 0] = torch.sum(kernel[:num_padding_used])
                # Remaining kernel elements multiply original sequence
                matrix_model.F.data[i, :seq_len] += kernel[num_padding_used:]
        
        # 2. Copy Gamma Matrix (linear layer weights)
        # Original: self.weights[i].weight [pred_len, ip_len] for i in range(depth)
        # Matrix: self.Gamma [pred_len, depth]
        
        for i, linear_layer in enumerate(original_model.weights):
            # linear_layer.weight is [pred_len, ip_len] where ip_len=1
            matrix_model.Gamma.data[:, i] = linear_layer.weight.data.squeeze(-1)  # Remove ip_len dim
        
        # 3. Compute combined bias = Gamma * conv_bias + linear_bias
        # Extract convolution bias
        conv_bias = torch.zeros(depth)
        if original_model.decomposition.moving_avg.avg.bias is not None:
            conv_bias = original_model.decomposition.moving_avg.avg.bias.data
        
        # Extract and sum linear biases
        linear_bias = torch.zeros(matrix_model.pred_len)
        for linear_layer in original_model.weights:
            if linear_layer.bias is not None:
                linear_bias += linear_layer.bias.data
        
        # Combine: bias = Gamma * conv_bias + linear_bias
        combined_bias = torch.matmul(matrix_model.Gamma.data, conv_bias) + linear_bias
        matrix_model.bias.data = combined_bias
        
        # 5. Copy RevIN parameters if present
        if hasattr(original_model, 'rin') and hasattr(matrix_model, 'rin'):
            matrix_model.rin.affine_weight.data = original_model.rin.affine_weight.data.clone()
            matrix_model.rin.affine_bias.data = original_model.rin.affine_bias.data.clone()

def debug_intermediate_outputs(original_model, matrix_model, x, tolerance=1e-5):
    """Debug intermediate outputs to identify where differences arise"""
    print("\nDebugging Intermediate Outputs...")
    print("-" * 40)
    
    batch_size = x.size(0)
    channels = x.size(2)
    
    # Set models to eval mode
    original_model.eval()
    matrix_model.eval()
    
    with torch.no_grad():
        # Original model intermediate outputs
        if original_model.use_revIN:
            x_orig = original_model.rin(x)
        else:
            x_orig = x
            
        # Get decomposition output from original model
        decomp_output = original_model.decomposition(x_orig)  # [Batch, Output_len, Channel, nf]
        decomp_permuted = decomp_output.permute(0, 3, 2, 1)  # [Batch, nf, Channel, Output_len]
        decomp_input_orig = decomp_permuted[:, :, :, -original_model.ip_len:]  # Last ip_len values
        
        print(f"Original decomp_input shape: {decomp_input_orig.shape}")
        print(f"Original decomp_input sample: {decomp_input_orig[0, 0, 0, :].cpu().numpy()}")
        
        # Matrix model intermediate outputs
        if matrix_model.use_revIN:
            x_matrix = matrix_model.rin(x)
        else:
            x_matrix = x
            
        # Apply F matrix manually to see intermediate results
        x_reshaped = x_matrix.permute(0, 2, 1).contiguous().view(batch_size * channels, matrix_model.seq_len)
        F_output = torch.matmul(matrix_model.F, x_reshaped.T)  # [depth, Batch * Channel]
        
        print(f"Matrix F_output shape: {F_output.shape}")
        print(f"Matrix F_output sample: {F_output[:, 0].cpu().numpy()}")
        
        # Compare the filter outputs (should be equivalent to decomp_input_orig squeezed)
        expected_F_output = decomp_input_orig.squeeze(-1)  # Remove ip_len dimension
        expected_F_output = expected_F_output.permute(1, 0, 2)  # [nf, Batch, Channel]
        expected_F_output = expected_F_output.reshape(matrix_model.depth, -1)  # [nf, Batch*Channel]
        
        F_diff = torch.max(torch.abs(F_output - expected_F_output)).item()
        print(f"F matrix output difference: {F_diff:.2e}")
        
        # Compare final outputs
        output_orig = original_model(x)
        output_matrix = matrix_model(x)
        
        final_diff = torch.max(torch.abs(output_orig - output_matrix)).item()
        print(f"Final output difference: {final_diff:.2e}")
        
        return F_diff, final_diff

def test_equivalence(batch_size=32, num_trials=5, tolerance=1e-5):
    """Test if both models produce equivalent outputs"""
    print("Testing Original vs Matrix Model Equivalence...")
    print("=" * 60)
    
    config = create_test_config()
    print(f"Config: seq_len={config.seq_len}, pred_len={config.pred_len}, "
          f"depth={config.depth}, filter_len={config.filter_len}")
    
    # Create models
    original_model = OriginalModel(config)
    matrix_model = MatrixModel(config)
    
    # Set models to evaluation mode
    original_model.eval()
    matrix_model.eval()
    
    # Copy weights from original to matrix model
    copy_weights_to_matrix_model(original_model, matrix_model)
    
    # Debug with a small example first
    print("\nDebugging with single example...")
    x_debug = torch.randn(2, config.seq_len, config.enc_in)
    F_diff, final_diff = debug_intermediate_outputs(original_model, matrix_model, x_debug, tolerance)
    
    # Test equivalence over multiple trials
    max_diff = 0.0
    all_close = True
    
    print(f"\nTesting equivalence over {num_trials} trials...")
    for trial in range(num_trials):
        # Create random input data
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        
        with torch.no_grad():
            # Get outputs from both models
            output_original = original_model(x)
            output_matrix = matrix_model(x)
            
            # Check if outputs are close
            diff = torch.max(torch.abs(output_original - output_matrix)).item()
            max_diff = max(max_diff, diff)
            
            is_close = torch.allclose(output_original, output_matrix, atol=tolerance, rtol=tolerance)
            all_close = all_close and is_close
            
            print(f"Trial {trial+1}: Max difference = {diff:.2e}, Close = {is_close}")
    
    print(f"\nOverall Results:")
    print(f"Max difference across all trials: {max_diff:.2e}")
    print(f"All outputs within tolerance ({tolerance}): {all_close}")
    print(f"Models are {'EQUIVALENT' if all_close else 'NOT EQUIVALENT'}")
    
    return all_close, max_diff

def benchmark_latency(batch_size=32, num_warmup=10, num_trials=100):
    """Benchmark the latency of both models"""
    print("\nBenchmarking Latency...")
    print("=" * 50)
    
    config = create_test_config()
    
    # Create models
    original_model = OriginalModel(config)
    matrix_model = MatrixModel(config)
    
    # Set models to evaluation mode
    original_model.eval()
    matrix_model.eval()
    
    # Copy weights
    copy_weights_to_matrix_model(original_model, matrix_model)
    
    # Create test input
    x = torch.randn(batch_size, config.seq_len, config.enc_in)
    
    # Warmup runs
    print(f"Performing {num_warmup} warmup runs...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = original_model(x)
            _ = matrix_model(x)
    
    # Benchmark original model
    print(f"Benchmarking original model ({num_trials} trials)...")
    original_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start_time = time.perf_counter()
            _ = original_model(x)
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
    
    # Benchmark matrix model
    print(f"Benchmarking matrix model ({num_trials} trials)...")
    matrix_times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start_time = time.perf_counter()
            _ = matrix_model(x)
            end_time = time.perf_counter()
            matrix_times.append(end_time - start_time)
    
    # Calculate statistics
    original_mean = np.mean(original_times) * 1000  # Convert to ms
    original_std = np.std(original_times) * 1000
    matrix_mean = np.mean(matrix_times) * 1000
    matrix_std = np.std(matrix_times) * 1000
    
    speedup = original_mean / matrix_mean
    
    print(f"\nLatency Results (batch_size={batch_size}):")
    print(f"Original Model: {original_mean:.3f} ± {original_std:.3f} ms")
    print(f"Matrix Model:   {matrix_mean:.3f} ± {matrix_std:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        'original_mean': original_mean,
        'original_std': original_std,
        'matrix_mean': matrix_mean,
        'matrix_std': matrix_std,
        'speedup': speedup
    }

def test_memory_usage():
    """Test memory usage of both models"""
    print("\nTesting Memory Usage...")
    print("=" * 50)
    
    config = create_test_config()
    
    # Test memory usage
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
    
    original_model = OriginalModel(config)
    matrix_model = MatrixModel(config)
    
    original_size = get_model_size(original_model)
    matrix_size = get_model_size(matrix_model)
    
    print(f"Original Model Size: {original_size:.3f} MB")
    print(f"Matrix Model Size:   {matrix_size:.3f} MB")
    print(f"Memory Ratio: {original_size/matrix_size:.2f}x")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("Original vs Matrix Model Comparison")
    print("=" * 70)
    
    # Test equivalence
    is_equivalent, max_diff = test_equivalence()
    
    if not is_equivalent:
        print("\nWARNING: Models are not equivalent! Stopping benchmark.")
        print("Check the weight copying logic or model implementation.")
        return
    
    # Test memory usage
    test_memory_usage()
    
    # Benchmark different batch sizes
    batch_sizes = [1, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n" + "-" * 30)
        print(f"Batch Size: {batch_size}")
        print("-" * 30)
        results[batch_size] = benchmark_latency(batch_size=batch_size, num_trials=50)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Models Equivalent: {'YES' if is_equivalent else 'NO'}")
    print(f"Max Difference: {max_diff:.2e}")
    print("\nLatency Speedup by Batch Size:")
    for batch_size, result in results.items():
        print(f"  Batch {batch_size:2d}: {result['speedup']:.2f}x")

if __name__ == "__main__":
    run_comprehensive_test() 