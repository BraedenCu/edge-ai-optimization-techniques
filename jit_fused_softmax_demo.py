#!/usr/bin/env python3
"""
JIT-Fused Softmax Demo

This script demonstrates the performance comparison between PyTorch's built-in softmax
and a TorchScript-compiled, fully fused softmax function. The goal is to show how
kernel fusion can potentially improve performance by reducing memory overhead and latency.

Key Concepts:
- Kernel Fusion: Combining multiple operations into a single GPU kernel
- TorchScript: JIT compilation for optimized execution
- Performance Benchmarking: Measuring execution time differences
- Memory Efficiency: Reducing memory bandwidth requirements

References:
- TorchScript: https://pytorch.org/docs/stable/jit.html
- Softmax Optimization: https://arxiv.org/abs/2008.03277
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np

def setup_device():
    """Setup and return the appropriate device (GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

@torch.jit.script
def jit_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled, fully fused softmax function.
    
    This function is compiled by TorchScript to potentially fuse operations
    into a single kernel, reducing memory overhead and latency.
    
    Args:
        x (torch.Tensor): Input tensor for softmax computation
        
    Returns:
        torch.Tensor: Softmax output with same shape as input
    """
    # Compute maximum for numerical stability
    max_val = x.max(dim=-1, keepdim=True)[0]
    # Subtract max value from x (broadcasted) to avoid large exponents
    x = x - max_val
    # Compute exponentials
    x = x.exp()
    # Sum over the last dimension
    sum_val = x.sum(dim=-1, keepdim=True)
    # Return the normalized softmax output
    return x / sum_val

def benchmark_softmax(func, input_tensor, iterations=100, warmup_iterations=10):
    """
    Benchmark a softmax function by measuring execution time.
    
    Args:
        func (callable): Function to benchmark
        input_tensor (torch.Tensor): Input tensor for the function
        iterations (int): Number of iterations for timing
        warmup_iterations (int): Number of warmup iterations
        
    Returns:
        float: Average execution time per iteration in milliseconds
    """
    device = input_tensor.device
    
    # Warm-up phase
    for _ in range(warmup_iterations):
        _ = func(input_tensor)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            _ = func(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)  # total time in ms
    else:
        # CPU timing
        start_time = time.time()
        for _ in range(iterations):
            _ = func(input_tensor)
        elapsed_time = (time.time() - start_time) * 1000  # convert to ms
    
    return elapsed_time / iterations  # average time per iteration

def create_attention_matrix(size=1024, device=None):
    """
    Create a large attention matrix for benchmarking.
    
    Args:
        size (int): Size of the square attention matrix
        device (torch.device): Device to place the tensor on
        
    Returns:
        torch.Tensor: Random attention scores matrix
    """
    return torch.randn(size, size, device=device)

def visualize_benchmark_results(builtin_time, fused_time, save_path=None):
    """
    Create a bar chart comparing the benchmark results.
    
    Args:
        builtin_time (float): Execution time for built-in softmax
        fused_time (float): Execution time for JIT fused softmax
        save_path (str, optional): Path to save the visualization
    """
    methods = ["Built-in Softmax", "JIT Fused Softmax"]
    times = [builtin_time, fused_time]
    colors = ["lightblue", "lightgreen"]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{time_val:.4f} ms", ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel("Average Time per Iteration (ms)", fontsize=12)
    plt.title("Softmax Performance Comparison", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add performance ratio
    ratio = fused_time / builtin_time
    plt.text(0.5, 0.95, f"JIT Fused is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than Built-in",
             transform=plt.gca().transAxes, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark visualization saved to {save_path}")
    
    plt.show()

def analyze_performance_differences(builtin_time, fused_time):
    """
    Analyze and print detailed performance differences.
    
    Args:
        builtin_time (float): Execution time for built-in softmax
        fused_time (float): Execution time for JIT fused softmax
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    ratio = fused_time / builtin_time
    speedup = builtin_time / fused_time if fused_time < builtin_time else fused_time / builtin_time
    
    print(f"Built-in Softmax:     {builtin_time:.4f} ms per iteration")
    print(f"JIT Fused Softmax:    {fused_time:.4f} ms per iteration")
    print(f"Performance Ratio:    {ratio:.2f}x")
    
    if ratio > 1:
        print(f"JIT Fused is {ratio:.2f}x slower than Built-in")
    else:
        print(f"JIT Fused is {speedup:.2f}x faster than Built-in")
    
    print(f"Time Difference:      {abs(fused_time - builtin_time):.4f} ms")
    print(f"Percentage Difference: {abs(fused_time - builtin_time) / builtin_time * 100:.1f}%")

def run_accuracy_test(input_tensor):
    """
    Test the accuracy of both softmax implementations.
    
    Args:
        input_tensor (torch.Tensor): Input tensor for testing
    """
    print("\n" + "="*60)
    print("ACCURACY TEST")
    print("="*60)
    
    # Compute results
    builtin_result = F.softmax(input_tensor, dim=-1)
    fused_result = jit_fused_softmax(input_tensor)
    
    # Calculate differences
    max_diff = torch.max(torch.abs(builtin_result - fused_result)).item()
    mean_diff = torch.mean(torch.abs(builtin_result - fused_result)).item()
    
    print(f"Maximum Difference:   {max_diff:.2e}")
    print(f"Mean Difference:      {mean_diff:.2e}")
    print(f"Results Match:        {max_diff < 1e-6}")

def main():
    """Main function to run the JIT-Fused Softmax demonstration."""
    print("JIT-Fused Softmax Demo")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Create large attention matrix for benchmarking
    print("\nCreating attention matrix for benchmarking...")
    attention_scores = create_attention_matrix(size=1024, device=device)
    print(f"Attention matrix shape: {attention_scores.shape}")
    
    # Run accuracy test first
    run_accuracy_test(attention_scores)
    
    # Benchmark both implementations
    print("\nRunning benchmarks...")
    print("Benchmarking built-in softmax...")
    builtin_time = benchmark_softmax(
        lambda x: F.softmax(x, dim=-1), 
        attention_scores, 
        iterations=100
    )
    
    print("Benchmarking JIT fused softmax...")
    fused_time = benchmark_softmax(
        jit_fused_softmax, 
        attention_scores, 
        iterations=100
    )
    
    # Analyze results
    analyze_performance_differences(builtin_time, fused_time)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_benchmark_results(builtin_time, fused_time, 
                               save_path="jit_fused_softmax_benchmark.png")
    
    print("\nDemo completed successfully!")
    print("\nKey Insights:")
    print("- PyTorch's built-in softmax is highly optimized with low-level CUDA kernels")
    print("- JIT compilation doesn't always guarantee better performance")
    print("- Production libraries often have years of optimization behind them")
    print("- Kernel fusion benefits depend on specific hardware and tensor sizes")

if __name__ == "__main__":
    main() 