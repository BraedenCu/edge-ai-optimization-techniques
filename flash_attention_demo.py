#!/usr/bin/env python3
"""
Flash Attention vs Traditional Attention Demo

This script compares two implementations of scaled dot-product attention:
1. Traditional Attention: Explicitly computes the full attention matrix
2. Flash Attention: Uses PyTorch's optimized F.scaled_dot_product_attention

The demo shows how Flash Attention can provide better performance and memory efficiency
by avoiding the construction of the full attention matrix.

Key Concepts:
- Scaled Dot-Product Attention: Core mechanism in transformer architectures
- Memory Efficiency: Avoiding O(n²) memory usage for attention matrices
- Kernel Fusion: Optimized GPU kernels for attention computation
- Performance Benchmarking: Measuring real-world speed differences

References:
- Flash Attention: https://arxiv.org/abs/2205.14135
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- xFormers: https://github.com/facebookresearch/xformers
"""

import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

def setup_device():
    """Setup and return the appropriate device (GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def traditional_attention(Q, K, V):
    """
    Traditional implementation of scaled dot-product attention.
    
    This approach explicitly computes the full attention matrix, which can be
    memory intensive and slower for large sequences.
    
    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, head_dim)
        K (torch.Tensor): Key tensor of shape (batch_size, seq_len, head_dim)
        V (torch.Tensor): Value tensor of shape (batch_size, seq_len, head_dim)
        
    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len, head_dim)
    """
    d = Q.size(-1)
    scaling = d ** -0.5
    
    # Compute full attention scores (B x T x T)
    scores = torch.bmm(Q, K.transpose(1, 2)) * scaling
    
    # Apply softmax along the last dimension
    attn = torch.softmax(scores, dim=-1)
    
    # Compute output
    output = torch.bmm(attn, V)
    return output

def flash_attention(Q, K, V):
    """
    Flash Attention implementation using PyTorch's optimized function.
    
    This function internally uses FlashAttention optimizations to compute
    attention more efficiently by avoiding the creation of the full attention matrix.
    
    Args:
        Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, head_dim)
        K (torch.Tensor): Key tensor of shape (batch_size, seq_len, head_dim)
        V (torch.Tensor): Value tensor of shape (batch_size, seq_len, head_dim)
        
    Returns:
        torch.Tensor: Attention output of shape (batch_size, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)

def create_attention_inputs(batch_size=1, seq_len=4096, head_dim=64, device=None):
    """
    Create random query, key, and value tensors for attention computation.
    
    Args:
        batch_size (int): Number of sequences in the batch
        seq_len (int): Length of each sequence
        head_dim (int): Dimension of each attention head
        device (torch.device): Device to place tensors on
        
    Returns:
        tuple: (Q, K, V) tensors
    """
    Q = torch.randn(batch_size, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, seq_len, head_dim, device=device)
    return Q, K, V

def measure_time(func, *args, iterations=10):
    """
    Measure execution time of a function over multiple iterations.
    
    Args:
        func (callable): Function to measure
        *args: Arguments to pass to the function
        iterations (int): Number of iterations to run
        
    Returns:
        list: List of execution times in milliseconds
    """
    times = []
    for i in range(iterations):
        # Synchronize GPU if using CUDA
        if args[0].device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        _ = func(*args)
        
        if args[0].device.type == 'cuda':
            torch.cuda.synchronize()
        
        times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    return times

def visualize_performance_comparison(traditional_times, flash_times, save_path=None):
    """
    Create visualizations comparing the performance of both attention methods.
    
    Args:
        traditional_times (list): Execution times for traditional attention
        flash_times (list): Execution times for flash attention
        save_path (str, optional): Path to save the visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot for distribution comparison
    box_data = [traditional_times, flash_times]
    box_labels = ['Traditional', 'Flash']
    colors = ['lightblue', 'lightgreen']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel("Execution Time per Iteration (ms)", fontsize=12)
    ax1.set_title("Attention Performance Distribution (Boxplot)", fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Line plot for iteration-wise comparison
    iterations = range(1, len(traditional_times) + 1)
    ax2.plot(iterations, traditional_times, marker='o', label='Traditional', linewidth=2, markersize=6)
    ax2.plot(iterations, flash_times, marker='s', label='Flash', linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Execution Time (ms)", fontsize=12)
    ax2.set_title("Iteration-wise Execution Times", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance visualization saved to {save_path}")
    
    plt.show()

def analyze_performance_results(traditional_times, flash_times):
    """
    Analyze and print detailed performance results.
    
    Args:
        traditional_times (list): Execution times for traditional attention
        flash_times (list): Execution times for flash attention
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    avg_traditional = np.mean(traditional_times)
    avg_flash = np.mean(flash_times)
    std_traditional = np.std(traditional_times)
    std_flash = np.std(flash_times)
    
    # Calculate performance ratio
    ratio = avg_flash / avg_traditional
    speedup = avg_traditional / avg_flash if avg_flash < avg_traditional else avg_flash / avg_traditional
    
    print(f"{'Metric':<25} {'Traditional':<15} {'Flash':<15}")
    print("-" * 55)
    print(f"{'Average Time (ms)':<25} {avg_traditional:<15.2f} {avg_flash:<15.2f}")
    print(f"{'Std Dev (ms)':<25} {std_traditional:<15.2f} {std_flash:<15.2f}")
    print(f"{'Min Time (ms)':<25} {min(traditional_times):<15.2f} {min(flash_times):<15.2f}")
    print(f"{'Max Time (ms)':<25} {max(traditional_times):<15.2f} {max(flash_times):<15.2f}")
    
    print(f"\nPerformance Ratio: {ratio:.2f}x")
    if ratio > 1:
        print(f"Flash Attention is {ratio:.2f}x slower than Traditional")
    else:
        print(f"Flash Attention is {speedup:.2f}x faster than Traditional")
    
    print(f"Time Difference: {abs(avg_flash - avg_traditional):.2f} ms")
    print(f"Percentage Difference: {abs(avg_flash - avg_traditional) / avg_traditional * 100:.1f}%")

def run_accuracy_test(Q, K, V):
    """
    Test the accuracy of both attention implementations.
    
    Args:
        Q (torch.Tensor): Query tensor
        K (torch.Tensor): Key tensor
        V (torch.Tensor): Value tensor
    """
    print("\n" + "="*60)
    print("ACCURACY TEST")
    print("="*60)
    
    # Compute results
    traditional_result = traditional_attention(Q, K, V)
    flash_result = flash_attention(Q, K, V)
    
    # Calculate differences
    max_diff = torch.max(torch.abs(traditional_result - flash_result)).item()
    mean_diff = torch.mean(torch.abs(traditional_result - flash_result)).item()
    
    print(f"Maximum Difference:   {max_diff:.2e}")
    print(f"Mean Difference:      {mean_diff:.2e}")
    print(f"Results Match:        {max_diff < 1e-5}")

def main():
    """Main function to run the Flash Attention demonstration."""
    print("Flash Attention vs Traditional Attention Demo")
    print("=" * 55)
    
    # Setup device
    device = setup_device()
    
    # Parameters for the demo
    batch_size = 1
    seq_len = 4096
    head_dim = 64
    
    print(f"\nDemo Parameters:")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_len}")
    print(f"Head Dimension: {head_dim}")
    print(f"Attention Matrix Size: {seq_len} x {seq_len}")
    
    # Create input tensors
    print("\nCreating attention input tensors...")
    Q, K, V = create_attention_inputs(batch_size, seq_len, head_dim, device)
    print(f"Input shapes: Q{K.shape}, K{K.shape}, V{V.shape}")
    
    # Run accuracy test first
    run_accuracy_test(Q, K, V)
    
    # Benchmark both implementations
    print("\nRunning performance benchmarks...")
    print("Benchmarking traditional attention...")
    traditional_times = measure_time(traditional_attention, Q, K, V, iterations=10)
    
    print("Benchmarking flash attention...")
    flash_times = measure_time(flash_attention, Q, K, V, iterations=10)
    
    # Print raw times
    print(f"\nTraditional Attention times (ms): {[f'{t:.2f}' for t in traditional_times]}")
    print(f"Flash Attention times (ms): {[f'{t:.2f}' for t in flash_times]}")
    
    # Analyze results
    analyze_performance_results(traditional_times, flash_times)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_performance_comparison(traditional_times, flash_times, 
                                   save_path="flash_attention_benchmark.png")
    
    print("\nDemo completed successfully!")
    print("\nKey Insights:")
    print("- Flash Attention can provide memory efficiency by avoiding full attention matrices")
    print("- Performance benefits depend on sequence length and hardware")
    print("- Both methods produce numerically equivalent results")
    print("- Flash Attention is particularly beneficial for long sequences")

if __name__ == "__main__":
    main() 