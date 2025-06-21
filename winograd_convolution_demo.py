#!/usr/bin/env python3
"""
Winograd Convolution Demo

This script implements Winograd convolution (specifically F(2x2, 3x3)) as an alternative
to standard convolution. Winograd algorithms reduce the number of multiplications
required for small convolutional kernels, potentially improving performance.

Key Concepts:
- Winograd Convolution: Algorithmic optimization for small kernels
- Transformation Matrices: Pre-computed matrices for input/kernel/output transformation
- Tile-based Processing: Dividing input into overlapping tiles
- Performance vs Accuracy: Trade-offs between speed and numerical precision

References:
- Fast Algorithms for Convolutional Neural Networks: https://arxiv.org/abs/1509.09308
- Winograd Convolution: https://arxiv.org/abs/1803.10986
- cuDNN Winograd: https://docs.nvidia.com/deeplearning/cudnn/api/index.html
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

def create_winograd_matrices(device=None):
    """
    Create Winograd transformation matrices for F(2x2, 3x3).
    
    These matrices are used to transform the input, kernel, and output
    to and from the Winograd domain where computation is more efficient.
    
    Args:
        device (torch.device): Device to place matrices on
        
    Returns:
        tuple: (B, G, A) transformation matrices
    """
    # B: Input transformation matrix (4×4)
    B = torch.tensor([
        [1,  0, -1,  0],
        [0,  1,  1,  0],
        [0, -1,  1,  0],
        [0,  1,  0, -1]
    ], dtype=torch.float32, device=device)
    
    # G: Kernel transformation matrix (4×3)
    G = torch.tensor([
        [1,    0,    0],
        [0.5,  0.5,  0.5],
        [0.5, -0.5,  0.5],
        [0,    0,    1]
    ], dtype=torch.float32, device=device)
    
    # A: Output inverse transformation matrix (4×2)
    A = torch.tensor([
        [1,  0],
        [1,  1],
        [1, -1],
        [0, -1]
    ], dtype=torch.float32, device=device)
    
    return B, G, A

def winograd_conv2d(input_img, kernel, B, G, A):
    """
    Implement Winograd convolution F(2x2, 3x3).
    
    This function applies the Winograd algorithm to compute convolution
    more efficiently by reducing the number of multiplications.
    
    Args:
        input_img (torch.Tensor): Input image of shape (N, C, H, W)
        kernel (torch.Tensor): Convolution kernel of shape (1, C, 3, 3)
        B (torch.Tensor): Input transformation matrix
        G (torch.Tensor): Kernel transformation matrix
        A (torch.Tensor): Output transformation matrix
        
    Returns:
        torch.Tensor: Convolution output
    """
    N, C, H, W = input_img.shape
    
    # Transform the kernel: assume single filter and channel
    g = kernel[0, 0]  # shape: (3,3)
    U = G @ g @ G.t()  # shape: (4,4)
    
    # For F(2x2, 3x3), tile size is 4x4 and each tile produces a 2x2 output
    tile_size = 4
    out_tile = 2  # Output tile size
    n_tiles_h = (H - tile_size) // out_tile + 1
    n_tiles_w = (W - tile_size) // out_tile + 1
    
    # Extract overlapping 4x4 tiles from the input
    tiles = input_img.unfold(2, tile_size, out_tile).unfold(3, tile_size, out_tile)
    # tiles shape: (N, C, n_tiles_h, n_tiles_w, 4, 4)
    
    # Transform input tiles: X = B * d * B^T for each tile
    X = torch.matmul(B, tiles)         # (4,4) x (N,C,n_tiles_h,n_tiles_w,4,4) -> (N,C,n_tiles_h,n_tiles_w,4,4)
    X = torch.matmul(X, B.t())         # (N,C,n_tiles_h,n_tiles_w,4,4)
    
    # Elementwise multiplication in Winograd domain
    Y = X * U  # Broadcasting U (4x4) to each tile
    
    # Inverse transform for each tile: y = A^T * Y * A
    # A^T: (2,4), A: (4,2)
    temp = torch.matmul(A.t(), Y)      # (2,4) x (N,C,n_tiles_h,n_tiles_w,4,4) -> (N,C,n_tiles_h,n_tiles_w,2,4)
    y = torch.matmul(temp, A)          # (N,C,n_tiles_h,n_tiles_w,2,4) x (4,2) -> (N,C,n_tiles_h,n_tiles_w,2,2)
    
    # Reassemble the output tiles into the full output
    # Permute dimensions to interleave the 2x2 outputs
    y = y.permute(0,1,2,4,3,5).contiguous()  # shape: (N,C,n_tiles_h,2,n_tiles_w,2)
    output = y.view(N, C, n_tiles_h * out_tile, n_tiles_w * out_tile)
    
    return output

def create_test_inputs(N=1, C=1, H=8, W=8, device=None):
    """
    Create test input image and kernel for convolution.
    
    Args:
        N (int): Batch size
        C (int): Number of channels
        H (int): Height of input image
        W (int): Width of input image
        device (torch.device): Device to place tensors on
        
    Returns:
        tuple: (input_img, kernel) tensors
    """
    input_img = torch.randn(N, C, H, W, device=device)
    kernel = torch.randn(1, C, 3, 3, device=device)  # Single 3x3 filter
    return input_img, kernel

def time_function(func, *args, iterations=1000):
    """
    Time a function over multiple iterations.
    
    Args:
        func (callable): Function to time
        *args: Arguments to pass to the function
        iterations (int): Number of iterations
        
    Returns:
        float: Average execution time per iteration in milliseconds
    """
    device = args[0].device
    
    # Warm-up
    for _ in range(10):
        _ = func(*args)
    
    # Synchronize GPU if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = func(*args)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - start) / iterations * 1000  # ms per iteration

def visualize_results(conv_output, winograd_output, save_path=None):
    """
    Visualize the outputs of both convolution methods side by side.
    
    Args:
        conv_output (torch.Tensor): Standard convolution output
        winograd_output (torch.Tensor): Winograd convolution output
        save_path (str, optional): Path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot standard convolution output
    im1 = ax1.imshow(conv_output[0, 0].detach().cpu().numpy(), cmap='viridis')
    ax1.set_title("Standard conv2d Output", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Width")
    ax1.set_ylabel("Height")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot Winograd convolution output
    im2 = ax2.imshow(winograd_output[0, 0].detach().cpu().numpy(), cmap='viridis')
    ax2.set_title("Winograd conv2d Output", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def analyze_accuracy(conv_output, winograd_output):
    """
    Analyze the numerical accuracy between standard and Winograd convolution.
    
    Args:
        conv_output (torch.Tensor): Standard convolution output
        winograd_output (torch.Tensor): Winograd convolution output
    """
    print("\n" + "="*60)
    print("ACCURACY ANALYSIS")
    print("="*60)
    
    # Calculate differences
    max_diff = torch.max(torch.abs(conv_output - winograd_output)).item()
    mean_diff = torch.mean(torch.abs(conv_output - winograd_output)).item()
    relative_diff = torch.mean(torch.abs(conv_output - winograd_output) / (torch.abs(conv_output) + 1e-8)).item()
    
    print(f"Maximum Absolute Difference: {max_diff:.2e}")
    print(f"Mean Absolute Difference:    {mean_diff:.2e}")
    print(f"Mean Relative Difference:    {relative_diff:.2e}")
    print(f"Results Match (tol=1e-6):    {max_diff < 1e-6}")
    
    # Print sample values for comparison
    print(f"\nSample Output Comparison (top-left 3x3 region):")
    print("Standard conv2d:")
    print(conv_output[0, 0, :3, :3].detach().cpu().numpy())
    print("\nWinograd conv2d:")
    print(winograd_output[0, 0, :3, :3].detach().cpu().numpy())

def analyze_performance(std_time, winograd_time):
    """
    Analyze and print performance comparison results.
    
    Args:
        std_time (float): Standard convolution time
        winograd_time (float): Winograd convolution time
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    ratio = winograd_time / std_time
    speedup = std_time / winograd_time if winograd_time < std_time else winograd_time / std_time
    
    print(f"Standard conv2d time:    {std_time:.2f} ms per iteration")
    print(f"Winograd conv2d time:    {winograd_time:.2f} ms per iteration")
    print(f"Performance Ratio:       {ratio:.2f}x")
    
    if ratio > 1:
        print(f"Winograd is {ratio:.2f}x slower than Standard")
    else:
        print(f"Winograd is {speedup:.2f}x faster than Standard")
    
    print(f"Time Difference:         {abs(winograd_time - std_time):.2f} ms")
    print(f"Percentage Difference:   {abs(winograd_time - std_time) / std_time * 100:.1f}%")

def main():
    """Main function to run the Winograd Convolution demonstration."""
    print("Winograd Convolution Demo")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Create Winograd transformation matrices
    print("\nCreating Winograd transformation matrices...")
    B, G, A = create_winograd_matrices(device)
    print(f"Transformation matrices created: B{B.shape}, G{G.shape}, A{A.shape}")
    
    # Setup test parameters
    N, C, H, W = 1, 1, 8, 8  # Single image, one channel, 8x8 input
    print(f"\nTest Parameters:")
    print(f"Input shape: ({N}, {C}, {H}, {W})")
    print(f"Kernel shape: (1, {C}, 3, 3)")
    print(f"Expected output shape: ({N}, {C}, {H-2}, {W-2}) = ({N}, {C}, 6, 6)")
    
    # Create test inputs
    print("\nCreating test input image and kernel...")
    input_img, kernel = create_test_inputs(N, C, H, W, device)
    
    # Compute standard convolution for comparison
    print("\nComputing standard convolution...")
    conv_output = F.conv2d(input_img, kernel, bias=None, stride=1, padding=0)
    print(f"Standard conv2d output shape: {conv_output.shape}")
    
    # Compute Winograd convolution
    print("Computing Winograd convolution...")
    winograd_output = winograd_conv2d(input_img, kernel, B, G, A)
    print(f"Winograd conv2d output shape: {winograd_output.shape}")
    
    # Analyze accuracy
    analyze_accuracy(conv_output, winograd_output)
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    print("Timing standard convolution...")
    std_time = time_function(lambda: F.conv2d(input_img, kernel, bias=None, stride=1, padding=0), iterations=1000)
    
    print("Timing Winograd convolution...")
    winograd_time = time_function(lambda: winograd_conv2d(input_img, kernel, B, G, A), iterations=1000)
    
    # Analyze performance
    analyze_performance(std_time, winograd_time)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(conv_output, winograd_output, save_path="winograd_convolution_comparison.png")
    
    print("\nDemo completed successfully!")
    print("\nKey Insights:")
    print("- Winograd convolution reduces multiplications but may not always be faster")
    print("- Numerical accuracy is maintained within expected floating-point precision")
    print("- Performance benefits depend on tensor size and hardware implementation")
    print("- Production libraries like cuDNN have highly optimized implementations")

if __name__ == "__main__":
    main() 