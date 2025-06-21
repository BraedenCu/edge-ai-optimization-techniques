#!/usr/bin/env python3
"""
GELU Activation with Group Normalization Demo

This script demonstrates how Group Normalization and GELU activation work together
in diffusion models like Stable Diffusion. It shows the transformation of feature maps
through these layers and visualizes the effects on the data distribution.

Key Concepts:
- Group Normalization: Normalizes features within groups of channels
- GELU: Gaussian Error Linear Unit activation function
- Feature Map Transformation: How intermediate tensors change through normalization and activation

References:
- Group Normalization: https://arxiv.org/abs/1803.08494
- GELU Activation: https://arxiv.org/abs/1606.08415
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def setup_device():
    """Setup and return the appropriate device (GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

class GroupNormGELU(nn.Module):
    """
    A module that applies Group Normalization followed by GELU activation.
    
    This combination is commonly used in modern neural networks, particularly
    in transformer architectures and diffusion models.
    """
    
    def __init__(self, num_channels, num_groups=8):
        """
        Initialize the GroupNormGELU module.
        
        Args:
            num_channels (int): Number of input channels
            num_groups (int): Number of groups for Group Normalization
        """
        super(GroupNormGELU, self).__init__()
        # Create a GroupNorm layer: split num_channels into num_groups
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        # Create a GELU activation function
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """
        Forward pass: apply Group Normalization then GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Apply Group Normalization to normalize feature maps across groups
        x_norm = self.group_norm(x)
        # Apply GELU activation to introduce non-linearity
        x_activated = self.gelu(x_norm)
        return x_activated

def create_dummy_feature_map(batch_size=4, channels=64, height=32, width=32, device=None):
    """
    Create a dummy input tensor to simulate a feature map.
    
    Args:
        batch_size (int): Number of samples in the batch
        channels (int): Number of channels
        height (int): Height of the feature map
        width (int): Width of the feature map
        device (torch.device): Device to place the tensor on
        
    Returns:
        torch.Tensor: Random feature map tensor
    """
    return torch.randn(batch_size, channels, height, width, device=device)

def visualize_feature_maps(input_tensor, output_tensor, save_path=None):
    """
    Visualize the input and output feature maps side by side.
    
    Args:
        input_tensor (torch.Tensor): Input feature map
        output_tensor (torch.Tensor): Output feature map after GroupNorm + GELU
        save_path (str, optional): Path to save the visualization
    """
    # Extract the first channel of the first image from both tensors
    input_channel = input_tensor[0, 0].detach().cpu().numpy()
    output_channel = output_tensor[0, 0].detach().cpu().numpy()
    
    # Create a side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot input feature map
    im1 = ax1.imshow(input_channel, cmap='viridis')
    ax1.set_title("Input Feature Map (Channel 0)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Width")
    ax1.set_ylabel("Height")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot output feature map
    im2 = ax2.imshow(output_channel, cmap='viridis')
    ax2.set_title("Output after GroupNorm + GELU (Channel 0)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Width")
    ax2.set_ylabel("Height")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def analyze_statistics(input_tensor, output_tensor):
    """
    Analyze and print statistical information about the feature maps.
    
    Args:
        input_tensor (torch.Tensor): Input feature map
        output_tensor (torch.Tensor): Output feature map
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Input statistics
    input_mean = input_tensor.mean().item()
    input_std = input_tensor.std().item()
    input_min = input_tensor.min().item()
    input_max = input_tensor.max().item()
    
    # Output statistics
    output_mean = output_tensor.mean().item()
    output_std = output_tensor.std().item()
    output_min = output_tensor.min().item()
    output_max = output_tensor.max().item()
    
    print(f"{'Metric':<20} {'Input':<15} {'Output':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {input_mean:<15.4f} {output_mean:<15.4f}")
    print(f"{'Std':<20} {input_std:<15.4f} {output_std:<15.4f}")
    print(f"{'Min':<20} {input_min:<15.4f} {output_min:<15.4f}")
    print(f"{'Max':<20} {input_max:<15.4f} {output_max:<15.4f}")
    
    # Calculate percentage of negative values
    input_negative_pct = (input_tensor < 0).float().mean().item() * 100
    output_negative_pct = (output_tensor < 0).float().mean().item() * 100
    
    print(f"{'Negative %':<20} {input_negative_pct:<15.2f} {output_negative_pct:<15.2f}")

def main():
    """Main function to run the GELU + GroupNorm demonstration."""
    print("GELU Activation with Group Normalization Demo")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Create dummy feature map
    print("\nCreating dummy feature map...")
    dummy_tensor = create_dummy_feature_map(device=device)
    print(f"Input shape: {dummy_tensor.shape}")
    
    # Create and apply GroupNormGELU module
    print("\nApplying GroupNorm + GELU...")
    groupnorm_gelu = GroupNormGELU(num_channels=64, num_groups=8).to(device)
    output = groupnorm_gelu(dummy_tensor)
    print(f"Output shape: {output.shape}")
    
    # Analyze statistics
    analyze_statistics(dummy_tensor, output)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_feature_maps(dummy_tensor, output, save_path="gelu_groupnorm_visualization.png")
    
    print("\nDemo completed successfully!")
    print("\nKey Insights:")
    print("- GroupNorm normalizes features within groups, improving training stability")
    print("- GELU provides smooth non-linearity, often preferred over ReLU in modern architectures")
    print("- The combination helps stabilize training in deep networks like diffusion models")

if __name__ == "__main__":
    main() 