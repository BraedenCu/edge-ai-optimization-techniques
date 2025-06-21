#!/usr/bin/env python3
"""
Run All Optimization Technique Demos

This script runs all the optimization technique demonstrations in sequence,
providing a comprehensive overview of all the implemented techniques.
"""

import subprocess
import sys
import time
import os

def run_demo(demo_name, demo_file):
    """Run a single demo with proper formatting."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {demo_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the demo
        result = subprocess.run([sys.executable, demo_file], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print(f"\n{demo_name} completed successfully!")
        else:
            print(f"\n{demo_name} failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"\nError running {demo_name}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration:.2f} seconds")
    
    return result.returncode == 0

def main():
    """Main function to run all demos."""
    print("Edge AI Optimization Techniques - Complete Demo Suite")
    print("=" * 80)
    print("This script will run all optimization technique demonstrations.")
    print("Each demo includes benchmarking, visualization, and analysis.")
    print("=" * 80)
    
    # Define all demos
    demos = [
        ("GELU Activation with Group Normalization", "gelu_groupnorm_demo.py"),
        ("JIT-Fused Softmax", "jit_fused_softmax_demo.py"),
        ("Flash Attention vs Traditional Attention", "flash_attention_demo.py"),
        ("Winograd Convolution", "winograd_convolution_demo.py"),
        ("Stable Diffusion with xFormers", "stable_diffusion_optimization_demo.py")
    ]
    
    # Check if all demo files exist
    missing_files = []
    for name, filename in demos:
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print(f"\nMissing demo files: {missing_files}")
        print("Please ensure all demo files are in the current directory.")
        return
    
    # Run all demos
    successful_demos = 0
    total_demos = len(demos)
    
    for name, filename in demos:
        success = run_demo(name, filename)
        if success:
            successful_demos += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("DEMO SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful_demos}/{total_demos}")
    print(f"Failed: {total_demos - successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("\nAll demos completed successfully!")
        print("\nKey Takeaways:")
        print("- Each optimization technique serves different purposes")
        print("- Performance benefits depend on hardware and use case")
        print("- Production libraries often have highly optimized implementations")
        print("- Real-world optimization requires careful benchmarking")
    else:
        print(f"\n{total_demos - successful_demos} demo(s) failed.")
        print("Check the output above for error details.")
    
    print(f"\nGenerated files:")
    print("- gelu_groupnorm_visualization.png")
    print("- jit_fused_softmax_benchmark.png")
    print("- flash_attention_benchmark.png")
    print("- winograd_convolution_comparison.png")
    print("- stable_diffusion_optimization_comparison.png")
    
    print(f"\nFor more information, see the README.md file.")

if __name__ == "__main__":
    main() 