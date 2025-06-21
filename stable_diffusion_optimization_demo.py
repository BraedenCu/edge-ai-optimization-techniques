#!/usr/bin/env python3
"""
Stable Diffusion Optimization Demo

This script demonstrates the performance comparison between standard and optimized
Stable Diffusion pipelines using xFormers memory-efficient attention. It shows how
different optimization techniques can improve inference speed and memory usage.

Key Concepts:
- Stable Diffusion: Text-to-image generation model
- xFormers: Memory-efficient attention implementation
- Pipeline Optimization: Comparing different inference strategies
- Performance Benchmarking: Measuring real-world speed differences

References:
- Stable Diffusion: https://arxiv.org/abs/2112.10752
- xFormers: https://github.com/facebookresearch/xformers
- Flash Attention: https://arxiv.org/abs/2205.14135
"""

import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler

def setup_device():
    """Setup and return the appropriate device (GPU if available, else CPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    return device

def load_pipelines(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Load both standard and optimized Stable Diffusion pipelines.
    
    Args:
        model_id (str): Hugging Face model ID
        device (str): Device to load models on
        
    Returns:
        tuple: (standard_pipeline, optimized_pipeline)
    """
    print(f"\nLoading model: {model_id}")
    
    # Pipeline A: Standard (baseline)
    print("Loading standard pipeline...")
    pipe_standard = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe_standard = pipe_standard.to(device)
    
    # Pipeline B: Optimized (xFormers memory-efficient attention)
    print("Loading optimized pipeline...")
    pipe_optimized = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe_optimized = pipe_optimized.to(device)
    
    # Enable xFormers for the "Optimized" pipeline (only works on GPU)
    if device == "cuda":
        try:
            pipe_optimized.enable_xformers_memory_efficient_attention()
            print("xFormers memory-efficient attention enabled for optimized pipeline")
        except Exception as e:
            print(f"Warning: Could not enable xFormers: {e}")
            print("  Optimized pipeline will use standard attention")
    
    return pipe_standard, pipe_optimized

def generate_image(pipeline, prompt, steps, width, height, guidance_scale, scheduler_name):
    """
    Generate an image using the specified pipeline and parameters.
    
    Args:
        pipeline: Stable Diffusion pipeline
        prompt (str): Text prompt for image generation
        steps (int): Number of inference steps
        width (int): Image width
        height (int): Image height
        guidance_scale (float): Classifier-free guidance scale
        scheduler_name (str): Name of the scheduler to use
        
    Returns:
        PIL.Image: Generated image
    """
    # Switch schedulers on-the-fly
    if scheduler_name == "Euler Ancestral":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == "DPM Solver":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # Otherwise, the pipeline's default scheduler remains
    
    # Convert to int in case user input is float
    steps = int(steps)
    width = int(width)
    height = int(height)
    
    device = next(pipeline.parameters()).device
    
    if device.type == "cuda":
        # For GPU, use autocast for half-precision
        with torch.autocast("cuda"):
            image = pipeline(
                prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale
            ).images[0]
    else:
        # CPU or MPS path, no autocast
        image = pipeline(
            prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale
        ).images[0]
    
    return image

def benchmark_generation(pipeline, prompt, steps, width, height, guidance_scale, 
                        scheduler_name, iterations=3):
    """
    Benchmark image generation performance.
    
    Args:
        pipeline: Stable Diffusion pipeline
        prompt (str): Text prompt
        steps (int): Number of inference steps
        width (int): Image width
        height (int): Image height
        guidance_scale (float): Guidance scale
        scheduler_name (str): Scheduler name
        iterations (int): Number of iterations to run
        
    Returns:
        list: List of generation times in seconds
    """
    times = []
    device = next(pipeline.parameters()).device
    
    print(f"Benchmarking {iterations} iterations...")
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...")
        
        # Synchronize GPU if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Generate image
        image = generate_image(pipeline, prompt, steps, width, height, 
                             guidance_scale, scheduler_name)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        generation_time = time.time() - start_time
        times.append(generation_time)
        
        print(f"    Generation time: {generation_time:.2f}s")
    
    return times

def compare_pipelines(pipe_standard, pipe_optimized, test_prompt, test_params):
    """
    Compare the performance of standard and optimized pipelines.
    
    Args:
        pipe_standard: Standard Stable Diffusion pipeline
        pipe_optimized: Optimized Stable Diffusion pipeline
        test_prompt (str): Test prompt for generation
        test_params (dict): Test parameters
        
    Returns:
        tuple: (standard_times, optimized_times)
    """
    print("\n" + "="*60)
    print("PIPELINE COMPARISON")
    print("="*60)
    
    # Benchmark standard pipeline
    print("\nBenchmarking Standard Pipeline:")
    standard_times = benchmark_generation(
        pipe_standard, test_prompt, **test_params
    )
    
    # Benchmark optimized pipeline
    print("\nBenchmarking Optimized Pipeline:")
    optimized_times = benchmark_generation(
        pipe_optimized, test_prompt, **test_params
    )
    
    return standard_times, optimized_times

def analyze_performance_results(standard_times, optimized_times):
    """
    Analyze and print detailed performance results.
    
    Args:
        standard_times (list): Generation times for standard pipeline
        optimized_times (list): Generation times for optimized pipeline
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    avg_standard = np.mean(standard_times)
    avg_optimized = np.mean(optimized_times)
    std_standard = np.std(standard_times)
    std_optimized = np.std(optimized_times)
    
    # Calculate performance ratio
    ratio = avg_optimized / avg_standard
    speedup = avg_standard / avg_optimized if avg_optimized < avg_standard else avg_optimized / avg_standard
    
    print(f"{'Metric':<25} {'Standard':<15} {'Optimized':<15}")
    print("-" * 55)
    print(f"{'Average Time (s)':<25} {avg_standard:<15.2f} {avg_optimized:<15.2f}")
    print(f"{'Std Dev (s)':<25} {std_standard:<15.2f} {std_optimized:<15.2f}")
    print(f"{'Min Time (s)':<25} {min(standard_times):<15.2f} {min(optimized_times):<15.2f}")
    print(f"{'Max Time (s)':<25} {max(standard_times):<15.2f} {max(optimized_times):<15.2f}")
    
    print(f"\nPerformance Ratio: {ratio:.2f}x")
    if ratio > 1:
        print(f"Optimized pipeline is {ratio:.2f}x slower than Standard")
    else:
        print(f"Optimized pipeline is {speedup:.2f}x faster than Standard")
    
    print(f"Time Difference: {abs(avg_optimized - avg_standard):.2f}s")
    print(f"Percentage Difference: {abs(avg_optimized - avg_standard) / avg_standard * 100:.1f}%")

def visualize_performance_comparison(standard_times, optimized_times, save_path=None):
    """
    Create visualizations comparing the performance of both pipelines.
    
    Args:
        standard_times (list): Generation times for standard pipeline
        optimized_times (list): Generation times for optimized pipeline
        save_path (str, optional): Path to save the visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Boxplot for distribution comparison
    box_data = [standard_times, optimized_times]
    box_labels = ['Standard', 'Optimized']
    colors = ['lightblue', 'lightgreen']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel("Generation Time (seconds)", fontsize=12)
    ax1.set_title("Pipeline Performance Distribution", fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar plot for average comparison
    methods = ['Standard', 'Optimized']
    avg_times = [np.mean(standard_times), np.mean(optimized_times)]
    colors = ['lightblue', 'lightgreen']
    
    bars = ax2.bar(methods, avg_times, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{time_val:.2f}s", ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel("Average Generation Time (seconds)", fontsize=12)
    ax2.set_title("Average Performance Comparison", fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance visualization saved to {save_path}")
    
    plt.show()

def test_different_schedulers(pipeline, test_prompt, test_params):
    """
    Test performance with different schedulers.
    
    Args:
        pipeline: Stable Diffusion pipeline
        test_prompt (str): Test prompt
        test_params (dict): Test parameters
        
    Returns:
        dict: Dictionary of scheduler performance results
    """
    schedulers = ["Default", "Euler Ancestral", "DPM Solver"]
    results = {}
    
    print("\n" + "="*60)
    print("SCHEDULER COMPARISON")
    print("="*60)
    
    for scheduler in schedulers:
        print(f"\nTesting {scheduler} scheduler...")
        times = benchmark_generation(
            pipeline, test_prompt, scheduler_name=scheduler, **test_params
        )
        results[scheduler] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'times': times
        }
        print(f"  Average time: {results[scheduler]['avg_time']:.2f}s")
    
    return results

def main():
    """Main function to run the Stable Diffusion optimization demonstration."""
    print("Stable Diffusion Optimization Demo")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Test parameters
    test_prompt = "A futuristic cityscape at night, neon lights, cyberpunk style"
    test_params = {
        'steps': 20,
        'width': 512,
        'height': 512,
        'guidance_scale': 7.5
    }
    
    print(f"\nTest Parameters:")
    print(f"Prompt: {test_prompt}")
    print(f"Steps: {test_params['steps']}")
    print(f"Resolution: {test_params['width']}x{test_params['height']}")
    print(f"Guidance Scale: {test_params['guidance_scale']}")
    
    # Load pipelines
    pipe_standard, pipe_optimized = load_pipelines(device=device)
    
    # Compare pipelines
    standard_times, optimized_times = compare_pipelines(
        pipe_standard, pipe_optimized, test_prompt, test_params
    )
    
    # Analyze results
    analyze_performance_results(standard_times, optimized_times)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_performance_comparison(standard_times, optimized_times, 
                                   save_path="stable_diffusion_optimization_comparison.png")
    
    # Test different schedulers with optimized pipeline
    scheduler_results = test_different_schedulers(pipe_optimized, test_prompt, test_params)
    
    # Print scheduler comparison
    print("\n" + "="*60)
    print("SCHEDULER PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Scheduler':<20} {'Avg Time (s)':<15} {'Std Dev (s)':<15}")
    print("-" * 50)
    for scheduler, result in scheduler_results.items():
        print(f"{scheduler:<20} {result['avg_time']:<15.2f} {result['std_time']:<15.2f}")
    
    print("\nDemo completed successfully!")
    print("\nKey Insights:")
    print("- xFormers can provide memory efficiency and potential speed improvements")
    print("- Different schedulers offer trade-offs between speed and quality")
    print("- Performance benefits depend on hardware and model configuration")
    print("- Real-world optimization requires careful benchmarking")

if __name__ == "__main__":
    main() 