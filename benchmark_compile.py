"""Benchmark script to compare performance with and without torch.compile.

This script tests the speedup from torch.compile on key optimizer functions.

Author: Claude
Date: 2025-10-16
"""

import time

import numpy as np
import torch

from spider.config import Config, compute_steps, compute_noise_schedule
from spider.optimizers.sampling import sample_ctrls, _compute_weights_impl


def benchmark_sample_ctrls(config: Config, num_iterations: int = 100):
    """Benchmark the sample_ctrls function."""
    # Create dummy control input
    ctrls = torch.randn(config.horizon_steps, config.nu, device=config.device)
    sample_params = {"global_noise_scale": 1.0}

    # Warmup
    for _ in range(10):
        _ = sample_ctrls(config, ctrls, sample_params)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = sample_ctrls(config, ctrls, sample_params)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    return avg_time


def benchmark_compute_weights(config: Config, num_iterations: int = 100):
    """Benchmark the compute_weights function."""
    # Create dummy rewards
    rews = torch.randn(config.num_samples, device=config.device)

    # Import the compiled version
    from spider.optimizers.sampling import _compute_weights_compiled

    # Choose which version to use
    compute_fn = (
        _compute_weights_compiled if config.use_torch_compile else _compute_weights_impl
    )

    # Warmup
    for _ in range(10):
        _ = compute_fn(rews, config.num_samples, config.temperature)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = compute_fn(rews, config.num_samples, config.temperature)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    return avg_time


def main():
    """Run benchmarks comparing compiled vs non-compiled versions."""
    print("=" * 80)
    print("torch.compile Benchmark")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Create a minimal config
    config = Config()
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.num_samples = 1024
    config.temperature = 1.0
    config.horizon = 1.2
    config.sim_dt = 0.01
    config.knot_dt = 0.4
    config.nu = 32  # Typical hand DOF
    config = compute_steps(config)
    config = compute_noise_schedule(config)

    num_iterations = 100

    print(f"\nConfig: num_samples={config.num_samples}, nu={config.nu}, "
          f"horizon_steps={config.horizon_steps}")
    print(f"Running {num_iterations} iterations for each benchmark...\n")

    # Benchmark sample_ctrls
    print("-" * 80)
    print("Benchmarking sample_ctrls...")
    print("-" * 80)

    config.use_torch_compile = False
    time_no_compile = benchmark_sample_ctrls(config, num_iterations)
    print(f"Without torch.compile: {time_no_compile:.3f} ms")

    config.use_torch_compile = True
    time_with_compile = benchmark_sample_ctrls(config, num_iterations)
    print(f"With torch.compile:    {time_with_compile:.3f} ms")

    speedup = time_no_compile / time_with_compile
    print(f"Speedup: {speedup:.2f}x")

    # Benchmark compute_weights
    print("\n" + "-" * 80)
    print("Benchmarking compute_weights...")
    print("-" * 80)

    config.use_torch_compile = False
    time_no_compile = benchmark_compute_weights(config, num_iterations)
    print(f"Without torch.compile: {time_no_compile:.3f} ms")

    config.use_torch_compile = True
    time_with_compile = benchmark_compute_weights(config, num_iterations)
    print(f"With torch.compile:    {time_with_compile:.3f} ms")

    speedup = time_no_compile / time_with_compile
    print(f"Speedup: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
