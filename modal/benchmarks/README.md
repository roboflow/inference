# Modal Custom Python Blocks Benchmarks

This directory contains benchmarking tools for the Modal Custom Python Blocks implementation.

## Available Benchmarks

### 1. Comprehensive Benchmark Suite (`benchmark.py`)

A comprehensive suite that tests various performance aspects:

- Cold start time
- Warm start latency
- Execution complexity impact
- Concurrency performance
- Maximum throughput

**Usage:**
```bash
# Run all benchmarks
python benchmarks/benchmark.py

# Run a specific benchmark
python benchmarks/benchmark.py --test cold_start

# Save results to output directory
python benchmarks/benchmark.py --test throughput --output ./benchmark_results
```

### 2. Execution Mode Comparison (`compare_execution_modes.py`)

Compares local (in-process) execution with Modal (remote) execution across different complexity levels:

- Simple arithmetic operations
- NumPy array processing
- Compute-intensive matrix operations
- Image processing operations

**Usage:**
```bash
python benchmarks/compare_execution_modes.py
```

## Prerequisites

- Modal credentials configured (either environment variables or `~/.modal.toml`)
- Deployed Modal App (`inference-custom-blocks`)
- Python packages: numpy, matplotlib, pandas

## Interpreting Results

The benchmarks provide various metrics:

- **Latency**: Execution time in seconds
- **Cold start overhead**: Time to initialize a new container
- **Throughput**: Requests per second
- **Concurrency**: Performance under parallel load
- **Overhead factor**: Ratio of Modal execution time to local execution time

## Example Output

```
EXECUTION MODE COMPARISON SUMMARY
============================================================

SIMPLE
  Local: mean=0.0007s, median=0.0006s, p95=0.0010s
  Modal: mean=0.2345s, median=0.2189s, p95=0.3034s
  Overhead factor: 312.61x (mean)

NUMPY
  Local: mean=0.0015s, median=0.0012s, p95=0.0026s
  Modal: mean=0.2487s, median=0.2390s, p95=0.2821s
  Overhead factor: 170.51x (mean)

COMPUTE_INTENSIVE
  Local: mean=0.2341s, median=0.2321s, p95=0.2398s
  Modal: mean=0.4876s, median=0.4725s, p95=0.5432s
  Overhead factor: 2.08x (mean)

IMAGE_PROCESSING
  Local: mean=0.0087s, median=0.0075s, p95=0.0126s
  Modal: mean=0.2765s, median=0.2654s, p95=0.3121s
  Overhead factor: 31.78x (mean)
```

This output shows that Modal execution has significant overhead for simple operations, but the relative overhead decreases dramatically for compute-intensive tasks where the actual computation time dominates the network latency.
