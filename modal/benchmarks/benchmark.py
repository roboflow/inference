#!/usr/bin/env python3
"""
Benchmarking script for Modal Custom Python Blocks implementation.

This script benchmarks various performance aspects of the Modal sandbox
execution for Custom Python Blocks in Roboflow Workflows, including:

- Cold start time
- Execution latency
- Throughput
- Concurrency
- Memory usage
- Error handling

Usage:
    python modal/benchmarks/benchmark.py [--test <test_name>]
"""

import argparse
import asyncio
import os
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

# Set default environment variables for tests
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "modal"
os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] = "False"  # Force remote execution

# Load Modal credentials
def load_modal_credentials():
    """Load Modal credentials from environment or ~/.modal.toml file.
    
    Returns:
        tuple: (token_id, token_secret) or (None, None) if not found
    """
    # First check environment variables
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if token_id and token_secret:
        print("✓ Using Modal credentials from environment variables")
        return token_id, token_secret
    
    # Try to read from ~/.modal.toml
    modal_toml_path = Path.home() / ".modal.toml"
    if modal_toml_path.exists():
        try:
            # Try using toml if available, otherwise parse manually
            try:
                import toml
                config = toml.load(modal_toml_path)
                
                # Find the active profile or use the first available profile
                active_profile = None
                for section, values in config.items():
                    if isinstance(values, dict) and values.get("active", False):
                        active_profile = section
                        break
                
                # Use active profile, or fall back to first available
                if not active_profile and config:
                    active_profile = list(config.keys())[0]
                
                if active_profile and active_profile in config:
                    profile = config[active_profile]
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                        
            except ImportError:
                # Fallback: Parse TOML file manually for basic key-value pairs
                import re
                content = modal_toml_path.read_text()
                
                # Find sections and their content
                sections = re.findall(r'\[([^\]]+)\](.*?)(?=\[|$)', content, re.DOTALL)
                
                active_profile = None
                profiles = {}
                
                for section_name, section_content in sections:
                    # Parse key-value pairs in section
                    profile_data = {}
                    
                    # Match lines like: key = "value" or key = 'value'
                    for match in re.finditer(r'^(\w+)\s*=\s*["\']([^"\']+)["\']', section_content, re.MULTILINE):
                        key, value = match.groups()
                        profile_data[key] = value
                    
                    # Also match boolean values
                    for match in re.finditer(r'^(\w+)\s*=\s*(true|false)', section_content, re.MULTILINE):
                        key, value = match.groups()
                        profile_data[key] = value == 'true'
                    
                    profiles[section_name] = profile_data
                    
                    # Check if this is the active profile
                    if profile_data.get('active', False):
                        active_profile = section_name
                
                # Use active profile or first available
                if not active_profile and profiles:
                    active_profile = list(profiles.keys())[0]
                
                if active_profile and active_profile in profiles:
                    profile = profiles[active_profile]
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                    
        except Exception as e:
            print(f"Warning: Could not parse {modal_toml_path}: {e}")
    
    print("\nERROR: Modal credentials not found")
    print("\nPlease provide credentials using one of these methods:")
    print("1. Set environment variables:")
    print("   export MODAL_TOKEN_ID='your_token_id'")
    print("   export MODAL_TOKEN_SECRET='your_token_secret'")
    print("\n2. Run 'modal setup' to create ~/.modal.toml")
    print("\n3. Create ~/.modal.toml manually with a profile like:")
    print("   [default]  # or [your-profile-name]")
    print("   token_id = \"your_token_id\"")
    print("   token_secret = \"your_token_secret\"")
    print("   active = true  # optional, marks this as the active profile")
    sys.exit(1)


# Load credentials before anything else
token_id, token_secret = load_modal_credentials()

# Make sure environment variables are set
if token_id and token_secret:
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret
    print("Modal credentials set in environment")

# Only now import Modal executor
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import ModalExecutor



class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
        self.durations = []
        self.errors = []
        self.cold_starts = []
        self.warm_starts = []
        self.results = []
    
    def add_result(self, duration: float, is_cold_start: bool, result: Any = None, error: Optional[Exception] = None):
        """Add a benchmark result."""
        self.durations.append(duration)
        if error:
            self.errors.append(error)
        
        if is_cold_start:
            self.cold_starts.append(duration)
        else:
            self.warm_starts.append(duration)
            
        if result is not None:
            self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        if not self.durations:
            return {"error": "No results collected"}
        
        stats = {
            "name": self.name,
            "total_runs": len(self.durations),
            "cold_starts": len(self.cold_starts),
            "warm_starts": len(self.warm_starts),
            "errors": len(self.errors),
            "total_duration_seconds": sum(self.durations),
            "min_duration_seconds": min(self.durations),
            "max_duration_seconds": max(self.durations),
            "mean_duration_seconds": statistics.mean(self.durations),
            "median_duration_seconds": statistics.median(self.durations),
            "stdev_duration_seconds": statistics.stdev(self.durations) if len(self.durations) > 1 else 0,
            "p95_duration_seconds": np.percentile(self.durations, 95) if self.durations else 0,
            "p99_duration_seconds": np.percentile(self.durations, 99) if self.durations else 0,
        }
        
        # Add cold start stats if available
        if self.cold_starts:
            stats["cold_start_mean_seconds"] = statistics.mean(self.cold_starts)
            stats["cold_start_median_seconds"] = statistics.median(self.cold_starts)
            stats["cold_start_p95_seconds"] = np.percentile(self.cold_starts, 95)
        
        # Add warm start stats if available
        if self.warm_starts:
            stats["warm_start_mean_seconds"] = statistics.mean(self.warm_starts)
            stats["warm_start_median_seconds"] = statistics.median(self.warm_starts)
            stats["warm_start_p95_seconds"] = np.percentile(self.warm_starts, 95)
            
        return stats
    
    def print_summary(self):
        """Print a formatted summary of benchmark results."""
        summary = self.summary()
        
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK RESULTS: {self.name}")
        print(f"{'=' * 60}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Cold starts: {summary['cold_starts']}")
        print(f"Warm starts: {summary['warm_starts']}")
        print(f"Errors: {summary['errors']}")
        print(f"Total duration: {summary['total_duration_seconds']:.2f} seconds")
        print("\nLatency statistics (seconds):")
        print(f"  Min: {summary['min_duration_seconds']:.4f}")
        print(f"  Max: {summary['max_duration_seconds']:.4f}")
        print(f"  Mean: {summary['mean_duration_seconds']:.4f}")
        print(f"  Median: {summary['median_duration_seconds']:.4f}")
        print(f"  Std Dev: {summary['stdev_duration_seconds']:.4f}")
        print(f"  P95: {summary['p95_duration_seconds']:.4f}")
        print(f"  P99: {summary['p99_duration_seconds']:.4f}")
        
        if self.cold_starts:
            print("\nCold start statistics (seconds):")
            print(f"  Mean: {summary.get('cold_start_mean_seconds', 0):.4f}")
            print(f"  Median: {summary.get('cold_start_median_seconds', 0):.4f}")
            print(f"  P95: {summary.get('cold_start_p95_seconds', 0):.4f}")
        
        if self.warm_starts:
            print("\nWarm start statistics (seconds):")
            print(f"  Mean: {summary.get('warm_start_mean_seconds', 0):.4f}")
            print(f"  Median: {summary.get('warm_start_median_seconds', 0):.4f}")
            print(f"  P95: {summary.get('warm_start_p95_seconds', 0):.4f}")
        
        print(f"{'=' * 60}")
    
    def plot_latency_distribution(self, save_path: Optional[str] = None):
        """Generate a latency distribution plot."""
        if not self.durations:
            print("No data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot histograms
        if self.cold_starts:
            plt.hist(self.cold_starts, alpha=0.7, label='Cold Starts', bins=10)
        if self.warm_starts:
            plt.hist(self.warm_starts, alpha=0.7, label='Warm Starts', bins=10)
            
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'Latency Distribution: {self.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def get_test_python_code(complexity: str = "simple") -> PythonCode:
    """Get test Python code with specified complexity."""
    if complexity == "simple":
        return PythonCode(
            type="PythonCode",
            imports=[],
            run_function_code="""
def compute(x: int, y: int) -> Dict[str, Any]:
    result = x + y
    squared = result ** 2
    return {"sum": result, "squared": squared}
""",
            run_function_name="compute",
            init_function_name="init",
            init_function_code=None,
        )
    elif complexity == "numpy":
        return PythonCode(
            type="PythonCode",
            imports=["import numpy as np"],
            run_function_code="""
def process_array(data: np.ndarray) -> Dict[str, Any]:
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    max_val = float(np.max(data))
    min_val = float(np.min(data))
    
    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "shape": list(data.shape)
    }
""",
            run_function_name="process_array",
            init_function_name="init",
            init_function_code=None,
        )
    elif complexity == "compute_intensive":
        return PythonCode(
            type="PythonCode",
            imports=["import numpy as np", "import time"],
            run_function_code="""
def heavy_compute(size: int, iterations: int) -> Dict[str, Any]:
    start_time = time.time()
    
    # Create large matrices and perform operations
    result = 0
    for i in range(iterations):
        matrix_a = np.random.rand(size, size)
        matrix_b = np.random.rand(size, size)
        result += np.sum(np.dot(matrix_a, matrix_b))
    
    process_time = time.time() - start_time
    
    return {
        "result": float(result),
        "process_time_seconds": process_time,
        "matrix_size": size,
        "iterations": iterations
    }
""",
            run_function_name="heavy_compute",
            init_function_name="init",
            init_function_code=None,
        )
    elif complexity == "memory_intensive":
        return PythonCode(
            type="PythonCode",
            imports=["import numpy as np", "import sys"],
            run_function_code="""
def memory_test(size_mb: int) -> Dict[str, Any]:
    # Calculate array size to consume roughly size_mb megabytes
    # Each float64 is 8 bytes
    array_size = int((size_mb * 1024 * 1024) / 8)
    
    # Create a large array
    large_array = np.random.rand(array_size)
    
    # Perform some operations
    result = np.mean(large_array)
    std = np.std(large_array)
    max_val = np.max(large_array)
    
    # Get memory usage info
    memory_usage = sys.getsizeof(large_array) / (1024 * 1024)  # in MB
    
    return {
        "mean": float(result),
        "std": float(std),
        "max": float(max_val),
        "requested_memory_mb": size_mb,
        "actual_memory_mb": memory_usage
    }
""",
            run_function_name="memory_test",
            init_function_name="init",
            init_function_code=None,
        )
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def benchmark_cold_start():
    """Benchmark cold start performance."""
    results = BenchmarkResults("Cold Start Performance")
    workspace_ids = [f"benchmark-workspace-{i}" for i in range(5)]
    
    print("\nMemory snapshot enabled: This should improve cold start times.")
    print("Benchmarking memory snapshot performance...")
    
    for i in range(10):
        # Use a different workspace for each run to ensure cold starts
        workspace_id = workspace_ids[i % len(workspace_ids)]
        
        # Create new executor each time
        executor = ModalExecutor(workspace_id=workspace_id)
        python_code = get_test_python_code("simple")
        
        # Measure execution time
        start_time = time.time()
        try:
            result = executor.execute_remote(
                block_type_name="cold_start_test",
                python_code=python_code,
                inputs={"x": 5, "y": 10},
                workspace_id=workspace_id
            )
            duration = time.time() - start_time
            results.add_result(duration, is_cold_start=True, result=result)
            print(f"Run {i+1}/10: Cold start took {duration:.4f} seconds")
        except Exception as e:
            duration = time.time() - start_time
            results.add_result(duration, is_cold_start=True, error=e)
            print(f"Run {i+1}/10: Error during cold start: {e}")
        
        # Sleep to ensure container is recycled
        time.sleep(2)
    
    results.print_summary()
    results.plot_latency_distribution()
    return results


def benchmark_warm_start():
    """Benchmark warm start performance."""
    results = BenchmarkResults("Warm Start Performance")
    workspace_id = "benchmark-workspace-warm"
    
    # Create executor
    executor = ModalExecutor(workspace_id=workspace_id)
    python_code = get_test_python_code("simple")
    
    # First run to warm up
    print("Warming up executor...")
    try:
        executor.execute_remote(
            block_type_name="warm_start_test",
            python_code=python_code,
            inputs={"x": 5, "y": 10},
            workspace_id=workspace_id
        )
    except Exception as e:
        print(f"Error during warm-up: {e}")
        return results
    
    # Wait a moment for container to stabilize
    time.sleep(1)
    
    # Run benchmark
    for i in range(20):
        start_time = time.time()
        try:
            result = executor.execute_remote(
                block_type_name="warm_start_test",
                python_code=python_code,
                inputs={"x": 5, "y": 10},
                workspace_id=workspace_id
            )
            duration = time.time() - start_time
            results.add_result(duration, is_cold_start=False, result=result)
            print(f"Run {i+1}/20: Warm start took {duration:.4f} seconds")
        except Exception as e:
            duration = time.time() - start_time
            results.add_result(duration, is_cold_start=False, error=e)
            print(f"Run {i+1}/20: Error during warm start: {e}")
    
    results.print_summary()
    results.plot_latency_distribution()
    return results


def benchmark_complexity_comparison():
    """Benchmark different complexity levels."""
    results = {}
    workspace_id = "benchmark-workspace-complexity"
    
    # Create executor
    executor = ModalExecutor(workspace_id=workspace_id)
    
    # Test different complexities
    complexities = {
        "simple": {"x": 5, "y": 10},
        "numpy": {"data": np.random.randn(50, 50)},
        "compute_intensive": {"size": 100, "iterations": 5},
        "memory_intensive": {"size_mb": 50}
    }
    
    for complexity, inputs in complexities.items():
        print(f"\nBenchmarking {complexity} complexity...")
        complexity_results = BenchmarkResults(f"Complexity: {complexity}")
        python_code = get_test_python_code(complexity)
        
        # First run (cold start)
        start_time = time.time()
        try:
            result = executor.execute_remote(
                block_type_name=f"complexity_test_{complexity}",
                python_code=python_code,
                inputs=inputs,
                workspace_id=workspace_id
            )
            duration = time.time() - start_time
            complexity_results.add_result(duration, is_cold_start=True, result=result)
            print(f"Cold start: {duration:.4f} seconds")
        except Exception as e:
            duration = time.time() - start_time
            complexity_results.add_result(duration, is_cold_start=True, error=e)
            print(f"Error during cold start: {e}")
        
        # Subsequent runs (warm starts)
        for i in range(5):
            start_time = time.time()
            try:
                result = executor.execute_remote(
                    block_type_name=f"complexity_test_{complexity}",
                    python_code=python_code,
                    inputs=inputs,
                    workspace_id=workspace_id
                )
                duration = time.time() - start_time
                complexity_results.add_result(duration, is_cold_start=False, result=result)
                print(f"Warm start {i+1}/5: {duration:.4f} seconds")
            except Exception as e:
                duration = time.time() - start_time
                complexity_results.add_result(duration, is_cold_start=False, error=e)
                print(f"Error during warm start {i+1}/5: {e}")
        
        complexity_results.print_summary()
        results[complexity] = complexity_results
    
    return results


def benchmark_concurrency():
    """Benchmark concurrent execution."""
    results = BenchmarkResults("Concurrency Performance")
    workspace_id = "benchmark-workspace-concurrency"
    
    def run_single_execution(i: int) -> Tuple[float, Any, Optional[Exception]]:
        """Run a single execution and return metrics."""
        executor = ModalExecutor(workspace_id=workspace_id)
        python_code = get_test_python_code("simple")
        
        start_time = time.time()
        try:
            result = executor.execute_remote(
                block_type_name="concurrency_test",
                python_code=python_code,
                inputs={"x": i, "y": i * 2},
                workspace_id=workspace_id
            )
            return time.time() - start_time, result, None
        except Exception as e:
            return time.time() - start_time, None, e
    
    # Run with increasing concurrency
    concurrency_levels = [1, 5, 10, 20]
    
    for concurrency in concurrency_levels:
        print(f"\nTesting concurrency level: {concurrency}")
        
        start_time = time.time()
        is_cold_start = True  # First batch is cold start
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_single_execution, i) for i in range(concurrency)]
            
            for future in concurrent.futures.as_completed(futures):
                duration, result, error = future.result()
                results.add_result(duration, is_cold_start=is_cold_start, result=result, error=error)
                print(f"Execution took {duration:.4f} seconds")
                is_cold_start = False  # Subsequent executions might be warm
        
        total_time = time.time() - start_time
        throughput = concurrency / total_time
        
        print(f"Concurrency {concurrency}: Total time {total_time:.4f}s, Throughput {throughput:.2f} req/s")
        
        # Short pause between batches
        time.sleep(2)
    
    results.print_summary()
    return results


def benchmark_throughput():
    """Benchmark maximum throughput."""
    results = BenchmarkResults("Throughput Performance")
    workspace_id = "benchmark-workspace-throughput"
    
    # Create executor
    executor = ModalExecutor(workspace_id=workspace_id)
    
    # Warm up
    python_code = get_test_python_code("simple")
    try:
        executor.execute_remote(
            block_type_name="throughput_test",
            python_code=python_code,
            inputs={"x": 1, "y": 2},
            workspace_id=workspace_id
        )
        print("Warm-up complete")
    except Exception as e:
        print(f"Error during warm-up: {e}")
        return results
    
    # Run benchmark
    num_requests = 50
    start_time = time.time()
    
    for i in range(num_requests):
        batch_start = time.time()
        try:
            result = executor.execute_remote(
                block_type_name="throughput_test",
                python_code=python_code,
                inputs={"x": i, "y": i * 2},
                workspace_id=workspace_id
            )
            duration = time.time() - batch_start
            results.add_result(duration, is_cold_start=False, result=result)
            print(f"Request {i+1}/{num_requests}: {duration:.4f} seconds")
        except Exception as e:
            duration = time.time() - batch_start
            results.add_result(duration, is_cold_start=False, error=e)
            print(f"Request {i+1}/{num_requests}: Error: {e}")
    
    total_time = time.time() - start_time
    throughput = num_requests / total_time
    
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Average throughput: {throughput:.2f} requests/second")
    
    results.print_summary()
    return results


def run_all_benchmarks():
    """Run all benchmark tests."""
    results = {}
    
    print("\n" + "=" * 60)
    print("MODAL CUSTOM PYTHON BLOCKS BENCHMARKS")
    print("=" * 60)
    
    print("\nRunning cold start benchmark...")
    results["cold_start"] = benchmark_cold_start()
    
    print("\nRunning warm start benchmark...")
    results["warm_start"] = benchmark_warm_start()
    
    print("\nRunning complexity comparison benchmark...")
    results["complexity"] = benchmark_complexity_comparison()
    
    print("\nRunning concurrency benchmark...")
    results["concurrency"] = benchmark_concurrency()
    
    print("\nRunning throughput benchmark...")
    results["throughput"] = benchmark_throughput()
    
    print("\nRunning cooldown impact benchmark...")
    try:
        from modal.benchmarks.cooldown_benchmark import benchmark_cooldown_impact
        results["cooldown"] = benchmark_cooldown_impact()
    except Exception as e:
        print(f"Error running cooldown benchmark: {e}")
    
    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)
    
    return results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark Modal Custom Python Blocks")
    parser.add_argument(
        "--test", 
        choices=["cold_start", "warm_start", "complexity", "concurrency", "throughput", "cooldown", "all"],
        default="all",
        help="Which benchmark test to run"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Directory to save benchmark results and plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to {output_dir}")
    
    # Run selected benchmark
    if args.test == "cold_start":
        results = benchmark_cold_start()
    elif args.test == "warm_start":
        results = benchmark_warm_start()
    elif args.test == "complexity":
        results = benchmark_complexity_comparison()
    elif args.test == "concurrency":
        results = benchmark_concurrency()
    elif args.test == "throughput":
        results = benchmark_throughput()
    elif args.test == "cooldown":
        from modal.benchmarks.cooldown_benchmark import benchmark_cooldown_impact
        results = benchmark_cooldown_impact()
    else:
        results = run_all_benchmarks()
    
    # Save results if output directory specified
    if args.output and isinstance(results, BenchmarkResults):
        import json
        output_path = Path(args.output) / f"{args.test}_results.json"
        with open(output_path, "w") as f:
            json.dump(results.summary(), f, indent=2)
        
        # Save plot
        plot_path = Path(args.output) / f"{args.test}_latency.png"
        results.plot_latency_distribution(save_path=str(plot_path))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
