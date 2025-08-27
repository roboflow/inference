#!/usr/bin/env python3
"""
Compare execution modes for Custom Python Blocks.

This script compares local (in-process) vs Modal (remote) execution for 
Python blocks with varying complexity levels.

Usage:
    python modal/benchmarks/compare_execution_modes.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Set environment variables first
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "modal"
os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] = "False"  # Force remote execution

def setup_environment():
    """Set up execution environment."""
    # Check for Modal credentials
    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if not modal_token_id or not modal_token_secret:
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
                        modal_token_id = profile.get("token_id")
                        modal_token_secret = profile.get("token_secret")
                        
                        if modal_token_id and modal_token_secret:
                            print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                            os.environ["MODAL_TOKEN_ID"] = modal_token_id
                            os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
                            print("Modal credentials set in environment")
                            return True
                            
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
                        modal_token_id = profile.get("token_id")
                        modal_token_secret = profile.get("token_secret")
                        
                        if modal_token_id and modal_token_secret:
                            print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                            os.environ["MODAL_TOKEN_ID"] = modal_token_id
                            os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
                            print("Modal credentials set in environment")
                            return True
                        
            except Exception as e:
                print(f"Warning: Could not parse {modal_toml_path}: {e}")
    
    if modal_token_id and modal_token_secret:
        print("Using Modal credentials from environment")
        # Make sure they're set in the environment for the modal_executor
        os.environ["MODAL_TOKEN_ID"] = modal_token_id
        os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
        print("Modal credentials set in environment")
        return True
    else:
        print("\nWARNING: Modal credentials not found")
        print("Modal execution mode will be skipped")
        return False

# Set up environment before importing
modal_available = setup_environment()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import the executors
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import ModalExecutor
from inference.core.workflows.execution_engine.v1.dynamic_blocks.local_executor import LocalExecutor


def get_test_code(complexity: str) -> PythonCode:
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
    elif complexity == "image_processing":
        return PythonCode(
            type="PythonCode", 
            imports=["import numpy as np", "import cv2"],
            run_function_code="""
def process_image(image: np.ndarray) -> Dict[str, Any]:
    # Apply some image processing operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate some statistics
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    
    return {
        "mean_pixel": mean,
        "std_pixel": std,
        "contour_count": len(contours),
        "output_shape": list(edges.shape)
    }
""",
            run_function_name="process_image",
            init_function_name="init",
            init_function_code=None,
        )
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def get_test_inputs(complexity: str) -> Dict[str, Any]:
    """Get test inputs for specified complexity."""
    if complexity == "simple":
        return {"x": 5, "y": 10}
    elif complexity == "numpy":
        return {"data": np.random.randn(50, 50)}
    elif complexity == "compute_intensive":
        return {"size": 100, "iterations": 5}
    elif complexity == "image_processing":
        # Create a test image
        return {"image": np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)}
    else:
        raise ValueError(f"Unknown complexity: {complexity}")


def compare_execution_modes():
    """Compare execution performance between local and Modal."""
    print("\n" + "=" * 60)
    print("COMPARING EXECUTION MODES")
    print("=" * 60)
    
    # Set up environment
    modal_available = setup_environment()
    
    # Create executors
    local_executor = LocalExecutor()
    modal_executor = ModalExecutor(workspace_id="mode-comparison") if modal_available else None
    
    # Test configurations
    complexities = ["simple", "numpy", "compute_intensive", "image_processing"]
    iterations = 10
    
    # Store results
    results = defaultdict(lambda: defaultdict(list))
    
    for complexity in complexities:
        print(f"\nTesting {complexity} operations:")
        
        # Get code and inputs
        python_code = get_test_code(complexity)
        inputs = get_test_inputs(complexity)
        
        # Benchmark local execution
        print(f"  Local execution ({iterations} iterations):")
        for i in range(iterations):
            start_time = time.time()
            try:
                result = local_executor.execute(
                    block_type_name=f"comparison_test_{complexity}",
                    python_code=python_code,
                    inputs=inputs
                )
                duration = time.time() - start_time
                results[complexity]["local"].append(duration)
                print(f"    Run {i+1}/{iterations}: {duration:.4f}s")
            except Exception as e:
                print(f"    Run {i+1}/{iterations}: Error: {e}")
        
        # Benchmark Modal execution
        if modal_executor:
            print(f"  Modal execution ({iterations} iterations):")
            for i in range(iterations):
                start_time = time.time()
                try:
                    result = modal_executor.execute_remote(
                        block_type_name=f"comparison_test_{complexity}",
                        python_code=python_code,
                        inputs=inputs,
                        workspace_id="mode-comparison"
                    )
                    duration = time.time() - start_time
                    results[complexity]["modal"].append(duration)
                    print(f"    Run {i+1}/{iterations}: {duration:.4f}s")
                except Exception as e:
                    print(f"    Run {i+1}/{iterations}: Error: {e}")
    
    # Calculate and display summary statistics
    summary_data = []
    
    print("\n" + "=" * 60)
    print("EXECUTION MODE COMPARISON SUMMARY")
    print("=" * 60)
    
    for complexity in complexities:
        local_times = results[complexity]["local"]
        modal_times = results[complexity]["modal"]
        
        if local_times:
            local_avg = np.mean(local_times)
            local_median = np.median(local_times)
            local_p95 = np.percentile(local_times, 95) if len(local_times) > 1 else local_avg
        else:
            local_avg = local_median = local_p95 = float('nan')
            
        if modal_times:
            modal_avg = np.mean(modal_times)
            modal_median = np.median(modal_times)
            modal_p95 = np.percentile(modal_times, 95) if len(modal_times) > 1 else modal_avg
            
            # Calculate overhead
            overhead_avg = (modal_avg / local_avg) if local_avg > 0 else float('nan')
            overhead_median = (modal_median / local_median) if local_median > 0 else float('nan')
        else:
            modal_avg = modal_median = modal_p95 = overhead_avg = overhead_median = float('nan')
        
        summary_data.append({
            "Complexity": complexity,
            "Local Mean (s)": local_avg,
            "Local Median (s)": local_median,
            "Local P95 (s)": local_p95,
            "Modal Mean (s)": modal_avg, 
            "Modal Median (s)": modal_median,
            "Modal P95 (s)": modal_p95,
            "Overhead Factor": overhead_avg
        })
        
        print(f"\n{complexity.upper()}")
        print(f"  Local: mean={local_avg:.4f}s, median={local_median:.4f}s, p95={local_p95:.4f}s")
        if modal_times:
            print(f"  Modal: mean={modal_avg:.4f}s, median={modal_median:.4f}s, p95={modal_p95:.4f}s")
            print(f"  Overhead factor: {overhead_avg:.2f}x (mean)")
        else:
            print("  Modal: Not tested")
    
    # Create summary dataframe
    df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(df.to_string(index=False, float_format="%.4f"))
    
    # Generate comparison plot
    if modal_available:
        plt.figure(figsize=(12, 8))
        
        bar_width = 0.35
        index = np.arange(len(complexities))
        
        # Extract data for plotting
        local_means = [d["Local Mean (s)"] for d in summary_data]
        modal_means = [d["Modal Mean (s)"] for d in summary_data]
        
        # Create bars
        plt.bar(index, local_means, bar_width, label='Local Execution')
        plt.bar(index + bar_width, modal_means, bar_width, label='Modal Execution')
        
        plt.xlabel('Operation Complexity')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Local vs. Modal Execution Performance')
        plt.xticks(index + bar_width / 2, complexities)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig("modal_comparison.png")
        print("\nComparison plot saved as 'modal_comparison.png'")
    
    return results, summary_data


if __name__ == "__main__":
    compare_execution_modes()
