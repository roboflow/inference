#!/usr/bin/env python3
"""
Test script for Modal Custom Python Blocks implementation.

This script tests the Modal sandbox execution for Custom Python Blocks
in Roboflow Workflows.

Usage:
    # Set environment variables first
    export MODAL_TOKEN_ID="your_token_id"
    export MODAL_TOKEN_SECRET="your_token_secret"
    export MODAL_WORKSPACE_NAME="your_workspace"
    export WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE="modal"
    
    # Run the test
    python modal/test_modal_blocks.py
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Set test configuration
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "modal"
os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] = "False"  # Force remote execution

from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import ModalExecutor
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData, Batch


def test_simple_computation():
    """Test a simple mathematical computation."""
    print("Testing simple computation...")
    
    python_code = PythonCode(
        imports=[],
        code="""
def compute(x: int, y: int) -> Dict[str, Any]:
    result = x + y
    squared = result ** 2
    return {"sum": result, "squared": squared}
""",
        run_function_name="compute",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    executor = ModalExecutor(workspace_id="test-workspace")
    
    inputs = {"x": 5, "y": 3}
    result = executor.execute_remote(
        block_type_name="simple_computation",
        python_code=python_code,
        inputs=inputs,
        workspace_id="test-workspace"
    )
    
    print(f"  Input: {inputs}")
    print(f"  Result: {result}")
    assert result["sum"] == 8
    assert result["squared"] == 64
    print("  ✅ Simple computation test passed!")


def test_numpy_processing():
    """Test numpy array processing."""
    print("\nTesting numpy processing...")
    
    python_code = PythonCode(
        imports=["import numpy as np"],
        code="""
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
        "shape": data.shape
    }
""",
        run_function_name="process_array",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    executor = ModalExecutor(workspace_id="test-workspace")
    
    test_array = np.random.randn(10, 20)
    inputs = {"data": test_array}
    
    result = executor.execute_remote(
        block_type_name="numpy_processing",
        python_code=python_code,
        inputs=inputs,
        workspace_id="test-workspace"
    )
    
    print(f"  Array shape: {test_array.shape}")
    print(f"  Result: {result}")
    assert result["shape"] == [10, 20]
    assert "mean" in result
    assert "std" in result
    print("  ✅ Numpy processing test passed!")


def test_workflow_image_processing():
    """Test WorkflowImageData processing."""
    print("\nTesting WorkflowImageData processing...")
    
    python_code = PythonCode(
        imports=["import cv2", "import numpy as np"],
        code="""
def process_image(image: WorkflowImageData) -> Dict[str, Any]:
    img = image.numpy_image
    if img is not None:
        height, width = img.shape[:2]
        mean_pixel = float(np.mean(img))
        return {
            "width": width,
            "height": height,
            "mean_pixel": mean_pixel,
            "has_image": True
        }
    return {"has_image": False}
""",
        run_function_name="process_image",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    executor = ModalExecutor(workspace_id="test-workspace")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    workflow_image = WorkflowImageData(
        numpy_image=test_image,
        parent_metadata={},
        workflow_root_ancestor_metadata={}
    )
    
    inputs = {"image": workflow_image}
    
    result = executor.execute_remote(
        block_type_name="image_processing",
        python_code=python_code,
        inputs=inputs,
        workspace_id="test-workspace"
    )
    
    print(f"  Image shape: {test_image.shape}")
    print(f"  Result: {result}")
    assert result["width"] == 150
    assert result["height"] == 100
    assert result["has_image"] is True
    print("  ✅ WorkflowImageData processing test passed!")


def test_error_handling():
    """Test error handling in Modal execution."""
    print("\nTesting error handling...")
    
    python_code = PythonCode(
        imports=[],
        code="""
def failing_function(x: int) -> Dict[str, Any]:
    # This will raise an error
    result = x / 0
    return {"result": result}
""",
        run_function_name="failing_function",
        run_function_code="",
        init_function_name=None,
        init_function_code=None,
    )
    
    executor = ModalExecutor(workspace_id="test-workspace")
    
    inputs = {"x": 10}
    
    try:
        result = executor.execute_remote(
            block_type_name="error_test",
            python_code=python_code,
            inputs=inputs,
            workspace_id="test-workspace"
        )
        print("  ❌ Should have raised an error!")
        sys.exit(1)
    except Exception as e:
        print(f"  Expected error caught: {e}")
        print("  ✅ Error handling test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Modal Custom Python Blocks Test Suite")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("MODAL_TOKEN_ID"):
        print("ERROR: MODAL_TOKEN_ID environment variable not set")
        print("Please set Modal credentials before running tests")
        sys.exit(1)
    
    # Run tests
    test_simple_computation()
    test_numpy_processing()
    test_workflow_image_processing()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
