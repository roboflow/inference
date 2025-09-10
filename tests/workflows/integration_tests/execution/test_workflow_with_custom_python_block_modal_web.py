"""
Tests for Custom Python Blocks execution via Modal Web Endpoints.

These tests verify that Custom Python Blocks work correctly when executed
via Modal web endpoints, ensuring the new implementation maintains compatibility
while removing the 2MB size limitation.

To run these tests:
1. Deploy the web endpoint first:
   export MODAL_TOKEN_ID="your_token_id"
   export MODAL_TOKEN_SECRET="your_token_secret"
   python deploy_modal_web.py
   
2. Run tests:
   pytest tests/workflows/integration_tests/execution/test_workflow_with_custom_python_block_modal_web.py
"""

import os
import json
import base64
from typing import Any, Dict

import numpy as np
import pytest
import cv2
import requests

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


# Skip all tests if Modal credentials or endpoint URL are not present
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")

# Get Modal web endpoint URL - try environment variable first, then dynamic retrieval
MODAL_WEB_ENDPOINT_URL = os.getenv("MODAL_WEB_ENDPOINT_URL")

if not MODAL_WEB_ENDPOINT_URL and MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    # Try to get URL dynamically from deployed Modal app
    try:
        import modal
        os.environ["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
        os.environ["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET
        
        cls = modal.Cls.from_name("inference-custom-blocks-web", "CustomBlockExecutor")
        instance = cls(workspace_id="test")
        if hasattr(instance, 'execute_block') and hasattr(instance.execute_block, 'get_web_url'):
            url = instance.execute_block.get_web_url()
            if url:
                MODAL_WEB_ENDPOINT_URL = url.split('?')[0]
                print(f"Dynamically retrieved Modal URL: {MODAL_WEB_ENDPOINT_URL}")
    except Exception as e:
        print(f"Could not dynamically retrieve Modal URL: {e}")
        # Use fallback URL as last resort
        MODAL_WEB_ENDPOINT_URL = "https://roboflow--inference-custom-blocks-web-customblockexecuto-4874a9.modal.run"

if not MODAL_WEB_ENDPOINT_URL:
    # Final fallback
    MODAL_WEB_ENDPOINT_URL = "https://roboflow--inference-custom-blocks-web-customblockexecuto-4874a9.modal.run"

SKIP_MODAL_WEB_TESTS = not (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
SKIP_REASON = "Modal credentials not present (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"


def call_modal_web_endpoint(
    code_str: str,
    inputs: Dict[str, Any],
    workspace_id: str = "test-workspace"
) -> Dict[str, Any]:
    """Call the Modal web endpoint directly."""
    
    # Serialize inputs to JSON
    inputs_json = json.dumps(inputs)
    
    # Prepare request
    request_payload = {
        "code_str": code_str,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": inputs_json,
    }
    
    # Make request
    url = f"{MODAL_WEB_ENDPOINT_URL}?workspace_id={workspace_id}"
    response = requests.post(url, json=request_payload, timeout=30)
    
    if response.status_code != 200:
        raise RuntimeError(f"HTTP error {response.status_code}: {response.text}")
    
    result = response.json()
    if not result.get("success"):
        raise RuntimeError(f"Execution failed: {result.get('error')}")
    
    # Parse and return the result
    return json.loads(result["result"])


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_simple_addition_web_endpoint():
    """Test basic addition via web endpoint."""
    
    code = """
def run(a: float, b: float):
    return {"sum": a + b}
"""
    
    inputs = {"a": 5.0, "b": 3.0}
    result = call_modal_web_endpoint(code, inputs)
    
    assert result["sum"] == 8.0


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_large_payload_web_endpoint():
    """Test handling of payloads > 2MB via web endpoint."""
    
    code = """
import numpy as np

def run(large_array):
    arr = np.array(large_array)
    return {
        "shape": list(arr.shape),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "size_mb": arr.nbytes / (1024 * 1024),
        "first_five": arr[:5].tolist()
    }
"""
    
    # Create a large array (>2MB)
    # 1M floats * 8 bytes/float = 8MB
    large_array = np.random.randn(1000000).tolist()
    
    inputs = {"large_array": large_array}
    result = call_modal_web_endpoint(code, inputs)
    
    # Verify the results
    assert result["shape"] == [1000000]
    assert result["size_mb"] > 2.0  # Should be around 8MB
    assert len(result["first_five"]) == 5
    assert isinstance(result["mean"], float)
    assert isinstance(result["std"], float)


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_numpy_operations_web_endpoint():
    """Test numpy operations via web endpoint."""
    
    code = """
import numpy as np

def run(matrix_a, matrix_b):
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    
    dot_product = np.dot(a, b)
    
    return {
        "dot_product": dot_product.tolist(),
        "shape_a": a.shape,
        "shape_b": b.shape,
        "result_shape": dot_product.shape
    }
"""
    
    inputs = {
        "matrix_a": [[1, 2], [3, 4]],
        "matrix_b": [[5, 6], [7, 8]]
    }
    
    result = call_modal_web_endpoint(code, inputs)
    
    expected_dot = [[19, 22], [43, 50]]
    assert result["dot_product"] == expected_dot
    assert result["shape_a"] == [2, 2]
    assert result["shape_b"] == [2, 2]
    assert result["result_shape"] == [2, 2]


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_error_handling_web_endpoint():
    """Test error handling via web endpoint."""
    
    code = """
def run(value: float):
    if value < 0:
        raise ValueError("Value must be non-negative")
    return {"result": value ** 0.5}
"""
    
    # Test success case
    inputs_success = {"value": 16.0}
    result = call_modal_web_endpoint(code, inputs_success)
    assert abs(result["result"] - 4.0) < 0.001
    
    # Test error case
    inputs_error = {"value": -5.0}
    with pytest.raises(RuntimeError) as exc_info:
        call_modal_web_endpoint(code, inputs_error)
    assert "Value must be non-negative" in str(exc_info.value)


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_authentication_failure_web_endpoint():
    """Test authentication failure handling."""
    
    code = """
def run():
    return {"test": "data"}
"""
    
    # Prepare request with wrong auth token
    request_payload = {
        "code_str": code,
        "imports": [],
        "run_function_name": "run",
        "inputs_json": "{}",
    }
    
    url = f"{MODAL_WEB_ENDPOINT_URL}?workspace_id=test"
    response = requests.post(url, json=request_payload, timeout=30)
    
    assert response.status_code == 200  # Still returns 200
    result = response.json()
    assert not result.get("success")
    assert result.get("error_type") == "AuthenticationError"


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_very_large_return_value_web_endpoint():
    """Test returning very large values (>10MB) via web endpoint."""
    
    code = """
import numpy as np

def run(size_multiplier):
    # Generate a large array
    large_result = np.random.randn(2000000) * size_multiplier
    
    return {
        "large_array": large_result.tolist(),
        "metadata": {
            "size": len(large_result),
            "multiplier": size_multiplier,
            "mean": float(np.mean(large_result))
        }
    }
"""
    
    inputs = {"size_multiplier": 2.5}
    result = call_modal_web_endpoint(code, inputs)
    
    # Verify the structure
    assert "large_array" in result
    assert "metadata" in result
    assert len(result["large_array"]) == 2000000
    assert result["metadata"]["size"] == 2000000
    assert result["metadata"]["multiplier"] == 2.5
    assert isinstance(result["metadata"]["mean"], float)


@pytest.mark.skipif(SKIP_MODAL_WEB_TESTS, reason=SKIP_REASON)
def test_modal_web_endpoint_with_workflow_integration():
    """Test the web endpoint integration with actual workflow execution."""
    
    # Set environment to use modal_web mode
    original_mode = os.environ.get("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE")
    original_endpoint = os.environ.get("MODAL_WEB_ENDPOINT_URL")
    
    try:
        os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "modal_web"
        os.environ["MODAL_WEB_ENDPOINT_URL"] = MODAL_WEB_ENDPOINT_URL
        
        # Create a simple workflow
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceParameter", "name": "a"},
                {"type": "InferenceParameter", "name": "b"},
            ],
            "steps": [
                {
                    "type": "PythonBlock",
                    "name": "add_numbers",
                    "code": """
def run(self, a: float, b: float):
    return {"sum": a + b}
""",
                    "inputs": {
                        "a": "$inputs.a",
                        "b": "$inputs.b",
                    },
                }
            ],
            "outputs": [
                {"type": "JsonField", "name": "result", "selector": "$steps.add_numbers.sum"}
            ],
        }
        
        # Execute workflow
        engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters={},
            max_concurrent_steps=1,
        )
        
        result = engine.run(runtime_parameters={"a": 10, "b": 20})
        
        # Verify result
        assert result[0]["result"] == 30
        
    finally:
        # Restore original environment
        if original_mode is not None:
            os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = original_mode
        elif "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE" in os.environ:
            del os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"]
            
        if original_endpoint is not None:
            os.environ["MODAL_WEB_ENDPOINT_URL"] = original_endpoint
        elif "MODAL_WEB_ENDPOINT_URL" in os.environ:
            del os.environ["MODAL_WEB_ENDPOINT_URL"]
