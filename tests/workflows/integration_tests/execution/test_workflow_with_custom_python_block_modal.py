"""
Tests for Custom Python Blocks execution via Modal.

These tests verify that Custom Python Blocks work correctly when executed
in Modal sandboxes, comparing results with local execution to ensure consistency.

To run these tests:
1. Set Modal credentials in environment:
   export MODAL_TOKEN_ID="your_token_id"
   export MODAL_TOKEN_SECRET="your_token_secret"
2. Run: pytest tests/workflows/integration_tests/execution/test_workflow_with_custom_python_block_modal.py
"""

import base64
import os
import subprocess
import sys
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# Skip all tests if Modal credentials are not present
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
SKIP_MODAL_TESTS = not (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
SKIP_REASON = "Modal credentials not present (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"


# Simple addition block for basic testing
SIMPLE_ADDITION_BLOCK = """
def run(self, a: float, b: float) -> BlockResult:
    result = a + b
    return {"sum": result}
"""

# Image inversion block
IMAGE_INVERT_BLOCK = """
def run(self, image: WorkflowImageData) -> BlockResult:
    # Convert WorkflowImageData to numpy array
    np_img = image.numpy_image
    # Invert image
    inverted_np_img = 255 - np_img
    # Convert back to WorkflowImageData
    inverted_image = WorkflowImageData.copy_and_replace(
        origin_image_data=image, 
        numpy_image=inverted_np_img
    )
    return {"inverted": inverted_image}
"""

# Complex calculation block with multiple operations
COMPLEX_MATH_BLOCK = """
def run(self, numbers: List[float]) -> BlockResult:
    if not numbers:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = variance ** 0.5
    
    return {
        "mean": mean,
        "std": std,
        "min": min(numbers),
        "max": max(numbers)
    }
"""

# Block that uses external libraries
NUMPY_OPERATIONS_BLOCK = """
def run(self, matrix_a: List[List[float]], matrix_b: List[List[float]]) -> BlockResult:
    # Convert to numpy arrays
    a = np.array(matrix_a)
    b = np.array(matrix_b)
    
    # Perform matrix multiplication
    result = np.dot(a, b)
    
    # Convert back to list for serialization
    return {"result": result.tolist()}
"""


def run_workflow_in_subprocess(
    workflow: dict, runtime_params: dict, execution_mode: str
) -> Any:
    """Run a workflow in a subprocess with specific execution mode."""
    import json
    import tempfile

    # Create a temporary Python script that runs the workflow
    script = f"""
import os
import sys
import json
import base64
import numpy as np

# Set execution mode BEFORE imports
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "{execution_mode}"
if "{execution_mode}" == "modal":
    # Modal credentials should already be in environment
    pass

# Now import
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

workflow = {json.dumps(workflow)}
runtime_params = {json.dumps(runtime_params)}

# Handle image data if present
if "image_data" in runtime_params:
    # Decode base64 image back to numpy array
    import cv2
    image_bytes = base64.b64decode(runtime_params["image_data"])
    nparr = np.frombuffer(image_bytes, np.uint8)
    numpy_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    runtime_params["image"] = numpy_image
    del runtime_params["image_data"]

engine = ExecutionEngine.init(
    workflow_definition=workflow,
    init_parameters={{}},
    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
)

result = engine.run(runtime_parameters=runtime_params)

# Convert any image results to base64 for serialization
for i, res in enumerate(result):
    for key, value in list(res.items()):
        if hasattr(value, 'numpy_image'):
            # Convert WorkflowImageData to base64 for serialization
            np_img = value.numpy_image
            _, buffer = cv2.imencode('.png', np_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result[i][key] = {{'type': 'image', 'data': img_base64}}

# Output as JSON for parsing
import json
print("RESULT_START")
print(json.dumps(result))
print("RESULT_END")
"""

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        temp_file = f.name

    try:
        # Run the script in a subprocess with clean environment
        env = os.environ.copy()
        if execution_mode == "modal":
            # Ensure Modal credentials are passed
            if not MODAL_TOKEN_ID or not MODAL_TOKEN_SECRET:
                raise ValueError("Modal credentials not found in environment")
            env["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
            env["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET

        # Ensure the subprocess can import the local 'inference' package on CI runners
        try:
            import pathlib

            repo_root = str(pathlib.Path(__file__).resolve().parents[4])
            existing_pythonpath = env.get("PYTHONPATH", "")
            if repo_root not in existing_pythonpath.split(
                ":" if os.name != "nt" else ";"
            ):
                sep = ":" if os.name != "nt" else ";"
                env["PYTHONPATH"] = (
                    (repo_root + sep + existing_pythonpath)
                    if existing_pythonpath
                    else repo_root
                )
        except Exception:
            pass

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr}")

        # Extract JSON result from output
        output = result.stdout
        if "RESULT_START" in output and "RESULT_END" in output:
            json_str = output.split("RESULT_START")[1].split("RESULT_END")[0].strip()
            parsed_result = json.loads(json_str)

            # Convert base64 images back to numpy arrays
            for i, res in enumerate(parsed_result):
                for key, value in list(res.items()):
                    if isinstance(value, dict) and value.get("type") == "image":
                        # Decode base64 back to numpy array
                        img_bytes = base64.b64decode(value["data"])
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        parsed_result[i][key] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return parsed_result
        else:
            raise RuntimeError(f"Could not parse result from output: {output}")

    finally:
        os.unlink(temp_file)


def create_workflow_with_custom_block(
    block_code: str, block_type: str, inputs: Dict, outputs: Dict
) -> dict:
    """Helper to create a workflow with a custom Python block."""
    workflow_inputs = []
    for name, definition in inputs.items():
        if (
            "selector_types" in definition
            and "input_image" in definition["selector_types"]
        ):
            workflow_inputs.append({"type": "InferenceImage", "name": name})
        else:
            workflow_inputs.append({"type": "WorkflowParameter", "name": name})

    return {
        "version": "1.0",
        "inputs": workflow_inputs,
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": block_type,
                    "inputs": inputs,
                    "outputs": outputs,
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": block_code,
                },
            }
        ],
        "steps": [
            {
                "type": block_type,
                "name": "custom_block",
                **{k: f"$inputs.{k}" for k in inputs.keys()},
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": output_name,
                "selector": f"$steps.custom_block.{output_name}",
            }
            for output_name in outputs.keys()
        ],
    }


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
class TestModalCustomPythonBlocks:
    """Test suite for Modal execution of Custom Python Blocks."""

    def test_simple_addition_block_local_vs_modal(self) -> None:
        """Test that a simple addition block produces the same results locally and in Modal."""
        workflow = create_workflow_with_custom_block(
            block_code=SIMPLE_ADDITION_BLOCK,
            block_type="SimpleAddition",
            inputs={
                "a": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
                "b": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            outputs={"sum": {"type": "DynamicOutputDefinition", "kind": []}},
        )

        runtime_params = {"a": 10.5, "b": 20.3}

        # Run in local mode
        local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

        # Run in Modal mode
        modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

        # Compare results
        assert local_result[0]["sum"] == modal_result[0]["sum"]
        assert abs(local_result[0]["sum"] - 30.8) < 0.001

    def test_image_invert_block_local_vs_modal(self) -> None:
        """Test that image inversion produces the same results locally and in Modal."""
        workflow = create_workflow_with_custom_block(
            block_code=IMAGE_INVERT_BLOCK,
            block_type="ImageInvert",
            inputs={
                "image": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_image"],
                },
            },
            outputs={
                "inverted": {"type": "DynamicOutputDefinition", "kind": ["image"]}
            },
        )

        # Create test image - solid color for easy verification
        test_image = np.ones((10, 10, 3), dtype=np.uint8) * 128

        # Encode image to base64 for subprocess transfer
        _, buffer = cv2.imencode(".png", test_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        runtime_params = {"image_data": image_base64}

        # Run in local mode
        local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

        # Run in Modal mode
        modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

        # Verify both returned inverted images
        local_inverted = local_result[0]["inverted"]
        modal_inverted = modal_result[0]["inverted"]

        # Expected inverted value: 255 - 128 = 127
        expected = np.ones((10, 10, 3), dtype=np.uint8) * 127

        # Compare results
        np.testing.assert_array_equal(
            local_inverted,
            expected,
            "Local execution should invert the image correctly",
        )
        np.testing.assert_array_equal(
            modal_inverted,
            expected,
            "Modal execution should invert the image correctly",
        )
        np.testing.assert_array_equal(
            local_inverted,
            modal_inverted,
            "Local and Modal should produce identical results",
        )

    def test_complex_math_operations_local_vs_modal(self) -> None:
        """Test that complex mathematical operations work consistently."""
        test_numbers = [1.5, 2.7, 3.9, 4.2, 5.8, 6.1, 7.3, 8.5, 9.2, 10.0]

        workflow = create_workflow_with_custom_block(
            block_code=COMPLEX_MATH_BLOCK,
            block_type="ComplexMath",
            inputs={
                "numbers": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            outputs={
                "mean": {"type": "DynamicOutputDefinition", "kind": []},
                "std": {"type": "DynamicOutputDefinition", "kind": []},
                "min": {"type": "DynamicOutputDefinition", "kind": []},
                "max": {"type": "DynamicOutputDefinition", "kind": []},
            },
        )

        runtime_params = {"numbers": test_numbers}

        # Run in local mode
        local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

        # Run in Modal mode
        modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

        # Compare results
        assert abs(local_result[0]["mean"] - modal_result[0]["mean"]) < 0.001
        assert abs(local_result[0]["std"] - modal_result[0]["std"]) < 0.001
        assert local_result[0]["min"] == modal_result[0]["min"]
        assert local_result[0]["max"] == modal_result[0]["max"]

    def test_numpy_matrix_operations_local_vs_modal(self) -> None:
        """Test that numpy matrix operations work consistently."""
        matrix_a = [[1, 2], [3, 4]]
        matrix_b = [[5, 6], [7, 8]]

        workflow = create_workflow_with_custom_block(
            block_code=NUMPY_OPERATIONS_BLOCK,
            block_type="NumpyOperations",
            inputs={
                "matrix_a": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
                "matrix_b": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            outputs={
                "result": {"type": "DynamicOutputDefinition", "kind": []},
            },
        )

        runtime_params = {"matrix_a": matrix_a, "matrix_b": matrix_b}

        # Run in local mode
        local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

        # Run in Modal mode
        modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

        # Compare results
        np.testing.assert_array_almost_equal(
            np.array(local_result[0]["result"]), np.array(modal_result[0]["result"])
        )

        # Verify the actual result is correct (matrix multiplication)
        expected = np.dot(np.array(matrix_a), np.array(matrix_b))
        np.testing.assert_array_almost_equal(
            np.array(modal_result[0]["result"]), expected
        )

    def test_error_handling_in_modal(self) -> None:
        """Test that errors in Modal execution are properly propagated."""
        error_block = """
def run(self, value: float) -> BlockResult:
    if value < 0:
        raise ValueError("Value must be non-negative")
    return {"result": value ** 0.5}
"""

        workflow = create_workflow_with_custom_block(
            block_code=error_block,
            block_type="ErrorTest",
            inputs={
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            outputs={
                "result": {"type": "DynamicOutputDefinition", "kind": []},
            },
        )

        # Test successful execution
        runtime_params_success = {"value": 16}
        modal_result = run_workflow_in_subprocess(
            workflow, runtime_params_success, "modal"
        )
        assert abs(modal_result[0]["result"] - 4.0) < 0.001

        # Test error case - this will raise in subprocess
        runtime_params_error = {"value": -5}
        with pytest.raises(RuntimeError) as exc_info:
            run_workflow_in_subprocess(workflow, runtime_params_error, "modal")
        assert "Value must be non-negative" in str(exc_info.value)


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
def test_modal_execution_verification():
    """Verify that Modal execution is actually happening."""
    verification_block = """
def run(self, value: float) -> BlockResult:
    import os
    # This env var is only set in Modal containers
    task_id = os.environ.get('MODAL_TASK_ID', 'LOCAL')
    return {"result": value * 2, "execution_context": task_id}
"""

    workflow = create_workflow_with_custom_block(
        block_code=verification_block,
        block_type="VerificationTest",
        inputs={
            "value": {
                "type": "DynamicInputDefinition",
                "selector_types": ["input_parameter"],
            },
        },
        outputs={
            "result": {"type": "DynamicOutputDefinition", "kind": []},
            "execution_context": {"type": "DynamicOutputDefinition", "kind": []},
        },
    )

    runtime_params = {"value": 5}

    # Run in local mode
    local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")
    assert local_result[0]["execution_context"] == "LOCAL"

    # Run in Modal mode
    modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")
    assert modal_result[0]["execution_context"] != "LOCAL"
    assert modal_result[0]["execution_context"].startswith(
        "ta-"
    )  # Modal task IDs start with ta-

    # Results should be the same
    assert local_result[0]["result"] == modal_result[0]["result"] == 10
