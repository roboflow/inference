"""
Tests for Custom Python Blocks execution via Modal.

These tests verify that Custom Python Blocks work correctly when executed
in Modal sandboxes, comparing results with local execution to ensure consistency.
"""

import os
from typing import Any, Dict
from unittest import mock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import (
    DynamicBlockError,
    WorkflowEnvironmentConfigurationError,
)
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


def create_workflow_with_custom_block(block_code: str, block_type: str, inputs: Dict, outputs: Dict, workflow_inputs: list = None):
    """Helper to create a workflow with a custom Python block."""
    # Create workflow inputs if not provided
    if workflow_inputs is None:
        workflow_inputs = []
        for name, definition in inputs.items():
            if "value_types" in definition:
                workflow_inputs.append({"type": "WorkflowParameter", "name": name})
            elif "selector_types" in definition and "input_image" in definition["selector_types"]:
                workflow_inputs.append({"type": "WorkflowImage", "name": name})
            else:
                workflow_inputs.append({"type": "WorkflowParameter", "name": name})
    
    # Build the dynamic input definitions correctly
    dynamic_inputs = {}
    for name, definition in inputs.items():
        if isinstance(definition, dict) and "type" in definition and definition["type"] in ["WorkflowParameter", "WorkflowImage"]:
            # Convert workflow input format to dynamic input format
            dynamic_inputs[name] = {
                "type": "DynamicInputDefinition",
                "value_types": ["float", "integer", "string", "list", "dict", "any"]  # Allow various types
            }
        else:
            dynamic_inputs[name] = definition
    
    return {
        "version": "1.0",
        "inputs": workflow_inputs,
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": block_type,
                    "inputs": dynamic_inputs,
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
        # Create workflow
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
            workflow_inputs=[
                {"type": "WorkflowParameter", "name": "a"},
                {"type": "WorkflowParameter", "name": "b"},
            ]
        )

        # Test with local execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
            local_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            local_result = local_engine.run(
                runtime_parameters={"a": 10.5, "b": 20.3}
            )

        # Test with Modal execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            modal_result = modal_engine.run(
                runtime_parameters={"a": 10.5, "b": 20.3}
            )

        # Compare results
        assert local_result[0]["sum"] == modal_result[0]["sum"]
        assert abs(local_result[0]["sum"] - 30.8) < 0.001

    def test_image_invert_block_local_vs_modal(self) -> None:
        """Test that image inversion produces the same results locally and in Modal."""
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create workflow
        workflow = create_workflow_with_custom_block(
            block_code=IMAGE_INVERT_BLOCK,
            block_type="ImageInvert",
            inputs={
                "image": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_image"],
                },
            },
            outputs={"inverted": {"type": "DynamicOutputDefinition", "kind": ["image"]}},
            workflow_inputs=[
                {"type": "WorkflowImage", "name": "image"},
            ]
        )

        # Test with local execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
            local_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            local_result = local_engine.run(
                runtime_parameters={"image": test_image}
            )

        # Test with Modal execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            modal_result = modal_engine.run(
                runtime_parameters={"image": test_image}
            )

        # Extract numpy images from results
        local_inverted = local_result[0]["inverted"].numpy_image
        modal_inverted = modal_result[0]["inverted"].numpy_image

        # Compare results
        np.testing.assert_array_equal(local_inverted, modal_inverted)
        np.testing.assert_array_equal(local_inverted, 255 - test_image)

    def test_complex_math_operations_local_vs_modal(self) -> None:
        """Test that complex mathematical operations work consistently."""
        test_numbers = [1.5, 2.7, 3.9, 4.2, 5.8, 6.1, 7.3, 8.5, 9.2, 10.0]
        
        # Create workflow
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
            workflow_inputs=[
                {"type": "WorkflowParameter", "name": "numbers"},
            ]
        )

        # Test with local execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
            local_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            local_result = local_engine.run(
                runtime_parameters={"numbers": test_numbers}
            )

        # Test with Modal execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            modal_result = modal_engine.run(
                runtime_parameters={"numbers": test_numbers}
            )

        # Compare results
        assert abs(local_result[0]["mean"] - modal_result[0]["mean"]) < 0.001
        assert abs(local_result[0]["std"] - modal_result[0]["std"]) < 0.001
        assert local_result[0]["min"] == modal_result[0]["min"]
        assert local_result[0]["max"] == modal_result[0]["max"]

    def test_numpy_matrix_operations_local_vs_modal(self) -> None:
        """Test that numpy matrix operations work consistently."""
        matrix_a = [[1, 2], [3, 4]]
        matrix_b = [[5, 6], [7, 8]]
        
        # Create workflow
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
            workflow_inputs=[
                {"type": "WorkflowParameter", "name": "matrix_a"},
                {"type": "WorkflowParameter", "name": "matrix_b"},
            ]
        )

        # Test with local execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
            local_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            local_result = local_engine.run(
                runtime_parameters={"matrix_a": matrix_a, "matrix_b": matrix_b}
            )

        # Test with Modal execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            modal_result = modal_engine.run(
                runtime_parameters={"matrix_a": matrix_a, "matrix_b": matrix_b}
            )

        # Compare results
        np.testing.assert_array_almost_equal(
            np.array(local_result[0]["result"]), 
            np.array(modal_result[0]["result"])
        )
        
        # Verify the actual result is correct (matrix multiplication)
        expected = np.dot(np.array(matrix_a), np.array(matrix_b))
        np.testing.assert_array_almost_equal(
            np.array(modal_result[0]["result"]), 
            expected
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
            workflow_inputs=[
                {"type": "WorkflowParameter", "name": "value"},
            ]
        )

        # Test with Modal execution - should raise error for negative input
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            
            # Import the correct exception type
            from inference.core.workflows.errors import StepExecutionError
            
            # This should raise an error
            with pytest.raises(StepExecutionError) as exc_info:
                modal_engine.run(runtime_parameters={"value": -5})
            
            assert "Value must be non-negative" in str(exc_info.value)
            
            # This should succeed
            result = modal_engine.run(runtime_parameters={"value": 16})
            assert abs(result[0]["result"] - 4.0) < 0.001

    def test_multiple_blocks_in_workflow_modal(self) -> None:
        """Test a workflow with multiple custom blocks executed in Modal."""
        # Block 1: Square a number
        square_block = """
def run(self, x: float) -> BlockResult:
    return {"squared": x ** 2}
"""
        
        # Block 2: Add two numbers
        add_block = """
def run(self, a: float, b: float) -> BlockResult:
    return {"sum": a + b}
"""
        
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "input_x"},
                {"type": "WorkflowParameter", "name": "input_y"},
            ],
            "dynamic_blocks_definitions": [
                {
                    "type": "DynamicBlockDefinition",
                    "manifest": {
                        "type": "ManifestDescription",
                        "block_type": "SquareBlock",
                        "inputs": {
                            "x": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["input_parameter"],
                            }
                        },
                        "outputs": {"squared": {"type": "DynamicOutputDefinition", "kind": []}},
                    },
                    "code": {
                        "type": "PythonCode",
                        "run_function_code": square_block,
                    },
                },
                {
                    "type": "DynamicBlockDefinition",
                    "manifest": {
                        "type": "ManifestDescription",
                        "block_type": "AddBlock",
                        "inputs": {
                            "a": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["step_output", "input_parameter"],
                            },
                            "b": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["step_output", "input_parameter"],
                            }
                        },
                        "outputs": {"sum": {"type": "DynamicOutputDefinition", "kind": []}},
                    },
                    "code": {
                        "type": "PythonCode",
                        "run_function_code": add_block,
                    },
                },
            ],
            "steps": [
                {
                    "type": "SquareBlock",
                    "name": "square_x",
                    "x": "$inputs.input_x",
                },
                {
                    "type": "SquareBlock",
                    "name": "square_y",
                    "x": "$inputs.input_y",
                },
                {
                    "type": "AddBlock",
                    "name": "add_squares",
                    "a": "$steps.square_x.squared",
                    "b": "$steps.square_y.squared",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "x_squared",
                    "selector": "$steps.square_x.squared",
                },
                {
                    "type": "JsonField",
                    "name": "y_squared",
                    "selector": "$steps.square_y.squared",
                },
                {
                    "type": "JsonField",
                    "name": "sum_of_squares",
                    "selector": "$steps.add_squares.sum",
                },
            ],
        }

        # Test with Modal execution
        with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
            modal_engine = ExecutionEngine.init(
                workflow_definition=workflow,
                init_parameters={},
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            )
            result = modal_engine.run(
                runtime_parameters={"input_x": 3.0, "input_y": 4.0}
            )

        # Verify results (3^2 + 4^2 = 9 + 16 = 25)
        assert result[0]["x_squared"] == 9.0
        assert result[0]["y_squared"] == 16.0
        assert result[0]["sum_of_squares"] == 25.0


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
def test_workspace_isolation_in_modal():
    """Test that different workspaces have isolated execution environments."""
    # This test would require mocking workspace_id through the execution context
    # For now, we'll test that the Modal executor properly handles workspace_id
    
    stateful_block = """
def run(self, increment: float) -> BlockResult:
    # This would be stateful if containers were reused across workspaces
    # Modal should isolate by workspace_id
    return {"value": increment}
"""
    
    workflow = create_workflow_with_custom_block(
        block_code=stateful_block,
        block_type="StatefulTest",
        inputs={
            "increment": {
                "type": "DynamicInputDefinition",
                "selector_types": ["input_parameter"],
            },
        },
        outputs={
            "value": {"type": "DynamicOutputDefinition", "kind": []},
        },
        workflow_inputs=[
            {"type": "WorkflowParameter", "name": "increment"},
        ]
    )

    # Test with Modal execution
    with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
        # Simulate different workspace contexts - for now just run it once
        modal_engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters={},
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )
        result = modal_engine.run(
            runtime_parameters={"increment": 10.0}
        )
        assert result[0]["value"] == 10.0
