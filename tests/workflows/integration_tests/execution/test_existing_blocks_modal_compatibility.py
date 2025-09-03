"""
Test existing dynamic Python blocks with Modal execution.

This test file ensures that dynamic blocks work correctly when executed via Modal.
"""

import os
import time
from unittest import mock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine


# Skip all tests if Modal credentials are not present
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
SKIP_MODAL_TESTS = not (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
SKIP_REASON = "Modal credentials not present (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
class TestExistingBlocksWithModal:
    """Test that existing dynamic blocks work correctly with Modal execution."""

    def test_simple_computation_modal_vs_local(self) -> None:
        """Test a simple computational block with both Modal and local execution."""
        
        # Simple block that processes a list of numbers
        compute_block = """
def run(self, numbers: list) -> BlockResult:
    if not numbers:
        return {"sum": 0, "product": 1, "count": 0}
    
    total_sum = sum(numbers)
    product = 1
    for n in numbers:
        product *= n
    
    return {
        "sum": total_sum,
        "product": product,
        "count": len(numbers)
    }
"""
        
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "numbers"},
            ],
            "dynamic_blocks_definitions": [
                {
                    "type": "DynamicBlockDefinition",
                    "manifest": {
                        "type": "ManifestDescription",
                        "block_type": "ComputeBlock",
                        "inputs": {
                            "numbers": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["input_parameter"],
                            },
                        },
                        "outputs": {
                            "sum": {"type": "DynamicOutputDefinition", "kind": []},
                            "product": {"type": "DynamicOutputDefinition", "kind": []},
                            "count": {"type": "DynamicOutputDefinition", "kind": []},
                        },
                    },
                    "code": {
                        "type": "PythonCode",
                        "run_function_code": compute_block,
                    },
                },
            ],
            "steps": [
                {
                    "type": "ComputeBlock",
                    "name": "compute",
                    "numbers": "$inputs.numbers",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "sum",
                    "selector": "$steps.compute.sum",
                },
                {
                    "type": "JsonField",
                    "name": "product",
                    "selector": "$steps.compute.product",
                },
                {
                    "type": "JsonField",
                    "name": "count",
                    "selector": "$steps.compute.count",
                },
            ],
        }
        
        test_numbers = [2, 3, 4, 5]
        
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
        assert local_result[0]["sum"] == modal_result[0]["sum"] == 14
        assert local_result[0]["product"] == modal_result[0]["product"] == 120
        assert local_result[0]["count"] == modal_result[0]["count"] == 4

    def test_string_manipulation_modal_vs_local(self) -> None:
        """Test string manipulation operations in Modal vs local."""
        
        string_block = """
def run(self, text: str, operation: str) -> BlockResult:
    if operation == "reverse":
        result = text[::-1]
    elif operation == "uppercase":
        result = text.upper()
    elif operation == "lowercase":
        result = text.lower()
    elif operation == "title":
        result = text.title()
    else:
        result = text
    
    return {
        "result": result,
        "length": len(result),
        "words": len(result.split())
    }
"""
        
        workflow = {
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowParameter", "name": "text"},
                {"type": "WorkflowParameter", "name": "operation"},
            ],
            "dynamic_blocks_definitions": [
                {
                    "type": "DynamicBlockDefinition",
                    "manifest": {
                        "type": "ManifestDescription",
                        "block_type": "StringManipulation",
                        "inputs": {
                            "text": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["input_parameter"],
                            },
                            "operation": {
                                "type": "DynamicInputDefinition",
                                "selector_types": ["input_parameter"],
                            },
                        },
                        "outputs": {
                            "result": {"type": "DynamicOutputDefinition", "kind": []},
                            "length": {"type": "DynamicOutputDefinition", "kind": []},
                            "words": {"type": "DynamicOutputDefinition", "kind": []},
                        },
                    },
                    "code": {
                        "type": "PythonCode",
                        "run_function_code": string_block,
                    },
                },
            ],
            "steps": [
                {
                    "type": "StringManipulation",
                    "name": "manipulate",
                    "text": "$inputs.text",
                    "operation": "$inputs.operation",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "result",
                    "selector": "$steps.manipulate.result",
                },
                {
                    "type": "JsonField",
                    "name": "length",
                    "selector": "$steps.manipulate.length",
                },
                {
                    "type": "JsonField",
                    "name": "words",
                    "selector": "$steps.manipulate.words",
                },
            ],
        }
        
        test_text = "Hello World from Modal"
        
        # Test different operations
        for operation in ["reverse", "uppercase", "lowercase", "title"]:
            # Test with local execution
            with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
                local_engine = ExecutionEngine.init(
                    workflow_definition=workflow,
                    init_parameters={},
                    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
                )
                local_result = local_engine.run(
                    runtime_parameters={"text": test_text, "operation": operation}
                )

            # Test with Modal execution
            with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
                modal_engine = ExecutionEngine.init(
                    workflow_definition=workflow,
                    init_parameters={},
                    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
                )
                modal_result = modal_engine.run(
                    runtime_parameters={"text": test_text, "operation": operation}
                )

            # Compare results
            assert local_result[0]["result"] == modal_result[0]["result"]
            assert local_result[0]["length"] == modal_result[0]["length"]
            assert local_result[0]["words"] == modal_result[0]["words"]


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
def test_performance_comparison_local_vs_modal() -> None:
    """Compare performance characteristics between local and Modal execution."""
    
    # Simple computational block for benchmarking
    compute_block = """
def run(self, iterations: int) -> BlockResult:
    import time
    start = time.time()
    
    # Perform some computation
    result = 0
    for i in range(iterations):
        result += i ** 2
    
    elapsed = time.time() - start
    return {"result": result, "elapsed": elapsed}
"""
    
    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "iterations"},
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "ComputeBlock",
                    "inputs": {
                        "iterations": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        },
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []},
                        "elapsed": {"type": "DynamicOutputDefinition", "kind": []},
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": compute_block,
                },
            },
        ],
        "steps": [
            {
                "type": "ComputeBlock",
                "name": "compute",
                "iterations": "$inputs.iterations",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.compute.result",
            },
            {
                "type": "JsonField",
                "name": "elapsed",
                "selector": "$steps.compute.elapsed",
            },
        ],
    }
    
    iterations = 100000
    
    # Measure local execution time
    with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}):
        local_engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters={},
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )
        
        local_start = time.time()
        local_result = local_engine.run(
            runtime_parameters={"iterations": iterations}
        )
        local_total = time.time() - local_start
    
    # Measure Modal execution time
    with mock.patch.dict(os.environ, {"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal"}):
        modal_engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters={},
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )
        
        modal_start = time.time()
        modal_result = modal_engine.run(
            runtime_parameters={"iterations": iterations}
        )
        modal_total = time.time() - modal_start
    
    # Verify results match
    assert local_result[0]["result"] == modal_result[0]["result"]
    
    # Log performance metrics (for informational purposes)
    print(f"\nPerformance Comparison:")
    print(f"Local execution: {local_total:.3f}s total, {local_result[0]['elapsed']:.3f}s compute")
    print(f"Modal execution: {modal_total:.3f}s total, {modal_result[0]['elapsed']:.3f}s compute")
    print(f"Modal overhead: {modal_total - modal_result[0]['elapsed']:.3f}s")
    
    # Modal will have overhead due to network and containerization
    # But computation results should be identical
    assert local_result[0]["result"] == modal_result[0]["result"]
