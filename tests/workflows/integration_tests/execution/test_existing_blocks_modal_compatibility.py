"""
Test existing dynamic Python blocks with Modal execution.

This test file ensures that dynamic blocks work correctly when executed via Modal.

To run these tests:
1. Set Modal credentials in environment:
   export MODAL_TOKEN_ID="your_token_id"
   export MODAL_TOKEN_SECRET="your_token_secret"
2. Run: pytest tests/workflows/integration_tests/execution/test_existing_blocks_modal_compatibility.py
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS

# Skip all tests if Modal credentials are not present
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
SKIP_MODAL_TESTS = not (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
SKIP_REASON = "Modal credentials not present (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"


def run_workflow_in_subprocess(
    workflow: dict, runtime_params: dict, execution_mode: str
) -> any:
    """Run a workflow in a subprocess with specific execution mode."""
    import json
    import tempfile

    # Create a temporary Python script that runs the workflow
    script = f"""
import os
import sys
import json

# Set execution mode BEFORE imports
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "{execution_mode}"

# Now import
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

workflow = {json.dumps(workflow)}
runtime_params = {json.dumps(runtime_params)}

engine = ExecutionEngine.init(
    workflow_definition=workflow,
    init_parameters={{}},
    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
)

result = engine.run(runtime_parameters=runtime_params)

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
            return json.loads(json_str)
        else:
            raise RuntimeError(f"Could not parse result from output: {output}")

    finally:
        os.unlink(temp_file)


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
        runtime_params = {"numbers": test_numbers}

        # Test with local execution
        local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

        # Test with Modal execution
        modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

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
            runtime_params = {"text": test_text, "operation": operation}

            # Test with local execution
            local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

            # Test with Modal execution
            modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

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
    runtime_params = {"iterations": iterations}

    # Measure local execution time
    local_start = time.time()
    local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")
    local_total = time.time() - local_start

    # Measure Modal execution time
    modal_start = time.time()
    modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")
    modal_total = time.time() - modal_start

    # Verify results match
    assert local_result[0]["result"] == modal_result[0]["result"]

    # Log performance metrics (for informational purposes)
    print(f"\nPerformance Comparison:")
    print(
        f"Local execution: {local_total:.3f}s total, {local_result[0]['elapsed']:.3f}s compute"
    )
    print(
        f"Modal execution: {modal_total:.3f}s total, {modal_result[0]['elapsed']:.3f}s compute"
    )
    print(f"Modal overhead: {modal_total - modal_result[0]['elapsed']:.3f}s")

    # Modal will have overhead due to network and containerization
    # But computation results should be identical
    assert local_result[0]["result"] == modal_result[0]["result"]


@pytest.mark.skipif(SKIP_MODAL_TESTS, reason=SKIP_REASON)
def test_actual_modal_execution() -> None:
    """Verify that Modal is actually being executed remotely."""

    verification_block = """
def run(self, x: int) -> BlockResult:
    import os
    import socket
    
    # These will be different in Modal vs local
    hostname = socket.gethostname()
    task_id = os.environ.get('MODAL_TASK_ID', 'LOCAL')
    
    return {
        "result": x * 2,
        "hostname": hostname,
        "task_id": task_id
    }
"""

    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "x"},
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "VerifyModal",
                    "inputs": {
                        "x": {
                            "type": "DynamicInputDefinition",
                            "selector_types": ["input_parameter"],
                        },
                    },
                    "outputs": {
                        "result": {"type": "DynamicOutputDefinition", "kind": []},
                        "hostname": {"type": "DynamicOutputDefinition", "kind": []},
                        "task_id": {"type": "DynamicOutputDefinition", "kind": []},
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": verification_block,
                },
            },
        ],
        "steps": [
            {
                "type": "VerifyModal",
                "name": "verify",
                "x": "$inputs.x",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.verify.result",
            },
            {
                "type": "JsonField",
                "name": "hostname",
                "selector": "$steps.verify.hostname",
            },
            {
                "type": "JsonField",
                "name": "task_id",
                "selector": "$steps.verify.task_id",
            },
        ],
    }

    runtime_params = {"x": 5}

    # Run locally
    local_result = run_workflow_in_subprocess(workflow, runtime_params, "local")

    # Run in Modal
    modal_result = run_workflow_in_subprocess(workflow, runtime_params, "modal")

    # Results should match
    assert local_result[0]["result"] == modal_result[0]["result"] == 10

    # But execution contexts should be different
    assert local_result[0]["task_id"] == "LOCAL"
    assert modal_result[0]["task_id"] != "LOCAL"
    assert modal_result[0]["task_id"].startswith("ta-")  # Modal task IDs start with ta-

    # Hostnames should be different
    import socket

    local_hostname = socket.gethostname()
    assert modal_result[0]["hostname"] != local_hostname

    print(f"\n✅ Modal execution confirmed:")
    print(f"  Local hostname: {local_hostname}")
    print(f"  Modal hostname: {modal_result[0]['hostname']}")
    print(f"  Modal task ID: {modal_result[0]['task_id']}")
