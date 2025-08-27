"""
Quick test of Modal executor
"""
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import ModalExecutor

print("Testing Modal executor...")

python_code = PythonCode(
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

print("Creating executor...")
executor = ModalExecutor(workspace_id="test-workspace")

print("Executing remote call...")
inputs = {"x": 5, "y": 3}
result = executor.execute_remote(
    block_type_name="simple_computation",
    python_code=python_code,
    inputs=inputs,
    workspace_id="test-workspace"
)

print(f"Result: {result}")
