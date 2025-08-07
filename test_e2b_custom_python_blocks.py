#!/usr/bin/env python3
"""
Test script for E2B Custom Python Blocks integration.
This tests both local and remote execution modes.
"""

import os
import json
import sys

# Add inference to path
sys.path.insert(0, '/Users/yeldarb/Code/inference')

# Set environment variables before importing
os.environ['WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE'] = 'local'
os.environ['ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS'] = 'True'

# Simple test workflow with Custom Python Block
test_workflow = {
    "version": "1.0",
    "inputs": [],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "TestBlock",
                "inputs": {
                    "test_input": {
                        "type": "DynamicInputDefinition",
                        "value_types": ["string"]
                    }
                },
                "outputs": {
                    "result": {
                        "type": "DynamicOutputDefinition",
                        "kind": []
                    }
                }
            },
            "code": {
                "type": "PythonCode",
                "run_function_code": """
def run(self, test_input: str):
    import time
    current_time = time.time()
    result = f"Processed '{test_input}' at {current_time}"
    return {"result": result}
"""
            }
        }
    ],
    "steps": [
        {
            "type": "TestBlock",
            "name": "test_step",
            "test_input": "Hello from E2B!"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "test_result",
            "selector": "$steps.test_step.result"
        }
    ]
}

def test_local_execution():
    """Test Custom Python Block with local execution."""
    print("Testing LOCAL execution mode...")
    os.environ['WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE'] = 'local'
    
    try:
        from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
        from inference.core.workflows.execution_engine.v1.executor.core import execute_workflow
        
        # Compile the workflow
        execution_graph = compile_workflow(workflow_definition=test_workflow)
        
        # Execute the workflow
        result = execute_workflow(
            execution_graph=execution_graph,
            runtime_parameters={}
        )
        
        print(f"Local execution result: {result}")
        assert "test_result" in result
        assert "Hello from E2B!" in result["test_result"]
        print("✅ Local execution test PASSED\n")
        return True
    except ImportError as e:
        print(f"⚠️  Cannot run test - missing dependencies: {e}")
        print("Please install requirements with: pip install -r requirements/_requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Local execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_remote_execution():
    """Test Custom Python Block with remote E2B execution."""
    print("Testing REMOTE (E2B) execution mode...")
    
    # Check if E2B API key is set
    if not os.environ.get('E2B_API_KEY'):
        print("⚠️  E2B_API_KEY not set - skipping remote execution test")
        print("To test remote execution, set E2B_API_KEY environment variable")
        return True
    
    os.environ['WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE'] = 'remote'
    
    try:
        from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
        from inference.core.workflows.execution_engine.v1.executor.core import execute_workflow
        
        # Compile the workflow
        execution_graph = compile_workflow(workflow_definition=test_workflow)
        
        # Execute the workflow
        result = execute_workflow(
            execution_graph=execution_graph,
            runtime_parameters={}
        )
        
        print(f"Remote execution result: {result}")
        assert "test_result" in result
        assert "Hello from E2B!" in result["test_result"]
        print("✅ Remote execution test PASSED\n")
        return True
    except ImportError as e:
        print(f"⚠️  Cannot run test - missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Remote execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_e2b_configuration():
    """Check E2B configuration and environment."""
    print("Checking E2B Configuration...")
    print("-" * 40)
    
    # Check environment variables
    from inference.core.env import (
        WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
        E2B_API_KEY,
        E2B_TEMPLATE_ID,
        E2B_SANDBOX_TIMEOUT,
        E2B_SANDBOX_IDLE_TIMEOUT,
    )
    
    print(f"Execution Mode: {WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE}")
    print(f"E2B API Key: {'Set' if E2B_API_KEY else 'Not set'}")
    print(f"E2B Template ID: {E2B_TEMPLATE_ID or 'Auto-detect from version'}")
    print(f"Sandbox Timeout: {E2B_SANDBOX_TIMEOUT} seconds")
    print(f"Sandbox Idle Timeout: {E2B_SANDBOX_IDLE_TIMEOUT} seconds")
    
    # Check if E2B package is installed
    try:
        import e2b_code_interpreter
        print(f"E2B Code Interpreter: Installed (version {e2b_code_interpreter.__version__ if hasattr(e2b_code_interpreter, '__version__') else 'unknown'})")
    except ImportError:
        print("E2B Code Interpreter: Not installed")
        print("Install with: pip install -r requirements/requirements.e2b.txt")
    
    print("-" * 40)
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("E2B Custom Python Blocks Integration Test")
    print("=" * 60)
    print()
    
    # Check configuration
    check_e2b_configuration()
    
    # Run tests
    tests_passed = []
    
    # Test local execution
    tests_passed.append(test_local_execution())
    
    # Test remote execution (if E2B API key is available)
    tests_passed.append(test_remote_execution())
    
    print("=" * 60)
    if all(tests_passed):
        print("All tests completed successfully! ✅")
    else:
        print("Some tests were skipped or failed ⚠️")
    print("=" * 60)
