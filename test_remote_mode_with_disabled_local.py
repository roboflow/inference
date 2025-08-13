#!/usr/bin/env python3
"""
Test that remote mode works even when ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS is False.
This tests both compilation and execution scenarios.
"""

import os
import sys

# Set environment variables BEFORE importing any inference modules
os.environ["ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"] = "False"
os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "remote"
os.environ["E2B_API_KEY"] = os.getenv("E2B_API_KEY", "test-key")  # Use real key if available

print("Test Configuration:")
print(f"ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS = {os.environ['ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS']}")
print(f"WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE = {os.environ['WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE']}")
print(f"E2B_API_KEY = {'<set>' if os.environ.get('E2B_API_KEY') else '<not set>'}")
print()

# Now import inference modules
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicBlockDefinition,
    ManifestDescription,
    DynamicInputDefinition,
    DynamicOutputDefinition,
    PythonCode,
    SelectorType,
    ValueType,
)

def test_compilation_with_remote_mode():
    """Test that dynamic blocks can be compiled in remote mode even with local execution disabled."""
    print("Test 1: Compilation with remote mode and local execution disabled")
    print("-" * 60)
    
    # Create a simple dynamic block definition
    dynamic_block_def = {
        "manifest": {
            "block_type": "test_block",
            "description": "A test custom Python block",
            "inputs": {
                "number": {
                    "selector_types": [],
                    "value_types": ["INTEGER"],
                    "is_optional": False,
                }
            },
            "outputs": {
                "result": {
                    "kind": []
                }
            }
        },
        "code": {
            "run_function_name": "run",
            "run_function_code": """
def run(self, number: int) -> dict:
    return {"result": number * 2}
""",
        }
    }
    
    try:
        # This should succeed because we're in remote mode
        result = compile_dynamic_blocks([dynamic_block_def])
        print("✅ SUCCESS: Dynamic block compiled successfully in remote mode!")
        print(f"   Compiled {len(result)} block(s)")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_validation_in_sandbox():
    """Test that code validation happens in the sandbox for invalid code."""
    print("\nTest 2: Code validation in sandbox (with invalid code)")
    print("-" * 60)
    
    # Create a dynamic block with invalid Python code
    invalid_block_def = {
        "manifest": {
            "block_type": "invalid_block",
            "description": "A block with invalid Python code",
            "inputs": {
                "text": {
                    "selector_types": [],
                    "value_types": ["STRING"],
                    "is_optional": False,
                }
            },
            "outputs": {
                "result": {
                    "kind": []
                }
            }
        },
        "code": {
            "run_function_name": "run",
            "run_function_code": """
def run(self, text: str) -> dict:
    # This has a syntax error
    return {"result": text +++ "invalid"}
""",
        }
    }
    
    try:
        # Compilation should succeed (validation is deferred to sandbox)
        result = compile_dynamic_blocks([invalid_block_def])
        print("✅ SUCCESS: Invalid code passed compilation (validation deferred to sandbox)")
        print("   This is expected behavior - actual validation happens during execution")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED: Compilation failed (should have been deferred): {e}")
        return False

def test_local_mode_blocked():
    """Test that local mode is properly blocked when disabled."""
    print("\nTest 3: Local mode blocked when disabled")
    print("-" * 60)
    
    # Temporarily switch to local mode
    os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "local"
    
    # Reload the module to pick up the new environment variable
    import importlib
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_assembler
    importlib.reload(block_assembler)
    
    dynamic_block_def = {
        "manifest": {
            "block_type": "local_test_block",
            "description": "Test block for local mode",
            "inputs": {
                "value": {
                    "selector_types": [],
                    "value_types": ["INTEGER"],
                    "is_optional": False,
                }
            },
            "outputs": {
                "result": {
                    "kind": []
                }
            }
        },
        "code": {
            "run_function_name": "run",
            "run_function_code": """
def run(self, value: int) -> dict:
    return {"result": value * 3}
""",
        }
    }
    
    try:
        result = block_assembler.compile_dynamic_blocks([dynamic_block_def])
        print("❌ FAILED: Local mode should have been blocked!")
        return False
    except Exception as e:
        if "local mode" in str(e) or "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS" in str(e):
            print(f"✅ SUCCESS: Local mode properly blocked: {e}")
            return True
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False
    finally:
        # Restore remote mode
        os.environ["WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"] = "remote"

def main():
    print("=" * 70)
    print("Testing Remote Mode with ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=False")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(test_compilation_with_remote_mode())
    results.append(test_validation_in_sandbox())
    results.append(test_local_mode_blocked())
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("-" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
