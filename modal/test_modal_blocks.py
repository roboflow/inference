#!/usr/bin/env python3
"""
Test script for Modal Custom Python Blocks implementation.

This script tests the Modal sandbox execution for Custom Python Blocks
in Roboflow Workflows.

Usage:
    # Option 1: Set environment variables
    export MODAL_TOKEN_ID="your_token_id"
    export MODAL_TOKEN_SECRET="your_token_secret"
    export WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE="modal"
    
    # Option 2: Use credentials from ~/.modal.toml (automatic fallback)
    
    # Run the test
    python modal/test_modal_blocks.py
"""

import os
import sys
import numpy as np
from typing import Dict, Any
from pathlib import Path


def load_modal_credentials():
    """Load Modal credentials from environment or ~/.modal.toml file.
    
    Returns:
        tuple: (token_id, token_secret) or (None, None) if not found
    """
    # First check environment variables
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if token_id and token_secret:
        print("✓ Using Modal credentials from environment variables")
        return token_id, token_secret
    
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
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                        
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
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                    
        except Exception as e:
            print(f"Warning: Could not parse {modal_toml_path}: {e}")
    
    return None, None


# Load credentials before importing Modal executor
token_id, token_secret = load_modal_credentials()

if not token_id or not token_secret:
    print("\nERROR: Modal credentials not found")
    print("\nPlease provide credentials using one of these methods:")
    print("1. Set environment variables:")
    print("   export MODAL_TOKEN_ID='your_token_id'")
    print("   export MODAL_TOKEN_SECRET='your_token_secret'")
    print("\n2. Run 'modal setup' to create ~/.modal.toml")
    print("\n3. Create ~/.modal.toml manually with a profile like:")
    print("   [default]  # or [your-profile-name]")
    print("   token_id = \"your_token_id\"")
    print("   token_secret = \"your_token_secret\"")
    print("   active = true  # optional, marks this as the active profile")
    sys.exit(1)

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
    assert result.get("sum") == 8
    assert result.get("squared") == 64
    print("  ✅ Simple computation test passed!")


def test_numpy_processing():
    """Test numpy array processing."""
    print("\nTesting numpy processing...")
    
    python_code = PythonCode(
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
        "shape": list(data.shape)  # Convert tuple to list for consistency
    }
""",
        run_function_name="process_array",
        init_function_name="init",
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
    
    print(f"  Input shape: {test_array.shape}")
    print(f"  Result: {result}")
    
    # Check shape - could be tuple (10, 20) or list [10, 20] depending on serialization
    shape = result.get("shape")
    if isinstance(shape, tuple):
        assert shape == (10, 20)
    else:
        assert shape == [10, 20]
    
    print("  ✅ Numpy processing test passed!")


def test_workspace_isolation():
    """Test that different workspaces get different executors."""
    print("\nTesting workspace isolation...")
    
    python_code = PythonCode(
        type="PythonCode",
        imports=[],
        run_function_code="""
def get_workspace_info(workspace: str) -> Dict[str, Any]:
    return {"workspace": workspace, "processed": True}
""",
        run_function_name="get_workspace_info",
        init_function_name="init",
        init_function_code=None,
    )
    
    # Test with workspace 1
    executor1 = ModalExecutor(workspace_id="workspace-1")
    result1 = executor1.execute_remote(
        block_type_name="workspace_test",
        python_code=python_code,
        inputs={"workspace": "workspace-1"},
        workspace_id="workspace-1"
    )
    
    # Test with workspace 2
    executor2 = ModalExecutor(workspace_id="workspace-2")
    result2 = executor2.execute_remote(
        block_type_name="workspace_test",
        python_code=python_code,
        inputs={"workspace": "workspace-2"},
        workspace_id="workspace-2"
    )
    
    print(f"  Workspace 1 result: {result1}")
    print(f"  Workspace 2 result: {result2}")
    assert result1.get("workspace") == "workspace-1"
    assert result2.get("workspace") == "workspace-2"
    print("  ✅ Workspace isolation test passed!")


def test_anonymous_fallback():
    """Test anonymous workspace fallback."""
    print("\nTesting anonymous workspace fallback...")
    
    python_code = PythonCode(
        type="PythonCode",
        imports=[],
        run_function_code="""
def anonymous_test(value: int) -> Dict[str, Any]:
    return {"value": value * 2, "anonymous": True}
""",
        run_function_name="anonymous_test",
        init_function_name="init",
        init_function_code=None,
    )
    
    # Test without workspace_id (should default to "anonymous")
    executor = ModalExecutor()  # No workspace_id provided
    result = executor.execute_remote(
        block_type_name="anonymous_test",
        python_code=python_code,
        inputs={"value": 21},
        workspace_id=None  # Explicitly None
    )
    
    print(f"  Result: {result}")
    assert result.get("value") == 42
    assert result.get("anonymous") == True
    print("  ✅ Anonymous fallback test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Modal Custom Python Blocks Implementation")
    print("=" * 60)
    
    try:
        test_simple_computation()
        test_numpy_processing()
        test_workspace_isolation()
        test_anonymous_fallback()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
