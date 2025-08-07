#!/usr/bin/env python3
"""
Simple test for E2B sandbox functionality.
"""

import os
import json
from e2b import Sandbox

# Set the E2B API key
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

def test_e2b_sandbox():
    """Test basic E2B sandbox functionality."""
    print("=" * 60)
    print("Testing E2B Sandbox Connection")
    print("=" * 60)
    
    # Template ID we created
    template_id = "qfupheopqmf6w7b36h6o"
    
    print(f"Creating sandbox with template: {template_id}")
    
    try:
        # Create a sandbox
        sandbox = Sandbox(template=template_id)
        print("✅ Sandbox created successfully!")
        
        # Test simple Python execution
        print("\nTesting Python execution...")
        result = sandbox.run_python("print('Hello from E2B sandbox!')")
        print(f"Output: {result.text if hasattr(result, 'text') else result}")
        
        # Test with numpy
        print("\nTesting numpy import...")
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array mean: {arr.mean()}")
print(f"Array sum: {arr.sum()}")
"""
        result = sandbox.run_python(code)
        print(f"Output: {result.text if hasattr(result, 'text') else result}")
        
        # Test with supervision
        print("\nTesting supervision import...")
        code = """
import supervision as sv
print(f"Supervision version: {sv.__version__}")
"""
        result = sandbox.run_python(code)
        print(f"Output: {result.text if hasattr(result, 'text') else result}")
        
        # Test with inference modules
        print("\nTesting inference module imports...")
        code = """
import sys
sys.path.append('/app')

from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult

print("✅ Inference modules imported successfully!")
"""
        result = sandbox.run_python(code)
        print(f"Output: {result.text if hasattr(result, 'text') else result}")
        
        # Clean up
        sandbox.close()
        print("\n✅ Sandbox closed successfully!")
        
        print("\n" + "=" * 60)
        print("✅ All E2B sandbox tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_e2b_sandbox()
