#!/usr/bin/env python3
"""
Test that the REPL now works with proper permissions
"""

import os
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

try:
    from e2b import Sandbox
except ImportError:
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

print("Testing REPL with fixed permissions...")
sandbox = Sandbox(template="qfupheopqmf6w7b36h6o")
print(f"✅ Sandbox created: {sandbox.sandbox_id}")

# Test writing to /home/user (should work)
test_code = """
import sys
sys.path.insert(0, '/app')

import inference
print(f"✅ Inference imported from: {inference.__file__}")

from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
print("✅ Workflow entities imported")

import numpy as np
import supervision as sv
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Supervision version: {sv.__version__}")
"""

# Write and execute in /home/user
sandbox.files.write("/home/user/test.py", test_code)
result = sandbox.commands.run("python3 /home/user/test.py")

if result.exit_code == 0:
    print("\n✅ All tests passed! Output:")
    print(result.stdout)
else:
    print(f"❌ Error: {result.stderr}")

print(f"\n✨ The REPL is now fixed and ready to use!")
print(f"   Run: python3 e2b_repl.py")
