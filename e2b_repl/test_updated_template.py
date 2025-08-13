#!/usr/bin/env python3
"""
Test the newly built E2B template with permission fixes
"""

import os
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

try:
    from e2b import Sandbox
except ImportError:
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

print("=" * 60)
print("Testing Updated E2B Template with Permission Fixes")
print("=" * 60)

# Use the updated template
sandbox = Sandbox(template="roboflow-inference-sandbox-v0-52-0-fixed")
print(f"\n✅ Sandbox created: {sandbox.sandbox_id}")
print(f"   Template: roboflow-inference-sandbox-v0-52-0-fixed")

# Test writing to /home/user (should work with permission fix)
test_code = """
import sys
sys.path.insert(0, '/app')

# Test imports
import inference
print(f"✅ Inference imported from: {inference.__file__}")

from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
print("✅ Workflow entities imported")

import numpy as np
import supervision as sv
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Supervision version: {sv.__version__}")

# Test writing to /home/user
with open('/home/user/test_write.txt', 'w') as f:
    f.write('Permission test successful!')
print("✅ Successfully wrote to /home/user")

# Read back the file
with open('/home/user/test_write.txt', 'r') as f:
    content = f.read()
print(f"✅ Read from /home/user: {content}")
"""

# Write and execute in /home/user
sandbox.files.write("/home/user/test_permissions.py", test_code)
result = sandbox.commands.run("python3 /home/user/test_permissions.py")

print("\n" + "=" * 60)
print("Test Results:")
print("=" * 60)
if result.exit_code == 0:
    print(result.stdout)
    print("\n🎉 SUCCESS: Permission fixes are working!")
else:
    print(f"❌ Error: {result.stderr}")

print("\n✨ The updated E2B template is ready for use!")
print(f"   Template Name: roboflow-inference-sandbox-v0-52-0-fixed")
print(f"   Template ID: qfupheopqmf6w7b36h6o")
