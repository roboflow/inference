#!/usr/bin/env python3
"""
Quick test to verify the fixed REPL works with inference imports
"""

import os
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

try:
    from e2b import Sandbox
except ImportError:
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

print("Testing fixed REPL with inference imports...")
sandbox = Sandbox(template="qfupheopqmf6w7b36h6o")
print(f"✅ Sandbox created: {sandbox.sandbox_id}")

# Test 1: Direct inference import
test_code1 = """
import inference
print(f"✅ Inference imported from: {inference.__file__}")
"""

# Test 2: Specific imports
test_code2 = """
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult
print("✅ Workflow entities imported successfully")
"""

# Test 3: Using inference with numpy and supervision
test_code3 = """
import numpy as np
import supervision as sv
from inference.core.workflows.core_steps.common.serializers import serialize_wildcard_kind

# Create test data
data = {"test": "value", "number": 42}
result = serialize_wildcard_kind(value=data)
print(f"✅ Serialization works: {type(result)}")
"""

# Execute tests
for i, code in enumerate([test_code1, test_code2, test_code3], 1):
    print(f"\nTest {i}:")
    sandbox.files.write("/tmp/test.py", f"import sys\nsys.path.insert(0, '/app')\n{code}")
    result = sandbox.commands.run("python3 /tmp/test.py")
    if result.exit_code == 0:
        print(result.stdout.strip())
    else:
        print(f"❌ Error: {result.stderr}")

print(f"\nℹ️  Now you can use the fixed REPL:")
print(f"    python3 e2b_repl.py")
print(f"    >>> import inference")
print(f"    >>> from inference.core.workflows.execution_engine.entities.base import Batch")
