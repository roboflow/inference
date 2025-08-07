#!/usr/bin/env python3
"""
Check E2B Sandbox API.
"""

import os
from e2b import Sandbox

# Set the E2B API key
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

template_id = "qfupheopqmf6w7b36h6o"
sandbox = Sandbox(template=template_id)

print("Available methods on Sandbox object:")
for attr in dir(sandbox):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Check if there's a code interpreter
if hasattr(sandbox, 'code_interpreter'):
    print("\nCode Interpreter methods:")
    for attr in dir(sandbox.code_interpreter):
        if not attr.startswith('_'):
            print(f"  - {attr}")

sandbox.close()
