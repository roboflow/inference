#!/usr/bin/env python3
"""
Check the E2B code interpreter library for default template info.
"""

import e2b_code_interpreter
print(f"E2B Code Interpreter version: {e2b_code_interpreter.__version__ if hasattr(e2b_code_interpreter, '__version__') else 'unknown'}")

# Check if there's a default template constant
import inspect
members = inspect.getmembers(e2b_code_interpreter)
print("\nConstants/variables in e2b_code_interpreter:")
for name, value in members:
    if not name.startswith('_') and isinstance(value, str):
        print(f"  {name}: {value}")

# Check Sandbox class attributes
from e2b_code_interpreter import Sandbox
print("\nSandbox class attributes:")
for attr in dir(Sandbox):
    if not attr.startswith('_') and attr.isupper():
        try:
            value = getattr(Sandbox, attr)
            print(f"  {attr}: {value}")
        except:
            pass

# Look for template-related info
print("\nChecking for template info in Sandbox:")
if hasattr(Sandbox, 'DEFAULT_TEMPLATE'):
    print(f"  DEFAULT_TEMPLATE: {Sandbox.DEFAULT_TEMPLATE}")
if hasattr(Sandbox, 'TEMPLATE'):
    print(f"  TEMPLATE: {Sandbox.TEMPLATE}")
