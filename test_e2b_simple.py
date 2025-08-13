#!/usr/bin/env python3
"""
Simple test of E2B code interpreter following their documentation pattern.
"""

import os

# This is the pattern from E2B documentation
from e2b_code_interpreter import Sandbox

api_key = os.getenv("E2B_API_KEY")
if not api_key:
    print("Please set E2B_API_KEY environment variable")
    exit(1)

print("E2B Code Interpreter - Documentation Pattern Test")
print("=" * 60)

# Following the exact pattern from e2b-code-interpreter docs
print("Creating sandbox using documentation pattern...")

# Create sandbox - no template specified (uses default)
sandbox = Sandbox()

print(f"✅ Sandbox created: {sandbox.sandbox_id}")

# Run simple code
print("\nRunning test code...")
execution = sandbox.run_code("print('Hello, World!')")

if execution.error:
    print(f"❌ Error: {execution.error}")
else:
    print("✅ Success!")
    if execution.logs and execution.logs.stdout:
        print(f"Output: {execution.logs.stdout}")

# Run code with result
print("\nRunning calculation...")
execution2 = sandbox.run_code("result = 2 + 3\nprint(f'Result: {result}')")

if execution2.error:
    print(f"❌ Error: {execution2.error}")
else:
    print("✅ Success!")
    if execution2.logs and execution2.logs.stdout:
        print(f"Output: {execution2.logs.stdout}")

# Cleanup
print("\nCleaning up...")
sandbox.close()
print("✅ Sandbox closed")

print("\n" + "=" * 60)
print("If this works, we should use the same pattern in e2b_executor.py")
