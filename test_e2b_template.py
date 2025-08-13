#!/usr/bin/env python3
"""
Test E2B Sandbox creation to see what template is being used.
"""

from e2b_code_interpreter import Sandbox
import os
import time

# Set your E2B API key
api_key = os.getenv("E2B_API_KEY")
if not api_key:
    print("❌ E2B_API_KEY not set")
    exit(1)

print("Testing E2B Sandbox Creation")
print("=" * 60)

try:
    # Create a sandbox with explicit logging
    print("Creating sandbox...")
    
    # The Sandbox from e2b_code_interpreter should use the right template
    # Let's try with NO template specified (should use default code interpreter)
    sandbox = Sandbox(api_key=api_key, timeout=60)
    
    print(f"✅ Sandbox created: {sandbox.sandbox_id}")
    print(f"📊 Sandbox domain: {sandbox.sandbox_domain if hasattr(sandbox, 'sandbox_domain') else 'N/A'}")
    print(f"📊 Is running: {sandbox.is_running if hasattr(sandbox, 'is_running') else 'N/A'}")
    
    # Try to get more info
    if hasattr(sandbox, 'envd_api_url'):
        print(f"📊 API URL: {sandbox.envd_api_url}")
    
    print("\nWaiting 2 seconds...")
    time.sleep(2)
    
    # Test if code execution works
    print("\nTesting code execution...")
    result = sandbox.run_code("print('Hello from E2B!')")
    
    if result.error:
        print(f"❌ Error: {result.error}")
    else:
        print(f"✅ Success! Output: {result.logs.stdout if result.logs else 'No output'}")
    
    # Try a simple calculation
    print("\nTesting calculation...")
    result2 = sandbox.run_code("result = 2 + 2; print(f'2 + 2 = {result}')")
    
    if result2.error:
        print(f"❌ Error: {result2.error}")
    else:
        print(f"✅ Success! Output: {result2.logs.stdout if result2.logs else 'No output'}")
    
    # Clean up
    print("\nCleaning up...")
    sandbox.close()
    print("✅ Sandbox closed")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("FINDINGS:")
print("If code execution works here but not in inference, the issue might be:")
print("1. The way we're creating the Sandbox in e2b_executor.py")
print("2. Template configuration differences")
print("3. Network/firewall issues in the inference environment")
