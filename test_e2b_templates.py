#!/usr/bin/env python3
"""
Test different E2B template configurations to see which one works.
"""

import os
from e2b_code_interpreter import Sandbox

api_key = os.getenv("E2B_API_KEY")
if not api_key:
    print("Set E2B_API_KEY environment variable")
    print("You can test with: export E2B_API_KEY='your-key'")
    exit(1)

print("Testing E2B Templates")
print("=" * 60)

# Test different template configurations
templates_to_test = [
    (None, "Default (None)"),
    ("", "Empty string"),
    ("base", "Base template"),
    ("code-interpreter", "code-interpreter"),
    ("python", "python"),
    ("python3", "python3"),
]

for template, description in templates_to_test:
    print(f"\nTesting template: {description}")
    print("-" * 40)
    
    try:
        # Create sandbox
        kwargs = {'api_key': api_key, 'timeout': 30}
        if template is not None:
            kwargs['template'] = template
            
        print(f"Creating sandbox with template={template}...")
        sandbox = Sandbox(**kwargs)
        print(f"✅ Created: {sandbox.sandbox_id}")
        
        # Test code execution
        result = sandbox.run_code("print('Hello')")
        
        if result.error:
            print(f"❌ Code execution failed: {result.error}")
        else:
            print(f"✅ Code execution works!")
            
        # Clean up
        sandbox.close()
        
    except Exception as e:
        print(f"❌ Failed to create/test: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("Use whichever template above shows '✅ Code execution works!'")
