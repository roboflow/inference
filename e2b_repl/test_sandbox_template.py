#!/usr/bin/env python3
"""
Test and debug E2B sandbox template
"""

import os
import sys

# Set E2B API key
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

# Install e2b if needed
try:
    from e2b import Sandbox
except ImportError:
    print("Installing e2b...")
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

def test_sandbox():
    print("=" * 60)
    print("Testing E2B Sandbox Template")
    print("=" * 60)
    
    # Create sandbox
    template_id = "qfupheopqmf6w7b36h6o"
    print(f"\n🚀 Creating sandbox with template: {template_id}")
    
    try:
        sandbox = Sandbox(template=template_id)
        print(f"✅ Sandbox created: {sandbox.sandbox_id}")
        
        # Test 1: Check Python version
        print("\n1. Checking Python version...")
        result = sandbox.commands.run("python3 --version")
        print(f"   Python: {result.stdout.strip()}")
        
        # Test 2: Check installed packages
        print("\n2. Checking installed packages...")
        result = sandbox.commands.run("pip3 list | grep -E 'numpy|supervision|opencv|inference'")
        print(f"   Packages found:")
        for line in result.stdout.strip().split('\n'):
            if line:
                print(f"   - {line}")
        
        # Test 3: Check directory structure
        print("\n3. Checking directory structure...")
        result = sandbox.commands.run("ls -la /")
        print("   Root directories:")
        for line in result.stdout.strip().split('\n')[1:6]:  # Show first 5 dirs
            print(f"   {line}")
        
        # Test 4: Check /app directory
        print("\n4. Checking /app directory...")
        result = sandbox.commands.run("ls -la /app")
        if result.exit_code == 0:
            print("   /app contents:")
            for line in result.stdout.strip().split('\n')[:10]:  # Show first 10 items
                print(f"   {line}")
        else:
            print(f"   ❌ /app not found: {result.stderr}")
        
        # Test 5: Test Python imports
        print("\n5. Testing Python imports...")
        
        # Test numpy
        test_code = """python3 -c "
import numpy as np
print('✅ numpy imported')
print(f'   Version: {np.__version__}')
" """
        result = sandbox.commands.run(test_code)
        print(result.stdout.strip() if result.exit_code == 0 else f"   ❌ numpy failed: {result.stderr}")
        
        # Test supervision
        test_code = """python3 -c "
import supervision as sv
print('✅ supervision imported')
print(f'   Version: {sv.__version__}')
" """
        result = sandbox.commands.run(test_code)
        print(result.stdout.strip() if result.exit_code == 0 else f"   ❌ supervision failed: {result.stderr}")
        
        # Test opencv
        test_code = """python3 -c "
import cv2
print('✅ opencv imported')
print(f'   Version: {cv2.__version__}')
" """
        result = sandbox.commands.run(test_code)
        print(result.stdout.strip() if result.exit_code == 0 else f"   ❌ opencv failed: {result.stderr}")
        
        # Test 6: Test inference modules
        print("\n6. Testing inference modules...")
        test_code = """python3 -c "
import sys
sys.path.insert(0, '/app')

try:
    from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
    print('✅ inference.core.workflows entities imported')
except ImportError as e:
    print(f'❌ Failed to import entities: {e}')

try:
    from inference.core.workflows.prototypes.block import BlockResult
    print('✅ inference.core.workflows BlockResult imported')
except ImportError as e:
    print(f'❌ Failed to import BlockResult: {e}')
    
try:
    import inference
    print(f'✅ inference package found at: {inference.__file__}')
except ImportError as e:
    print(f'❌ inference package not found: {e}')
" """
        result = sandbox.commands.run(test_code)
        print(result.stdout.strip() if result.stdout else f"   Error: {result.stderr}")
        
        # Test 7: Check PYTHONPATH
        print("\n7. Checking PYTHONPATH...")
        result = sandbox.commands.run("echo $PYTHONPATH")
        print(f"   PYTHONPATH: {result.stdout.strip() or '(empty)'}")
        
        # Test 8: Look for inference files
        print("\n8. Looking for inference files...")
        result = sandbox.commands.run("find /app -name '*.py' -path '*/inference/*' 2>/dev/null | head -5")
        if result.stdout:
            print("   Found inference Python files:")
            for line in result.stdout.strip().split('\n'):
                print(f"   - {line}")
        else:
            print("   No inference files found in /app")
            
        # Test 9: Check working directory
        print("\n9. Checking working directory...")
        result = sandbox.commands.run("pwd")
        print(f"   Current directory: {result.stdout.strip()}")
        
        # Test 10: Check startup script
        print("\n10. Checking startup script...")
        result = sandbox.commands.run("ls -la /root/.e2b/")
        if result.exit_code == 0:
            print("   /root/.e2b contents:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
                
            # Check startup.py content
            result = sandbox.commands.run("head -20 /root/.e2b/startup.py 2>/dev/null")
            if result.exit_code == 0:
                print("\n   startup.py (first 20 lines):")
                for line in result.stdout.strip().split('\n'):
                    print(f"   {line}")
        else:
            print(f"   /root/.e2b not found")
        
        print("\n" + "=" * 60)
        print("✅ Template testing complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'sandbox' in locals():
            print(f"\n🧹 Cleaning up sandbox {sandbox.sandbox_id}...")

if __name__ == "__main__":
    test_sandbox()
