#!/usr/bin/env python3
"""
Test that E2B dependencies are properly installed.
"""

def test_e2b_import():
    """Test that we can import the E2B sandbox."""
    try:
        from e2b_code_interpreter import Sandbox
        print("✅ SUCCESS: e2b_code_interpreter imported successfully!")
        
        # Check for E2B API key
        import os
        if os.getenv("E2B_API_KEY"):
            print("✅ E2B_API_KEY is set")
        else:
            print("⚠️  E2B_API_KEY is not set - you'll need this to run sandboxes")
            print("   Get your API key from: https://e2b.dev/dashboard")
            print("   Then set it: export E2B_API_KEY='your-key-here'")
        
        # Check execution mode
        mode = os.getenv("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local")
        print(f"\n📍 Current execution mode: {mode}")
        
        if mode == "remote":
            print("✅ Ready for remote execution in E2B sandboxes")
        else:
            print("ℹ️  To enable remote execution, set:")
            print("   export WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=remote")
        
        return True
        
    except ImportError as e:
        print(f"❌ FAILED to import e2b_code_interpreter: {e}")
        print("\nTo fix this, run:")
        print("  pip install -r requirements/requirements.e2b.txt")
        return False

if __name__ == "__main__":
    import sys
    success = test_e2b_import()
    sys.exit(0 if success else 1)
