#!/usr/bin/env python3
"""
Test that E2B requirements are properly included in setup.py
"""

def test_setup_includes_e2b():
    """Verify that setup.py includes the E2B requirements."""
    
    with open("setup.py", "r") as f:
        setup_content = f.read()
    
    if "requirements/requirements.e2b.txt" in setup_content:
        print("✅ SUCCESS: E2B requirements are included in setup.py")
        print("\nNow when you run 'pip install -e .' it will install:")
        print("  - e2b>=0.14.0")
        print("  - e2b-code-interpreter>=0.0.10")
        print("\nNo need to manually install E2B dependencies anymore!")
        return True
    else:
        print("❌ FAILED: E2B requirements not found in setup.py")
        return False

if __name__ == "__main__":
    import sys
    import os
    os.chdir("/Users/yeldarb/Code/inference")
    success = test_setup_includes_e2b()
    
    if success:
        print("\n📦 To reinstall inference with E2B support:")
        print("   pip install -e .")
    
    sys.exit(0 if success else 1)
