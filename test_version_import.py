#!/usr/bin/env python3
"""
Test that the version import is fixed.
"""

def test_version_import():
    """Test that we can import the version correctly."""
    
    try:
        from inference.core.version import __version__ as inference_version
        print(f"✅ SUCCESS: Version imported successfully: {inference_version}")
        
        # Test the template ID generation logic
        version_str = inference_version.replace('.', '-')
        template_id = f"inference-sandbox-v{version_str}"
        print(f"✅ E2B Template ID would be: {template_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ FAILED to import version: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_version_import()
    sys.exit(0 if success else 1)
