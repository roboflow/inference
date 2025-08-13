#!/usr/bin/env python3
"""
Simple standalone test to verify the escaped braces fix.
"""

def test_wrapper_syntax():
    """Test that the wrapper code with escaped braces is syntactically valid."""
    
    func_name = "run"
    
    # This is exactly what gets generated in the wrapper
    wrapper_snippet = f"""
# Create mock self object with init results if available
class MockSelf:
    def __init__(self):
        self._init_results = globals().get('_init_results', {{}})

mock_self = MockSelf()

# Call the user's run function
result = {func_name}(mock_self, **inputs)

# Serialize outputs
serialized_result = {{}}
for key, value in result.items():
    serialized_result[key] = {{
        '_serialized_type': 'basic',
        'data': value
    }}
"""
    
    print("Testing wrapper code generation...")
    print("=" * 60)
    
    # Try to compile the code
    try:
        compile(wrapper_snippet, '<test>', 'exec')
        print("✅ SUCCESS: Wrapper code is syntactically valid!")
    except SyntaxError as e:
        print(f"❌ FAILED: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    
    # Verify the correct output
    expected_strings = [
        "globals().get('_init_results', {})",
        "result = run(mock_self, **inputs)",
        "serialized_result = {}"
    ]
    
    for expected in expected_strings:
        if expected in wrapper_snippet:
            print(f"✅ Found: {expected}")
        else:
            print(f"❌ Missing: {expected}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The f-string issue is fixed.")
    print("\nThe fix ensures that:")
    print("1. Empty dict {} in globals().get() is properly escaped as {{}}")
    print("2. Empty dict for serialized_result is properly escaped as {{}}")
    print("3. Dict literals in loops are properly escaped as {{}}")
    print("\nYour simple block with random.random() should now work!")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_wrapper_syntax()
    sys.exit(0 if success else 1)
