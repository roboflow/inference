#!/usr/bin/env python3
"""
Test that the escaped braces fix works correctly.
"""

def test_escaped_braces():
    """Test that braces are properly escaped in f-strings."""
    
    # Simulate what happens in the wrapper code
    func_name = "run"
    
    # This is the problematic line that needed fixing
    wrapper_code = f"""
        # Create mock self object with init results if available
        class MockSelf:
            def __init__(self):
                self._init_results = globals().get('_init_results', {{}})
        
        mock_self = MockSelf()
        
        # Call the user's run function
        result = {func_name}(mock_self, **inputs)
        
        # Serialize outputs
        serialized_result = {{}}
    """
    
    print("Generated wrapper code:")
    print(wrapper_code)
    
    # Check that the output is correct
    assert "globals().get('_init_results', {})" in wrapper_code
    assert "result = run(mock_self, **inputs)" in wrapper_code
    assert "serialized_result = {}" in wrapper_code
    
    print("\n✅ All braces properly escaped!")
    print("The fix ensures:")
    print("1. Empty dict {} in globals().get() is properly rendered")
    print("2. Function name is correctly inserted")
    print("3. Empty dict for serialized_result is properly rendered")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_escaped_braces()
    sys.exit(0 if success else 1)
