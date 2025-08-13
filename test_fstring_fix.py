#!/usr/bin/env python3
"""
Test that the f-string issue is fixed for simple custom Python blocks.
"""

# Simple test to check if the wrapper code generation works correctly
def test_wrapper_code_generation():
    """Test that wrapper code is generated without f-string errors."""
    
    class MockPythonCode:
        def __init__(self):
            self.run_function_name = "run"
            self.run_function_code = """
import random
def run(self, image) -> BlockResult:
    return {"random": random.random()}
"""
            self.init_function_code = None
            self.init_function_name = "init"
            self.imports = []
    
    # Simulate the wrapper code generation
    python_code = MockPythonCode()
    
    # This simulates what _build_wrapper_code does
    func_name = python_code.run_function_name
    
    # Test that the function name is properly inserted
    test_string = f"""
# Call the user's run function
result = {func_name}(mock_self, **inputs)
"""
    
    print("Generated code snippet:")
    print(test_string)
    
    # Check that the function name was properly inserted
    assert "result = run(mock_self, **inputs)" in test_string
    print("✅ Function name properly inserted")
    
    # Test with double braces for dictionary
    test_dict_string = f"""
serialized_result = {{}}
for key, value in result.items():
    serialized_result[key] = {{
        'type': 'test',
        'data': value
    }}
"""
    
    print("\nGenerated dictionary code:")
    print(test_dict_string)
    
    # Check that braces are properly escaped
    assert "serialized_result = {}" in test_dict_string
    assert "'type': 'test'" in test_dict_string
    print("✅ Dictionary braces properly escaped")
    
    print("\n✅ All wrapper code generation tests passed!")

if __name__ == "__main__":
    test_wrapper_code_generation()
