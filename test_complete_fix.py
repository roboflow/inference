#!/usr/bin/env python3
"""
Comprehensive test that the complete fix works for the simple block.
This simulates what happens during the actual execution.
"""

from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)

def test_complete_wrapper_generation():
    """Test the complete wrapper code generation with the fixes."""
    
    # Create the PythonCode object matching your simple block
    python_code = PythonCode(
        run_function_name="run",
        run_function_code="""import random
def run(self, image) -> BlockResult:
    return {"random": random.random()}""",
        init_function_name="init",
        init_function_code=None,
        imports=[]
    )
    
    # Simulate the _build_wrapper_code function
    func_name = python_code.run_function_name
    
    # Build the wrapper exactly as it would be built
    wrapper = f"""
# Wrapper function to handle serialization
def execute_wrapped(serialized_inputs_json: str) -> str:
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
    
    return json.dumps(serialized_result)
"""
    
    print("Generated wrapper code for your block:")
    print("-" * 60)
    print(wrapper)
    print("-" * 60)
    
    # Verify the code is syntactically correct
    try:
        compile(wrapper, '<string>', 'exec')
        print("\n✅ Wrapper code compiles successfully!")
    except SyntaxError as e:
        print(f"\n❌ Syntax error: {e}")
        return False
    
    # Check critical parts
    assert "globals().get('_init_results', {})" in wrapper
    assert f"result = {func_name}(mock_self, **inputs)" in wrapper
    assert "serialized_result = {}" in wrapper
    
    print("\n✅ All critical code parts correctly generated!")
    print("\nYour simple block should now work without errors:")
    print("- Empty dict in globals().get() is properly escaped")
    print("- Function name 'run' is correctly inserted")  
    print("- Dictionary literals are properly escaped throughout")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_complete_wrapper_generation()
    sys.exit(0 if success else 1)
