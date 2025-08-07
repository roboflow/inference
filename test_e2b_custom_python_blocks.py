#!/usr/bin/env python3
"""
Test script for E2B sandbox integration with Custom Python Blocks.
"""

import os
import json
import base64
import numpy as np

# Set the E2B API key
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'
os.environ['WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE'] = 'remote'

# Now import the executor
from inference.core.workflows.core_steps.dynamic_blocks.e2b_executor import E2BExecutor

def test_simple_execution():
    """Test basic code execution in E2B sandbox."""
    print("Testing simple code execution...")
    
    executor = E2BExecutor()
    
    # Simple test code
    code = """
result = 2 + 2
print(f"Result: {result}")
output = result * 10
"""
    
    result = executor.run_custom_code(
        code=code,
        inputs={},
        global_parameters={}
    )
    
    print(f"Execution result: {result}")
    assert result['output'] == 40, f"Expected 40, got {result['output']}"
    print("✅ Simple execution test passed!")
    
    return result

def test_with_inputs():
    """Test code execution with inputs."""
    print("\nTesting code execution with inputs...")
    
    executor = E2BExecutor()
    
    # Test code with inputs
    code = """
import numpy as np

# Process the input image
image = inputs['image']
prediction = inputs['prediction']

# Calculate average confidence
avg_confidence = np.mean([det['confidence'] for det in prediction['predictions']])

# Create output
output = {
    'image_shape': image.shape,
    'num_detections': len(prediction['predictions']),
    'avg_confidence': float(avg_confidence)
}
"""
    
    # Create test inputs
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    test_prediction = {
        'predictions': [
            {'confidence': 0.9, 'class': 'person', 'x': 100, 'y': 100},
            {'confidence': 0.8, 'class': 'car', 'x': 200, 'y': 200},
            {'confidence': 0.7, 'class': 'dog', 'x': 300, 'y': 300}
        ]
    }
    
    result = executor.run_custom_code(
        code=code,
        inputs={
            'image': test_image,
            'prediction': test_prediction
        },
        global_parameters={}
    )
    
    print(f"Execution result: {result}")
    assert result['output']['num_detections'] == 3, "Expected 3 detections"
    assert result['output']['image_shape'] == [640, 480, 3], "Expected correct image shape"
    assert 0.7 <= result['output']['avg_confidence'] <= 0.9, "Expected correct avg confidence"
    print("✅ Input test passed!")
    
    return result

def test_with_supervision():
    """Test code execution with supervision library."""
    print("\nTesting code execution with supervision...")
    
    executor = E2BExecutor()
    
    # Test code using supervision
    code = """
import supervision as sv
import numpy as np

# Access the detection from inputs
detections = inputs['detections']

# Get detection count
num_detections = len(detections.xyxy) if hasattr(detections, 'xyxy') else 0

# Create output
output = {
    'num_detections': num_detections,
    'has_confidence': detections.confidence is not None
}
"""
    
    # Create test supervision detections
    import supervision as sv
    test_detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [100, 100, 200, 200]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 1])
    )
    
    result = executor.run_custom_code(
        code=code,
        inputs={'detections': test_detections},
        global_parameters={}
    )
    
    print(f"Execution result: {result}")
    assert result['output']['num_detections'] == 2, "Expected 2 detections"
    assert result['output']['has_confidence'] == True, "Expected confidence to be present"
    print("✅ Supervision test passed!")
    
    return result

def test_error_handling():
    """Test error handling in E2B sandbox."""
    print("\nTesting error handling...")
    
    executor = E2BExecutor()
    
    # Code with intentional error
    code = """
# This will raise an error
result = 1 / 0
output = result
"""
    
    try:
        result = executor.run_custom_code(
            code=code,
            inputs={},
            global_parameters={}
        )
        print("Error: Should have raised an exception!")
        assert False, "Expected an exception"
    except Exception as e:
        print(f"Caught expected exception: {e}")
        print("✅ Error handling test passed!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("E2B Custom Python Blocks Integration Test")
    print("=" * 60)
    
    try:
        # Run tests
        test_simple_execution()
        test_with_inputs()
        test_with_supervision()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
