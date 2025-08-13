#!/usr/bin/env python3
"""
Test E2B sandbox - verify inference modules work correctly
"""

import os
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

try:
    from e2b import Sandbox
except ImportError:
    print("Installing e2b...")
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

# Create sandbox
print("Creating sandbox...")
sandbox = Sandbox(template="qfupheopqmf6w7b36h6o")
print(f"✅ Sandbox created: {sandbox.sandbox_id}")

# Test complete Custom Python Block code
test_code = """
import sys
sys.path.insert(0, '/app')

import numpy as np
import supervision as sv
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult

# Simulate inputs that would come from a workflow
class MockInputs:
    def __init__(self):
        self.image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        self.prediction = {
            'predictions': [
                {'confidence': 0.9, 'class': 'person', 'x': 100, 'y': 100, 'width': 50, 'height': 100},
                {'confidence': 0.8, 'class': 'car', 'x': 200, 'y': 200, 'width': 100, 'height': 60},
                {'confidence': 0.7, 'class': 'dog', 'x': 300, 'y': 300, 'width': 40, 'height': 40}
            ]
        }

# Create mock inputs
inputs = MockInputs()

# Example Custom Python Block code
print("Testing Custom Python Block code...")

# Process the input image
image = inputs.image
prediction = inputs.prediction

# Calculate average confidence
avg_confidence = np.mean([det['confidence'] for det in prediction['predictions']])

# Count detections per class
class_counts = {}
for det in prediction['predictions']:
    cls = det['class']
    class_counts[cls] = class_counts.get(cls, 0) + 1

# Create output
output = {
    'image_shape': list(image.shape),
    'num_detections': len(prediction['predictions']),
    'avg_confidence': float(avg_confidence),
    'class_counts': class_counts
}

print(f"✅ Output: {output}")

# Test with supervision
detections = sv.Detections(
    xyxy=np.array([[100, 100, 150, 200], [200, 200, 300, 260]]),
    confidence=np.array([0.9, 0.8]),
    class_id=np.array([0, 1])
)

print(f"✅ Supervision detections created: {len(detections)} detections")

# Test BlockResult
result = BlockResult(output=output)
print(f"✅ BlockResult created successfully")

print("\\n✅ All Custom Python Block operations work correctly!")
"""

# Save and run the test
sandbox.files.write("/tmp/test_custom_block.py", test_code)
result = sandbox.commands.run("python3 /tmp/test_custom_block.py")

print("\n" + "=" * 60)
print("Test Results:")
print("=" * 60)
if result.exit_code == 0:
    print(result.stdout)
    print("\n✅ SUCCESS: Template works correctly for Custom Python Blocks!")
else:
    print(f"❌ Error: {result.stderr}")

print(f"\n🧹 Sandbox ID: {sandbox.sandbox_id}")
