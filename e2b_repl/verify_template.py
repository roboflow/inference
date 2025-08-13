#!/usr/bin/env python3
"""
Final test to verify the E2B template works for Custom Python Blocks
"""

import os
os.environ['E2B_API_KEY'] = 'e2b_c57c2691de57c1a7a6112cb2d0973f2f51e2ee8e'

try:
    from e2b import Sandbox
except ImportError:
    os.system("pip3 install -q e2b")
    from e2b import Sandbox

print("=" * 60)
print("E2B Template Verification for Custom Python Blocks")
print("=" * 60)

# Create sandbox
sandbox = Sandbox(template="qfupheopqmf6w7b36h6o")
print(f"\n✅ Sandbox created: {sandbox.sandbox_id}")

# Test code that represents a real Custom Python Block
test_code = '''
import sys
sys.path.insert(0, '/app')

import numpy as np
import supervision as sv
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData

print("Testing Custom Python Block functionality...")

# Simulate typical Custom Python Block operations
# 1. Process numpy arrays
image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
print(f"✅ Created image array: shape={image.shape}, dtype={image.dtype}")

# 2. Work with detections
detections = sv.Detections(
    xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
    confidence=np.array([0.95, 0.87]),
    class_id=np.array([0, 1])
)
print(f"✅ Created detections: {len(detections)} objects detected")

# 3. Process predictions (typical workflow input)
predictions = {
    "predictions": [
        {"x": 150, "y": 150, "width": 100, "height": 100, "confidence": 0.95, "class": "person"},
        {"x": 350, "y": 350, "width": 100, "height": 100, "confidence": 0.87, "class": "car"}
    ],
    "image": {"width": 640, "height": 480}
}

# 4. Calculate statistics (common operation)
confidences = [p["confidence"] for p in predictions["predictions"]]
avg_confidence = np.mean(confidences)
print(f"✅ Calculated average confidence: {avg_confidence:.3f}")

# 5. Create output dictionary (required for Custom Python Blocks)
output = {
    "num_detections": len(predictions["predictions"]),
    "average_confidence": float(avg_confidence),
    "classes_detected": list(set(p["class"] for p in predictions["predictions"])),
    "image_dimensions": [predictions["image"]["width"], predictions["image"]["height"]]
}

print(f"✅ Generated output: {output}")

# 6. Test serialization helpers
from inference.core.workflows.core_steps.common.serializers import serialize_wildcard_kind
test_data = {"key": "value", "number": 42}
serialized = serialize_wildcard_kind(value=test_data)
print(f"✅ Serialization works: {type(serialized)}")

print("\\n✨ All Custom Python Block operations completed successfully!")
'''

# Write and execute the test
sandbox.files.write("/tmp/final_test.py", test_code)
result = sandbox.commands.run("python3 /tmp/final_test.py")

print("\nTest Output:")
print("-" * 60)
if result.exit_code == 0:
    print(result.stdout)
    print("\n" + "=" * 60)
    print("🎉 SUCCESS: E2B Template is fully functional!")
    print("=" * 60)
    print("\nTemplate Details:")
    print(f"  Template ID: qfupheopqmf6w7b36h6o")
    print(f"  Sandbox ID: {sandbox.sandbox_id}")
    print(f"  Status: Ready for Custom Python Blocks")
else:
    print(f"Error Output:\n{result.stderr}")
    print("\n❌ Test failed")

print(f"\nℹ️  To connect to this sandbox:")
print(f"    python3 e2b_inference_repl.py {sandbox.sandbox_id}")
