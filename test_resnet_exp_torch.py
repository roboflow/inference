"""
Test script for ResNet classification using inference-exp with PyTorch backend.

This script mirrors resnet.py but uses the experimental inference library (inference-exp)
with explicit PyTorch backend for the classification model.
"""
from inference_exp import AutoModel
import cv2
import numpy as np
import onnxruntime as ort

# Same image and API key as resnet.py
API_KEY = "Em3CODkJT1vbZ52wlWwk"
IMAGE = "blazenek-scaled.jpg"

# Load image
image = cv2.imread(IMAGE)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE}")

# Test ResNet models with torch backend
# Note: Model names in inference-exp may differ from standard inference
# Using the Roboflow model IDs similar to the original resnet.py
model_names = ["resnet18", "resnet34", "resnet50", "resnet101"]

print("Testing ResNet models with Torch backend...")
print("=" * 60)

for model_name in model_names:
    try:
        print(f"\nLoading {model_name}...")
        model = AutoModel.from_pretrained(
            model_name,
            api_key=API_KEY,
            backend="torch",  # Explicitly use PyTorch backend
        )

        # Run inference
        predictions = model(image)

        # Get top prediction
        # For classification, predictions has confidence and class_id attributes
        if hasattr(predictions, 'confidence'):
            # Single prediction object (multi-class)
            top_class_idx = predictions.class_id[0].item()
            top_confidence = predictions.confidence[0, top_class_idx].item()
            class_name = model.class_names[top_class_idx] if hasattr(model, 'class_names') else str(top_class_idx)
            print(f"{model_name}: {class_name} ({top_confidence:.4f})")
        else:
            # List of predictions (multi-label)
            pred = predictions[0]
            top_class_idx = pred.class_ids[0].item()
            top_confidence = pred.confidence[top_class_idx].item()
            class_name = model.class_names[top_class_idx] if hasattr(model, 'class_names') else str(top_class_idx)
            print(f"{model_name}: {class_name} ({top_confidence:.4f})")

    except Exception as e:
        print(f"{model_name}: Error - {e}")

print("\n" + "=" * 60)
print("Done!")
