"""
Test script for instance segmentation using inference-exp with PyTorch/TorchScript backend.

This script mirrors test_seg.py but uses the experimental inference library (inference-exp)
with explicit PyTorch backend for the segmentation model.
"""
from inference_exp import AutoModel
import cv2
import supervision as sv
import urllib.request
import numpy as np

# Image URL (same as test_seg.py)
image_url = "https://media.roboflow.com/dog.jpeg"

# Load image for visualization
resp = urllib.request.urlopen(image_url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# Load model using AutoModel with torch backend
print("Loading rfdetr-seg-preview model with Torch backend...")
model = AutoModel.from_pretrained(
    "rfdetr-seg-preview",
    backend="torch",  # Explicitly use PyTorch backend
)

# Run inference
print("Running inference...")
predictions = model(image)

# Convert to supervision format
result = predictions[0].to_supervision()

print(f"Detections found: {len(result)}")
# Map class IDs to names using model's class_names
labels = [f"{model.class_names[cid]}: {conf:.2f}" for cid, conf in zip(result.class_id, result.confidence)]
print(f"Detections: {labels}")

# Visualize and save with class names as labels
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(scene=image.copy(), detections=result)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=result, labels=labels)

cv2.imwrite("rfdetr_seg_exp_torch_result.png", annotated_image)
print("Saved visualization to rfdetr_seg_exp_torch_result.png")
