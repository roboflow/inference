# YOLOLite - Object Detection

YOLOLite is a lightweight, efficient object detection model family designed for both GPU and edge (CPU) deployment. It uses EfficientNet and MobileNetV4 backbones with an anchor-free decoupled detection head.

## Overview

YOLOLite for object detection features:

- **Lightweight architecture** - EfficientNet-Lite backbones for GPU, MobileNetV4 for edge/CPU
- **Anchor-free detection** - Decoupled head with separate box, objectness, and class branches
- **Multiple model sizes** - From nano to extra-large, plus dedicated edge variants
- **ImageNet normalization** - Uses standard ImageNet mean/std preprocessing

## Model Variants

**GPU-optimized:**

| Variant | Backbone |
|---------|----------|
| `yololite_n` | EfficientNetV2-B0 |
| `yololite_s` | EfficientNetV2-B1 |
| `yololite_m` | EfficientNetV2-B2 |
| `yololite_l` | ConvNeXtV2-Tiny |
| `yololite_xl` | EfficientNetV2-S |

**Edge/CPU-optimized:**

| Variant | Backbone |
|---------|----------|
| `edge_n` | MobileNetV4-Small-050 |
| `edge_s` | MobileNetV4-Small |
| `edge_m` | MobileNetV4-Small |
| `edge_l` | MobileNetV4-Small |
| `edge_xl` | HGNetV2-B0 |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | -- Not yet supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Object Detection block |
| **Edge Deployment (Jetson)** | -- Not yet supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Installation

Install with ONNX extras depending on your hardware:

- **CPU**: `pip install inference-models[onnx-cpu]`
- **CUDA 12.x**: `pip install inference-models[onnx-cu12]`

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("my-project-abc123/2", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

# Run inference and convert to supervision Detections
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

## Prediction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | `float` | `0.25` | Minimum confidence threshold |
| `iou_threshold` | `float` | `0.45` | IoU threshold for NMS |
| `max_detections` | `int` | `300` | Maximum number of detections |
| `class_agnostic_nms` | `bool` | `False` | Whether to use class-agnostic NMS |
