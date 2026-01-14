# RF-DETR - Object Detection

RF-DETR is a state-of-the-art, real-time object detection model developed by Roboflow. It is the first real-time detection transformer to achieve breakthrough performance on COCO, setting a new standard for object detection accuracy and speed.

Developed entirely in-house at Roboflow, RF-DETR represents a major advancement in computer vision, designed to transfer exceptionally well across diverse domains and dataset sizes—from small custom datasets to large-scale benchmarks.

## Overview

RF-DETR features:

- **State-of-the-art accuracy** - Leading performance on COCO and real-world benchmarks
- **Transformer-based architecture** - First real-time detection transformer architecture
- **Exceptional domain transfer** - Designed to excel across diverse domains and dataset sizes
- **Multiple model sizes** - From nano to large variants for different deployment scenarios
- **Real-time performance** - Optimized for speed without sacrificing accuracy
- **Production-ready** - Built for deployment on edge devices and cloud infrastructure

## License

**Apache 2.0**

RF-DETR is released under the Apache 2.0 license, making it free for both commercial and non-commercial use.

Learn more: [Apache 2.0 License](https://github.com/roboflow/inference/blob/main/inference_models/models/rfdetr/LICENSE.txt)

## Pre-trained Model IDs

All pre-trained RF-DETR object detection models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | Model ID | Parameters |
|------------|----------|------------|
| Nano | `rfdetr-nano` | ~10M |
| Small | `rfdetr-small` | ~25M |
| Base | `rfdetr-base` | 29M |
| Medium | `rfdetr-medium` | ~75M |
| Large | `rfdetr-large` | 129M |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Object Detection block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Installation

Install with one of the following extras depending on your backend:

- **PyTorch**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`
- **ONNX**: `onnx-cpu`, `onnx-cu12`
- **TensorRT**: `trt10` (requires CUDA 12.x)

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("rfdetr-base")
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

