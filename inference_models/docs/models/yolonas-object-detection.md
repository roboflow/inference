# YOLO-NAS - Object Detection

YOLO-NAS is an object detection model developed by Deci AI using Neural Architecture Search (NAS), optimized for high accuracy and efficient inference on edge devices.

## Overview

YOLO-NAS for object detection features:

- **Neural Architecture Search** - Automatically optimized architecture
- **Quantization-aware training** - Optimized for INT8 deployment
- **Edge-optimized** - Designed for efficient edge device deployment
- **Multiple model sizes** - Small, medium, and large variants

## License

**Apache 2.0**

!!! info "Important Licensing Information"
    The YOLO-NAS **code** is released under the Apache 2.0 license, but the **Deci-provided pre-trained weights** are under a special license.

    - **Training with Roboflow Train**: Commercial usage is allowed because Roboflow does not use the Deci weights.
    - **Self-training outside Roboflow**: If you train your own YOLO-NAS model outside the Roboflow platform, ensuring adherence to the [Deci YOLO-NAS license](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) is your responsibility.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing)

## Pre-trained Model IDs

All pre-trained YOLO-NAS object detection models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Small | `yolo-nas-s-640` |
| Medium | `yolo-nas-m-640` |
| Large | `yolo-nas-l-640` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
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

Install with one of the following extras:

- **ONNX**: `onnx-cpu`, `onnx-cu12`

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("yolo-nas-s-640")
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

