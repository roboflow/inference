# YOLOv10 - Object Detection

YOLOv10 is an object detection model developed by Tsinghua University, featuring an NMS-free architecture for end-to-end object detection with reduced latency.

## Overview

YOLOv10 for object detection is designed for real-time detection with unique architectural improvements:

- **NMS-free architecture** - End-to-end detection without non-maximum suppression
- **Dual assignments** - Improved training strategy
- **Efficient backbone** - Optimized for speed and accuracy
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv10 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv10 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv10 License Details](https://roboflow.com/model-licenses/yolov10)

## Pre-trained Model IDs

All pre-trained YOLOv10 object detection models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Nano | `yolov10n-640` |
| Small | `yolov10s-640` |
| Medium | `yolov10m-640` |
| Balanced | `yolov10b-640` |
| Large | `yolov10l-640` |
| Extra-Large | `yolov10x-640` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ✅ Upload pre-trained weights ([guide](https://docs.roboflow.com/deploy/upload-custom-weights)) |
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
model = AutoModel.from_pretrained("yolov10n-640")
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

