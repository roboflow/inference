# YOLOv11 - Object Detection

YOLOv11 is the latest version in the YOLO family developed by Ultralytics, offering improved accuracy and speed over YOLOv8. It features enhanced architecture and better performance across all tasks.

## Overview

YOLOv11 for object detection is designed for real-time detection and localization of objects in images. It features:

- **Enhanced anchor-free detection head** - Further optimized architecture
- **Improved feature pyramid network** - Better multi-scale feature fusion
- **Efficient backbone** - Optimized for speed and accuracy
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv11 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv11 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv11 License Details](https://roboflow.com/model-licenses/yolo11)

## Pre-trained Model IDs

All pre-trained YOLOv11 object detection models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 | 1280×1280 |
|------------|---------|-----------|
| Nano | `yolov11n-640` | `yolov11n-1280` |
| Small | `yolov11s-640` | `yolov11s-1280` |
| Medium | `yolov11m-640` | `yolov11m-1280` |
| Large | `yolov11l-640` | `yolov11l-1280` |
| Extra-Large | `yolov11x-640` | `yolov11x-1280` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch-script` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
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

- **ONNX**: `onnx-cpu`, `onnx-cu12`
- **TensorRT**: `trt10` (requires CUDA 12.x)
- **TorchScript**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("yolov11n-640")
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

