# YOLOv9 - Object Detection

YOLOv9 is an object detection model featuring Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) for improved accuracy and efficiency.

## Overview

YOLOv9 for object detection introduces novel architectural improvements:

- **Programmable Gradient Information (PGI)** - Better gradient flow during training
- **GELAN architecture** - Efficient layer aggregation
- **Improved feature extraction** - Better multi-scale feature fusion
- **Multiple model sizes** - Various model variants available

## License

**GPL-3.0**

!!! info "Licensing Options"
    - **GPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - For commercial use, you need to follow AGPL-3.0 conditions or purchase a license for commercial use, modifications, and distribution.

    See [Roboflow Model Licenses - YOLOv9](https://roboflow.com/model-licenses/yolov9) for details.

## Pre-trained Model IDs

No pre-trained model aliases are available. Train custom models on Roboflow.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch-script` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
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

