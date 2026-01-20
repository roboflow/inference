# YOLACT - Instance Segmentation

YOLACT (You Only Look At CoefficienTs) is a real-time instance segmentation model that introduced a novel approach to generating masks in parallel with object detection.

## Overview

YOLACT is designed for real-time instance segmentation with a unique architecture. Key features include:

- **Parallel mask generation** - Generates prototype masks and coefficients simultaneously
- **Real-time performance** - Fast inference suitable for video applications
- **Single-stage architecture** - Simplified pipeline compared to two-stage methods
- **ResNet backbone** - Available in ResNet-50 and ResNet-101 variants
- **Proven approach** - Well-established model with strong community support

## License

**MIT**

!!! info "Open Source License"
    YOLACT is licensed under MIT, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [MIT License](https://opensource.org/licenses/MIT)

## Pre-trained Model IDs

YOLACT instance segmentation models must be trained on Roboflow or uploaded as custom weights. There are no pre-trained public model IDs available.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights ([guide](https://docs.roboflow.com/deploy/upload-custom-weights)) |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Instance Segmentation block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load your custom model (requires Roboflow API key)
model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key"
)
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image with masks
mask_annotator = sv.MaskAnnotator()
annotated_image = mask_annotator.annotate(image.copy(), detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

