# Roboflow Instant - Object Detection

Roboflow Instant is a proprietary object detection model trained on the Roboflow platform. It's a specialized model designed to work seamlessly with your Roboflow account using `inference` and `inference-models`.

## Overview

Roboflow Instant for object detection features:

- **Roboflow-exclusive** - Proprietary model trained on the Roboflow platform
- **Almost no labeling required** - Quickly prototype the model using Roboflow platform 
- **Seamless integration** - Works directly with your Roboflow account

## Pre-trained Model IDs

Roboflow Instant models are accessed through your Roboflow workspace. Model IDs follow a special format specific to Roboflow Instant (not the standard `project-url/version` format).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `hugging-face` | Included in default installation |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train on Roboflow (proprietary model) |
| **Upload Weights** | ❌ Not applicable (Roboflow-exclusive) |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Object Detection block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Note:** Model ID structure differs from standard Roboflow models.

## Installation

Roboflow Instant support is included with the default `inference-models` installation. No additional packages required.

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
# Replace with your actual Roboflow Instant model ID
model = AutoModel.from_pretrained("your-roboflow-instant-model-id", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

