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

### Using Pre-trained Models

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

## Trained RF-DETR Outside Roboflow? Use with `inference-models`

RF-DETR offers a **seamless training-to-deployment workflow** that makes it incredibly easy to go from training to production.

### Step 1: Train Your Model

Train RF-DETR on your custom dataset using the official [rf-detr repository](https://github.com/roboflow/rf-detr):

```bash
# Clone the RF-DETR training repository
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# Install dependencies
pip install -r requirements.txt

# Train on your custom dataset
python train.py --config configs/rfdetr_base.yaml --data path/to/your/dataset
```

After training completes, you'll have a checkpoint file (e.g., `checkpoint_best.pth`) containing your trained weights.

### Step 2: Deploy Instantly with inference-models

Here's where the magic happens - **no conversion, no export, no hassle**. Simply point `AutoModel` directly at your training checkpoint:

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load your freshly trained model directly from the checkpoint
# You MUST specify model_type for checkpoint loading
model = AutoModel.from_pretrained(
    "/path/to/checkpoint_best.pth",
    model_type="rfdetr-base",  # Required: specify the model architecture
    labels=["class1", "class2", "class3"]  # Optional: your custom class names
)

# That's it! Use it exactly like any other model
image = cv2.imread("path/to/image.jpg")
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate and visualize
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)
cv2.imwrite("annotated.jpg", annotated_image)
```

**Important parameters:**

- **`model_type`** (required) - Specifies the RF-DETR architecture variant. Must be one of: `rfdetr-nano`, `rfdetr-small`, `rfdetr-base`, `rfdetr-medium`, `rfdetr-large`
- **`labels`** (optional) - Class names for your model. Can be:
    - A list of class names: `["person", "car", "dog"]`
    - A registered label set name: `"coco"` (for COCO dataset classes)
    - If not provided, defaults to COCO labels

### Why This Matters

**Frictionless training-to-production workflow:**

- ✅ **No model conversion** - Use training checkpoints directly
- ✅ **No export step** - Skip ONNX/TensorRT export complexity
- ✅ **Instant deployment** - From training to production in seconds
- ✅ **Same API** - Identical interface for pre-trained and custom models
- ✅ **Production-ready** - Leverage all `inference-models` features (multi-backend, caching, optimization)

This seamless workflow eliminates the traditional friction between training and deployment, letting you iterate faster and deploy with confidence.

