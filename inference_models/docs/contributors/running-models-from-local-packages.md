# Running Models from Local Packages

This guide explains how to run models available in the `inference-models` library from local model packages, including when and how to create `inference_config.json` files.

## Overview

Local model packages allow you to use custom-trained models or models not hosted on the Roboflow platform. The package structure and requirements depend on the model architecture and backend.

## Package Structure

A typical local model package has the following structure:

```
my-model-package/
├── model_config.json          # Required: Model metadata
├── inference_config.json      # Optional: Backend-specific configuration
├── weights.onnx              # Model weights (format depends on backend)
└── class_names.txt           # Optional: Class names for detection/classification
```

## Required Files

### `model_config.json`

**Always required.** This file contains essential model metadata:

```json
{
  "model_architecture": "yolov8",
  "task_type": "object-detection",
  "input_width": 640,
  "input_height": 640,
  "num_classes": 80,
  "class_names": ["person", "bicycle", "car", ...]
}
```

**Required fields:**
- `model_architecture`: Architecture identifier (e.g., `"yolov8"`, `"resnet"`, `"clip"`)
- `task_type`: Task identifier (e.g., `"object-detection"`, `"classification"`)
- `input_width`: Model input width in pixels
- `input_height`: Model input height in pixels

**Task-specific fields:**
- **Object Detection/Instance Segmentation**: `num_classes`, `class_names`
- **Classification**: `num_classes`, `class_names`
- **Keypoint Detection**: `num_keypoints`, `keypoint_names`

### Model Weights File

The weights file format depends on the backend:

| Backend | File Extension | Notes |
|---------|---------------|-------|
| ONNX | `.onnx` | Standard ONNX format |
| TensorRT | `.engine` or `.trt` | Platform-specific, must match GPU |
| PyTorch | `.pt` or `.pth` | TorchScript format |
| Hugging Face | N/A | Loaded from HF Hub or local directory |

**Naming conventions:**
- ONNX: `weights.onnx` or `model.onnx`
- TensorRT: `weights.engine` or `model.engine`
- PyTorch: `weights.pt` or `model.pt`

## Optional Files

### `inference_config.json`

**When is it needed?**

The `inference_config.json` file is **optional** and only required when you need to override default inference behavior or provide backend-specific configuration.

**Common use cases:**

1. **Custom NMS parameters** (Object Detection):
```json
{
  "nms_threshold": 0.45,
  "confidence_threshold": 0.25,
  "max_detections": 300
}
```

2. **Custom preprocessing** (Classification):
```json
{
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "normalize": true
}
```

3. **Backend-specific settings** (TensorRT):
```json
{
  "precision": "fp16",
  "workspace_size": 1073741824
}
```

4. **Model-specific parameters** (SAM):
```json
{
  "points_per_side": 32,
  "pred_iou_thresh": 0.88,
  "stability_score_thresh": 0.95
}
```

**When NOT needed:**

- Using default inference parameters
- Standard preprocessing (most YOLO models, ResNet, ViT)
- Models loaded from Roboflow platform (config is fetched automatically)

### `class_names.txt`

Alternative to specifying `class_names` in `model_config.json`:

```
person
bicycle
car
motorcycle
...
```

One class name per line, in order of class indices.

## Backend-Specific Requirements

### ONNX Backend

**Minimum requirements:**
- `model_config.json`
- `weights.onnx`

**Optional:**
- `inference_config.json` for custom NMS/preprocessing

**Example:**
```python
from inference_models import AutoModel

model = AutoModel.from_local_package(
    package_path="./my-yolov8-model",
    backend="onnx"
)
```

### TensorRT Backend

**Minimum requirements:**
- `model_config.json`
- `weights.engine` (must be built on target GPU)

**Important:** TensorRT engines are platform-specific. You must build the engine on the same GPU architecture where it will run.

**Example:**
```python
from inference_models import AutoModel

model = AutoModel.from_local_package(
    package_path="./my-yolov8-model",
    backend="trt"
)
```

### PyTorch/TorchScript Backend

**Minimum requirements:**
- `model_config.json`
- `weights.pt` (TorchScript format)

**Example:**
```python
from inference_models import AutoModel

model = AutoModel.from_local_package(
    package_path="./my-model",
    backend="torch-script"
)
```

### Hugging Face Backend

**Minimum requirements:**
- `model_config.json`
- Hugging Face model directory or Hub model ID

**Example:**
```python
from inference_models import AutoModel

# From local HF directory
model = AutoModel.from_local_package(
    package_path="./my-hf-model",
    backend="hugging-face"
)

# From HF Hub
model = AutoModel.from_pretrained(
    "username/model-name",
    backend="hugging-face"
)
```

## Complete Examples

### Example 1: YOLOv8 Object Detection (ONNX)

```
yolov8-custom/
├── model_config.json
└── weights.onnx
```

`model_config.json`:
```json
{
  "model_architecture": "yolov8",
  "task_type": "object-detection",
  "input_width": 640,
  "input_height": 640,
  "num_classes": 3,
  "class_names": ["cat", "dog", "bird"]
}
```

Usage:
```python
from inference_models import AutoModel

model = AutoModel.from_local_package("./yolov8-custom", backend="onnx")
results = model.infer("image.jpg")
```

### Example 2: ResNet Classification with Custom Preprocessing

```
resnet-custom/
├── model_config.json
├── inference_config.json
└── weights.onnx
```

`model_config.json`:
```json
{
  "model_architecture": "resnet",
  "task_type": "classification",
  "input_width": 224,
  "input_height": 224,
  "num_classes": 10,
  "class_names": ["class0", "class1", ..., "class9"]
}
```

`inference_config.json`:
```json
{
  "mean": [0.5, 0.5, 0.5],
  "std": [0.5, 0.5, 0.5],
  "normalize": true
}
```

## Validation

To validate your package structure:

```python
from inference_models.models.auto_loaders.package_loader import validate_package

validate_package("./my-model-package")
```

This will check:
- Required files exist
- `model_config.json` is valid JSON with required fields
- Model architecture and task type are supported
- Weights file exists and has correct extension

## Troubleshooting

### "Missing model_config.json"
Ensure `model_config.json` exists in the package root directory.

### "Unsupported model architecture"
Check that `model_architecture` in `model_config.json` matches a registered architecture. See [Models Overview](../models/index.md) for supported architectures.

### "Backend not available"
Install the required backend dependencies. See [Installation Guide](../getting-started/installation.md).

### "TensorRT engine incompatible"
TensorRT engines must be built on the target GPU. Rebuild the engine on your deployment hardware.

## Next Steps

- [Models Overview](../models/index.md) - See all supported model architectures
- [Installation Guide](../getting-started/installation.md) - Install backend dependencies
- [How-To: Load Local Packages](../how-to/local-packages.md) - Detailed usage examples

