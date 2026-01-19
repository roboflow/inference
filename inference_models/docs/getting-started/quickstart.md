# Quick Start Guide

Get up and running with `inference-models` in minutes.

## Installation

```bash
# CPU installation
pip install inference-models

# GPU installation with ONNX support
pip install "inference-models[torch-cu128,onnx-cu12]"
```

See the [Installation Guide](installation.md) for more options.

## Basic Usage

### Load and Run a Model

```python
from inference_models import AutoModel
import cv2

# Load a pre-trained model
model = AutoModel.from_pretrained("rfdetr-base")

# Load an image
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)

# Access results
for detection in predictions:
    print(f"Detected: {detection.class_id} with confidence {detection.confidence}")
```

### Integration with Supervision

The library integrates seamlessly with [Supervision](https://github.com/roboflow/supervision):

```python
from inference_models import AutoModel
import cv2
import supervision as sv

# Load model
model = AutoModel.from_pretrained("yolov8n-640")

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)

# Convert to Supervision format
detections = predictions[0].to_supervision()

# Annotate image
annotator = sv.BoxAnnotator()
annotated_image = annotator.annotate(image.copy(), detections)

# Save or display
cv2.imwrite("output.jpg", annotated_image)
```

## Working with Different Model Types

### Object Detection

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)

# predictions is a list of Detections objects
for detection in predictions[0]:
    print(f"Box: {detection.xyxy}")
    print(f"Class: {detection.class_id}")
    print(f"Confidence: {detection.confidence}")
```

### Instance Segmentation

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-seg-640")
predictions = model(image)

# predictions is a list of InstanceDetections objects
for detection in predictions[0]:
    print(f"Box: {detection.xyxy}")
    print(f"Mask: {detection.mask.shape}")
    print(f"Class: {detection.class_id}")
```

### Classification

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("resnet50-imagenet")
prediction = model(image)

print(f"Top class: {prediction.class_ids[0]}")
print(f"Confidence: {prediction.confidence[0]}")
```

### Embeddings

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("clip/ViT-B-32")

# Image embeddings
image_embedding = model.embed_image(image)

# Text embeddings
text_embedding = model.embed_text("a photo of a cat")

# Compare similarity
similarity = model.compare(image, "a photo of a cat")
```

## Batch Processing

Process multiple images efficiently:

```python
from inference_models import AutoModel
import cv2

model = AutoModel.from_pretrained("yolov8n-640")

# Load multiple images
images = [
    cv2.imread("image1.jpg"),
    cv2.imread("image2.jpg"),
    cv2.imread("image3.jpg"),
]

# Batch inference
predictions = model(images)

# predictions is a list with one result per image
for i, pred in enumerate(predictions):
    print(f"Image {i}: {len(pred.xyxy)} detections")
```

## Model Information

### Describe a Model

Get detailed information about a model before loading:

```python
from inference_models import AutoModel

# Show model details
AutoModel.describe_model("rfdetr-base")
```

This displays:
- Available model packages
- Backend requirements
- Model architecture
- Task type
- Package sizes

### List Available Models

```python
from inference_models import AutoModel

# List all registered models
AutoModel.list_available_models()
```

### Runtime Information

Check your environment and available backends:

```python
from inference_models import AutoModel

# Show runtime environment details
AutoModel.describe_runtime()
```

## Loading from Different Sources

### From Roboflow (Default)

```python
model = AutoModel.from_pretrained("rfdetr-base")
```

### From Local Directory

```python
model = AutoModel.from_pretrained(
    "/path/to/model/directory",
    weights_provider="local"
)
```

### With API Key

For private models on Roboflow:

```python
model = AutoModel.from_pretrained(
    "your-workspace/your-model/version",
    api_key="your_api_key"
)
```

## Advanced Options

### Specify Backend

Force a specific backend:

```python
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend_type="onnx"  # or "torch", "trt"
)
```

### Custom Device

```python
model = AutoModel.from_pretrained(
    "rfdetr-base",
    device="cuda:1"  # Use specific GPU
)
```

### Disable Caching

```python
model = AutoModel.from_pretrained(
    "yolov8n-640",
    use_auto_resolution_cache=False
)
```

## Next Steps

- [Understand Core Concepts](../how-to/understand-core-concepts.md) - Understand the design philosophy
- [Installation Guide](installation.md) - Detailed installation instructions
- [Models Overview](../models/index.md) - Explore all available models
- [How-To Guides](../how-to/local-packages.md) - Advanced usage patterns

