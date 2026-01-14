# Moondream2 - Vision Language Model

Moondream2 is a compact vision-language model designed for efficient multimodal understanding with specialized capabilities for detection, captioning, and visual question answering.

## Overview

Moondream2 is a lightweight VLM with unique capabilities:

- **Object Detection** - Detect objects through natural language queries
- **Visual Question Answering** - Answer questions about image content
- **Image Captioning** - Generate captions with adjustable length
- **Point Detection** - Locate specific objects and return coordinates
- **Efficient Design** - Small model size for faster inference

!!! warning "Device Compatibility"
    Moondream2 **cannot run on Apple devices with MPS** due to a bug in the original implementation. Use NVIDIA GPU or x86 CPU instead.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [vikhyatk/moondream](https://github.com/vikhyat/moondream)

## Pre-trained Model IDs

Moondream2 pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `moondream2` | Latest version - general purpose vision-language model |

You can also use fine-tuned models from Roboflow by specifying `project/version` as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128` |

!!! warning "MPS Not Supported"
    Moondream2 does not support Apple MPS acceleration. Use CPU or CUDA only.

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA fine-tuning only |
| **Upload Weights** | ✅ Upload fine-tuned models |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Moondream2 block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Supported Tasks

Moondream2 supports multiple vision-language tasks through specialized methods:

| Task | Method | When to Use |
|------|--------|-------------|
| Object Detection | `detect()` | Detect and locate objects by specifying class names |
| Visual Question Answering | `prompt()` | Answer questions about image content |
| Image Captioning | `caption()` | Generate descriptive captions with adjustable length |
| Point Detection | `point()` | Locate specific objects and return their coordinates |

## Usage Examples

### Object Detection

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("moondream2")
image = cv2.imread("path/to/image.jpg")

# Detect objects by class name
detections = model.detect(
    images=image,
    classes=["person", "car", "dog"],
    max_tokens=700
)

# Access detection results
for detection in detections[0]:
    print(f"Class: {detection.class_name}")
    print(f"Confidence: {detection.confidence}")
    print(f"Box: {detection.xyxy}")
```

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("moondream2")
image = cv2.imread("path/to/image.jpg")

# Ask a question
answers = model.query(
    images=image,
    question="What is the person doing?",
    max_tokens=700
)
print(f"Answer: {answers[0]}")
```

### Image Captioning

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("moondream2")
image = cv2.imread("path/to/image.jpg")

# Generate caption with adjustable length
captions = model.caption(
    images=image,
    length="normal",  # Options: "short", "normal", "long"
    max_tokens=700
)
print(f"Caption: {captions[0]}")
```

### Point Detection

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("moondream2")
image = cv2.imread("path/to/image.jpg")

# Find specific object location
points = model.point(
    images=image,
    object="the red car",
    max_tokens=700
)

# Access point coordinates
for point in points[0]:
    print(f"Location: x={point.x}, y={point.y}")
    print(f"Confidence: {point.confidence}")
```

### Using Fine-tuned Models

```python
import cv2
from inference_models import AutoModel

# Load your fine-tuned model from Roboflow
model = AutoModel.from_pretrained(
    "your-project/version",
    api_key="your_roboflow_api_key"
)

image = cv2.imread("path/to/image.jpg")

# Use any of the methods above
answers = model.query(
    images=image,
    question="your custom question",
    max_tokens=700
)
print(f"Answer: {answers[0]}")
```

## Workflows Integration

Moondream2 can be used in Roboflow Workflows for complex computer vision pipelines.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use CUDA GPU** - Moondream2 benefits from GPU acceleration (MPS not supported)
2. **Adjust max_tokens** - Default is 700; increase for more detailed responses
3. **Use specific prompts** - Clear object names in `detect()` and `point()` yield better results
4. **Choose caption length** - Use "short" for speed, "long" for detail
5. **Batch processing** - Process multiple images by passing a list

