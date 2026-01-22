# DeepLabV3+ - Semantic Segmentation

DeepLabV3+ is a state-of-the-art semantic segmentation model that assigns a class label to every pixel in an image, enabling precise scene understanding and object delineation.

## Overview

DeepLabV3+ is designed for pixel-level classification tasks:

- **Semantic Segmentation** - Classify every pixel in an image into predefined categories
- **Scene Understanding** - Understand the complete layout and composition of scenes
- **Custom Classes** - Train on your own dataset with custom semantic classes
- **Flexible Encoders** - Choose from multiple backbone architectures for different speed/accuracy trade-offs
- **Multi-Backend Support** - Run on PyTorch, ONNX, or TensorRT for optimal performance

!!! info "Architecture"
    **Paper**: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)<br>**Implementation**: Based on [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)<br>**License**: MIT

## Model Access

DeepLabV3+ models are **custom-trained only** and require a Roboflow API key. Train your own semantic segmentation model on [Roboflow](https://roboflow.com).

### Model IDs

Custom models use the format:

```
{project-id}/{version}
```

## Supported Backends

| Backend | Extras Required | When to Use |
|---------|----------------|-------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` | Development, maximum flexibility |
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` | Production, cross-platform deployment |
| `trt` | `trt10` | Maximum GPU performance on NVIDIA hardware |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train on Roboflow platform |
| **Upload Weights** | ✅ Upload custom weights |
| **Serverless API (v2)** | ✅ Available for inference |
| **Edge Deployment (Jetson)** | ✅ Supported with appropriate backend |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Examples

### Basic Inference

```python
import cv2
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained(
    "<project-id>/<version>",
    api_key="<your-api-key>"
)

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference
results = model(image)

# Access segmentation map and confidence
segmentation_map = results[0].segmentation_map  # Class ID for each pixel
confidence = results[0].confidence  # Confidence for each pixel
```

### Visualizing Results

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "<project-id>/<version>",
    api_key="<your-api-key>"
)

# Run inference
image = cv2.imread("path/to/image.jpg")
results = model(image)

# Get segmentation map
seg_map = results[0].segmentation_map.cpu().numpy()

# Create colored visualization
num_classes = len(model.class_names)
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
colored_mask = colors[seg_map]

# Overlay on original image
alpha = 0.5
overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

cv2.imwrite("segmentation_result.jpg", overlay)
```

### Batch Processing

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "<project-id>/<version>",
    api_key="<your-api-key>"
)

# Load multiple images
images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(10)]

# Run batch inference
results = model(images)

# Process each result
for i, result in enumerate(results):
    seg_map = result.segmentation_map
    confidence = result.confidence
    print(f"Image {i}: segmentation shape {seg_map.shape}")
```

## Output Format

The model returns a list of `SemanticSegmentationResult` objects with:

| Field | Type | Description |
|-------|------|-------------|
| `segmentation_map` | `torch.Tensor` | Class ID for each pixel (H x W) |
| `confidence` | `torch.Tensor` | Confidence score for each pixel (H x W) |
| `image_metadata` | `dict` | Optional metadata about the image |

## Performance Tips

1. **Choose the right encoder** - Balance speed and accuracy based on your use case
2. **Use appropriate backend** - ONNX for production, TensorRT for maximum GPU performance
3. **Batch processing** - Process multiple images together for better throughput
4. **Resize strategy** - Choose appropriate resize mode (stretch, letterbox, center crop)
5. **GPU acceleration** - Use CUDA-enabled backends for faster inference
6. **Model optimization** - Consider quantization or pruning for edge deployment

## Common Use Cases

### Autonomous Driving
- Road segmentation (road, sidewalk, lane markings)
- Vehicle and pedestrian detection
- Traffic sign and signal recognition

### Medical Imaging
- Organ segmentation in CT/MRI scans
- Tumor detection and delineation
- Cell segmentation in microscopy images

### Agriculture
- Crop and weed segmentation
- Disease detection on plants
- Field boundary detection

### Aerial/Satellite Imagery
- Land use classification
- Building and road extraction
- Vegetation mapping

### Industrial Inspection
- Defect segmentation
- Surface quality assessment
- Component identification

## Comparison with Instance Segmentation

| Feature | Semantic Segmentation | Instance Segmentation |
|---------|----------------------|----------------------|
| **Output** | Class per pixel | Individual object masks |
| **Distinguishes instances** | ❌ No | ✅ Yes |
| **Use case** | Scene understanding | Object counting, tracking |
| **Example** | All cars labeled as "car" | Each car gets unique ID |
| **Models** | DeepLabV3+ | YOLOv8-Seg, RF-DETR-Seg |

!!! note "Choosing the Right Task"
    - Use **semantic segmentation** when you need to understand scene composition
    - Use **instance segmentation** when you need to count or track individual objects

