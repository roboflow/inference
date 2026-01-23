# Depth Anything V3 - Monocular Depth Estimation

Depth Anything V3 is the latest version of the Depth Anything series, offering improved monocular depth estimation from single RGB images.

## Overview

Depth Anything V3 builds upon V2 with enhanced architecture and training for better depth prediction. Key capabilities include:

- **Monocular Depth Estimation** - Predict depth from a single image
- **Improved Accuracy** - Enhanced performance over V2
- **Multiple Model Sizes** - Choose from small or base variants
- **Zero-Shot Generalization** - Works on diverse image types without fine-tuning

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Depth-Anything-V3](https://github.com/ByteDance-Seed/Depth-Anything-3)<br>**Paper**: [Depth Anything V3](https://arxiv.org/abs/2511.10647)

!!! note "Architecture Differences"
    Unlike V2, V3 uses a custom architecture that is not HuggingFace Transformers compatible. The model uses PyTorch backend with vendored architecture components.

## Pre-trained Model IDs

Depth Anything V3 pre-trained models are available and **require a Roboflow API key**.

| Model ID | Description |
|----------|-------------|
| `depth-anything-v3/small` | Small model - fastest inference |
| `depth-anything-v3/base` | Base model - better accuracy |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Depth Anything block |
| **Edge Deployment (Jetson)** | ⚠️ Limited support (potential VRAM issues) |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Basic Depth Estimation

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained("depth-anything-v3/small", api_key="your_roboflow_api_key")

# Load image
image = cv2.imread("path/to/image.jpg")

# Predict depth
depth_maps = model(image)
depth_map = depth_maps[0].cpu().numpy()

# Normalize for visualization
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

# Save visualization
cv2.imwrite("depth_map.jpg", depth_colored)
```

### Processing Multiple Images

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("depth-anything-v3/base", api_key="your_roboflow_api_key")

# Load multiple images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Batch inference
depth_maps = model(images)

# Process each depth map
for i, depth_map in enumerate(depth_maps):
    depth_np = depth_map.cpu().numpy()
    print(f"Image {i}: depth shape = {depth_np.shape}")
```

## Workflows Integration

Depth Anything V3 can be used in Roboflow Workflows for complex computer vision pipelines. The Depth Anything block outputs depth maps that can be used for 3D reconstruction, scene understanding, and more.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Depth estimation models require GPU for acceptable performance
2. **Choose the right model** - Use `small` for speed, `base` for accuracy
3. **Batch processing** - Process multiple images together when possible
4. **Normalize outputs** - Depth maps are relative; normalize for visualization or comparison

## Output Format

The model returns a list of PyTorch tensors, one per input image:
- **Shape**: `(height, width)` - same as input image dimensions
- **Values**: Relative depth values (not metric depth)
- **Range**: Varies per image; normalize for visualization

## Common Use Cases

- **3D Reconstruction** - Estimate scene geometry from images
- **Augmented Reality** - Understand scene depth for AR applications
- **Robotics** - Navigation and obstacle avoidance
- **Photography** - Depth-based effects and bokeh simulation
- **Scene Understanding** - Analyze spatial relationships in images

