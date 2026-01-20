# Depth Anything V2 - Monocular Depth Estimation

Depth Anything V2 is a state-of-the-art monocular depth estimation model that predicts depth maps from single RGB images.

## Overview

Depth Anything V2 uses a vision transformer architecture to estimate relative depth from images. Key capabilities include:

- **Monocular Depth Estimation** - Predict depth from a single image
- **High Quality** - State-of-the-art performance on depth estimation benchmarks
- **Multiple Model Sizes** - Choose from small, base, or large variants
- **Zero-Shot Generalization** - Works on diverse image types without fine-tuning

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)<br>**Paper**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)

## Pre-trained Model IDs

Depth Anything V2 pre-trained models are available and **require a Roboflow API key**.

| Model ID | Description |
|----------|-------------|
| `depth-anything-v2/small` | Small model - fastest inference |
| `depth-anything-v2/base` | Base model - balanced performance |
| `depth-anything-v2/large` | Large model - best accuracy |
| `depth-anything-v2` | Alias for `depth-anything-v2/small` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `hf` (Hugging Face) | Included in base installation |

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
model = AutoModel.from_pretrained("depth-anything-v2/small", api_key="your_roboflow_api_key")

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
model = AutoModel.from_pretrained("depth-anything-v2/base", api_key="your_roboflow_api_key")

# Load multiple images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Batch inference
depth_maps = model(images)

# Process each depth map
for i, depth_map in enumerate(depth_maps):
    depth_np = depth_map.cpu().numpy()
    print(f"Image {i}: depth shape = {depth_np.shape}")
```

### Using the Large Model

```python
import cv2
from inference_models import AutoModel

# Load larger model for better accuracy
model = AutoModel.from_pretrained("depth-anything-v2/large", api_key="your_roboflow_api_key")

# Load image
image = cv2.imread("path/to/image.jpg")

# Predict depth
depth_maps = model(image)
depth_map = depth_maps[0]

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
```

## Workflows Integration

Depth Anything V2 can be used in Roboflow Workflows for complex computer vision pipelines. The Depth Anything block outputs depth maps that can be used for 3D reconstruction, scene understanding, and more.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Depth estimation models require GPU for acceptable performance
2. **Choose the right model** - Use `small` for speed, `large` for accuracy
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

