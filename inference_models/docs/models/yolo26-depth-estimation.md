# YOLO26 - Depth Estimation

YOLO26 is the latest addition to the Ultralytics YOLO model series. The depth estimation variant predicts a **metric** depth value in meters for every pixel, producing a dense monocular depth map from a single RGB image.

## Overview

YOLO26 for depth estimation pairs the efficient YOLO26 backbone with a dense log-depth prediction head. Key features include:

- **Metric depth in meters** - The log-depth head predicts unbounded absolute depth (~0.02-150 m), unlike relative-depth models such as Depth Anything.
- **Efficient YOLO26 backbone** - Shares the NMS-free, DFL-free YOLO26 architecture for fast inference and broad edge compatibility.
- **Broadly pre-trained checkpoints** - Public weights trained on ~2.19M indoor and outdoor images, evaluated on NYU Depth V2, available across all model sizes.
- **768×768 inference** - Checkpoints are trained and exported at 768×768; the head predicts at input/4 resolution and upsamples inside the exported graph.
- **Multiple model sizes** - From nano to extra-large variants.

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLO26 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLO26 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLO26 License Details](https://roboflow.com/model-licenses/yolo26)

## Pre-trained Model IDs

Public YOLO26 depth estimation checkpoints are trained at 768×768 and are **open access** (no API key required).

| Model Size | 768×768 |
|------------|---------|
| Nano | `yolo26n-depth-768` |
| Small | `yolo26s-depth-768` |
| Medium | `yolo26m-depth-768` |
| Large | `yolo26l-depth-768` |
| Extra-Large | `yolo26x-depth-768` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch-script` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not yet available |
| **Upload Weights** | ❌ Not yet available |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load a public checkpoint (open access, no API key required)
model = AutoModel.from_pretrained("yolo26n-depth-768")
image = cv2.imread("path/to/image.jpg")

# Run inference - depth in meters at the original image resolution
results = model(image)
depth_meters = results[0]  # torch.Tensor, (H x W), float32

print(f"Depth range: {depth_meters.min():.2f}m - {depth_meters.max():.2f}m")

# Colormap for visualization (near = bright, matching Depth Anything renders)
depth = depth_meters.cpu().numpy()
normalized = (depth.max() - depth) / (depth.max() - depth.min())
colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cv2.imwrite("depth_result.jpg", colored)
```

## Output Format

The model returns a list of `torch.Tensor` depth maps, one per input image:

| Field | Type | Description |
|-------|------|-------------|
| `results[i]` | `torch.Tensor` | Metric depth in meters for each pixel (H x W, float32), at the original image resolution |

!!! note "Metric vs relative depth"
    `AutoModel` returns **absolute metric depth in meters**. When served through the
    Roboflow hosted API or the `depth_estimation@v1` workflow block, the output is
    min-max normalized per image to the disparity-style convention shared with
    Depth Anything (1.0 = nearest, 0.0 = farthest) so the models are drop-in
    interchangeable; the metric scale is not exposed on that path.

!!! note "Choosing between YOLO26 depth and Depth Anything"
    YOLO26 predicts at input/4 native resolution and is substantially faster at a
    matched budget, with metric calibration; [Depth Anything V3](depth-anything-v3.md)
    decodes at full network resolution, producing sharper relative-depth maps at
    higher compute cost. Prefer YOLO26 for speed and absolute distances; prefer
    Depth Anything for edge fidelity in relative maps.
