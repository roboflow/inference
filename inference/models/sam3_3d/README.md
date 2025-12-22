# SAM3-3D Model

3D object generation model that converts 2D images with masks into 3D assets (meshes and Gaussian splats).

## Installation

```bash
pip install --no-cache-dir --no-build-isolation -r requirements/requirements.sam3_3d.txt
```

## Docker

```
docker/dockerfiles/Dockerfile.onnx.gpu.3d
```

## Input

- **image**: RGB image (PIL Image, numpy array, or inference request format)
- **mask_input**: Mask(s) defining object regions. Supports multiple formats:

| Format | Single | Multiple |
|--------|--------|----------|
| Binary mask | `np.ndarray` (H, W) | `np.ndarray` (N, H, W) or `List[np.ndarray]` |
| Polygon (COCO flat) | `[x1, y1, x2, y2, ...]` | `[[x1, y1, ...], [x1, y1, ...]]` |
| Polygon (points) | `[[x1, y1], [x2, y2], ...]` | N/A |
| RLE | `{"counts": "...", "size": [H, W]}` | `[{"counts": ...}, ...]` |
| sv.Detections | From SAM2 or other segmentation models | Extracts all masks |

## Output

- **mesh_glb**: Combined 3D scene mesh (GLB binary)
- **gaussian_ply**: Combined Gaussian splatting (PLY binary)
- **objects**: List of individual objects:
  - `mesh_glb`: Object mesh (GLB binary)
  - `gaussian_ply`: Object Gaussian splat (PLY binary)
  - `metadata`: `{rotation, translation, scale}`
- **time**: Inference time in seconds

## Keep in mind to set 'os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"' otherwise the model won't be available.

## Example

```python
import os
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"

from inference import get_model
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
import numpy as np

model = get_model("sam3-3d-objects", api_key="YOUR_API_KEY")
image = {"type": "file", "value": "path/to/image.jpg"}

# Option 1: Polygon (COCO flat format)
mask = [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]

# Option 2: Binary mask
mask = np.zeros((480, 640), dtype=np.uint8)
mask[100:200, 100:200] = 255

# Option 3: Multiple masks
mask = [
    [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],
    [300.0, 300.0, 400.0, 300.0, 400.0, 400.0, 300.0, 400.0],
]

request = Sam3_3D_Objects_InferenceRequest(image=image, mask_input=mask)
response = model.infer_from_request(request)

# Save outputs
if response.mesh_glb:
    with open("scene.glb", "wb") as f:
        f.write(response.mesh_glb)

for i, obj in enumerate(response.objects):
    if obj.mesh_glb:
        with open(f"object_{i}.glb", "wb") as f:
            f.write(obj.mesh_glb)
```
