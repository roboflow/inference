# SAM3-3D (3D Object Generation)

SAM3-3D is a 3D object generation model that converts 2D images with masks into 3D assets, including meshes (GLB format) and Gaussian splats (PLY format).

!!! warning "Beta Feature"

    This model is currently in Beta state. It is only available when the `SAM3_3D_OBJECTS_ENABLED` environment flag is enabled.

!!! warning "Hardware Requirements"

    SAM3-3D requires a GPU with **32GB+ VRAM** to run.

## How to Use SAM3-3D with Inference

You can use SAM3-3D via the Inference Python SDK or run it in Docker.

### Prerequisites

To use SAM3-3D, you will need:

1. A Roboflow API key. [Sign up for a free Roboflow account](https://app.roboflow.com) to retrieve your key.
2. A GPU with 32GB+ VRAM
3. Python 3.10 (recommended)

### Installation

```bash
pip install --no-cache-dir --no-build-isolation -r requirements/requirements.sam3_3d.txt
```

### Environment Configuration

You must enable the SAM3-3D feature flag. For optimal performance, also configure the attention backends:

```python
import os
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"
os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
os.environ["ATTN_BACKEND"] = "flash_attn"
```

### Python SDK

Here is an example of how to use SAM3-3D to generate 3D objects from an image with masks.

```python
import os
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"
os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
os.environ["ATTN_BACKEND"] = "flash_attn"

from inference.models.sam3_3d.segment_anything_3d import SegmentAnything3_3D_Objects
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import Sam3_3D_Objects_Response
from inference import get_model
import json

def load_coco_masks(annotations_path, image_id):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] == image_id
    ]

    segmentations = [ann['segmentation'] for ann in annotations]

    return segmentations

image_path = "path/to/your/image.jpg"
annotations_path = "path/to/annotations.coco.json"
image_id = 0

mask_polygon = load_coco_masks(annotations_path, image_id)

image = {
    "type": "file",
    "value": image_path
}

model = get_model("sam3-3d-objects", api_key="<YOUR_ROBOFLOW_API_KEY>")

request = Sam3_3D_Objects_InferenceRequest(
    image=image,
    mask_input=mask_polygon,
)

print("Running SAM3_3D inference...")
response: Sam3_3D_Objects_Response = model.infer_from_request(request)

print(f"SAM3_3D inference completed in {response.time:.2f} seconds!")

# Save Scene Mesh (GLB)
if response.mesh_glb is not None:
    with open("out_mesh.glb", "wb") as f:
        f.write(response.mesh_glb)
    print(f"Saved mesh to out_mesh.glb ({len(response.mesh_glb):,} bytes)")

# Save Combined Gaussian Splatting (PLY)
if response.gaussian_ply is not None:
    with open("out_gaussian.ply", "wb") as f:
        f.write(response.gaussian_ply)
    print(f"Saved gaussian to out_gaussian.ply ({len(response.gaussian_ply):,} bytes)")

# Save Individual Objects
for i, obj in enumerate(response.objects):
    if obj.mesh_glb is not None:
        with open(f"out_object_{i}_mesh.glb", "wb") as f:
            f.write(obj.mesh_glb)

    if obj.gaussian_ply is not None:
        with open(f"out_object_{i}_gaussian.ply", "wb") as f:
            f.write(obj.gaussian_ply)
```

### Docker

You can build and run SAM3-3D using Docker:

#### 1. Build the Docker Image

```bash
docker build -t roboflow/roboflow-inference-server-gpu:dev -f docker/dockerfiles/Dockerfile.onnx.gpu.3d .
```

#### 2. Run the Container

```bash
docker run --gpus all -p 9001:9001 roboflow/roboflow-inference-server-gpu:dev
```

## Input Format

SAM3-3D accepts the following inputs:

- **image**: RGB image (PIL Image, numpy array, or inference request format)
- **mask_input**: Mask(s) defining object regions. Supports multiple formats:

| Format | Single | Multiple |
|--------|--------|----------|
| Binary mask | `np.ndarray` (H, W) | `np.ndarray` (N, H, W) or `List[np.ndarray]` |
| Polygon (COCO flat) | `[x1, y1, x2, y2, ...]` | `[[x1, y1, ...], [x1, y1, ...]]` |
| Polygon (points) | `[[x1, y1], [x2, y2], ...]` | N/A |
| RLE | `{"counts": "...", "size": [H, W]}` | `[{"counts": ...}, ...]` |
| sv.Detections | From SAM2 or other segmentation models | Extracts all masks |

## Output Format

SAM3-3D returns:

- **mesh_glb**: Combined 3D scene mesh (GLB binary)
- **gaussian_ply**: Combined Gaussian splatting (PLY binary)
- **objects**: List of individual objects, each containing:
    - `mesh_glb`: Object mesh (GLB binary)
    - `gaussian_ply`: Object Gaussian splat (PLY binary)
    - `metadata`: Object transform data (`rotation`, `translation`, `scale`)
- **time**: Inference time in seconds

## Workflow Integration

SAM3-3D can be used in [Inference Workflows](https://inference.roboflow.com/workflows/core_steps/) as part of a local inference server, allowing you to chain it with other models like SAM2 for mask generation.
