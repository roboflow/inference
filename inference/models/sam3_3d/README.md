# SAM3-3D Model

3D object generation model that converts 2D images with polygon masks into 3D assets (meshes and Gaussian splats).

## Installation

Install dependencies from the project root:

```bash
pip install --no-cache-dir --no-build-isolation -r requirements/requirements.sam3_3d.txt
```

## Docker

A GPU container for running the model is available at:

```
docker/dockerfiles/Dockerfile.onnx.gpu.3d
```

## Input

- **image**: RGB image (PIL Image, numpy array, or inference request format)
- **mask_input**: Polygon mask(s) defining object regions in COCO polygon format:
  - Single mask: flat list of coordinates `[x1, y1, x2, y2, x3, y3, ...]`
  - Multiple masks: list of flat lists `[[x1, y1, ...], [x1, y1, ...], ...]`

## Output

Returns per request:
- **mesh_glb**: Combined 3D scene mesh in GLB format (binary)
- **gaussian_ply**: Combined Gaussian splatting data in PLY format (binary)
- **objects**: List of individual 3D objects, each containing:
  - `mesh_glb`: Individual object mesh (GLB binary)
  - `gaussian_ply`: Individual object Gaussian splat (PLY binary)
  - `metadata`: Transformation data (rotation quaternion, translation, scale)
- **time**: Inference time in seconds

## Example

```python
import os
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"

from inference import get_model
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest

# Load model
model = get_model("sam3-3d-objects", api_key="YOUR_API_KEY")

# Prepare image and mask (COCO polygon format)
image = {"type": "file", "value": "path/to/image.jpg"}
mask_polygon = [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0] #Your actaul polygon here, single object
# Or multiple objects: [[x1, y1, ...], [x1, y1, ...]]

# Run inference
request = Sam3_3D_Objects_InferenceRequest(image=image, mask_input=mask_polygon)
response = model.infer_from_request(request)

print(f"Inference completed in {response.time:.2f} seconds")

# Save outputs
if response.mesh_glb:
    with open("scene.glb", "wb") as f:
        f.write(response.mesh_glb)

if response.gaussian_ply:
    with open("scene.ply", "wb") as f:
        f.write(response.gaussian_ply)

# Individual objects
for i, obj in enumerate(response.objects):
    if obj.mesh_glb:
        with open(f"object_{i}.glb", "wb") as f:
            f.write(obj.mesh_glb)
```
