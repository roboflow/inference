# RF-DETR - Oriented Object Detection

RF-DETR Oriented 2X Large (preview) detects oriented boxes: each object is a rotated rectangle, and for classes that
define a front (such as a ship's bow or a forklift's forks) the box also says which way the object faces.

## Overview

- **Oriented boxes** - each prediction is `[cx, cy, width, height, theta]` in original-image pixels, `theta` in
  clockwise radians.
- **Class fronts** - classes with a heading are predicted with their front along the box's local +x axis; other
  classes keep the box as predicted. Which classes have a heading is fixed per trained model (its dataset version).
- **Preview** - one size, RF-DETR 2X Large, trained on the Roboflow platform from oriented-box datasets.

## License

The RF-DETR Oriented 2X Large weights are released under the
[Platform Model License 1.0](https://github.com/roboflow/rf-detr/blob/develop/rfdetr/platform/LICENSE.platform).

## Pre-trained Model IDs

There are no public pre-trained model IDs. Models are trained on the Roboflow platform from oriented-box projects.

**Custom model ID format:** `project-url/version` (e.g., `my-obb-project/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12` (CPU, CUDA and TensorRT execution providers) |
| `trt` | `trt10` (engines built by TensorRT compilation) |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow (preview) |
| **Self-Hosting** | ✅ Deploy with `inference-models` or the inference server (`/infer/oriented_object_detection`) |

## Input and Output

The package declares the dataset version's resize, which the runtime reproduces exactly as in training: `letterbox`
(padding in the declared value) or `center-crop`. Resize modes that change the aspect ratio are refused, since a
rotated rectangle cannot follow them.

Loading needs the model's frozen class heading policy, which the Roboflow weights provider supplies with the model
metadata (`oriented_box_metadata`); loading a package without it raises an error.

## Usage Example

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained("my-obb-project/2", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

detections = model(image, confidence=0.5)[0]  # OrientedDetections
for obb, name, score, has_front in zip(
    detections.obb.tolist(), detections.class_names, detections.confidence.tolist(), detections.heading.tolist()
):
    cx, cy, width, height, theta = obb
    print(name, round(score, 3), obb, "front along +x" if has_front else "no front")

# Supervision detections with the ordered corners in data["xyxyxyxy"]
sv_detections = detections.to_supervision()
```
