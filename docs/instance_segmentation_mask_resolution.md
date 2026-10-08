# Opt into configurable instance-segmentation masks

On servers running `USE_INFERENCE_MODELS=True`, existing requests keep masks at
image resolution. To use `mask_decode_mode="fast"` or `"tradeoff"`, also set
`allow_reduced_mask_resolution=true`. This applies to both
`POST /infer/instance_segmentation` (JSON body) and the legacy
`POST /{dataset_id}/{version_id}` route (query parameters).

```python
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

client = InferenceHTTPClient(api_url="http://localhost:9001").configure(
    InferenceConfiguration(
        allow_reduced_mask_resolution=True,
        mask_decode_mode="tradeoff",
        tradeoff_factor=0.5,
        response_mask_format="polygon",  # or "rle"
    )
)
result = client.infer("image.jpg", model_id="yolov8n-seg-640")
```

With the opt-in, polygons and RLE use the same mask grid. When its dimensions
differ from the response image, `mask_metadata` explicitly describes that grid:

```json
{
  "image": {"width": 640, "height": 480},
  "mask_metadata": {
    "coordinate_system": "mask_grid",
    "width": 160,
    "height": 120,
    "scale_x": 4.0,
    "scale_y": 4.0
  }
}
```

These dimensions illustrate the contract, not an expected result for every
model or factor. `image` and bounding boxes remain in image coordinates.
Polygon points are in mask coordinates; RLE `size` is `[height, width]` from
`mask_metadata`. Multiply polygon x by `scale_x` and y by `scale_y` to recover
image coordinates. The metadata is omitted when the grids agree, including
normal accurate-mode responses. Empty predictions still include grid metadata
when their mask grid differs. Each image in a batch has its own metadata.

Existing Supervision conversion does not consume the new polygon metadata.
Restore polygon coordinates before passing an opted-in polygon response to
`sv.Detections.from_inference`:

```python
from copy import deepcopy
import supervision as sv

image_result = deepcopy(result)
metadata = image_result.get("mask_metadata")
if metadata:
    for prediction in image_result["predictions"]:
        for point in prediction.get("points", []):
            point["x"] *= metadata["scale_x"]
            point["y"] *= metadata["scale_y"]
    image_result.pop("mask_metadata")
detections = sv.Detections.from_inference(image_result)
```

Decode RLE on its declared grid; never replace its `size` with image dimensions
without resampling the mask. The SDK preserves mask-grid points and RLE during
client-side image resizing, adjusting bounding boxes and metadata scales to the
client image. Server-rendered polygon visualizations restore image coordinates
without changing the returned points.

Direct `inference_models` calls keep their `masks_resolution_factor` interface.
The legacy model backend retains its existing decode-mode behavior and ignores
the new opt-in flag. Values from 0 to 1 interpolate model-grid and image-grid
dimensions; a lower factor does not always mean a smaller grid or faster execution.
Damian's Triton optimization is separate; the existing fallback remains for now.

Workflow versions v1–v4 retain their compatibility guards. The new v5 block,
reduced-mask Supervision/workflow handling, and full tensor support are deferred
to a separate PR. This change does not introduce reduced-grid workflow outputs.
