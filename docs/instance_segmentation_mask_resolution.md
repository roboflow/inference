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

With the opt-in, contours can be extracted from reduced masks, but polygon
points and bounding boxes always use the original image's pixel coordinates.
The server scales contour coordinates before returning polygon predictions;
callers do not need to transform the points.

RLE responses retain the encoded mask grid. When its dimensions differ from
the response image, `mask_metadata` explicitly describes that RLE grid:

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
model or factor. RLE `size` is `[height, width]` from `mask_metadata`.
The scales map positions on the encoded RLE grid to image coordinates.
Metadata is omitted for polygon responses and for RLE grids matching the
image. Empty RLE predictions still include metadata when their grid differs.
Each image in a batch has its own metadata.

Polygon responses can be passed directly to Supervision:

```python
import supervision as sv

detections = sv.Detections.from_inference(result)
```

Decode RLE on its declared grid; never replace its `size` with image dimensions
without resampling the mask. During client-side image resizing, the SDK scales
polygon points and bounding boxes together. It preserves encoded RLE data and
updates its metadata scales to the client image. Server-rendered polygon
visualizations use the returned points directly.

Direct `inference_models` calls keep their `masks_resolution_factor` interface.
The legacy model backend retains its existing decode-mode behavior and ignores
the new opt-in flag. Values from 0 to 1 interpolate model-grid and image-grid
dimensions; a lower factor does not always mean a smaller grid or faster execution.
Damian's Triton optimization is separate; the existing fallback remains for now.

Workflow versions v1–v4 retain their compatibility guards. The new v5 block,
reduced-mask Supervision/workflow handling, and full tensor support are deferred
to a separate PR. This change does not introduce reduced-grid workflow outputs.
