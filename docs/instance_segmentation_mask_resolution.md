# Opt into configurable instance-segmentation masks

On servers running `USE_INFERENCE_MODELS=True`, existing requests keep masks,
polygon points and bounding boxes in original-image coordinates. The HTTP
request and SDK configuration both default `allow_reduced_mask_resolution` to
`False`. Set it to `True` to return **all prediction geometry on the selected
mask grid**, regardless of response format.

This applies to `POST /infer/instance_segmentation` (JSON body) and
`POST /{dataset_id}/{version_id}` (query parameters).

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

For `mask_decode_mode="tradeoff"` and `tradeoff_factor=0.5`:

| Opt-in | Format | Returned coordinate frame |
|---|---|---|
| `False` | `rle` | RLE and boxes use original-image coordinates |
| `True` | `rle` | RLE and boxes use the selected mask grid |
| `False` | `polygon` | Polygon points and boxes use original-image coordinates |
| `True` | `polygon` | Polygon points and boxes use the selected mask grid |

The factor interpolates between the unpadded model mask grid and the image grid;
it does **not** multiply image dimensions or coordinates directly. With a
160×160 model grid and a 1000×800 image, factor `0.5` selects a 580×480 grid
without cropping. Boxes scale by `580/1000` horizontally and `480/800`
vertically. Polygon contours are extracted on that grid and stay there.
Static crops are aligned onto a canvas representing the original image; its
actual dimensions, including rounding, determine the output mapping.

## Response dimensions and mapping

With opt-in, `image` describes the output coordinate frame. `original_image`
retains the input dimensions and `mask_metadata` describes the mapping back to
that input. These fields are included for both formats, including empty results
and when the output grid equals the input dimensions. Each batch item carries
its own dimensions and mapping. Without opt-in, the extra fields are omitted.

```json
{
  "image": {"width": 580, "height": 480},
  "original_image": {"width": 1000, "height": 800},
  "mask_metadata": {
    "coordinate_system": "mask_grid",
    "width": 580,
    "height": 480,
    "scale_x": 1.7241379310344827,
    "scale_y": 1.6666666666666667
  }
}
```

All boxes and polygon points use `image` coordinates. RLE `size` is
`[image.height, image.width]`. To map geometry back to the input, multiply x
coordinates and widths by `scale_x`, and y coordinates and heights by `scale_y`.
Mapping coordinates does not recover detail lost through reduced mask resolution.

`sv.Detections.from_inference(result)` produces detections for the **output
grid**. To annotate the original image, first map boxes and polygon points back,
or resize decoded RLE masks and map their boxes. Do not annotate the original
scene directly with output-grid detections. Workflow integration of reduced
masks remains deferred.

When SDK client downsizing is enabled, the server selects the output grid from
the image it receives. The SDK preserves that grid, boxes, points and encoded
RLE; it updates only `original_image` and mapping scales to refer to the client
image. Server-rendered visualizations project predictions onto the server's
input image without changing the returned geometry.

The active-learning manager also projects its own copy back to image coordinates
before handing predictions to sampling and registration. RLE responses are converted
to the largest-contour polygon annotations used by that path; the HTTP response
retains its original RLE representation and output grid.

## Compatibility and follow-up work

Direct `inference_models` calls keep their `masks_resolution_factor` interface
and existing image-space box contract. The HTTP output-coordinate transformation
happens at the response boundary. The legacy model backend retains its existing
decode-mode behavior and ignores the new opt-in flag.

Values from 0 to 1 interpolate model-grid and image-grid dimensions. A lower
factor does not always mean a smaller grid or faster execution. `accurate`
selects factor 1.0, `fast` selects 0.0, and only `tradeoff` reads `tradeoff_factor`.
The RF-DETR Triton RLE post-processor supports native and upsampled mask grids
at every factor. It preserves image-space boxes and records the encoded grid in
`mask_size`, including for empty and deferred results. It uses the same target-size
rounding as the reference path. Antialiased downsampling, unsupported preprocessing
transforms, and other existing compatibility limits use the reference post-processor.
The existing Triton enablement flag still controls this path; no execution plan is
required. Sparse-record capacity limits are unchanged, including the existing
deferred-mode overflow error.

Workflow versions v1–v4 retain their compatibility guards. Workflow v5,
reduced-mask downstream handling, and full tensor support are deferred to a
separate PR. This change does not introduce reduced-grid workflow outputs.
