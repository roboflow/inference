# Resolved model metadata

When `USE_INFERENCE_MODELS=true`, standard computer vision HTTP responses include
`resolved_model` if the loaded model has package metadata. The metadata identifies
the model instance that produced that result.

To receive metadata, install an `inference-models` release that provides it
and update the server dependency pin.
With an older library, ordinary inference works and omits `resolved_model`.
For source-based testing, install this repository's `inference_models/` package
into the server environment.

```json
{
  "image": {"width": 640, "height": 480},
  "predictions": [],
  "resolved_model": {
    "model_id": "project/3",
    "model_package_id": "trtpackage",
    "backend": "trt",
    "quantization": "fp16"
  }
}
```

| Field | Meaning |
| --- | --- |
| `model_id` | The canonical model ID returned by the weights provider. An input alias can differ. |
| `model_package_id` | The ID of the package that loaded successfully. |
| `backend` | The backend of that package, such as `onnx` or `trt`. |
| `quantization` | The package quantization, such as `fp32`, `fp16`, or `unknown` when unavailable. |

Quantization describes the package. It does not describe the input tensor dtype
or guarantee that every runtime operation uses that precision.

The v0 `/{dataset_id}/{version_id}` endpoint and standard v1 `/infer/...` endpoints
return this metadata. Supported tasks are object detection, classification,
instance segmentation, semantic segmentation, and keypoint detection. JSON results
include the metadata. Responses that contain only a rendered image do not.

Each result in a v1 batch has its own descriptor. The Python HTTP SDK preserves
this field for both API versions, including batches of inference inputs.

If a preferred package fails to load and another package succeeds, the response
identifies the successful package. A cache reload preserves its package identity.
An unrelated model removal or replacement does not change metadata for an active request.

When `USE_INFERENCE_MODELS=false`, responses omit `resolved_model`. Models without
package identity, including direct local-path loads, also omit the field.
