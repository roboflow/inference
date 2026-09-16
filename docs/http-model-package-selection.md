# Select a model package over HTTP

Model package selection requires a server with `USE_INFERENCE_MODELS=true`.
Explicit selection requires the `inference-models` release that exposes the resolved package descriptor and an updated dependency pin.
With an older library, ordinary inference works, but explicit selection fails because the server cannot verify the loaded package.
Before that release, a Docker build with the updated library source can validate the change.
A request can specify `model_package_id`, or `backend`, `quantization`, or both.
The server rejects a package ID combined with either other selector.
A successful request uses a package that satisfies all explicit selectors.
An unavailable, incompatible, or disabled package produces an error.

| SDK configuration field | Selection |
| --- | --- |
| `model_package_id` | Exact package ID; cannot accompany either other field. |
| `backend` | Required backend, such as `onnx` or `trt`. |
| `quantization` | Required precision, such as `fp32` or `fp16`; can accompany `backend`. |

Malformed or conflicting selectors return HTTP 422.
Unsupported selector values and unavailable backend/precision combinations return HTTP 400.
Unknown package IDs return HTTP 404 with `Model or model package not found.`

Standard computer vision endpoints accept these fields in the v1 JSON body.
The legacy `/{project}/{version}` endpoint accepts them as query parameters.
`/model/add` and `/model/remove` also accept them in the JSON body.

```python
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

client = InferenceHTTPClient(api_url="http://localhost:9001", api_key="YOUR_API_KEY")
with client.use_configuration(InferenceConfiguration(backend="trt", quantization="fp16")):
    result = client.infer("image.jpg", model_id="project/1")
    client.unload_model("project/1")

with client.use_configuration(InferenceConfiguration(model_package_id="PACKAGE_ID")):
    client.load_model("project/1")
    result = client.infer("image.jpg", model_id="project/1")
```

The configuration applies to synchronous and asynchronous SDK calls.
For inference and model loading, the SDK requires a server acknowledgment for explicit selection.
It raises an error if an older server silently ignores the selectors.
HTTP success responses include `X-Roboflow-Model-Selection: applied` for selected inference and model loading.

Each model ID, selector combination, and client credential identifies a separate cache entry.
Each entry counts toward the existing LRU limit.
The registry exposes its cache handle as `model_id`.
The server generates opaque handles with a private random key.
`/model/add` returns `selected_model_id` for explicit selections; the SDK uses this identifier to find the loaded model.
Handles can change after a server restart. They cannot be used for inference or model loading.
Two selector combinations can load separate instances of the same package.
Requests without selectors retain their existing automatic selection and cache entry.
Registration of a preferred package does not replace a resident instance.

`unload_model()` removes the entry for the active configuration.
The SDK sends the public model ID, selectors, and credential so the server can find the current entry.
`/model/remove` accepts the original model ID with the same selectors and credential, or the registry handle without selectors.
An active request retains its model reference until inference completes.
Concurrent package loads still require sufficient device memory.
