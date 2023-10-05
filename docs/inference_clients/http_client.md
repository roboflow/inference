# `InferenceHTTPClient`

`InferenceHTTPClient` was created to make it easy for users to consume HTTP API exposed by `inference` server. You
can think of it, as a friendly wrapper over `requests` that you can use, instead of creating calling logic on
your own.

## 🔥 quickstart
```python
from inference_clients.http.client import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

#Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
predictions = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

print(predictions)
```

## What are the client capabilities?
* Executing inference for models hosted at Roboflow platform (use client version `v0`)
* Executing inference for models hosted in local (or on-prem) docker images with `inference` HTTP API

## Why client has two modes - `v0` and `v1`?
We are constantly improving our `infrence` package - initial version (`v0`) is compatible with
models deployed at Roboflow platform (task types: `classification`, `object-detection` and `instance-segmentation`)
are supported. Version `v1` is available in locally hosted Docker images with HTTP API. 

Locally hosted `inference` server exposes endpoints for model manipulations, but those endpoints are not available
at the moment for models deployed at Roboflow platform.

`api_url` parameter passed to `InferenceHTTPClient` will decide on default client mode - URLs with `*.roboflow.com`
will be defaulted to version `v0`.
