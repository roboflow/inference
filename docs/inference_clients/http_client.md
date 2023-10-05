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
