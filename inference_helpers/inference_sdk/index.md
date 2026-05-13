# Inference SDK

The `InferenceHTTPClient` enables you to interact with an [Inference Server](../quickstart/docker.md) over HTTP - hosted either by Roboflow or on your own hardware. `inference-sdk` can be installed via pip:

```bash
pip install inference-sdk
```

## Quickstart

You can run inference on images from URLs, file paths, PIL images, and NumPy arrays.

=== "URL"

    ```python
    from inference_sdk import InferenceHTTPClient
    import os

    image_url = "https://media.roboflow.com/inference/soccer.jpg"

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.environ["API_KEY"],
    )

    results = client.infer(image_url, model_id="soccer-players-5fuqs/1")
    print(results)
    ```

=== "NumPy Array"

    ```python
    from inference_sdk import InferenceHTTPClient
    import cv2
    import os

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.environ["API_KEY"],
    )

    numpy_image = cv2.imread("path/to/local/image.jpg")
    results = client.infer(numpy_image, model_id="soccer-players-5fuqs/1")
    print(results)
    ```

=== "PIL Image"

    ```python
    from inference_sdk import InferenceHTTPClient
    from PIL import Image
    import os

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.environ["API_KEY"],
    )

    pil_image = Image.open("path/to/local/image.jpg")
    results = client.infer(pil_image, model_id="soccer-players-5fuqs/1")
    print(results)
    ```

On the first request, the model weights will be downloaded and set up with your local inference server. This request may take some time depending on your network connection and the size of the model. Once your model has downloaded, subsequent requests will be much faster. You can also [pre-load models and manage loaded weights](inference_sdk/model_management.md) to control this process.

### Self-Hosted server

You can also self-host the Inference server (see [Inference CLI page](../inference_helpers/inference_cli/)), and then change `api_url` in the `InferenceHTTPClient`:

```python
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["API_KEY"],
)
```

### AsyncIO client
```python
import asyncio
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"
loop = asyncio.get_event_loop()
result = loop.run_until_complete(
  CLIENT.infer_async(image_url, model_id="soccer-players-5fuqs/1")
)
```

## Parallel / Batch inference

You may want to predict against multiple images at single call. There are two parameters of [`InferenceConfiguration`](inference_sdk/configuration.md)
that specifies batching and parallelism options:
- `max_concurrent_requests` - max number of concurrent requests that can be started
- `max_batch_size` - max number of elements that can be injected into single request

Thanks to that the following improvements can be achieved:

- if you run inference container with API on prem on powerful GPU machine - setting `max_batch_size` properly
may bring performance / throughput benefits
- if you run inference against hosted Roboflow API - setting `max_concurrent_requests` will cause multiple images
being served at once bringing performance / throughput benefits
- combination of both options can be beneficial for clients running inference container with API on cluster of machines,
then the load of single node can be optimised and parallel requests to different nodes can be made at a time

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
predictions = CLIENT.infer([image_url] * 5, model_id="soccer-players-5fuqs/1")

print(predictions)
```

Methods that support batching / parallelism:

- `infer(...)` and `infer_async(...)`
- `ocr_image(...)` and `ocr_image_async(...)` (enforcing `max_batch_size=1`)
- `detect_gazes(...)` and `detect_gazes_async(...)`
- `get_clip_image_embeddings(...)` and `get_clip_image_embeddings_async(...)`

The client also supports [core foundation models](inference_sdk/core_models.md) (CLIP, DocTR, Gaze) and [running Workflows](inference_sdk/workflows.md) for multi-step pipelines.

## Inference against stream

One may want to infer against video or directory of images - and that modes are supported in `inference-client`

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
for frame_id, frame, prediction in CLIENT.infer_on_stream("video.mp4", model_id="soccer-players-5fuqs/1"):
    # frame_id is the number of frame
    # frame - np.ndarray with video frame
    # prediction - prediction from the model
    pass

for file_path, image, prediction in CLIENT.infer_on_stream("local/dir/", model_id="soccer-players-5fuqs/1"):
    # file_path - path to the image
    # frame - np.ndarray with video frame
    # prediction - prediction from the model
    pass
```

## What is actually returned as prediction?

`inference_client` returns plain Python dictionaries that are responses from model serving API. Modification
is done only in context of `visualization` key that keep server-generated prediction visualisation (it
can be transcoded to the format of choice) and in terms of client-side re-scaling.
