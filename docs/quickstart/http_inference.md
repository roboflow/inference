# HTTP Inference

The Roboflow Inference Server provides a standard API through which to run inference on computer vision models.

In this guide, we show how to run inference on object detection, classification, and segmentation models using the Inference Server.

Currently, the server is compatible with models trained on Roboflow, but stay tuned as we actively develop support for bringing your own models.

To run inference with the server, we will:

1. Install the server
2. Download a model for use on the server
3. Run inference

## Step #1: Install the Inference Server

The Inference Server runs in Docker. Before we begin, make sure you have installed Docker on your system. To learn how to install Docker, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

Once you have Docker installed, you are ready to download Roboflow Inference. The command you need to run depends on what device you are using.

[Run the relevant command for your device](docker.md#run). After you have installed the Inference Server, the Docker container will start running the server at `localhost:9001`.

Now we are ready to run inference!

## Step #2: Run Inference

> The easiest way to interact with the `inference` API is using the [`inference-sdk`](../inference_sdk/http_client.md)

To run inference on a model, there are two routes we can make an HTTP request to:

```url
http://localhost:9001/{project_id}/{model_version}
```

or

```url
http://localhost:9001/infer/{task}
```

where task is one of `object_detection`, `instance_segmentation`, or `classification`, depending on your project type. The former is the route used by our hosted inference. It doesn't require that you know or pass your task type, just your project ID and model version number and it is fully interchangable with Roboflow's hosted inference offerings at `detect.roboflow.com`, `outline.roboflow.com`, and `classify.roboflow.com`. The latter is a newer route within the `inference` API. It assumes you know your task type, and it is more flexible (e.g. batching is possible).

To find your project ID, model version number, and task type refer to the Roboflow documentation, [Workspace and Project IDs](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

### via CLI

First, [Install the CLI](../index.md#cli).

To run inference, use the `inference infer` command:

```bash
inference infer {image_path} \
    --project-id {project_id} \
    --model-version {model_version} \
    --api-key {api_key}
```

You can also specify a host option to run inference on the Roboflow Hosted API.

```bash
inference infer {image_path} \
    --project-id {project_id} \
    --model-version {model_version} \
    --api-key {api_key} \
    --host https://detect.roboflow.com
```

{image_path} can be a local path to an image, or a URL to a hosted image.

E.g. `./image.jpg` or `https://[YOUR_HOSTED_IMAGE_URL]`

### Python script

To run inference, make a HTTP request to the routes:

*V1 Route*
```python
import requests

project_id = ""
model_version = ""
image_url = ""
confidence = 0.75
api_key = ""

params = {
    "image": image_url,
    "confidence": confidence,
    "overlap": iou_thresh,
    "api_key": api_key,
}
res = requests.post(
    f"http://localhost:9001/{project_id}/{model_version}",
    params=infer_object_detection_payload,
)

predictions = res.json()
```

You can also run using an image you have in memory.

```python
import base64
import requests
from io import BytesIO
from PIL import Image

project_id = ""
model_version = ""
confidence = 0.75
api_key = ""

params = {
    "confidence": confidence,
    "overlap": iou_thresh,
    "api_key": api_key,
}

image = Image.open("/path/to/file.jpg")
# image = Image.fromarray(numpy_array)

buffered = BytesIO()
image.save(buffered, quality=100, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode("ascii")

res = requests.post(
    f"http://localhost:9001/{project_id}/{model_version}",
    params=infer_object_detection_payload,
    data=img_str,
)

predictions = res.json()
```

*V2 Route*
```python
import requests

project_id = ""
model_version = ""
image_url = ""
confidence = 0.75
api_key = ""
task = "object_detection

infer_payload = {
    "image": {
        "type": "url",
        "value": image_url,
    },
    "confidence": confidence,
    "iou_threshold": iou_thresh,
    "api_key": api_key,
}
res = requests.post(
    f"http://localhost:9001/infer/{task}",
    json=infer_object_detection_payload,
)

predictions = res.json()
```

Object detection models trained with Roboflow support batching.

```python
infer_payload = {
    "image": [
        {
            "type": "url",
            "value": image_url_1,
        },
        {
            "type": "url",
            "value": image_url_2,
        },
        {
            "type": "url",
            "value": image_url_3,
        },
    ],
    "confidence": confidence,
    "iou_threshold": iou_thresh,
    "api_key": api_key,
}
```

To run with an image in memory:

```python
image = Image.open("/path/to/file.jpg")
# image = Image.fromarray(numpy_array)

buffered = BytesIO()
image.save(buffered, quality=100, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode("ascii")

infer_payload = {
    "image": {
        "type": "base64",
        "value": img_str,
    },
    "confidence": confidence,
    "iou_threshold": iou_thresh,
    "api_key": api_key,
}
```

The code snippets above will run inference on a computer vision model. On the first request, the model weights will be downloaded and set up with your local inference server. This request may take some time depending on your network connection and the size of the model. Once your model has downloaded, subsequent requests will be much faster.

Above, set your project ID and model version number. Also configure your confidence and IoU threshold values as needed. If you are using classification, you can omit the IoU threshold value. You will also need to set your Roboflow API key. To learn how to retrieve your Roboflow API key, refer to the Roboflow API documentation, [Authentication - Retrieve an API Key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

You can post either a URL, a base64-encoded image, or a pickled NumPy array to the server.

The Inference Server comes with a `/docs` route at `localhost:9001/docs` or `localhost:9001/redoc` that provides OpenAPI-powered documentation. You can use this to reference the routes available, and the configuration options for each route.
