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

To run inference on a model, we will make a HTTP request to:

```url
http://localhost:9001/{project_id}/{model_version}
```

To find your project ID and model version number, refer to the Roboflow documentation, [Workspace and Project IDs](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

This route works for all supported task types: object detection, classification, and segmentation.

To run inference, make a HTTP request to the route:

```python
import requests

project_id = ""
model_version = ""
image_url = ""
confidence = 0.75
api_key = ""

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
    f"http://localhost:9001/{project_id}/{model_version}",
    json=infer_object_detection_payload,
)

predictions = res.json()
```

This code will run inference on a computer vision model. On the first request, the model weights will be downloaded and set up with your local inference server. This request may take some time depending on your network connection and the size of the model. Once your model has downloaded, subsequent requests will be much faster.

Above, set your project ID and model version number. Also configure your confidence and IoU threshold values as needed. If you are using classification, you can omit the IoU threshold value. You will also need to set your Roboflow API key. To learn how to retrieve your Roboflow API key, refer to the Roboflow API documentation, [Authentication - Retrieve an API Key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

You can post either a URL, a base64-encoded image, or a pickled NumPy array to the server.

The Inference Server comes with a `/docs` route at `localhost:9001/docs` that provides OpenAPI-powered documentation. You can use this to reference the routes available, and the configuration options for each route.
