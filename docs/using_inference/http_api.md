The HTTP Inference API provides a standard API through which to run inference on computer vision models. The HTTP API is a helpful way to treat your machine learning models as their own microservice. With this interface, you will run a docker container and make requests over HTTP. The requests contain all of the information Inference needs to run a computer vision model including the model ID, the input image data, and any configurable parameters used during processing (e.g. confidence threshold).

## Quickstart

### Install the Inference Server

_You can skip this step if you already have Inference installed and running._

The Inference Server runs in Docker. Before we begin, make sure you have installed Docker on your system. To learn how to install Docker, refer to the <a href="https://docs.docker.com/get-docker/" target="_blank">official Docker installation guide</a>.

Once you have Docker installed, you are ready to download Roboflow Inference. The command you need to run depends on what device you are using.

Start the server using `inference server start`. After you have started the Inference Server, the Docker container will start running the server at `localhost:9001`.

### Run Inference

You can send a URL with an image, a NumPy array, or a base64-encoded image to an Inference server. The server will return a JSON response with the predictions. The easiest way to interact with the Roboflow Inference server is to sue the Inference SDK. To do this first install it with pip:

```bash
pip install inference-sdk
```

Next, instantiate a client and use the `infer(...)` method:

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

project_id = "soccer-players-5fuqs"
model_version = "1"
model_id = project_id + "/" + model_version
image_url = "https://media.roboflow.com/inference/soccer.jpg"

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)


results = client.infer(image_url, model_id=model_id)
```
{% include 'model_id.md' %}
!!! Hint

    See [full docs for the Inference SDK](../../inference_helpers/inference_sdk).

### Visualize Results

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

model_id = "soccer-players-5fuqs/1"
image_file = "soccer.jpg"

image = cv2.imread(image_file)

#Configure client
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # route for local inference server
    api_key=os.environ["ROBOFLOW_API_KEY"], # api key for your workspace
)

#Run inference
results = client.infer(image, model_id=model_id)

#Load results into Supervision Detection API
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

#Create Supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

#Extract labels array from inference results
labels = [p['class'] for p in results[0]['predictions']]

#Apply results to image using Supervision annotators
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

#Write annotated image to file or display image
cv2.imwrite("soccer-annotated.jpg", annotated_image)
#or sv.plot_image(annotated_image)

```

## Hosted API

Roboflow hosts a powerful and infinitely scalable version of the Roboflow Inference Server. This makes it even easier to integrate computer vision models into your software without adding any maintenance burden. And, since the Roboflow hosted APIs are running using the Inference package, it's easy to switch between using a hosted server and an on prem server without having to reinvent your client code. To use the hosted API, simply replace the `api_url` parameter passed to the Inference SDK client configuration. The hosted API base URL is `https://detect.roboflow.com`.

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

project_id = "soccer-players-5fuqs/1"
image_url = "https://media.roboflow.com/inference/soccer.jpg"

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

results = client.infer(image_url, model_id=f"{model_id}")
```

## Custom Client

For interacting with the HTTP API from a language other than Python it is possible to write your own client code (for python applications, we highly recommend sticking with the Inference SDK). Below, we show examples of formulating requests in Python. These examples can be emulated in other languages for sending requests to the Roboflow Inference server.

There are two generations of routes in a Roboflow inference server. To see what routes are available for a running inference server instance, visit the `/docs` route in a browser. Roboflow hosted inference endpoints (`detect.roboflow.com`) only support V1 routes.

### Run Inference on a v2 Route

!!! V2

    === "URL"

        ```python
        import requests

        task = "object_detection"
        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        image_url = "https://storage.googleapis.com/com-roboflow-marketing/inference/soccer2.jpg"

        payload = {
            "model_id": model_id,
            "image": {
                "type": "url",
                "value": image_url,
            },
            "api_key": api_key,
        }
        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=payload,
        )
        ```

    === "Base64 Encoded Image"

        ```python
        import base64
        import requests
        from io import BytesIO

        from PIL import Image

        task = "object_detection"
        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        image = Image.open(...)

        buffered = BytesIO()
        image.save(buffered, quality=100, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        payload = {
            "model_id": model_id,
            "image": {
                "type": "base64",
                "value": img_str,
            },
            "api_key": api_key,
        }
        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=payload,
        )
        ```

    === "NumPy Array"

        ```python
        import base64
        import requests
        from io import BytesIO

        import cv2

        task = "object_detection"
        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        numpy_image = cv2.imread(...)

        numpy_data = pickle.dumps(numpy_image)
        numpy_str = numpy_data.decode("ascii")

        payload = {
            "model_id": model_id,
            "image": {
                "type": "numpy",
                "value": numpy_str,
            },
            "api_key": api_key,
        }
        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=payload,
        )
        ```

    === "Batch Inference"

        Roboflow object detection models support batching. Utilize batch inference by passing a list of image objects in a request payload:

        ```python
        import requests
        import base64
        from PIL import Image
        from io import BytesIO

        task = "object_detection"
        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        image = Image.open(...)

        buffered = BytesIO()
        image.save(buffered, quality=100, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        infer_payload = {
            "model_id": model_id,
            "image": [
                {
                    "type": "base64",
                    "value": img_str,
                },
                {
                    "type": "base64",
                    "value": img_str,
                },
                {
                    "type": "base64",
                    "value": img_str,
                }
            ],
            "api_key": api_key,
        }

        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=infer_payload,
        )
        ```

### Run Inference on a v1 Route

!!! V1

    === "URL"

        ```python
        import requests
        import base64
        from PIL import Image
        from io import BytesIO

        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"
        image_url = "https://storage.googleapis.com/com-roboflow-marketing/inference/soccer.jpg

        res = requests.post(
            f"https://detect.roboflow.com/{model_id}?api_key={api_key}&image={image_url}",
        )
        ```

    === "Base64 Image"

        The Roboflow hosted API uses the V1 route and requests take a slightly different form:

        ```python
        import requests
        import base64
        from io import BytesIO

        from PIL import Image

        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        image = Image.open(...)

        buffered = BytesIO()
        image.save(buffered, quality=100, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        res = requests.post(
            f"https://detect.roboflow.com/{model_id}?api_key={api_key}",
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        ```

    === "NumPy Array"

        Numpy arrays can be pickled and passed to the inference server for quicker processing. Note, Roboflow hosted APIs to not accept numpy inputs for security reasons:

        ```python
        import pickle
        import requests

        import cv2

        model_id = "soccer-players-5fuqs/1"
        api_key = "YOUR ROBOFLOW API KEY"

        image = cv2.imread(...)

        numpy_data = pickle.dumps(image)

        res = requests.post(
            f"https://detect.roboflow.com/{model_id}?api_key={api_key}",
            data=numpy_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        ```

    === "Batch Inference"

        Batch inference is not currently supported by V1 routes.

The Inference Server comes with a `/docs` route at `localhost:9001/docs` or `localhost:9001/redoc` that provides OpenAPI-powered documentation. You can use this to reference the routes available, and the configuration options for each route.
