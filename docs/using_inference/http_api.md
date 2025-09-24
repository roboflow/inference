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
import os

import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

model_id = "soccer-players-5fuqs/1"
image_file = "soccer.jpg"

image = cv2.imread(image_file)

#Configure client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", # route for local inference server
    api_key=os.environ["ROBOFLOW_API_KEY"], # api key for your workspace
)

#Run inference
result = client.infer(image, model_id=model_id)

#Load results into Supervision Detection API
detections = sv.Detections.from_inference(result)

#Create Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

#Extract labels array from inference results
labels = [p['class'] for p in result['predictions']]

#Apply results to image using Supervision annotators
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

#Write annotated image to file or display image
with sv.ImageSink(target_dir_path="./results/", overwrite=True) as sink:
    sink.save_image(annotated_image)
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