<div align="center">
  <p>
    <a align="center" href="" target="https://inference.roboflow.com/">
      <img
        width="100%"
        src="https://github.com/roboflow/inference/blob/main/banner.png"
      >
    </a>
  </p>

  <br>

[notebooks](https://github.com/roboflow/notebooks) | [supervision](https://github.com/roboflow/supervision) | [autodistill](https://github.com/autodistill/autodistill) | [maestro](https://github.com/roboflow/multimodal-maestro)

  <br>

[![version](https://badge.fury.io/py/inference.svg)](https://badge.fury.io/py/inference)
[![downloads](https://img.shields.io/pypi/dm/inference)](https://pypistats.org/packages/inference)
![docker pulls](https://img.shields.io/docker/pulls/roboflow/roboflow-inference-server-cpu)
[![license](https://img.shields.io/pypi/l/inference)](https://github.com/roboflow/inference/blob/main/LICENSE.md)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/workflows)
[![discord](https://img.shields.io/discord/1159501506232451173)](https://discord.gg/GbfgXGJ8Bk)

</div>

## 👋 hello

Roboflow Inference is an open-source, powerful, and versatile platform designed to simplify the deployment of computer vision models. It enables developers to easily run object detection, classification, and instance segmentation and utilize foundation models like CLIP and SAM through a Python-native package, a self-hosted inference server, or a fully managed API.

Explore our enterprise [options](https://roboflow.com/sales) for advanced features like server deployment, device management, active learning, and commercial licenses for YOLOv5 and YOLOv8.

## 💻 install

Inference package requires [**Python>=3.8,<=3.11**](https://www.python.org/).

```bash
pip install inference
```

<details>
<summary>👉 additional considerations</summary>

<br>

- hardware

  Improve model performance in GPU-accelerated environments.
  
  ```bash
  pip install inference-gpu
  ```

- models

  Install model-specific dependencies, ensuring code compatibility and license compliance. Learn more about the [models](https://inference.roboflow.com/#extras) supported by Inference.

  ```bash
  pip install inference[yolo-world]
  ```

</details>

Click [here](https://inference.roboflow.com/quickstart/docker/) to learn more about running Inference inside Docker.

## 🔥 quickstart

TODO

## 📟 inference server

TODO

## ⏩ workflows

TODO

## 🧪 examples

TODO

## 📚 documentation

Visit our [documentation](https://inference.roboflow.com) to explore comprehensive guides, detailed API references, and a wide array of tutorials designed to help you harness the full potential of the Inference package.

## © license

The Roboflow Inference code is distributed under the [Apache 2.0](https://github.com/roboflow/inference/blob/master/LICENSE.md) license. However, each supported model is subject to its licensing. Detailed information on each model's license can be found [here](https://inference.roboflow.com/quickstart/licensing/#model-code-licenses).

## 🏆 contribution

We would love your input to improve Roboflow Inference! Please see our [contributing guide](https://github.com/roboflow/inference/blob/master/CONTRIBUTING.md) to get started. Thank you to all of our contributors! 🙏    

## 🎬 pip install inference

[Roboflow](https://roboflow.com) Inference is the easiest way to use and deploy computer vision models.
Inference supports running object detection, classification, instance segmentation, and even foundation models (like CLIP and SAM).
You can [train and deploy your own custom model](https://github.com/roboflow/notebooks) or use one of the 50,000+
[fine-tuned models shared by the community](https://universe.roboflow.com).

There are three primary `inference` interfaces:

- A Python-native package (`pip install inference`)
- A self-hosted inference server (`inference server start`)
- A [fully-managed, auto-scaling API](https://docs.roboflow.com).

You can run Inference on an edge device like an NVIDIA Jetson, or on cloud computing platforms like AWS, GCP, and Azure.

## 🏃 Getting Started

Get up and running with `inference` on your local machine in 3 minutes.

```sh
pip install inference # or inference-gpu if you have CUDA
```

Setup [your Roboflow Private API Key](https://app.roboflow.com/settings/api)
by exporting a `ROBOFLOW_API_KEY` environment variable or
adding it to a `.env` file.

```sh
export ROBOFLOW_API_KEY=your_key_here
```

Run [an open-source Rock, Paper, Scissors model](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw)
on your webcam stream:

```python
import inference

inference.Stream(
    source="webcam", # or rtsp stream or camera id
    model="rock-paper-scissors-sxsw/11", # from Universe

    on_prediction=lambda predictions, image: (
        print(predictions) # now hold up your hand: 🪨 📄 ✂️
    )
)
```

> [!NOTE]  
> Currently, the stream interface only supports object detection

Now let's extend the example to use [Supervision](https://roboflow.com/supervision)
to visualize the predictions and display them on screen with OpenCV:

```python
import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

inference.Stream(
    source="webcam", # or rtsp stream or camera id
    model="rock-paper-scissors-sxsw/11", # from Universe

    output_channel_order="BGR",
    use_main_thread=True, # for opencv display

    on_prediction=lambda predictions, image: (
        print(predictions), # now hold up your hand: 🪨 📄 ✂️

        cv2.imshow(
            "Prediction",
            annotator.annotate(
                scene=image,
                detections=sv.Detections.from_inference(predictions)
            )
        ),
        cv2.waitKey(1)
    )
)

```

## 👩‍🏫 More Examples

The [`/examples`](https://github.com/roboflow/inference/tree/main/examples/) directory contains code samples for working with and extending `inference` including using foundation models like CLIP, HTTP and UDP clients, and an insights dashboard, along with community examples (PRs welcome)!



## Inference Client

To consume predictions from inference server in Python you can
use the `inference-sdk` package.

```bash
pip install inference-sdk
```

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://media.roboflow.com/inference/soccer.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
client = InferenceHTTPClient(
    api_url="http://localhost:9001", # or https://detect.roboflow.com for Hosted API
    api_key="ROBOFLOW_API_KEY"
)
with client.use_model("soccer-players-5fuqs/1"):
    predictions = client.infer(image_url)

print(predictions)
```

Visit our [documentation](https://inference.roboflow.com/) to discover capabilities of `inference-clients` library.

## Single Image Inference

After installing `inference` via pip, you can run a simple inference
on a single image (vs the video stream example above) by instantiating
a `model` and using the `infer` method (don't forget to setup your
`ROBOFLOW_API_KEY` environment variable or `.env` file):

```python
from inference.models.utils import get_model

model = get_model(
    model_id="soccer-players-5fuqs/1"
)

# you can also infer on local images by passing a file path,
# a PIL image, or a numpy array
results = model.infer(
  image="https://media.roboflow.com/inference/soccer.jpg",
  confidence=0.5,
  iou_threshold=0.5
)

print(results)
```

## Getting CLIP Embeddings

You can run inference with [OpenAI's CLIP model](https://blog.roboflow.com/openai-clip) using:

```python
from inference.models import Clip

image_url = "https://media.roboflow.com/inference/soccer.jpg"

model = Clip()
embeddings = model.embed_image(image_url)

print(embeddings)
```

## Using SAM

You can run inference with [Meta's Segment Anything model](https://blog.roboflow.com/segment-anything-breakdown/) using:

```python
from inference.models import SegmentAnything

image_url = "https://media.roboflow.com/inference/soccer.jpg"

model = SegmentAnything()
embeddings = model.embed_image(image_url)

print(embeddings)
```

## ✅ Supported Models

### Load from Roboflow

You can use models hosted on Roboflow with the following architectures through Inference:

- YOLOv5 Object Detection
- YOLOv5 Instance Segmentation
- YOLOv8 Object Detection
- YOLOv8 Classification
- YOLOv8 Segmentation
- YOLACT Segmentation
- ViT Classification

### Core Models

Core Models are foundation models and models that have not been fine-tuned on a specific dataset.

The following core models are supported:

1. CLIP
2. L2CS (Gaze Detection)
3. Segment Anything (SAM)

## Inference CLI

We've created a CLI tool with useful commands to make the `inference` usage easier. Check out [docs](./inference_cli/README.md).

<br>

<div align="center">
  <div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://disuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
