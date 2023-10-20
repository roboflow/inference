![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

## 🎬 pip install inference

[Roboflow](https://roboflow.com) Inference is the easiest way to use and deploy computer vision models.
Inference supports running object detection, classification, instance segmentation, and even foundation models (like CLIP and SAM).
You can [train and deploy your own custom model](https://github.com/roboflow/notebooks) or use one of the 50,000+
[fine-tuned models shared by the community](https://universe.roboflow.com).

There are three primary `inference` interfaces:
* A Python-native package (`pip install inference`)
* A self-hosted inference server (`inference server start`)
* A [fully-managed, auto-scaling API](https://docs.roboflow.com).

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
                detections=sv.Detections.from_roboflow(predictions)
            )
        ),
        cv2.waitKey(1)
    )
)

```

## 👩‍🏫 More Examples

The [`/examples`](https://github.com/roboflow/inference/tree/main/examples/) directory contains code samples for working with and extending `inference` including using foundation models like CLIP, HTTP and UDP clients, and an insights dashboard, along with community examples (PRs welcome)!

## 🎥 Inference in action

Check out Inference running on a video of a football game:

https://github.com/roboflow/inference/assets/37276661/121ab5f4-5970-4e78-8052-4b40f2eec173

## 💻 Why Inference?

Inference provides a scalable method through which you can manage inferences for your vision projects.

Inference is composed of:

- Thousands of [pre-trained community models](https://universe.roboflow.com) that you can use as a starting point.

- Foundation models like CLIP, SAM, and OCR.

- A tight integration with [Supervision](https://roboflow.com/supervision).

- An HTTP server, so you don’t have to reimplement things like image processing and prediction visualization on every project and you can scale your GPU infrastructure independently of your application code, and access your model from whatever language your app is written in.

- Standardized APIs for computer vision tasks, so switching out the model weights and architecture can be done independently of your application code.

- A model registry, so your code can be independent from your model weights & you don't have to re-build and re-deploy every time you want to iterate on your model weights.

- Active Learning integrations, so you can collect more images of edge cases to improve your dataset & model the more it sees in the wild.

- Seamless interoperability with [Roboflow](https://roboflow.com) for creating datasets, training & deploying custom models.

And more!

### 📌 Use the Inference Server

You can learn more about Roboflow Inference Docker Image build, pull and run in our [documentation](https://inference.roboflow.com/quickstart/docker/).

- Run on x86 CPU:

```bash
docker run --net=host roboflow/roboflow-inference-server-cpu:latest
```

- Run on NVIDIA GPU:

```bash
docker run --network=host --gpus=all roboflow/roboflow-inference-server-gpu:latest
```

<details close>
<summary>👉 more docker run options</summary>

- Run on arm64 CPU:

```bash
docker run -p 9001:9001 roboflow/roboflow-inference-server-arm-cpu:latest
```

- Run on NVIDIA GPU with TensorRT Runtime:

```bash
docker run --network=host --gpus=all roboflow/roboflow-inference-server-trt:latest
```

- Run on NVIDIA Jetson with JetPack `4.x`:

```bash
docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson:latest
```

- Run on NVIDIA Jetson with JetPack `5.x`:

```bash
docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson-5.1.1:latest
```

</details>

### Extras:

Some functionality requires extra dependencies. These can be installed by specifying the desired extras during installation of Roboflow Inference.
| extra | description |
|:-------|:-------------------------------------------------|
| `clip` | Ability to use the core `CLIP` model (by OpenAI) |
| `gaze` | Ability to use the core `Gaze` model |
| `http` | Ability to run the http interface |
| `sam` | Ability to run the core `Segment Anything` model (by Meta AI) |

**_Note:_** Both CLIP and Segment Anything require pytorch to run. These are included in their respective dependencies however pytorch installs can be highly environment dependent. See the [official pytorch install page](https://pytorch.org/get-started/locally/) for instructions specific to your enviornment.

Example install with CLIP dependencies:

```bash
pip install "inference[clip]"
```

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
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(
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

## 🏗️ inference Process

To standardize the inference process throughout all our models, Roboflow Inference has a structure for processing inference requests. The specifics can be found on each model's respective page, but overall it works like this for most models:

<img width="900" alt="inference structure" src="https://github.com/stellasphere/inference/assets/29011058/abf69717-f852-4655-9e6e-dae19fc263dc">

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

## 📝 License

The Roboflow Inference code is distributed under an [Apache 2.0 license](https://github.com/roboflow/inference/blob/master/LICENSE.md). The models supported by Roboflow Inference have their own licenses. View the licenses for supported models below.

| model                     |                                                                license                                                                |
| :------------------------ | :-----------------------------------------------------------------------------------------------------------------------------------: |
| `inference/models/clip`   |                                        [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)                                        |
| `inference/models/gaze`   | [MIT](https://github.com/Ahmednull/L2CS-Net/blob/main/LICENSE), [Apache 2.0](https://github.com/google/mediapipe/blob/master/LICENSE) |
| `inference/models/sam`    |                         [Apache 2.0](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)                          |
| `inference/models/vit`    |                         [Apache 2.0](https://github.com/roboflow/inference/main/inference/models/vit/LICENSE)                         |
| `inference/models/yolact` |                                     [MIT](https://github.com/dbolya/yolact/blob/master/README.md)                                     |
| `inference/models/yolov5` |                                 [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE)                                 |
| `inference/models/yolov7` |                                  [GPL-3.0](https://github.com/WongKinYiu/yolov7/blob/main/README.md)                                  |
| `inference/models/yolov8` |                              [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/master/LICENSE)                               |

## Inference CLI
We've created a CLI tool with useful commands to make the `inference` usage easier. Check out [docs](./inference_cli/README.md).

## 🚀 Enterprise

With a Roboflow Inference Enterprise License, you can access additional Inference features, including:

- Server cluster deployment
- Device management
- Active learning
- YOLOv5 and YOLOv8 commercial license

To learn more, [contact the Roboflow team](https://roboflow.com/sales).

## 📚 documentation

Visit our [documentation](https://inference.roboflow.com) for usage examples and reference for Roboflow Inference.

## 🏆 contribution

We would love your input to improve Roboflow Inference! Please see our [contributing guide](https://github.com/roboflow/inference/blob/master/CONTRIBUTING.md) to get started. Thank you to all of our contributors! 🙏

## 💻 explore more Roboflow open source projects

| Project                                                           | Description                                                                                                                                            |
| :---------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [supervision](https://roboflow.com/supervision)                   | General-purpose utilities for use in computer vision projects, from predictions filtering and display to object tracking to model evaluation.          |
| [Autodistill](https://github.com/autodistill/autodistill)         | Automatically label images for use in training computer vision models.                                                                                 |
| [Inference](https://github.com/roboflow/inference) (this project) | An easy-to-use, production-ready inference server for computer vision supporting deployment of many popular model architectures and fine-tuned models. |
| [Notebooks](https://roboflow.com/notebooks)                       | Tutorials for computer vision tasks, from training state-of-the-art models to tracking objects to counting objects in a zone.                          |
| [Collect](https://github.com/roboflow/roboflow-collect)           | Automated, intelligent data collection powered by CLIP.                                                                                                |

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
  </div>
