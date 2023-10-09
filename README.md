![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

## 👋 hello

[Roboflow](https://roboflow.com) Inference is an opinionated tool for running inference on state-of-the-art computer vision models. With no prior
knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

## 🎥 Inference in action

Check out Inference running on a video of a football game:

https://github.com/roboflow/inference/assets/37276661/121ab5f4-5970-4e78-8052-4b40f2eec173

## 👩‍🏫 Examples

The [`/examples` directory](https://github.com/roboflow/inference/blob/main/examples/) contains example code for working with and extending `inference`, including HTTP and UDP client code and an insights dashboard, along with community examples (PRs welcome)!

## 💻 Why Inference?

Inference provides a scalable method through which you can manage inferences for your vision projects.

Inference is backed by:

- A server, so you don’t have to reimplement things like image processing and prediction visualization on every project.

- Standardized APIs for computer vision tasks, so switching out the model weights and architecture can be done independently of your application code.

- Model architecture implementations, which implement the tensor parsing glue between images and predictions for supervised models that you've fine-tuned to perform custom tasks.

- A model registry, so your code can be independent from your model weights & you don't have to re-build and re-deploy every time you want to iterate on your model weights.

- Data management integrations, so you can collect more images of edge cases to improve your dataset & model the more it sees in the wild.

And more!

### 📌 Install pip vs Docker:

- **pip**: Installs `inference` into your Python environment. Lightweight, good for Python-centric projects.
- **Docker**: Packages `inference` with its environment. Ensures consistency across setups; ideal for scalable deployments.

## 💻 install

### With ONNX CPU Runtime:

For CPU powered inference:

```bash
pip install inference
```

or

```bash
pip install inference-cpu
```

### With ONNX GPU Runtime:

If you have an NVIDIA GPU, you can accelerate your inference with:

```bash
pip install inference-gpu
```

### Without ONNX Runtime:

Roboflow Inference uses Onnxruntime as its core inference engine. Onnxruntime provides an array of different [execution providers](https://onnxruntime.ai/docs/execution-providers/) that can optimize inference on differnt target devices. If you decide to install onnxruntime on your own, install inference with:

```bash
pip install inference-core
```

Alternatively, you can take advantage of some advanced execution providers using one of our published docker images.

### Extras:

Some functionality requires extra dependencies. These can be installed by specifying the desired extras during installation of Roboflow Inference.
| extra | description |
|:-------|:-------------------------------------------------|
| `clip` | Ability to use the core `CLIP` model (by OpenAI) |
| `gaze` | Ability to use the core `Gaze` model |
| `http` | Ability to run the http interface |
| `sam` | Ability to run the core `Segment Anything` model (by Meta AI) |

**_Note:_** Both CLIP and Segment Anything require pytorch to run. These are included in their respective dependencies however pytorch installs can be highly environment dependent. See the [official pytorch install page](https://pytorch.org/get-started/locally/) for instructions specific to your enviornment.

Example install with http dependencies:

```bash
pip install inference[http]
```

## 🐋 docker

You can learn more about Roboflow Inference Docker Image build, pull and run in our [documentation](https://roboflow.github.io/inference/quickstart/docker/).

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

<br/>
[test.yml](.github%2Fworkflows%2Ftest.yml)
### Inference clients

If you seek for Python client to consume predictions from inference server - you can do it via `inference-sdk`
package.

```bash
pip install inference-sdk
```


## 🔥 quickstart

**Docker Quickstart**:

We've made calling our models easy with Python client for HTTP API exposed by `inference`.

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
with CLIENT.use_model("soccer-players-5fuqs/1"):
    predictions = CLIENT.infer(image_url)

print(predictions)
```
Visit our [documentation](https://roboflow.github.io/inference) to discover capabilities of `inference-clients` library.


**pip Quickstart**:

After installing via pip, you can run a simple inference using:

```python
from inference.models.utils import get_roboflow_model

model = get_roboflow_model(
    model_id="soccer-players-5fuqs/1",
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key="ROBOFLOW_API_KEY"
)

results = model.infer(image="https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg", confidence=0.5, iou_threshold=0.5)

print(results)

```

**CLIP Quickstart**:

You can run inference with OpenAI's CLIP model using:

```python
from inference.models import Clip

model = Clip(
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key = "ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/7fLqS2r1SV8mm0YzyI0c/yy6hjtPUFFkq4yAvhkvs/original.jpg"

embeddings = model.embed_image(image_url)

print(embeddings)
```

**SAM Quickstart**:

You can run inference with Meta's Segment Anything model using:

```python
from inference.models import SegmentAnything

model = SegmentAnything(
    #Replace ROBOFLOW_API_KEY with your Roboflow API Key
    api_key = "ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/7fLqS2r1SV8mm0YzyI0c/yy6hjtPUFFkq4yAvhkvs/original.jpg"

embeddings = model.embed_image(image_url)

print(embeddings)
```

## 🏗️ inference process

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

## 📝 license

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

## 🚀 enterprise

With a Roboflow Inference Enterprise License, you can access additional Inference features, including:

- Server cluster deployment
- Device management
- Active learning
- YOLOv5 and YOLOv8 model sub-license

To learn more, [contact the Roboflow team](https://roboflow.com/sales).

## 📚 documentation

Visit our [documentation](https://roboflow.github.io/inference) for usage examples and reference for Roboflow Inference.

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
