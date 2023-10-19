<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />

  <br/>

### [inference package](https://pypi.org/project/inference/) | [inference repo](https://github.com/roboflow/inference)

  <br/>

[![version](https://badge.fury.io/py/roboflow.svg)](https://badge.fury.io/py/inference-cli)
[![downloads](https://img.shields.io/pypi/dm/inference-cli)](https://pypistats.org/packages/inference-cli)
[![license](https://img.shields.io/pypi/l/inference-cli)](https://github.com/roboflow/inference/blob/main/LICENSE)
[![python-version](https://img.shields.io/pypi/pyversions/inference-cli)](https://badge.fury.io/py/inference-cli)

</div>

# Roboflow Inference CLI

Roboflow Inference CLI offers a lightweight interface for running the Roboflow inference server locally or the Roboflow Hosted API.

To create custom inference server Docker images, go to the parent package, [Roboflow Inference](https://pypi.org/project/inference/).

[Roboflow](https://roboflow.com) has everything you need to deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

## 👩‍🏫 Examples

### inference server start

Starts a local inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment, 
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If 
you haven't installed Docker yet, you can get it from [Docker's official website](https://www.docker.com/get-started).

The CLI will automatically detect the device you are running on and pull the appropriate Docker image.

```bash
inference server start --port 9001
```

### inference server status

Checks the status of the local inference server.

```bash
inference server status
```

### inference infer

Runs inference on a single image. It takes a path to an image, a Roboflow project name, model version, and API key, and will return a JSON object with the model's predictions. You can also specify a host to run inference on our hosted inference server.

#### Local image

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key
```

#### Hosted image

```bash
inference infer https://[YOUR_HOSTED_IMAGE_URL] --project-id my-project --model-version 1 --api-key my-api-key
```

#### Hosted API inference

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key --host https://detect.roboflow.com
```

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson specific inference server images, check out the [Roboflow Inference](https://pypi.org/project/inference/) package, or pull the images directly following instructions in the official [Roboflow Inference documentation](https://inference.roboflow.com/quickstart/docker/#pull-from-docker-hub).

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

## 🚀 enterprise

With a Roboflow Inference Enterprise License, you can access additional Inference features, including:

- Server cluster deployment
- Device management
- Active learning
- YOLOv5 and YOLOv8 model sub-license

To learn more, [contact the Roboflow team](https://roboflow.com/sales).

## 📚 documentation

Visit our [documentation](https://roboflow.github.io/inference) for usage examples and reference for Roboflow Inference.

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
