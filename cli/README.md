<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/blob/main/banner.png?raw=true"
    />

<br>

[notebooks](https://github.com/roboflow/notebooks) | [inference](https://github.com/roboflow/inference) | [autodistill](https://github.com/autodistill/autodistill) | [collect](https://github.com/roboflow/roboflow-collect)

<br>

[![version](https://badge.fury.io/py/roboflow.svg)](https://badge.fury.io/py/inference-cli)
[![downloads](https://img.shields.io/pypi/dm/inference-cli)](https://pypistats.org/packages/inference-cli)
[![license](https://img.shields.io/pypi/l/inference-cli)](https://github.com/roboflow/inference/blob/main/LICENSE)
[![python-version](https://img.shields.io/pypi/pyversions/inference-cli)](https://badge.fury.io/py/inference-cli)

</div>

# Roboflow Inference CLI

[Roboflow](https://roboflow.com) Inference is an opinionated tool for running inference on state-of-the-art computer vision models. With no prior
knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

## 🎥 Inference in action

Check out Inference running on a video of a football game:

https://github.com/roboflow/inference/assets/37276661/121ab5f4-5970-4e78-8052-4b40f2eec173

## 👩‍🏫 Examples

The [`/examples` directory](https://github.com/roboflow/inference/blob/main/examples/) contains example code for working with and extending `inference`, including HTTP and UDP client code and an insights dashboard, along with community examples (PRs welcome)!

### `inference serve`

`inference serve` is the main command for starting a local inference server. It takes a port number and will only start the docker container if there is not already a container running on that port.

```bash
inference serve --port 9001
```

### `inference infer`

`inference infer` is the main command for running inference on a single image. It takes a path to an image, a Roboflow project name, model version, and API key, and will return a JSON object with the model's predictions. You can also specify a host to run inference on our hosted inference server.

#### Local image

```bash
inference infer --image ./image.jpg --project_id my-project --model-version 1 --api-key my-api-key
```

#### Hosted image

```bash
inference infer --image https://[your-hosted-image-url] --project_id my-project --model-version 1 --api-key my-api-key
```

#### Hosted inference

```bash
inference infer --image ./image.jpg --project_id my-project --model-version 1 --api-key my-api-key --host https://infer.roboflow.com
```

## 💻 Why Inference?

Inference provides a scalable method through which you can manage inferences for your vision projects.

Inference is backed by:

- A server, so you don’t have to reimplement things like image processing and prediction visualization on every project.

- Standardized APIs for computer vision tasks, so switching out the model weights and architecture can be done independently of your application code.

- Model architecture implementations, which implement the tensor parsing glue between images and predictions for supervised models that you've fine-tuned to perform custom tasks.

- A model registry, so your code can be independent from your model weights & you don't have to re-build and re-deploy every time you want to iterate on your model weights.

- Data management integrations, so you can collect more images of edge cases to improve your dataset & model the more it sees in the wild.

And more!

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
