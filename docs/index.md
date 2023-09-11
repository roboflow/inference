![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

# Roboflow Inference 💻

[Roboflow](https://roboflow.com) Inference is an opinionated tool for running inference on state-of-the-art computer vision models. With no prior knowledge of machine learning or device-specific deployment, you can deploy a computer vision model to a range of devices and environments.

By using Roboflow Inference, you can:

1. Write inference logic without having to configure environments or install dependencies.
2. Deploy models to a range of devices and environments.
3. Use state-of-the-art models with your own weights.
4. Query models with HTTP.
5. Scale up your inference server as your production needs grow with Docker.

You can use Inference wherever you can use Docker: on CPUs, GPUs, TRT and more.

You can run inference using the `inference` pip package, or using the HTTP server available by running a Roboflow Inference Docker containers.

## Installation 🛠️

The Roboflow Inference runs in a Docker container. If you do not already have Docker installed on your computer, follow the official Docker installation instructions to install Docker.

For all installation options, you will need a [free Roboflow account](https://app.roboflow.com) and a Roboflow API key. You can learn how to retrieve your API key in the [Roboflow API documentation](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

### Install Using Pip

To install the Inference using pip, run one of the following commands, depending on the device on which you want to run the Inference:

```bash
pip install inference[arm]
pip install inference[jetson]
pip install inference[trt]
```

### Install Using Docker

See the [docker quickstart](docs/quickstart/docker.md).

## Quickstart 🚀

API documentation are hosted by a running instance of the inference server at the `/docs` and `/redoc` endpoints. For an inference server running locally, see `https://localhost/docs` or `https://localhost/redoc`.

## Model Support 🖼️

The Roboflow Inference supports the models listed below.

You can also run any model hosted on Roboflow using the Inference.

### Classification

- [ViT](https://inference.roboflow.com/library/models/vit_classification)
- [YOLOv8](https://inference.roboflow.com/library/models/yolov8_classification/)
- [CLIP](https://inference.roboflow.com/library/models/clip/)

### Object Detection

- [YOLOv5](https://inference.roboflow.com/library/models/yolov5/)
- [YOLOv8](https://inference.roboflow.com/library/models/yolov8/)

### Segmentation

- [YOLOv7](https://inference.roboflow.com/library/models/yolov7_instance_segmentation/)
- [YOLOv8](https://inference.roboflow.com/library/models/yolov8_segmentation/)
- [YOLOACT](https://inference.roboflow.com/library/models/yoloact_segmentation/)
- [Segment Anything](https://inference.roboflow.com/library/models/segment_anything/)

## Environment Variable Control 🌐

Use these environment variables to control pingback and to Roboflow as well as other features; if you are running a Docker container, [pass these](https://docs.docker.com/engine/reference/commandline/run/#env) into the docker run command.

| ENV Variable    | Description |
| -------- | ------- |
|   PINGBACK_ENABLED  | Default is true; if set to the string "false", pingback messages are not sent back to Roboflow.   |
|   PINGBACK_URL  | Default is `https://api.roboflow.com/pingback`   |
| PINGBACK_INTERVAL_SECONDS | Frequency of sending pingback messages, default is 3600 seconds |
| ROBOFLOW_SERVER_UUID  | If this is set, the ID of the process reported back to Roboflow's UI is the value of this environment variable. Omitting this causes the process (docker container) to generate a new UUID.    |
| ENABLE_PROMETHEUS | if set to any value, this will cause a /metrics endpoint to be created with some FastAPI metrics for Prometheus to scrape; not applicable to the lambda inference server     |

## Community Resources 📚

- [Roboflow Inference Documentation](https://inference.roboflow.com/)

## Roadmap

- [ ] Add support for more models

## Contributing ⌨️

Thank you for your interest in contributing to the Roboflow Inference! You can learn more about how to start contributing in our [contributing guide](https://github.com/roboflow/roboflow-inference-server/blob/master/CONTRIBUTING.md).

## License 📝

The Roboflow Inference code, enclosed within `inference/core`, as well as the documentation in `docs/`, is licnsed under an [Apache 2.0 license](LICENSE).

The following models, accessible through, Roboflow Inference comes with their own licenses:

- `inference/models/clip`: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE).
- `inference/models/sam`: [Apache 2.0](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE).
- `inference/models/yolact`: [MIT](https://github.com/dbolya/yolact/blob/master/README.md).
- `inference/models/yolov5`: [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE).
- `inference/models/yolov7`: [GPL-3.0](https://github.com/WongKinYiu/yolov7/blob/main/README.md).
- `inference/models/yolov8`: [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/master/LICENSE).

Roboflow Inference offers more features to Enterprise License holders, including:

1. Running Inference on a cluster of multiple servers
2. Handling auto-batched inference
3. Device management
4. Active learning
5. Sub-license YOLOv5 and YOLOv8 models for enterprise use
6. And more.

To learn more about a Roboflow Inference Enterprise License, [contact us](https://roboflow.com/sales).

## Build the Documentation

Roboflow Inference uses `mkdocs` and `mike` to offer versioned documentation. The project documentation is hosted at [https://inference.roboflow.com](https://inference.roboflow.com)

To build the Inference documentation, first install the project development dependencies:

```bash
pip install -r requirements/requirements.docs.txt
```

To run the latest version of the documentation, run:

```bash
mike serve
```

Before a new release is published, a new version of the documentation should be built. To create a new version, run:

```bash
mike deploy <version-number>
```
