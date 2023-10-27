![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

[Roboflow](https://roboflow.com) Inference enables you to deploy computer vision models faster than ever.

With a `pip install inference` and `inference server start`, you can start a server to run a fine-tuned model on images, videos, and streams.

Inference supports running object detection, classification, instance segmentation, and foundation models (i.e. SAM, CLIP).

You can [train and deploy your own custom model](https://github.com/roboflow/notebooks) or use one of the 50,000+
[fine-tuned models shared by the Roboflow Universe community](https://universe.roboflow.com).

<a href="https://docs.roboflow.com/inference/quickstart/run_a_model" class="button">Get started with our "Run your first model" guide</a>

<style>
  .button {
    background-color: var(--md-primary-fg-color);
    display: block;
    padding: 10px;
    color: white !important;
    border-radius: 5px;
    text-align: center;
  }
</style>

Here is an example of a model running on a video using Inference:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/football-video.mp4" type="video/mp4">
</video>

## 💻 Features

Inference provides a scalable method through which you can use computer vision models.

Inference is backed by:

- A server, so you don’t have to reimplement things like image processing and prediction visualization on every project.

- Standard APIs for computer vision tasks, so switching out the model weights and architecture can be done independently of your application code.

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
| `sam`  | Ability to run the core `Segment Anything` model (by Meta AI) |
| `doctr` | Ability to use the core `doctr` model (by [Mindee](https://github.com/mindee/doctr)) |

**_Note:_** Both CLIP and Segment Anything require PyTorch to run. These are included in their respective dependencies however PyTorch installs can be highly environment dependent. See the [official PyTorch install page](https://pytorch.org/get-started/locally/) for instructions specific to your enviornment.

Example install with CLIP dependencies:

```bash
pip install inference[clip]
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

## 📟 CLI

To use the CLI you will need python 3.7 or higher. To ensure you have the correct version of python, run `python --version` in your terminal. To install python, follow the instructions [here](https://www.python.org/downloads/).

After you have python installed, install the pypi package `inference-cli` or `inference`:

```bash
pip install inference-cli
```

From there you can run the inference server. See [Docker quickstart via CLI](./quickstart/docker.md/#via-cli) for more information.

```bash
inference server start
```

To use the CLI to make inferences, first find your project ID and model version number in the Roboflow documentation, [Workspace and Project IDs](https://docs.roboflow.com/api-reference/workspace-and-project-ids).

See more detailed documentation on [HTTP Inference quickstart via CLI](./quickstart/http_inference.md/#via-cli).

```bash
inference infer {image_path} \
    --project-id {project_id} \
    --model-version {model_version} \
    --api-key {api_key}
```
## Enterprise License

With a Roboflow Inference Enterprise License, you can access additional Inference features, including:

- Server cluster deployment
- Device management
- Active learning
- YOLOv5 and YOLOv8 model sub-license

To learn more, [contact the Roboflow team](https://roboflow.com/sales).

## More Roboflow Open Source Projects

|Project | Description|
|:---|:---|
|[supervision](https://roboflow.com/supervision) | General-purpose utilities for use in computer vision projects, from predictions filtering and display to object tracking to model evaluation.
|[Autodistill](https://github.com/autodistill/autodistill) | Automatically label images for use in training computer vision models. |
|[Inference](https://github.com/roboflow/inference) (this project) | An easy-to-use, production-ready inference server for computer vision supporting deployment of many popular model architectures and fine-tuned models.
|[Notebooks](https://roboflow.com/notebooks) | Tutorials for computer vision tasks, from training state-of-the-art models to tracking objects to counting objects in a zone.
|[Collect](https://github.com/roboflow/roboflow-collect) | Automated, intelligent data collection powered by CLIP.