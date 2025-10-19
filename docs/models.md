![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

<a href="https://roboflow.com" target="_blank">Roboflow</a> Inference enables you to deploy computer vision models faster than ever.

With a `pip install inference` and `inference server start`, you can start a server to run a fine-tuned model on images, videos, and streams.

Inference supports running object detection, classification, instance segmentation, and foundation models (i.e. SAM, CLIP).

You can <a href="https://github.com/roboflow/notebooks" target="_blank">train and deploy your own custom model</a> or use one of the 50,000+
<a href="https://universe.roboflow.com" target="_blank">fine-tuned models shared by the Roboflow Universe community</a>.

You can run Inference on an edge device like an NVIDIA Jetson, or on cloud computing platforms like AWS, GCP, and Azure.

<a href="https://inference.roboflow.com/quickstart/run_a_model/" class="button">Get started with our "Run your first model" guide</a>

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

- A server, so you don’t have to reinvent the wheel when it comes to serving your model to disperate parts of your application.

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

Roboflow Inference uses Onnxruntime as its core inference engine. Onnxruntime provides an array of different <a href="https://onnxruntime.ai/docs/execution-providers/" target="_blank">execution providers</a> that can optimize inference on differnt target devices. If you decide to install onnxruntime on your own, install inference with:

```bash
pip install inference-core
```

Alternatively, you can take advantage of some advanced execution providers using one of our published docker images.

### Extras:

Some functionality requires extra dependencies. These can be installed by specifying the desired extras during installation of Roboflow Inference. e.x. `pip install inference[extra]`

| extra | description                                                                                                                                                                                                                         |
|:-------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `clip` | Ability to use the core `CLIP` model (by OpenAI)                                                                                                                                                                                    |
| `gaze` | Ability to use the core `Gaze` model                                                                                                                                                                                                |
| `http` | Ability to run the http interface                                                                                                                                                                                                   |
| `sam`  | Ability to run the core `Segment Anything` model (by Meta AI)                                                                                                                                                                       |
| `doctr` | Ability to use the core `doctr` model (by <a href="https://github.com/mindee/doctr" target="_blank">Mindee</a>)                                                                                                                     |
| `easy-ocr` | Ability to use the core `easy-ocr` model (by <a href="https://github.com/JaidedAI/EasyOCRr" target="_blank">JaidedAI</a>)                                                                                                                  |
| `transformers` | Ability to use transformers based multi-modal models such as `Florence2` and `PaliGemma`. To use Florence2 you will need to manually install <a href="https://github.com/Dao-AILab/flash-attention/" target="_blank">flash_attn</a> |

**_Note:_** Both CLIP and Segment Anything require PyTorch to run. These are included in their respective dependencies however PyTorch installs can be highly environment dependent. See the <a href="https://pytorch.org/get-started/locally/" target="_blank">official PyTorch install page</a> for instructions specific to your enviornment.

Example install with CLIP dependencies:

```bash
pip install inference[clip]
```

## 🐋 docker

You can learn more about Roboflow Inference Docker Image build, pull and run in our <a href="https://roboflow.github.io/inference/quickstart/docker/" target="_blank">documentation</a>.

- Run on x86 CPU:

```bash
docker run -it --net=host roboflow/roboflow-inference-server-cpu:latest
```

- Run on NVIDIA GPU:

```bash
docker run -it --network=host --gpus=all roboflow/roboflow-inference-server-gpu:latest
```

<details close>
<summary>👉 more docker run options</summary>

- Run on arm64 CPU:

```bash
docker run -p 9001:9001 roboflow/roboflow-inference-server-arm-cpu:latest
```

- Run on NVIDIA Jetson with JetPack `4.x` (Deprecated):

```bash
docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson:latest
```

- Run on NVIDIA Jetson with JetPack `5.x`:

```bash
docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson-5.1.1:latest
```

- Run on NVIDIA Jetson with JetPack `6.x`:

```bash
docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson-6.0.0:latest
```

</details>

<br/>

## 📟 CLI

To use the CLI you will need python 3.7 or higher. To ensure you have the correct version of python, run `python --version` in your terminal. To install python, follow the instructions <a href="https://www.python.org/downloads/" target="_blank">here</a>.

After you have python installed, install the pypi package `inference-cli` or `inference`:

```bash
pip install inference-cli
```

From there you can run the inference server. See [Docker quickstart via CLI](./quickstart/docker.md#set-up-a-docker-inference-server-via-inference-server-start) for more information.

```bash
inference server start
```

CLI supports also stopping the server via:
```bash
inference server stop
```

To use the CLI to make inferences, first <a href="https://docs.roboflow.com/api-reference/workspace-and-project-ids" target="_blank">find your project ID and model version number in Roboflow</a>.

See more detailed documentation on [HTTP Inference quickstart](./quickstart/http_inference.md).

```bash
inference infer {image_path} \
    --project-id {project_id} \
    --model-version {model_version} \
    --api-key {api_key}
```
## Enterprise License

With a Roboflow Inference Enterprise License, you can access additional Inference features, including:

- Server cluster deployment
- Active learning
- YOLOv5 and YOLOv8 model sub-license

To learn more, <a href="https://roboflow.com/sales" target="_blank">contact the Roboflow team</a>.

## More Roboflow Open Source Projects

|Project | Description|
|:---|:---|
|<a href="https://roboflow.com/supervision" target="_blank">supervision</a> | General-purpose utilities for use in computer vision projects, from predictions filtering and display to object tracking to model evaluation.
|<a href="https://github.com/autodistill/autodistill" target="_blank">Autodistill</a> | Automatically label images for use in training computer vision models. |
|<a href="https://github.com/roboflow/inference" target="_blank">Inference</a> (this project) | An easy-to-use, production-ready inference server for computer vision supporting deployment of many popular model architectures and fine-tuned models.
|<a href="https://roboflow.com/notebooks" target="_blank">Notebooks</a> | Tutorials for computer vision tasks, from training state-of-the-art models to tracking objects to counting objects in a zone.
|<a href="https://github.com/roboflow/roboflow-collect" target="_blank">Collect</a> | Automated, intelligent data collection powered by CLIP.