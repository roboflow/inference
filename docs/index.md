---
hide:
  - path
  - navigation
  - toc
---

<style>
/* Hide <h1> on homepage */
.md-typeset h1 {
  display: none;
}
.md-main__inner {
  margin-top: -1rem;
}

/* constrain to same width even w/o sidebar */
.md-content {
  max-width: 50rem;
  margin: auto;
}

/* hide edit button */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>

![Roboflow Inference banner](https://github.com/roboflow/inference/blob/main/banner.png?raw=true)

Roboflow Inference is an open-source platform designed to simplify the deployment of computer vision models. It enables developers to perform object detection, classification, instance segmentation and [keypoint detection](/quickstart/run_keypoint_detection.md), and utilize foundation models like [CLIP](/foundation/clip.md), [Segment Anything](/foundation/sam.md), and [YOLO-World](/foundation/yolo_world.md) through a Python-native package, a self-hosted inference server, or a fully [managed API](https://docs.roboflow.com/).

Explore our [enterprise options](https://roboflow.com/sales) for advanced features like server deployment, active learning, and commercial licenses for YOLOv5 and YOLOv8.

<a href="/quickstart/run_a_model/" class="button">Get started with our "Run your first model" guide</a>

<div class="button-holder">
<a href="/quickstart/inference_101/" class="button half-button">Learn about the various ways you can use Inference</a>
<a href="/workflows/about/" class="button half-button">Build a visual agent with Workflows</a>
</div>

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

## 💻 install

Inference package requires [**Python>=3.8,<=3.11**](https://www.python.org/). Click [here](/quickstart/docker.md) to learn more about running Inference inside Docker.

```bash
pip install inference
```

<details>
<summary>👉 running on a GPU</summary>

  To enhance model performance in GPU-accelerated environments, install CUDA-compatible dependencies instead:
  
  ```bash
  pip install inference-gpu
  ```
</details>

<details>
<summary>👉 advanced models</summary>

  Inference supports multiple model types for specialized tasks. From Grounding DINO for identifying objects with a text prompt, to DocTR for OCR, to CogVLM for asking questions about images - you can find out more in the <a href="/foundation/about">Foundation Models</a> page.

  <br/><br/>

  Note that <code>inference</code> and <code>inference-gpu</code> packages install only the minimal shared dependencies. <b>Instead</b>, install model-specific dependencies to ensure code compatibility and license compliance.

  <br/><br/>

  The <code>inference</code> and <code>inference-gpu</code> packages install only the minimal shared dependencies. Install model-specific dependencies to ensure code compatibility and license compliance. Learn more about the <a href="#extras">models</a> supported by Inference.

  ```bash
  pip install inference[yolo-world]
  ```

</details>

## 🔥 quickstart

Use Inference SDK to run models locally with just a few lines of code. The image input can be a URL, a numpy array, or a PIL image.

```python
from inference import get_model

model = get_model(model_id="yolov8n-640")

results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")
```

<details>
<summary>👉 roboflow models</summary>

<br>

Set up your <code>ROBOFLOW_API_KEY</code> to access thousands of fine-tuned models shared by the <a href="https://universe.roboflow.com/">Roboflow Universe</a> community and your custom model. Navigate to 🔑 keys section to learn more.

```python
from inference import get_model

model = get_model(model_id="soccer-players-5fuqs/1")

results = model.infer(
    image="https://media.roboflow.com/inference/soccer.jpg",
    confidence=0.5,
    iou_threshold=0.5
)
```

</details>

<details>
<summary>👉 foundational models</summary>

- <a href="/foundation/clip">CLIP Embeddings</a> - generate text and image embeddings that you can use for zero-shot classification or assessing image similarity.

  ```python
  from inference.models import Clip

  model = Clip()

  embeddings_text = clip.embed_text("a football match")
  embeddings_image = model.embed_image("https://media.roboflow.com/inference/soccer.jpg")
  ```

- <a href="/foundation/sam">Segment Anything</a> - segment all objects visible in the image or only those associated with selected points or boxes.

  ```python
  from inference.models import SegmentAnything

  model = SegmentAnything()

  result = model.segment_image("https://media.roboflow.com/inference/soccer.jpg")
  ```

- <a href="/foundation/yolo_world">YOLO-World</a> - an almost real-time zero-shot detector that enables the detection of any objects without any training.

  ```python
  from inference.models import YOLOWorld

  model = YOLOWorld(model_id="yolo_world/l")

  result = model.infer(
      image="https://media.roboflow.com/inference/dog.jpeg",
      text=["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
      confidence=0.03
  )
  ```

</details>

## 📟 inference server

You can also run Inference as a microservice with Docker.

### deploy server

The inference server is distributed via Docker. Behind the scenes, inference will download and run the image that is appropriate for your hardware. [Here](/quickstart/docker.md#advanced-build-a-docker-container-from-scratch), you can learn more about the supported images.

```bash
inference server start
```

### run client

Consume inference server predictions using the HTTP client available in the Inference SDK.

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=<ROBOFLOW_API_KEY>
)
with client.use_model(model_id="soccer-players-5fuqs/1"):
    predictions = client.infer("https://media.roboflow.com/inference/soccer.jpg")
```

If you're using the hosted API, change the local API URL to `https://detect.roboflow.com`. Accessing the hosted inference server and/or using any of the fine-tuned models require a `ROBOFLOW_API_KEY`. For further information, visit the 🔑 keys section.

## 🎥 inference pipeline

The inference pipeline is an efficient method for processing static video files and streams. Select a model, define the video source, and set a callback action. You can choose from predefined callbacks that allow you to [display results](/reference/inference/core/interfaces/stream/sinks.md#inference.core.interfaces.stream.sinks.render_boxes) on the screen or [save them to a file](/reference/inference/core/interfaces/stream/sinks.md#inference.core.interfaces.stream.sinks.VideoFileSink).

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://media.roboflow.com/inference/people-walking.mp4",
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()
```

## 🔑 keys

Inference enables the deployment of a wide range of pre-trained and foundational models without an API key. To access thousands of fine-tuned models shared by the [Roboflow Universe](https://universe.roboflow.com/) community, [configure your](https://app.roboflow.com/settings/api) API key.

```bash
export ROBOFLOW_API_KEY=<YOUR_API_KEY>
```

## 📚 documentation

Visit our [documentation](/) to explore comprehensive guides, detailed API references, and a wide array of tutorials designed to help you harness the full potential of the Inference package.

## © license

The Roboflow Inference code is distributed under the [Apache 2.0](https://github.com/roboflow/inference/blob/master/LICENSE.core) license. However, each supported model is subject to its licensing. Detailed information on each model's license can be found [here](https://roboflow.com/licensing).


## ⚡️ extras

Below you can find list of extras available for `inference` and `inference-gpu`

<table>
<tr>
  <th>Name</th>
  <th style="width:30%">Description</th>
  <th style="width:50%">Notes</th>
</tr>
<tr>
  <td><code>clip</code></td>
  <td><a href="/foundation/clip">CLIP model</a></td>
  <td><code>N/A</code></td>
</tr>
<tr>
  <td><code>gaze</code></td>
  <td><a href="/foundation/gaze">L2CS-Net model</a></td>
  <td><code>N/A</code></td>
</tr>
<tr>
  <td><code>grounding-dino</code></td>
  <td><a href="/foundation/grounding_dino/">Grounding Dino model</a></td>
  <td><code>N/A</code></td>
</tr>
<tr>
  <td><code>sam</code></td>
  <td><a href="/foundation/sam">SAM</a> and <a href="/foundation/sam2">SAM2</a> models</td>
  <td>The extras depend on <code>rasterio</code> which require <code>GDAL</code> library to work. If the installation fails with <code>gdal-config</code> command error - run <code>sudo apt-get install libgdal-dev</code> for Linux or follow <a href="https://gdal.org/en/latest/download.html#binaries">official installation guide</a></td>
</tr>
<tr>
  <td><code>yolo-world</code></td>
  <td><a href="/foundation/yolo_world/">Yolo-World model</a></td>
  <td><code>N/A</code></td>
</tr>
<tr>
  <td><code>transformers</code></td>
  <td><code>transformers</code> based models, like <a href="/foundation/florence2/">Florence-2</a></td>
  <td><code>N/A</code></td>
</tr>
</table>

??? note "Installing extras"

    To install specific extras you need to run

    ```bash
    pip install inferenence[extras-name]
    ```
    or 

    ```bash
    pip install inferenence-gpu[extras-name]
    ```