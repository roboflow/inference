# About

The `inference` Python package is the core library that powers Roboflow's computer vision deployment stack. It provides model loading, pre/post-processing, GPU/CPU optimization, and [Workflows](../workflows/about.md) execution, callable directly from Python.

The [Inference Server](../quickstart/docker.md) wraps this package and exposes it over HTTP (distributed as a Docker image with all dependencies installed), but you can also use `inference` directly in your own scripts and applications.

## Multi-Backend Support

Inference 1.0 supports multiple inference runtime backends: ONNX, TensorRT, Hugging Face, and PyTorch. It automatically selects the fastest available backend for your hardware. For example, if you have an NVIDIA GPU or running on Jetson device and a TensorRT engine is available for the model on your platform, Inference will use TensorRT by default.

## Installation

--8<-- "include/install.md"

## Quick Example

`get_model()` loads a model by its ID, downloads the weights, and returns a model object you can call `.infer()` on.

```python
from inference import get_model

model = get_model(model_id="rfdetr-small")
results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")
```

See [Model ID](../quickstart/run_a_model.md#model-ids) to use models that require an API key, set the `ROBOFLOW_API_KEY` environment variable or pass it directly:

```python
model = get_model(model_id="my-project/1", api_key="ROBOFLOW_API_KEY")
```

See the [Run a Model](../quickstart/run_a_model.md) guide for a more detailed walkthrough with visualization.

## Inference Pipeline

`InferencePipeline` provides a streaming interface for running inference on video sources - webcams, RTSP streams, video files, and more. 

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="rfdetr-large",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=render_boxes,
    api_key="ROBOFLOW_API_KEY",
)

pipeline.start()
pipeline.join()
```

The code above will do object detection annotation directly (through the `render_boxes` sink).
For more information, see the [Inference Pipeline](inference_pipeline.md) page.
