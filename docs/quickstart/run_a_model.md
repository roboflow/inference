Let's run a computer vision model with Inference. There are two ways to do this: the [inference Python package](../using_inference/about.md) which loads and runs models directly in your process, or the [inference-sdk](../inference_helpers/inference_sdk.md) which sends requests to an Inference Server over HTTP.

## Install

=== "inference-sdk (HTTP client)"

    ```
    pip install inference-sdk
    ```

    This will install lightweight HTTP client that sends requests to Inference Server.

=== "inference (native)"

    ```
    pip install inference
    ```
    Or, if you have NVIDIA GPU, you can install `inference-gpu` package instead:
    ```
    pip install --extra-index-url https://download.pytorch.org/whl/cu124 inference-gpu
    # please adjust the --extra-index-url to CUDA version installed in your OS
    # https://download.pytorch.org/whl/cu<major><minor>, for instance https://download.pytorch.org/whl/cu130 for CUDA 13.0
    # alternativelly use
    uv pip install inference-gpu
    ```
    GPU installation requires CUDA available in the OS - check 
    [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) or 
    [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) CUDA installation guide if
    your environment lacks required dependencies.

    Starting from `inference 1.2.0`, the new inference engine — called `inference-models` — is used by default.
    It brings support for different model backends, like TensorRT. By default, `inference` installs the dependencies
    required to support `torch` and `onnx` models. Additional dependencies can be installed via `inference-models` 
    package extras. For instance, to install TRT dependencies:
    ```
    pip install inference-models[trt10]
    ```
    See the [full installation guide](https://inference-models.roboflow.com/getting-started/installation/) for 
    more details.

## Load a Model and Run Inference

=== "inference-sdk (HTTP client)"

    ```python
    from inference_sdk import InferenceHTTPClient

    image = "https://media.roboflow.com/inference/people-walking.jpg"
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",  # or "http://localhost:9001" for self-hosted
        api_key="ROBOFLOW_API_KEY",
    )
    results = client.infer(image, model_id="rfdetr-small")
    ```

    `InferenceHTTPClient` sends requests to an Inference Server (Roboflow-hosted or [self-hosted](../inference_helpers/inference_cli.md)). See the [`inference-sdk` docs](../inference_helpers/inference_sdk.md) for more details.

=== "inference (native)"

    ```python
    from inference import get_model

    image = "https://media.roboflow.com/inference/people-walking.jpg"
    model = get_model(model_id="rfdetr-small")
    results = model.infer(image)
    ```

    `get_model()` downloads model weights and runs inference locally. See the [`inference` package docs](../using_inference/about.md) for more details.

When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

## Model IDs

The `model_id` parameter can be:

- A [pre-trained model](../quickstart/aliases.md) alias (e.g. `rfdetr-small`, `rfdetr-large`)
- Your own [fine-tuned model](../quickstart/explore_models.md) from Roboflow (e.g. `my-project/1`)
- A [Universe model](../quickstart/load_from_universe.md) (e.g. [soccer-players-xy9vk/2](https://universe.roboflow.com/soccer-players/soccer-players-xy9vk/model/2))

Fine-tuned models and Universe models require an [API key](https://app.roboflow.com/settings/api).

## Visualize Results

To visualize results, also install [Supervision](https://supervision.roboflow.com):

```
pip install supervision
```

=== "inference-sdk (HTTP client)"

    ```python
    from io import BytesIO

    import requests
    import supervision as sv
    from inference_sdk import InferenceHTTPClient
    from PIL import Image

    image = Image.open(
        BytesIO(requests.get("https://media.roboflow.com/inference/people-walking.jpg").content)
    )

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="ROBOFLOW_API_KEY",
    )
    results = client.infer(image, model_id="rfdetr-medium")

    detections = sv.Detections.from_inference(results)

    annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)
    ```

=== "inference (native)"

    ```python
    from io import BytesIO

    import requests
    import supervision as sv
    from inference import get_model
    from PIL import Image

    image = Image.open(
        BytesIO(requests.get("https://media.roboflow.com/inference/people-walking.jpg").content)
    )

    model = get_model(model_id="rfdetr-medium")
    results = model.infer(image)[0]

    detections = sv.Detections.from_inference(results)

    annotated_image = sv.BoxAnnotator().annotate(scene=image, detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    sv.plot_image(annotated_image)
    ```

![People Walking Annotated](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg)

## Next Steps

There are many different ways to use Inference depending on your use case and deployment environment. [Learn more about how to use inference here](../quickstart/inference_101.md).
