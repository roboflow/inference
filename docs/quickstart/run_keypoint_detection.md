# Keypoint Detection

Running a keypoint detection model on Roboflow is very similar to segmentation or detection.

You may run it locally, hosted on our inference servers, or using a docker container.

<details>
<summary>💡 model weights</summary>

In all cases, model weights need to be downloaded from Roboflow's servers first.

If you have the weights locally, you may upload the weights to our servers using the [From Local Weights](https://inference.roboflow.com/models/from_local_weights/) guide.

For offline usage, run inference with the Python API once. The weights will be downloaded and cached in the format our inference runtime can parse.

</details>

## Quickstart

Install dependencies:

```bash
pip install inference
```

Set up the [API key](https://inference.roboflow.com/quickstart/configure_api_key/):

```bash
export ROBOFLOW_API_KEY=MY_ROBOFLOW_API_KEY
```

Run with Python API:

```python
from inference import get_model


image = "https://media.roboflow.com/inference/people-walking.jpg"

model = get_model(model_id="yolov8x-pose-640")
results = model.infer(image)[0]
```

## Details

### Inference Setup

In all cases, you'll need the `inference` package.

```bash
pip install inference
```

By default, it runs on the CPU. Instead, you may install the GPU module with the following command:

```bash
pip install inference-gpu
```

### API Keys

You'll need the API key to access the [fine-tuned models](https://app.roboflow.com/) or models on the [Roboflow Universe](https://universe.roboflow.com/). A guide can be found in [Retrieve Your API Key](https://inference.roboflow.com/quickstart/configure_api_key/).

```bash
export ROBOFLOW_API_KEY=MY_ROBOFLOW_API_KEY
```

### Available Pretrained Models

You may use keypoint detection models available on the [Universe](https://universe.roboflow.com/search?q=keypoint+detection+model&t=metadata). Alternatively, here's a few `model_ids` that we support out-of-the-box:

- `yolov8x-pose-1280` (largest)
- `yolov8x-pose-640`
- `yolov8l-pose-640`
- `yolov8m-pose-640`
- `yolov8s-pose-640`
- `yolov8n-pose-640` (smallest)

!!! Run Keypoint Detection

    === "Python API - Image"

        Run the model locally, without needing to set up a docker container. This pulls the model from roboflow servers and runs it on your machine. It can take both images and videos as input.

        Run:

        ```python
        from inference import get_model


        # This can be a URL, a np.ndarray or a PIL image.
        image = "https://media.roboflow.com/inference/people-walking.jpg"

        model = get_model(model_id="yolov8x-pose-640")
        results = model.infer(image)[0]
        ```

    === "Inference Pipeline - Stream"

        Inference Pipeline allows running inference on videos, webcams and RTSP streams. You may define a custom sink to extract pose results.

        More details can be found on [Predict on a Video, Webcam or RTSP Stream](https://inference.roboflow.com/quickstart/run_model_on_rtsp_webcam/)

        ```python
        from inference import InferencePipeline
        from inference.core.interfaces.camera.entities import VideoFrame

        def my_custom_sink(predictions: dict, video_frame: VideoFrame):
            print(predictions)

        pipeline = InferencePipeline.init(
            model_id="yolov8x-pose-640", # Roboflow model to use
            video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
            on_prediction=my_custom_sink, # Function to run after each prediction
        )
        pipeline.start()
        pipeline.join()
        ```

    === "Hosted"

        Send an image to our servers and get the detected keypoint response. Only images are supported (URL, `np.ndarray`, `PIL`).

        ```python
        import os
        from inference_sdk import InferenceHTTPClient


        # This can be a URL, a np.ndarray or a PIL image.
        image = "https://media.roboflow.com/inference/people-walking.jpg"

        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.environ["ROBOFLOW_API_KEY"]
        )
        results = client.infer(image, model_id="yolov8x-pose-640")
        ```

    === "Docker server"

        With this method, you may self-host a server container, similar to Hosted model API. Only images are supported (URL, `np.ndarray`, `PIL`).

        Note that the model weights still need to be retrieved from our servers at least once. Check out [From Local Weights](https://inference.roboflow.com/models/from_local_weights/) for instructions on how to upload yours.

        Start the inference server:

        ```bash
        inference server start
        ```

        Run:
        ```python
        import os
        from inference_sdk import InferenceHTTPClient


        # This can be a URL, a np.ndarray or a PIL image.
        image = "https://media.roboflow.com/inference/people-walking.jpg"

        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=os.environ["ROBOFLOW_API_KEY"]
        )
        results = client.infer(image, model_id="yolov8x-pose-640")
        ```

### Visualize

With [supervision](https://supervision.roboflow.com/latest/) you may visualize the results, carry out post-processing. Supervision library standardizes results from various keypoint detection and pose estimation models into a consistent format, using adaptors such as `from_inference`.

Example usage:

```python
import os
import cv2
from inference import get_model
import supervision as sv


# Model accepts URLs, np.arrays (cv2.imread), and PIL images.
# Annotators accept np.arrays (cv2.imread), and PIL images
image = "https://media.roboflow.com/inference/people-walking.jpg"

model = get_model(model_id="yolov8x-pose-640")
results = model.infer(image)[0]

# Any results object would work, regardless of which inference API is used
keypoints = sv.KeyPoints.from_inference(results)

# Convert to numpy image
img_name = "people-walking.jpg"
if not os.path.exists(img_name):
    os.system(f"wget -O {img_name} {image}")
image_np = cv2.imread(img_name)

annotated_image = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=5
).annotate(image, keypoints)
```
