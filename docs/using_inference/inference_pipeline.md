The Inference Pipeline interface is made for streaming and is likely the best route to go for real time use cases. It is an asynchronous interface that can consume many different video sources including local devices (like webcams), RTSP video streams, video files, etc. With this interface, you define the source of a video stream and sinks.

## Quickstart

First, install Inference:

```bash
pip install inference
```

Next, create an Inference Pipeline:

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=render_boxes,
    api_key=api_key,
)
pipeline.start()
pipeline.join()
```

Let's break down the example line by line:

#### `pipeline = InferencePipeline.init(...)`

Here, we are calling a class method of InferencePipeline.

#### `model_id="yolov8x-1280"`

We set the model ID to a YOLOv8x model with input resolution `1280x1280`.

#### `video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4"`

We set the video reference to a URL. Later we will show the various values that can be used as a video reference.

#### `on_prediction=render_boxes`

The `on_prediction` argument defines our sink (or a list of sinks).

#### `pipeline.start(); pipeline.join()`

Here, we start and join the thread that processes the video stream.

## Sinks

Sinks define what an Inference Pipeline should do with each prediction. A sink is a function with definition:

```python
def on_prediction(
    predictions: dict,
    video_frame: VideoFrame,
    **kwargs
)
```

The arguments are:

- `predictions`: A dictionary that is the response object resulting from a call to a model's `infer(...)` method.
- `video_frame`: A [VideoFrame object](../../docs/reference/inference/core/interfaces/camera/entities/#inference.core.interfaces.camera.entities.VideoFrame) containing metadata and pixel data from the video frame.
- `**kwargs`: Other keyward arguments can be defined for the ability to configure a sink.

### Built In Sinks

Inference has [several sinks built in](../../docs/reference/inference/core/interfaces/stream/sinks/) that are ready to use.

#### `render_boxes(...)`

The [render boxes sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.render_boxes) is made to visualize predictions and overlay them on a stream. It uses Supervision annotators to render the predictions and display the annotated frame.

#### `UDPSink(...)`

The [UDP sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.UDPSink) is made to broadcast predictions with a UDP port. This port can be listened to by client code for further processing.

#### `multi_sink(...)`

The [Multi-Sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.multi_sink) is a way to combine multiple sinks so that multiple actions can happen on a single inference result.

## Other Pipeline Configuration

Inference Pipelines are highly configurable. Configurations include:

- `max_fps`: Used to set the maximum rate of frame processing.
- `confidence`: Confidence threshold used for inference.
- `iou_threshold`: IoU threshold used for inference.

```python
pipeline = InferencePipeline.init(
    ...,
    max_fps=10,
    confidence=0.75,
    iou_threshold=0.4,
)
```

See the reference docs for the [full list of Inference Pipeline parameters](../../docs/reference/inference/core/interfaces/stream/inference_pipeline/#inference.core.interfaces.stream.inference_pipeline.InferencePipeline).
