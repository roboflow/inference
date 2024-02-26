The Inference Pipeline interface is made for streaming and is likely the best route to go for real time use cases. It is an asynchronous interface that can consume many different video sources including local devices (like webcams), RTSP video streams, video files, etc. With this interface, you define the source of a video stream and sinks.

## Quickstart

First, install Inference:

{% include 'install.md' %}

Next, create an Inference Pipeline:

```python
# import the InferencePipeline interface
from inference import InferencePipeline
# import a built in sink called render_boxes (sinks are the logic that happens after inference)
from inference.core.interfaces.stream.sinks import render_boxes

# create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="yolov8x-1280", # set the model id to a yolov8x model with in put size 1280
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4", # set the video reference (source of video), it can be a link/path to a video file, an RTSP stream url, or an integer representing a device id (usually 0 for built in webcams)
    on_prediction=render_boxes, # tell the pipeline object what to do with each set of inference by passing a function
    api_key=api_key, # provide your roboflow api key for loading models from the roboflow api
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
```

Let's break down the example line by line:

**`pipeline = InferencePipeline.init(...)`**

Here, we are calling a class method of InferencePipeline.

**`model_id="yolov8x-1280"`**

We set the model ID to a YOLOv8x model pre-trained on COCO with input resolution `1280x1280`.

**`video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4"`**

We set the video reference to a URL. Later we will show the various values that can be used as a video reference.

**`on_prediction=render_boxes`**

The `on_prediction` argument defines our sink (or a list of sinks).

**`pipeline.start(); pipeline.join()`**

Here, we start and join the thread that processes the video stream.

## Video Reference

Inference Pipelines can consume many different types of video streams.

- Device Id (integer): Providing an integer instructs a pipeline to stream video from a local device, like a webcam. Typically, built in webcams show up as device `0`.
- Video File (string): Providing the path to a video file will result in the pipeline reading every frame from the file, running inference with the specified model, then running the `on_prediction` method with each set of resulting predictions.
- Video URL (string): Providing the path to a video URL is equivalent to providing a video file path and voids needing to first download the video.
- RTSP URL (string): Providing an RTSP URL will result in the pipeline streaming frames from an RTSP stream as fast as possible, then running the `on_prediction` callback on the latest available frame.

## Sinks

Sinks define what an Inference Pipeline should do with each prediction. A sink is a function with signature:

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

### Custom Sinks

To create a custom sink, define a new function with the appropriate signature.

```python
# import the VideoFrame object for type hints
from inference.core.interfaces.camera.entities import VideoFrame

def my_custom_sink(
    prediction: dict, # predictions are dictionaries
    video_frame: VideoFrame, # video frames are python objects with metadata and the video frame itself
):
    # put your custom logic here
    ...
```

!!! Info

    See our [tutorial on creating a custom Inference Pipeline sink!](/quickstart/create_a_custom_inference_pipeline_sink/)

**prediction**

Predictions are provided to the sink as a dictionary containing keys:

- `predictions` (list): A list of prediction dictionaries

Each prediction dictionary contains keys:

- `x`: The center x coordinate of the predicted bounding box in pixels
- `y`: The center y coordinate of the predicted bounding box in pixels
- `width`: The width of the predicted bounding box in pixels
- `height`: The height of the predicted bounding box in pixels
- `confidence`: The confidence value of the prediction (between 0 and 1)
- `class`: The predicted class name
- `class_id`: The predicted class ID

**video_frame**

The video frame is provided as a video frame object with attributes:

- `image`: A numpy array containing the image pixels
- `frame_id`: An integer of the frame ID, a monotonically increasing integer starting at 0 from the time the pipeline was started
- `frame_timestamp`: A python datetime object of when the frame was captured

### Built In Sinks

Inference has [several sinks built in](../../docs/reference/inference/core/interfaces/stream/sinks/) that are ready to use.

#### `render_boxes(...)`

The [render boxes sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.render_boxes) is made to visualize predictions and overlay them on a stream. It uses Supervision annotators to render the predictions and display the annotated frame.

#### `UDPSink(...)`

The [UDP sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.UDPSink) is made to broadcast predictions with a UDP port. This port can be listened to by client code for further processing.

#### `multi_sink(...)`

The [Multi-Sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.multi_sink) is a way to combine multiple sinks so that multiple actions can happen on a single inference result.

#### `VideoFileSink(...)`

The [Video File Sink](../../docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.VideoFileSink) visualizes predictions, similar to the `render_boxes(...)` sink, however, instead of displaying the annotated frames, it saves them to a video file.

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

## Performance

We tested the performance of Inference on a variety of hardware devices.

Below are the results of our benchmarking tests for Inference.

### MacBook M2

| Test     | FPS |
| -------- | :-: |
| yolov8-n | ~26 |
| yolov8-s | ~12 |
| yolov8-m | ~5  |

Tested against the same 1080p 60fps RTSP stream emitted by localhost.

### Jetson Orin Nano

| Test     | FPS |
| -------- | :-: |
| yolov8-n | ~25 |
| yolov8-s | ~18 |
| yolov8-m | ~8  |

With old version reaching at max 6-7 fps. This test was executed against 4K@60fps stream, which is not possible to
be decoded in native pace due to resource constrains. New implementation proved to run without stability issues
for few hours straight.

### Tesla T4

GPU workstation with Tesla T4 was able to run 4 concurrent HD streams at 15FPS utilising ~80% GPU - reaching
over 60FPS throughput per GPU (against `yolov8-n`).

## Migrating from `inference.Stream` to `InferencePipeline`

Inference is deprecating support for `inference.Stream`, our video stream inference interface. `inference.Stream` is being replaced with `InferencePipeline`, which has feature parity and achieves better performance. There are also new, more advanced features available in `InferencePipeline`.

### New Features in `InferencePipeline`

#### Stability

New implementation allows `InferencePipeline` to re-connect to a video source, eliminating the need to create
additional logic to run inference against streams for long hours in fault-tolerant mode.

#### Granularity of control

New implementation let you decide how to handle video sources - and provided automatic selection of mode.
Your videos will be processed frame-by-frame with each frame being passed to model, and streams will be
processed in a way to provide continuous, up-to-date predictions on the most fresh frames - and the system
will automatically adjust to performance of the hardware to ensure best experience.

#### Observability

New implementation allows to create reports about InferencePipeline state in runtime - providing an easy way to
build monitoring on top of it.

### Migrate from `inference.Stream` to `InferencePipeline`

Let's assume you used `inference.Stream(...)` with your custom handlers:

```python
import numpy as np


def on_prediction(predictions: dict, image: np.ndarray) -> None:
    pass
```

Now, the structure of handlers has changed into:

```python
import numpy as np

def on_prediction(predictions, video_frame) -> None:
    pass
```

With predictions being still dict (passed as second parameter) in the same, standard Roboflow format,
but `video_frame` is a dataclass with the following property:

- `image`: which is video frame (`np.ndarray`)
- `frame_id`: int value representing the place of the frame in stream order
- `frame_timestamp`: time of frame grabbing - the exact moment when frame appeared in the file/stream
  on the receiver side (`datetime.datetime`)

Additionally, it eliminates the need of grabbing `.frame_id` from `inference.Stream()`.

`InferencePipeline` exposes interface to manage its state (possibly from different thread) - including
functions like `.start()`, `.pause()`, `.terminate()`.
