The Inference Pipeline interface is made for streaming and is likely the best route to go for real time use cases. 
It is an asynchronous interface that can consume many different video sources including local devices (like webcams), 
RTSP video streams, video files, etc. With this interface, you define the source of a video stream and sinks.

Now, since version `v0.9.18` `InferencePipeline` supports multiple sources of video at the same time! 

## Quickstart

First, install Inference:

{% include 'install.md' %}

Next, create an Inference Pipeline:

```python
# import the InferencePipeline interface
from inference import InferencePipeline
# import a built-in sink called render_boxes (sinks are the logic that happens after inference)
from inference.core.interfaces.stream.sinks import render_boxes

api_key = "YOUR_ROBOFLOW_API_KEY"

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

## What is video reference?

Inference Pipelines can consume many different types of video streams.

- Device Id (integer): Providing an integer instructs a pipeline to stream video from a local device, like a webcam. Typically, built in webcams show up as device `0`.
- Video File (string): Providing the path to a video file will result in the pipeline reading every frame from the file, running inference with the specified model, then running the `on_prediction` method with each set of resulting predictions.
- Video URL (string): Providing the path to a video URL is equivalent to providing a video file path and voids needing to first download the video.
- RTSP URL (string): Providing an RTSP URL will result in the pipeline streaming frames from an RTSP stream as fast as possible, then running the `on_prediction` callback on the latest available frame.
- Since version `0.9.18` - list of elements that may be any of values described above.

## How the `InferencePipeline` works?
![inference pipeline diagram](https://media.roboflow.com/inference/inference-pipeline-diagram.jpg)

`InferencePipeline` spins a video source consumer thread for each provided video reference. Frames from videos are
grabbed by video multiplexer that awaits `batch_collection_timeout` (if source will not provide frame, smaller batch 
will be passed to `on_video_frame(...)`, but missing frames and predictions will be filled with `None` before passing
to `on_prediction(...)`). `on_prediction(...)` may work in `SEQUENTIAL` mode (only one element at once), or `BATCH` 
mode - all batch elements at a time and that can be controlled by `sink_mode` parameter.

For static video files, `InferencePipeline` processes all frames by default, for streams - it is possible to drop
frames from the buffers - in favour of always processing the most recent data (when model inference is slow, more
frames can be accumulated in buffer - stream processing drop older frames and only processes the most recent one).

To enhance stability, in case of streams processing - video sources will be automatically re-connected once 
connectivity is lost during processing. That is meant to prevent failures in production environment when the pipeline
can run long hours and need to gracefully handle sources downtimes.

## How to provide a custom inference logic to `InferencePipeline`

As of `inference>=0.9.16`, Inference Pipelines support running custom inference logic. This means, instead of passing 
a model ID, you can pass a custom callable. This callable should accept and `VideoFrame` return a dictionary with 
results from the processing (as `on_video_frame` handler). It can be model predictions or results of any other processing you wish to execute.
It is **important to note** that the sink being used (`on_prediction` handler you use) - must be adjusted to the
specific format of `on_video_frame(...)` response. This way, you can shape video processing in a way you want.

```python
# This is example, reference implementation - you need to adjust the code to your purposes
import os
import json
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from typing import Any, List

TARGET_DIR = "./my_predictions"

class MyModel:
  
  def __init__(self, weights_path: str):
    self._model = your_model_loader(weights_path)

  # before v0.9.18  
  def infer(self, video_frame: VideoFrame) -> Any:
    return self._model(video_frame.image)
  
  # after v0.9.18  
  def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
    # result must be returned as list of elements representing model prediction for single frame
    # with order unchanged.
    return self._model([v.image for v in video_frames])
  
def save_prediction(prediction: dict, video_frame: VideoFrame) -> None:
  with open(os.path.join(TARGET_DIR, f"{video_frame.frame_id}.json")) as f:
    json.dump(prediction, f)

my_model = MyModel("./my_model.pt")

pipeline = InferencePipeline.init_with_custom_logic(
  video_reference="./my_video.mp4",
  on_video_frame=my_model.infer,
  on_prediction=save_prediction,
)

# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
```

## `InferencePipeline` and Roboflow `workflows`

!!! Info

    This is feature preview. Please refer to [workflows docs](https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows).
  
    Feature preview do not support multiple videos input!

We are working to make `workflows` compatible with `InferencePipeline`. Since version `0.9.16` we introduce 
an initializer to be used with workflow definitions. Here is the example:

```python
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes

def workflows_sink(
    predictions: dict,
    video_frame: VideoFrame,
) -> None:
    render_boxes(
        predictions["predictions"][0],
        video_frame,
        display_statistics=True,
    )


# here you may find very basic definition of workflow - with a single object detection model.
# Please visit workflows docs: https://github.com/roboflow/inference/tree/main/inference/enterprise/workflows to
# find more examples.
workflow_specification = {
    "specification": {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.5,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.*"},
        ],
    }
}
pipeline = InferencePipeline.init_with_workflow(
    video_reference="./my_video.mp4",
    workflow_specification=workflow_specification,
    on_prediction=workflows_sink,
    image_input_name="image",  # adjust according to name of WorkflowImage input you define
    video_metadata_input_name="video_metadata" # AVAILABLE from v0.17.0! adjust according to name of WorkflowVideoMetadata input you define
)

# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
```

Additionally, since `v0.9.21`, you can initialise `InferencePipeline` with `workflow` registered
in Roboflow App - providing your `workspace_name` and `workflow_id`:

```python
pipeline = InferencePipeline.init_with_workflow(
    video_reference="./my_video.mp4",
    workspace_name="<your_workspace>",
    workflow_id="<your_workflow_id_to_be_found_in_workflow_url>",
    on_prediction=workflows_sink,
)
```

!!! tip "Workflows profiling"

    Since `inference v0.22.0`, you may profile your Workflow execution inside `InferencePipeline` when 
    you export environmental variable `ENABLE_WORKFLOWS_PROFILING=True`. Additionally, you can tune the 
    number of frames you keep in profiler buffer via another environmental variable `WORKFLOWS_PROFILER_BUFFER_SIZE`.
    `init_with_workflow(...)` was also given a new parameter `profiling_directory` which can be adjusted to 
    dictate where to save the trace. 

## Sinks

Sinks define what an Inference Pipeline should do with each prediction. A sink is a function with signature:

### Before `v0.9.18`
```python
from inference.core.interfaces.camera.entities import VideoFrame


def on_prediction(
    predictions: dict,
    video_frame: VideoFrame,
) -> None:
    pass
```

The arguments are:

- `predictions`: A dictionary that is the response object resulting from a call to a model's `infer(...)` method.
- `video_frame`: A [VideoFrame object](../../docs/reference/inference/core/interfaces/camera/entities/#inference.core.interfaces.camera.entities.VideoFrame) containing metadata and pixel data from the video frame.

### After `v0.9.18`
```python
from typing import Union, List, Optional
from inference.core.interfaces.camera.entities import VideoFrame

def on_prediction(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            # EMPTY FRAME
            continue
        # SOME PROCESSING
```

See more info in **Custom Sink** section on how to create sink.

### Usage

You can also make `on_prediction` accepting other parameters that configure its behaviour, but those needs to be 
latched in function closure before injection into `InferencePipeline` init methods.

```python
from functools import partial
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline


def on_prediction(
    predictions: dict,
    video_frame: VideoFrame,
    my_parameter: int,
) -> None:
    # you need to implement your logic here, with `my_parameter` used
    pass

pipeline = InferencePipeline.init(
  video_reference="./my_video.mp4",
  model_id="yolov8n-640",
  on_prediction=partial(on_prediction, my_parameter=42),
)
```

### Custom Sinks

To create a custom sink, define a new function with the appropriate signature.

```python
from typing import Union, List, Optional, Any
from inference.core.interfaces.camera.entities import VideoFrame

def on_prediction(
    predictions: Union[Any, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    if not issubclass(type(predictions), list):
      # this is required to support both sequential and batch processing with single code
      # if you use only one mode - you may create function that handles with only one type
      # of input
      predictions = [predictions]
      video_frame = [video_frame]
    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            # EMPTY FRAME
            continue
        # SOME PROCESSING
```

In `v0.9.18` we introduced `InferencePipeline` parameter called `sink_mode` - here is how it works.
With `SinkMode.SEQUENTIAL` - each frame and prediction triggers separate call for sink, in case of `SinkMode.BATCH` - 
list of frames and predictions will be provided to sink, always aligned in the order of video sources - with None 
values in the place of vide_frames / predictions that were skipped due to `batch_collection_timeout`. 
`SinkMode.ADAPTIVE` is a middle ground (and default mode) - all old sources will work in that mode against a single 
video input, as the pipeline will behave as if running in `SinkMode.SEQUENTIAL`. To handle multiple videos - 
sink needs to accept `predictions: List[Optional[dict]]` and `video_frame: List[Optional[VideoFrame]]`. It is also 
possible to process multiple videos using old sinks - but then `SinkMode.SEQUENTIAL` is to be used, causing
sink to be called on each prediction element.

#### Why there is `Optional` in  `List[Optional[dict]]` and `List[Optional[VideoFrame]]`?
It may happen that it is not possible to collect video frames from all the video sources (for instance when one of the 
source disconnected and re-connection is attempted). `predictions` and `video_frame` are ordered matching the order of
`video_reference` list of `InferencePipeline` and `None` elements will appear in position of missing frames. We
provide this information to sink, as some sinks may require all predictions and video frames from the batch to
be provided (even if missing) - for example: `render_boxes(...)` sink needs that information to maintain the position
of frames in tiles mosaic.

!!! Info

    See our [tutorial on creating a custom Inference Pipeline sink!](#custom-sinks)

**prediction**

Predictions are provided to the sink as a dictionary containing keys:

- `predictions`: predictions - either for single frame or batch of frames. Content depends on which model runs behind 
`InferencePipeline` - for Roboflow models - it will come as dict or list of dicts. The schema of elements is given 
below.

Depending on the model output, predictions look differently. You must adjust sink to the prediction format.
For instance, Roboflow object-detection prediction contains the following keys:

- `x`: The center x coordinate of the predicted bounding box in pixels
- `y`: The center y coordinate of the predicted bounding box in pixels
- `width`: The width of the predicted bounding box in pixels
- `height`: The height of the predicted bounding box in pixels
- `confidence`: The confidence value of the prediction (between 0 and 1)
- `class`: The predicted class name
- `class_id`: The predicted class ID

## Other Pipeline Configuration

Inference Pipelines are highly configurable. Configurations include:

- `max_fps`: Used to set the maximum rate of frame processing.
- `confidence`: Confidence threshold used for inference.
- `iou_threshold`: IoU threshold used for inference.
- `video_source_properties`: Optional dictionary of properties to configure the video source, corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. See the [OpenCV Documentation](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d) for a list of all possible properties.

```python
from inference import InferencePipeline
pipeline = InferencePipeline.init(
    ...,
    max_fps=10,
    confidence=0.75,
    iou_threshold=0.4,
    video_source_properties={
        "frame_width": 1920.0,
        "frame_height": 1080.0,
        "fps": 30.0,
    },
)
```

See the reference docs for the [full list of Inference Pipeline parameters](../../docs/reference/inference/core/interfaces/stream/inference_pipeline/#inference.core.interfaces.stream.inference_pipeline.InferencePipeline).

!!! Warning "Breaking change planned at the **end of Q4 2024**"

    We've discovered that the behaviour of `max_fps` parameter is not in line with `inference` clients expectations
    regarding processing of video files. Current implementation for vides waits before processing the next 
    video frame, instead droping the frames to *modulate* video FPS. 

    We have added a way to change this suboptimal behaviour in release `v0.26.0` - new behaviour of 
    `InferencePipeline` can be enabled setting environmental variable flag 
    `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True`. 

    Please note that the new behaviour will be the default one end of Q4 2024!
    

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

### Migrate to changes introduced in `v0.9.18`

List of changes:
1. `VideoFrame` got new parameter: `source_id` - indicating which video source yielded the frame
2. `on_prediction` callable signature changed:
```python
from typing import Callable, Any, Optional, List, Union
from inference.core.interfaces.camera.entities import VideoFrame
# OLD
SinkHandler = Callable[[Any, VideoFrame], None]

# NEW
SinkHandler = Optional[
    Union[
        Callable[[Any, VideoFrame], None],
        Callable[[List[Optional[Any]], List[Optional[VideoFrame]]], None],
    ]
]
```
this change is non-breaking, as there is new parameter of `InferencePipeline.init*()` functions - `sink_mode` with default 
value on `ADAPTIVE` - which forces single video frame and prediction to be provided for sink invocation if one video 
only is specified. Old sinks were adjusted to work in dual mode - for instance in the demo you see `render_boxes(...)` 
displaying image tiles.

Example:
```python
from typing import Union, List, Optional
import json

from inference.core.interfaces.camera.entities import VideoFrame

def save_prediction(predictions: dict, file_name: str) -> None:
  with open(file_name, "w") as f:
    json.dump(predictions, f)

def on_prediction_old(predictions: dict, video_frame: VideoFrame) -> None:
  save_prediction(
    predictions=predictions,
    file_name=f"frame_{video_frame.frame_id}.json"
  )

def on_prediction_new(
    predictions: Union[dict, List[Optional[dict]]],
    video_frame: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    for prediction, frame in zip(predictions, video_frame):
        if prediction is None:
            # EMPTY FRAME
            continue
        save_prediction(
        predictions=prediction,
        file_name=f"source_{frame.source_id}_frame_{frame.frame_id}.json"
      )
```

2. `on_video_frame` callable used in InferencePipeline.init_with_custom_logic(...)` changed:
Previously:  `InferenceHandler = Callable[[VideoFrame], Any]`
Now: `InferenceHandler = Callable[[List[VideoFrame]], List[Any]]`

Example:
```python
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Any, List

MY_MODEL = ...

# before v0.9.18  
def on_video_frame_old(video_frame: VideoFrame) -> Any:
  return MY_MODEL(video_frame.image)
  
# after v0.9.18  
def on_video_frame_new(video_frames: List[VideoFrame]) -> List[Any]: 
  # result must be returned as list of elements representing model prediction for single frame
  # with order unchanged.
  return MY_MODEL([v.image for v in video_frames])
```

3. The interface for `PipelineWatchdog` changed - and there is also a side effect change in form of pipeline state report 
that is emitted being changed.

Old watchdog:
```python
class PipelineWatchDog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def register_video_source(self, video_source: VideoSource) -> None:
        pass

    @abstractmethod
    def on_status_update(self, status_update: StatusUpdate) -> None:
        pass

    @abstractmethod
    def on_model_inference_started(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def on_model_prediction_ready(
        self, frame_timestamp: datetime, frame_id: int
    ) -> None:
        pass

    @abstractmethod
    def get_report(self) -> Optional[PipelineStateReport]:
        pass
```

New watchdog:
```python
class PipelineWatchDog(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def register_video_sources(self, video_sources: List[VideoSource]) -> None:
        pass

    @abstractmethod
    def on_status_update(self, status_update: StatusUpdate) -> None:
        pass

    @abstractmethod
    def on_model_inference_started(
        self,
        frames: List[VideoFrame],
    ) -> None:
        pass

    @abstractmethod
    def on_model_prediction_ready(
        self,
        frames: List[VideoFrame],
    ) -> None:
        pass

    @abstractmethod
    def get_report(self) -> Optional[PipelineStateReport]:
        pass
```

Old report:
```python
@dataclass(frozen=True)
class PipelineStateReport:
    video_source_status_updates: List[StatusUpdate]
    latency_report: LatencyMonitorReport
    inference_throughput: float
    source_metadata: Optional[SourceMetadata]
```

New report:
```python
@dataclass(frozen=True)
class PipelineStateReport:
    video_source_status_updates: List[StatusUpdate]
    latency_reports: List[LatencyMonitorReport]  # now - one report for each source
    inference_throughput: float
    sources_metadata: List[SourceMetadata] # now - one metadata for each source
```

If there was custom watchdog created on your end - reimplementation should be easy, as all the data passed to methods
previously for single video source / frame are now provided for all sources / frames.