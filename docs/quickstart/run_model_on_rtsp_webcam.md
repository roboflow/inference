You can run computer vision models on webcam stream frames, RTSP stream frames, and video frames with Inference.

Webcam inference is ideal if you want to run a model on an edge device (i.e. an NVIDIA Jetson or Raspberry Pi).

RTSP inference is ideal for using models with internet connected cameras that support RTSP streaming.

You can run Inference on video frames from `.mp4` and `.mov` files.

You can run both fine-tuned models and foundation models on the above three input types. See the "Foundation Models" section in the sidebar to learn how to import and run foundation models.

!!! tip "Tip"
    Follow our [Run a Fine-Tuned Model on Images](/docs/quickstart/run_model_on_image) guide to learn how to find a model to run.

## Run a Vision Model on Video Frames

To use fine-tuned with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

Once you have selected a model to run, create a new Python file and add the following code:

```python
import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    cv2.imshow(
        "Prediction", 
        annotator.annotate(
            scene=image, 
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

inference.Stream(
    source="webcam", # or "rstp://0.0.0.0:8000/password" for RTSP stream, or "file.mp4" for video
    model="rock-paper-scissors-sxsw/11", # from Universe
    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    on_prediction=on_prediction, 
)
```

This code will run a model on frames from a webcam stream. To use RTSP, set the `source` value to an RTSP stream URL. To use video, set the `source` value to a video file path.

Predictions will be annotated using the [supervision Python package](https://github.com/roboflow/supervision).

Replace `rock-paper-scissors-sxsw/11` with the model ID associated with the mode you want to run.

Then, run the Python script:

```
python app.py
```

Your webcam will open and you can see the model running:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/rock-paper-scissors.mp4" type="video/mp4">
</video>

## New stream interface!

### Motivation
We've identified certain problems with our previous implementation of `inference.Stream`:
* it could not achieve high throughput of processing
* in case of source disconnection - it was not attempting to re-connect automatically

That's why we've re-designed API and provided new abstraction - called `InferencePipeline`. At the moment, we
are testing and improving implementation - hoping that over time it will be a replacement for `inference.Stream`.

### Why to migrate?
We understand that each breaking change may be hard to adopt on your end, but the changes we introduced were meant
to improve the quality. Here are the results:

#### Performance 

##### MacBook M2
| Test     | OLD (FPS) | NEW (FPS) |
|----------|:---------:|:---------:|
| yolov8-n |    ~6     |    ~26    |
| yolov8-s |   ~4.5    |    ~12    |
| yolov8-m |   ~3.5    |    ~5     |

Tested against the same 1080p 60fps RTSP stream emitted by localhost. 
For `yolov8-n` we also measured that new implementation operates on stream frames that are on average ~25ms old (
measured from frame grabbing) compared to ~60ms for old implementation.

##### Jetson Orin Nano
At Jetson, new implementation is also more performant:

| Test     | NEW (FPS) |
|----------|:---------:|
| yolov8-n |    ~25    |
| yolov8-s |    ~18    |
| yolov8-m |    ~8     |

With old version reaching at max 6-7 fps. This test was executed against 4K@60fps stream, which is not possible to
be decoded in native pace due to resource constrains. New implementation proved to run without stability issues
for few hours straight.

##### Tesla T4
GPU workstation with Tesla T4 was able to run 4 concurrent HD streams at 15FPS utilising ~80% GPU - reaching
over 60FPS throughput per GPU (against `yolov8-n`).

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

### How to migrate?

Let's assume you used `inference.Stream(...)` with your custom handlers:

```python
import numpy as np


def on_prediction(predictions: dict, image: np.ndarray) -> None:
    pass
```

Now, the structure of handlers has changed into:
```python
import numpy as np

from inference.core.interfaces.camera.entities import VideoFrame

def on_prediction(predictions: dict, video_frame: VideoFrame) -> None:
    pass
```

With predictions being still dict (passed as second parameter) in the same, standard Roboflow format, 
but `video_frame` is a dataclass with the following property:
* `image`: which is video frame (`np.ndarray`)
* `frame_id`: int value representing the place of the frame in stream order
* `frame_timestamp`: time of frame grabbing - the exact moment when frame appeared in the file/stream 
on the receiver side (`datetime.datetime`)

Additionally, it eliminates the need of grabbing `.frame_id` from `inference.Stream()`.

So the re-implementation work should be relatively easy. There is new package:
`inference.core.interfaces.stream.sinks` - with handful of useful `on_prediction()` implementations ready to be
used!

Initialisation of stream also has been changed:
```python
import inference

inference.Stream(
    source="webcam", # or "rstp://0.0.0.0:8000/password" for RTSP stream, or "file.mp4" for video
    model="rock-paper-scissors-sxsw/11", # from Universe
    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    on_prediction=on_prediction, 
)
```

will change into:
```python
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11",
    video_reference=0,
    on_prediction=render_boxes,
)
pipeline.start()
pipeline.join()
```
`InferencePipeline` exposes interface to manage its state (possibly from different thread) - including
functions like `.start()`, `.pause()`, `.terminate()`.

### I want to know more!
Obviously, as with all changes there is a lot to be learned! We've prepared detailed docs of new API elements, 
which can be found in functions and classes docstrings. We encourage to acknowledge, especially the
part related to new `VideoSoure` abstraction that is meant to replace `interface.Camera` - mainly to understand
the notion of new configuration possibilities related to buffered decoding and different behaviour of system
appropriate in different cases (that can be tuned via configuration).
