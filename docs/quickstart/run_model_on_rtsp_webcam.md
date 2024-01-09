You can run computer vision models on webcam stream frames, RTSP stream frames, and video frames with Inference.

Webcam inference is ideal if you want to run a model on an edge device (i.e. an NVIDIA Jetson or Raspberry Pi).

RTSP inference is ideal for using models with internet connected cameras that support RTSP streaming.

You can run Inference on video frames from `.mp4` and `.mov` files.

You can run both fine-tuned models and foundation models on the above three input types. See the "Foundation Models" section in the sidebar to learn how to import and run foundation models.

!!! tip "Follow our [Run a Fine-Tuned Model on Images](/quickstart/run_model_on_image) guide to learn how to find a model to run."

## Run a Vision Model on Video Frames

To use fine-tuned with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Once you have selected a model to run, create a new Python file and add the following code:

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11",
    video_reference=0,
    on_prediction=render_boxes,
)
pipeline.start()
pipeline.join()
```

This code will run a model on frames from a webcam stream. To use RTSP, set the `source` value to an RTSP stream URL. To use video, set the `source` value to a video file path.

Predictions are annotated using the `render_boxes` helper function. You can specify any function to process each prediction in the `on_prediction` parameter.

Replace `rock-paper-scissors-sxsw/11` with the model ID associated with the mode you want to run.

Then, run the Python script:

```
python app.py
```

Your webcam will open and you can see the model running:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/rock-paper-scissors.mp4" type="video/mp4">
</video>

!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

## Define Custom Prediction Handlers

The `on_prediction` parameter in the `InferencePipeline` constructor allows you to define custom prediction handlers. You can use this to define custom logic for how predictions are processed.

This function provides two parameters:

- `predictions`: A dictionary that contains all predictions returned by the model for the frame, and;
- `video_frame`: A dataclass that contains:
  - `image`: The video frame as a NumPy array,
  - `frame_id`: The frame ID, and;
  - `frame_timestamp`: The timestamp of the frame.

For example, you can use the following code to print the predictions to the console:

```python
from inference import InferencePipeline
import numpy as np

def on_prediction(predictions: dict, video_frame: np.array) -> None:
    print(predictions)
    pass

pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11",
    video_reference=0,
    on_prediction=on_prediction,
)
pipeline.start()
pipeline.join()
```

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

def on_prediction(predictions video_frame) -> None:
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
