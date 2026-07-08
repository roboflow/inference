---
description: Stream video to an Inference server over WebRTC and receive live predictions, using a model ID or a Workflow, from webcams, RTSP cameras, video files, or manually sent frames.
---

# WebRTC Streaming

The `inference-sdk` includes a WebRTC client for low-latency, real-time video inference. Instead of sending images one HTTP request at a time, you open a streaming session: video frames flow to the server over a WebRTC connection, and processed frames plus prediction data flow back continuously.

!!! warning "Experimental"

    The WebRTC SDK is experimental and under active development. The API may change in future
    releases. Please report issues at [github.com/roboflow/inference/issues](https://github.com/roboflow/inference/issues).

WebRTC streaming requires extra dependencies:

```bash
pip install "inference-sdk[webrtc]"
```

## Quickstart: stream a model

The fastest way to get live predictions is to pass a `model_id` — no Workflow required. The SDK builds a minimal single-model Workflow under the hood, and your `on_frame` handler receives each video frame together with the raw predictions dict for that frame:

```python
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY",
)

session = client.webrtc.stream(
    source=WebcamSource(),
    model_id="rfdetr-nano",
)

box_annotator = sv.BoxAnnotator()

@session.on_frame
def show(frame, data):
    # data is the raw predictions dict, exactly as returned by the server
    # (None when predictions are unavailable for this frame)
    if data is None:
        return
    detections = sv.Detections.from_inference(data)
    annotated = box_annotator.annotate(frame.copy(), detections)
    cv2.imshow("Preview", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

session.run()  # blocks until the stream ends or session.close() is called
```

`model_id` works for any task type with a generic Workflow model block. The model's task type is resolved automatically via a Roboflow API lookup, and the matching model block is selected for you. Supported task types:

- `object-detection`
- `instance-segmentation`
- `semantic-segmentation`
- `classification`
- `multi-label-classification`
- `keypoint-detection`

`data` is the serialized predictions dict passed through verbatim — its shape follows the task type. For detection-family models it is inference-response-shaped, so you can convert it with the matching [`supervision`](https://supervision.roboflow.com/) helper — `sv.Detections.from_inference(data)` for object detection and instance segmentation, `sv.KeyPoints.from_inference(data)` for keypoint models. Classification predictions carry `top`/`confidence` keys, and semantic-segmentation predictions carry run-length-encoded masks (`rle_mask`) that you decode yourself.

When predictions are unavailable for a frame (e.g. the paired prediction message never arrived for a live stream frame), `data` is `None` — check for it in your handler before use.

VLMs are not supported in `model_id` mode (each VLM family has its own dedicated Workflow block, so there is no generic block to wrap them with) — stream them with a full [`workflow`](#streaming-a-workflow) instead.

**Skipping the task-type lookup:** pass `task_type` explicitly to avoid the network call — useful for air-gapped or self-hosted deployments:

```python
session = client.webrtc.stream(
    source=WebcamSource(),
    model_id="my-project/3",
    task_type="object-detection",
)
```

In `model_id` mode, `on_frame` handlers can take either `(frame, data)` or `(frame, data, metadata)` — the third argument is the [`VideoMetadata`](#frame-metadata) for the frame.

## Streaming a Workflow

For multi-step pipelines, pass a `workflow` instead of a `model_id`. Reference a Workflow saved in your Roboflow workspace by ID, or provide a full specification dict:

=== "Workflow ID"

    ```python
    import cv2
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.webrtc import WebcamSource, StreamConfig

    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="ROBOFLOW_API_KEY",
    )

    session = client.webrtc.stream(
        source=WebcamSource(),
        workflow="my-workflow-id",
        workspace="my-workspace-name",
        config=StreamConfig(
            stream_output=["output_image"],   # workflow output streamed back as video
            data_output=["predictions"],      # workflow outputs delivered via data channel
        ),
    )

    @session.on_frame
    def show(frame, metadata):
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

    @session.on_data("predictions")
    def handle_predictions(predictions, metadata):
        print(f"Frame {metadata.frame_id}: {predictions}")

    session.run()
    ```

=== "Workflow specification"

    ```python
    import cv2
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.webrtc import WebcamSource, StreamConfig

    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="ROBOFLOW_API_KEY",
    )

    workflow_spec = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "model",
                "images": "$inputs.image",
                "model_id": "rfdetr-nano",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            },
            {"type": "JsonField", "name": "image", "selector": "$inputs.image"},
        ],
    }

    session = client.webrtc.stream(
        source=WebcamSource(),
        workflow=workflow_spec,
        config=StreamConfig(
            stream_output=["image"],
            data_output=["predictions"],
        ),
    )

    @session.on_frame
    def show(frame, metadata):
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

    session.run()
    ```

Notes:

- `workflow` and `model_id` are mutually exclusive — pass exactly one.
- `workspace` is required when `workflow` is an ID string; it is not needed for a specification dict.
- `image_input` (default `"image"`) names the Workflow image input the video frames are bound to.
- In workflow mode, `on_frame` handlers receive `(frame, metadata)` — prediction data arrives separately through [`on_data`](#receiving-data-on_data) handlers, routed by the `data_output` names in `StreamConfig`.

## Video sources

The first argument to `stream()` selects where video comes from:

```python
from inference_sdk.webrtc import (
    WebcamSource,
    RTSPSource,
    LocalStreamSource,
    MJPEGSource,
    VideoFileSource,
    ManualSource,
)
```

### WebcamSource

Captures frames from a local camera device and sends them to the server:

```python
source = WebcamSource()                                    # default camera
source = WebcamSource(device_id=1, resolution=(1920, 1080))
```

The camera's FPS is auto-detected and reported to the server.

### RTSPSource

The **server** connects to the RTSP camera and streams processed video back to you — use this when the camera is reachable from the server:

```python
source = RTSPSource("rtsp://user:pass@camera.local/stream")
```

### LocalStreamSource

Captures an RTSP/RTMP stream **locally** (on the client machine) and sends frames to the server — use this when the camera is only reachable from your machine, not from the server:

```python
source = LocalStreamSource("rtsp://192.168.1.10/stream")   # also rtsps://, rtmp://, rtmps://
```

### MJPEGSource

Like `RTSPSource`, but for MJPEG streams captured by the server:

```python
source = MJPEGSource("http://camera.local/mjpeg")
```

### VideoFileSource

Uploads a video file to the server over the data channel; the server processes it and streams results back. More efficient than frame-by-frame streaming for pre-recorded video:

```python
source = VideoFileSource("video.mp4")

# Track upload progress and process at original FPS (live-preview pacing)
source = VideoFileSource(
    "video.mp4",
    on_upload_progress=lambda uploaded, total: print(f"{uploaded}/{total} chunks"),
    realtime_processing=True,   # default False = process as fast as possible
)
```

By default frames come back through the data channel (guaranteed order and quality). Pass `use_datachannel_frames=False` to receive them via a hardware-accelerated WebRTC video track instead (lower bandwidth).

### ManualSource

Send frames programmatically — useful when frames come from a custom pipeline:

```python
import threading
import time

import cv2
from inference_sdk.webrtc import ManualSource, StreamConfig

source = ManualSource()
session = client.webrtc.stream(
    source=source,
    model_id="rfdetr-nano",
    config=StreamConfig(declared_fps=30),
)

@session.on_frame
def handle(frame, data):
    print(data)

# run() establishes the connection and dispatches handlers; start it in a
# background thread so this thread can feed frames.
threading.Thread(target=session.run, daemon=True).start()

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        source.send(frame)   # BGR numpy array
    except RuntimeError:
        pass                 # session still connecting — frame skipped
    time.sleep(1 / 30)       # pace sends to the declared FPS

session.close()
```

`send()` raises `RuntimeError` until the connection is established, and queued frames are dropped oldest-first if you send faster than the stream is consumed. `ManualSource` has no FPS auto-detection, so declare the frame rate via `StreamConfig(declared_fps=...)`.

## Consuming results

### The session lifecycle

`stream()` returns a `WebRTCSession`. The connection starts lazily on first use (`run()`, `video()`, or `wait()`) and must be closed to release resources. Three equivalent patterns:

```python
# 1. run() — auto-closes when it exits (recommended with handlers)
session.run()

# 2. Context manager — auto-closes on exit (recommended with the video() iterator)
with client.webrtc.stream(source=source, model_id="rfdetr-nano") as session:
    for frame, data in session.video():
        ...

# 3. Manual — you must call close() yourself
session = client.webrtc.stream(source=source, model_id="rfdetr-nano")
for frame, data in session.video():
    ...
session.close()
```

`session.close()` is idempotent and safe to call from inside a handler — it ends `run()` and the `video()` iterator. `session.wait(timeout=None)` blocks until the stream ends without consuming frames yourself.

### Receiving frames: `on_frame` and `video()`

`@session.on_frame` registers a handler invoked for every processed video frame when using `run()`. `session.video()` is the iterator equivalent — same data, pull-based:

```python
# model_id mode: (frame, data) — data is the raw predictions dict
# (None when predictions are unavailable for the frame)
for frame, data in session.video():
    ...

# workflow mode: (frame, metadata)
for frame, metadata in session.video():
    ...
```

Frames are BGR numpy arrays. If your handler falls behind in realtime mode, the oldest frames are dropped so the stream stays live.

### Receiving data: `on_data`

Workflow outputs listed in `StreamConfig.data_output` arrive over the data channel. Register handlers per output name, or one global handler for the whole payload:

```python
@session.on_data("predictions")           # a single output field
def handle_predictions(predictions, metadata):
    print(f"Frame {metadata.frame_id}: {predictions}")

@session.on_data                          # global: full output dict
def handle_all(data, metadata):
    print(data)
```

Handlers may accept `(value, metadata)` or just `(value)` — the signature is auto-detected.

### Handling errors: `on_error`

The server reports per-frame errors (workflow execution failures, output serialization failures) alongside each data channel message. `on_error` handlers fire only for frames with a non-empty error list:

```python
@session.on_error
def on_err(errors, metadata):
    print(f"Frame {metadata.frame_id} failed: {errors}")
    session.close()   # e.g. bail out on first error
```

These are server-side per-frame failures; connection and setup errors surface as exceptions from `run()` instead. Errors are also attached to `metadata.errors` on every frame, so `on_frame` / `on_data` handlers can inspect them directly.

### Frame metadata

`VideoMetadata` accompanies each frame and data message:

| Attribute | Description |
|-----------|-------------|
| `frame_id` | Unique identifier of the frame in the stream |
| `received_at` | When the server received the frame |
| `pts` / `time_base` | Presentation timestamp of the video stream |
| `declared_fps` / `measured_fps` | Declared vs. measured stream FPS |
| `errors` | Per-frame errors reported by the server (empty when the frame processed cleanly) |

## StreamConfig

`StreamConfig` controls output routing, processing behavior, and network settings:

```python
from inference_sdk.webrtc import StreamConfig

config = StreamConfig(
    stream_output=["output_image"],
    data_output=["predictions"],
    realtime_processing=True,
)
session = client.webrtc.stream(source=source, workflow="...", workspace="...", config=config)
```

| Field | Default | Description |
|-------|---------|-------------|
| `stream_output` | `[]` | Workflow output names streamed back as video |
| `data_output` | `[]` | Workflow output names delivered via the data channel |
| `realtime_processing` | `True` | Drop frames to keep up in real time; set `False` to queue and process every frame |
| `declared_fps` | `None` | FPS declaration for sources without auto-detection (e.g. `ManualSource`) |
| `turn_server` | `None` | TURN server config: `{"urls": "turn:...", "username": "...", "credential": "..."}` |
| `workflow_parameters` | `{}` | Parameters passed to the Workflow execution |
| `requested_plan` | `None` | Compute plan for Roboflow serverless endpoints (e.g. `"webrtc-gpu-small"`) |
| `requested_region` | `None` | Processing region for serverless endpoints (e.g. `"us"`, `"eu"`) |
| `processing_timeout` | `None` | Server-side session time limit in seconds (serverless endpoints) |

In `model_id` mode, empty `stream_output` / `data_output` are filled automatically (`["image"]` and `["predictions"]`); any other settings you provide are preserved.

**TURN servers:** when connecting to Roboflow-hosted endpoints, TURN configuration is fetched automatically. For self-hosted servers behind restrictive NATs or firewalls, provide `turn_server` explicitly; direct connection is attempted when it is not set.
