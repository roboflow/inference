# WebRTC Real-time Video Processing with Roboflow Inference

## Overview

The WebRTC feature in Roboflow Inference enables real-time video processing through WebRTC connections.

Following is the list of supported sources:
 * webcam (via `WebcamSource`)
 * RTSP (via `RTSPSource`)
 * video files (via `VideoFileSource`)

## Getting Started

### Installation

We recommend creating a virtual environment and installing the inference-sdk package:

```bash
python -m venv venv
source venv/bin/activate
pip install inference-sdk
```

## Basic Usage Examples

Find complete working examples in the [examples/webrtc_sdk/](https://github.com/roboflow/inference/tree/main/examples/webrtc_sdk) directory:
- [webcam_basic.py](https://github.com/roboflow/inference/blob/main/examples/webrtc_sdk/webcam_basic.py) - Basic webcam streaming
- [rtsp_basic.py](https://github.com/roboflow/inference/blob/main/examples/webrtc_sdk/rtsp_basic.py) - RTSP stream processing
- [video_file_basic.py](https://github.com/roboflow/inference/blob/main/examples/webrtc_sdk/video_file_basic.py) - Video file processing with output saving

## Minimal Example

```python
import cv2 as cv

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoMetadata, StreamConfig, WebcamSource

API_KEY = "<your API key>"
WORKFLOW = "<your workflow id>"
WORKSPACE = "<your workspace id>"
STREAM_OUTPUT = "visualization" # must be valid video output as defined in workflow
DATA_OUTPUT = "count" # must be valid data output as defined in workflow

client = InferenceHTTPClient.init(
   api_url="https://serverless.roboflow.com",  # or "http://127.0.0.1:9001" for local server
   api_key=API_KEY,
)

source = WebcamSource()
config = StreamConfig(
   stream_output=[STREAM_OUTPUT],
   data_output=[DATA_OUTPUT],
   requested_region="us"
)
session = client.webrtc.stream(
   source=source,
   workflow=WORKFLOW,
   workspace=WORKSPACE,
   image_input="image",
   config=config,
)

@session.on_frame
def show_frame(frame, metadata):
   cv.imshow("WebRTC SDK - Webcam", frame)
   if cv.waitKey(1) & 0xFF == ord("q"):
       session.close()

@session.on_data()
def on_message(data: dict, metadata: VideoMetadata):
   print(
       f"Frame {metadata.frame_id}: {data[DATA_OUTPUT]}"
   )

session.run()
```

## Local Container Setup (optional)

If you want to run the Inference server locally, you can use the following commands:

 * To start a local Inference container with WebRTC support:
```bash
docker run -p 9001:9001 roboflow/roboflow-inference-server-cpu:latest
```

 * For GPU support:
```bash
docker run --gpus all -p 9001:9001 roboflow/roboflow-inference-server-gpu:latest
```

## Advanced Configuration

### Stream Configuration Options

```python
config = StreamConfig(
    # Video outputs - specify workflow output names, must be valid video output as defined in workflow
    stream_output=["annotated_image", "cropped_detections"],
    
    # Data outputs - use ["*"] for all outputs, must be valid data output as defined in workflow
    data_output=["predictions", "confidence_scores"],
    
    # Performance options
    realtime_processing=True,  # Minimize latency
    declared_fps=30           # Expected frame rate
)
```

### Workflow Parameters

Pass parameters to your workflow:

```python
session = client.webrtc.stream(
    source=source,
    workflow="your-workflow-id",
    workspace="your-workspace",
    config=config,
    workflow_params={
        "confidence_threshold": 0.5,
        "max_detections": 10
    }
)
```

## Support

For questions and support:
- Documentation: https://docs.roboflow.com
- Community Forum: https://discuss.roboflow.com
- GitHub Issues: https://github.com/roboflow/inference/issues