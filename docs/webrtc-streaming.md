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

### 1. Webcam Streaming

Stream from your webcam to a workflow:

```python
import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",  # Replace with http://localhost:9001 if running locally
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# Configure what outputs to receive
config = StreamConfig(
    stream_output=["annotated_image"],  # Must be valid video output as defined in workflow
    data_output=["predictions"]         # Must be valid data output as defined in workflow
)

# Create streaming session
session = client.webrtc.stream(
    source=WebcamSource(),
    workflow="your-workflow-id",
    workspace="your-workspace",
    config=config
)

# Handle processed frames
@session.on_frame
def show_frame(frame, metadata):
    # frame is the processed video frame
    cv2.imshow("Processed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()  # Close session and cleanup resources

# Handle data outputs
@session.on_data()
def on_message(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

# Run the session
session.run()
```

### 2. RTSP Stream Processing

Process an RTSP stream, client initiates the worker which pulls frames from RTSP stream (note: RTSP stream must be accessible from the worker, that means if you are processing in Roboflow cloud, RTSP stream must be publicly accessible):

```python
import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import RTSPSource, StreamConfig

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",  # Replace with http://localhost:9001 if running locally
    api_key="YOUR_ROBOFLOW_API_KEY"
)

source = RTSPSource("rtsp://demo.roboflow.com:8554")  # Roboflow demo stream, this will only work if running the worker in Roboflow cloud

config = StreamConfig(
    stream_output=["annotated_image"],  # Must be valid video output as defined in workflow
    data_output=["predictions"]         # Must be valid data output as defined in workflow
)

session = client.webrtc.stream(
    source=source,
    workflow="your-workflow-id",
    workspace="your-workspace",
    config=config
)

# Handle processed frames
@session.on_frame
def show_frame(frame, metadata):
    # frame is the processed video frame
    cv2.imshow("Processed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()  # Close session and cleanup resources

# Handle data outputs
@session.on_data()
def on_message(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

# Run the session
session.run()
```

### 3. Video File Processing

Process a pre-recorded video:

```python
import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig

source_path = "path/to/video.mp4"
source = VideoFileSource(
    source_path,
    on_upload_progress=lambda uploaded, total: print(f"Upload: {uploaded}/{total}")
)

# get source fps
cap = cv2.VideoCapture(source_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

config = StreamConfig(
    stream_output=["annotated_image"],  # Must be valid video output as defined in workflow
    data_output=["predictions"]         # Must be valid data output as defined in workflow
)

session = client.webrtc.stream(
    source=source,
    workflow="your-workflow-id",
    workspace="your-workspace",
    config=config
)

# Save processed video
writer = None

@session.on_frame
def save_frame(frame, metadata):
    global writer
    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(c1="m", c2="p", c3="4", c4="v")
        writer = cv2.VideoWriter("path/to/result.mp4", fourcc=fourcc, apiPreference=cv2.CAP_FFMPEG, fps=fps, frameSize=(w, h))
    writer.write(frame)

session.run()
writer.release()
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
    # Video outputs - specify workflow output names
    stream_output=["annotated_image", "cropped_detections"],
    
    # Data outputs - use ["*"] for all outputs
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

## Examples Repository

Find complete working examples in the `examples/webrtc_sdk/` directory:
- `webcam_basic.py` - Basic webcam streaming
- `rtsp_basic.py` - RTSP stream processing
- `video_file_basic.py` - Video file processing with output saving

## Support

For questions and support:
- Documentation: https://docs.roboflow.com
- Community Forum: https://discuss.roboflow.com
- GitHub Issues: https://github.com/roboflow/inference/issues