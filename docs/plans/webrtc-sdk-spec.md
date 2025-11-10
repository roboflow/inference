# WebRTC SDK – Minimal, Prescriptive API

Status: Draft-to-Adopt
Scope: Python SDK, namespaced WebRTC client
Audience: SDK users and implementers

---

## Overview

A minimal, opinionated WebRTC API for the Python SDK that is easy to use and OpenCV‑friendly. The WebRTC surface lives under `client.webrtc` to keep concerns and optional dependencies isolated.

- Default frame type: `numpy.ndarray` (H, W, 3) BGR `uint8`
- Sync and async context managers: `with` and `async with`
- Evented data channel and iterable video frames

---

## Main API

### Entry Point
- `client.webrtc: WebRTCClient` – namespaced accessor on `InferenceHTTPClient`

### Session Factories (Context Managers)
- `client.webrtc.use_webcam(workflow_id: str, config: Optional[WebcamConfig] = None) -> WebRTCSession`
- `client.webrtc.use_video_file(path: str, workflow_id: str, config: Optional[VideoFileConfig] = None) -> WebRTCSession`
- `client.webrtc.use_rtsp_stream(url: str, workflow_id: str, config: Optional[RTSPConfig] = None) -> WebRTCSession`

### Manual Source (Send OpenCV Frames)
- `client.webrtc.use_manual(workflow_id: str, config: Optional[WebRTCBaseConfig] = None) -> WebRTCSession`
  - Call `session.send(frame: np.ndarray)` to push frames to the workflow

### Session Essentials
- `WebRTCSession.video.stream()` → iterator of `np.ndarray` BGR frames (if remote stream enabled)
- `WebRTCSession.data.on(event: str)` → decorator to register handlers (e.g., `"message"`)
- `WebRTCSession.send(frame: np.ndarray)` → available on manual sessions
- `WebRTCSession.wait_for_disconnect(timeout: Optional[float] = None)`

Note: Pause/resume are not available in worker mode. They are only feasible when using the stream manager pipeline endpoints (see server notes below), and are out of scope for this minimal spec.


### Config Basics (Matches Server Init APIs)
- `WebRTCBaseConfig(`
  `webrtc_realtime_processing: bool = True,`
  `webrtc_turn_config: dict | None = None,  # {urls, username, credential}`
  `stream_output: list[str] = [],`
  `data_output: list[str] = [],`
  `declared_fps: float | None = None,`
  `workflows_parameters: dict = {}`
  `)`
- `WebcamConfig(resolution: tuple[int, int] | None = None)`  # local‑only, not sent to server
- `VideoFileConfig()`  # no extra fields; path is provided to `use_video_file`
- `RTSPConfig()`       # no extra fields; credentials go in the URL if needed

---

## Samples

### 1) Webcam – Data Only
```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient.init(api_url="http://localhost:9001", api_key="...")

with client.webrtc.use_webcam(workflow_id="object-detection") as s:
    @s.data.on("message")
    def on_message(msg):
        print("Detections:", msg.predictions)

    s.wait_for_disconnect()
```

### 2) Webcam – Annotated Video Display
```python
import cv2
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient.init(api_url="http://localhost:9001", api_key="...")

with client.webrtc.use_webcam(workflow_id="object-detection") as s:
    for frame in s.video.stream():  # np.ndarray BGR
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### 3) RTSP – Video + Alerts
```python
import cv2
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient.init(api_url="http://localhost:9001", api_key="...")

with client.webrtc.use_rtsp_stream(
    url="rtsp://camera.local/stream",
    workflow_id="intrusion-detection"
) as s:
    @s.data.on("message")
    def on_message(msg):
        if msg.data.get("intrusion_detected"):
            print("ALERT: Intrusion detected")

    for frame in s.video.stream():
        cv2.imshow("Security Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break
```

### 4) Manual – Send OpenCV Frames to Workflow
```python
import cv2
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient.init(api_url="http://localhost:9001", api_key="...")
cap = cv2.VideoCapture(0)

with client.webrtc.use_manual(workflow_id="object-detection") as s:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # any OpenCV preprocessing
        s.send(frame)  # np.ndarray BGR → server workflow
```

---

## Notes (Non‑Negotiable Defaults)
- Frames are yielded as `np.ndarray` BGR.
- All session factories (including manual) are context managers and clean up automatically.
- Data channel exposes a single `"message"` event by default for workflow outputs.

---

## Out‑of‑Scope (Future)
- Pipeline builder DSL
- Alternative frame types beyond `np.ndarray`
- Room/participant abstractions or multi‑peer topologies

---

## Implementation Steps (Webcam‑First)

- Scaffold WebRTC namespace + types
- Implement `webrtc.use_webcam()` worker initialization
- Add ndarray BGR frame conversion
- Implement `video.stream()` and `data.on()`
- Create `webcam_basic` sample script
- Add unit tests with mocks
- Run tests and validate sample
- Polish API + minimal docs

## Implementation Steps (RTSP)

- Implement `webrtc.use_rtsp_stream(url, workflow_id, config)` mapping to worker init
- Map config -> request: `rtsp_url`, `stream_output`, `data_output`, `webrtc_turn_config`, `webrtc_realtime_processing`, `declared_fps`, `workflows_parameters`
- Add URL validation and credential pass‑through (username:password@host in URL)
- Reuse `video.stream()` and `data.on()` paths (frames come from server)
- Create `rtsp_basic` sample script (display video + handle `"message"`)
- Add unit tests with mocks (valid URL, auth URL, unreachable host, SDP success/failure)
- Run tests and validate sample
- Add brief docs for RTSP specifics (auth in URL, NAT/TURN note)
