# Changelog

This is the canonical changelog for the `streamvision` package.

## Unreleased

## `0.1.5rc1`

### Fixed

- Jetson USB camera capture honors requested format, dimensions and frame rate before startup, including automatic format selection and rounded NTSC rates. Bridge diagnostics are safe before deferred capture starts and after release.
- When tensor-native Jetson USB capture falls back to OpenCV, startup fails with an error state if the effective camera format, dimensions or frame rate differ from the requested mode, including drivers that silently ignore property writes.

## `0.1.4`

### Changed

- Security maintenance release, no functional change. The lockfile is refreshed for three advisories in transitive dependencies: `certifi` 2026.7.22 -> 2026.10.23 (GHSA-6q48-6c7w-9gw2, bundled emSign root CAs whose intermediate CA keys were compromised), `GitPython` 3.1.62 -> 3.2.1 (GHSA-83vg-56qc-22m7, remote code execution through a submodule `.git` path) and `setuptools` 84.0.0 -> 84.2.1 (GHSA-xmj3-9gf7-h52w, Windows path escape during legacy wheel installation). Declared dependency ranges are unchanged.

## `0.1.3`

### Changed

- The `workflows` extra requires `roboflow-workflows>=0.2.13` instead of
  `>=0.2.11rc1`, providing the execution-engine cleanup used below.

### Documentation

- Clarify `max_fps` behavior: video files retain every frame and wait by default; streams drop intermediate frames. `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING=True` enables frame dropping for files, and the Inference CLI sets this flag automatically.

### Fixed

- Joining a workflow `InferencePipeline` (after end of stream, `terminate()` or an inference error) now closes the workflow's blocks through `ExecutionEngine.close()`, after the workflow thread pools are shut down. Blocks with background connections, such as the OPC UA Reader, MQTT Reader and Kafka Consumer, no longer keep their threads and broker or server sessions while a stopped pipeline object is still referenced. With a `roboflow-workflows` that has no `ExecutionEngine.close()`, nothing changes.

## `0.1.2`

### Changed

- The `workflows` extra requires `aiohttp~=3.14.4`, the aiohttp release that fixes GHSA-2g87-pp6c-29x5 and GHSA-5wr8-gcqf-gqf3.
- The `workflows` extra requires `roboflow-workflows>=0.2.11rc1` (previously `>=0.2.4rc3`).

## `0.1.1`

### Added

- `VideoFrame.total_frames` carries the source video file's frame count (`None` for streams, unknown counts, realtime WebRTC uploads, and after a source restart or initial seek). `InferencePipeline` workflow runs and WebRTC video-file uploads pass it into workflow `VideoMetadata.total_frames`.

### Fixed

- WebRTC sessions await peer/transport cleanup on cancellation and setup failures, finish in-flight inference, and join workflow thread pools before returning. Normal connection closure is no longer logged as fatal.
- WebRTC worker logs TURN channel binds the relay refuses (403) at debug level instead of asyncio "Task exception was never retrieved" tracebacks.

## `0.1.0`

### Added

- Camera, stream and stream-manager runtime moved here from `inference.core.interfaces`.
- `python -m streamvision --host-factory ...` launches the stream manager standalone (needs the `webrtc` extra).
- `scripts/streamvision_isolation_probe.py` verifies an installed wheel runs without `inference`.
- `nvdec` extra installs PyNvVideoCodec for NVIDIA hardware decoding of video files; messages shown when it is missing name the extra.
- WebRTC worker engine (`streamvision.webrtc_worker`: `webrtc`, `entities`, `serializers`, `sources`, `utils`) moved here from `inference.core.interfaces.webrtc_worker`; it reaches its host through `streamvision.webrtc_worker.host`.
- WebRTC worker watchdog and Modal session runner (`streamvision.webrtc_worker.watchdog`, `.modal_session`) moved here from `inference.core.interfaces.webrtc_worker`.

### Changed

- `roboflow-workflows` moved from base dependencies to the `workflows` extra; install `streamvision[workflows]` for `InferencePipeline.init_with_workflow` and the stream manager.
- `StreamsConfiguration` manager address fields default to `None`; the manager then reads `STREAM_MANAGER_HOST`, `STREAM_MANAGER_PORT` and `STREAM_MANAGER_SOCKET_TIMEOUT`.
- Requires `supervision>=0.30.6,<0.31.0` (was `>=0.29.0,<0.30.0`); the `workflows` extra requires `roboflow-workflows>=0.2.4rc3`.
- The `webrtc` extra installs `orjson` and `requests`.
- `StreamsConfiguration` carries the WebRTC worker settings (data channel, preview compression, STUN, RTSP placeholder, time limits, Modal watchdog, session heartbeat and usage quota).

### Fixed

- Declare Pillow directly for camera imports and Torch in the `workflows` and supported-platform `nvdec` extras; GPU decoding still requires a compatible CUDA-enabled Torch build.

- Video source properties are applied in a camera-safe order: `fourcc` first, `fps` last, others in the given order. USB/V4L2 cameras (e.g. Logitech C920) no longer stay at the frame rate clamped for their startup pixel format when `fps` arrived before `fourcc`. `fourcc` also accepts a case-sensitive four-character code such as `"MJPG"`; the stream manager API rejects invalid values with a validation error, other invalid values are logged and skipped, and a `fourcc` the device rejects is logged as a warning.
- The stream manager no longer blocks forever on a command sent to a pipeline process that has exited; the caller gets a `not_found` error.
- `GstreamerCudaVideoFrameProducer` decodes container files such as MP4 and MKV; the previous pipeline never produced a frame for them.
- Declared dependencies are enough to import the workflows runtime, and the `test` extra installs `pytest-asyncio` and `pytest-timeout`.
- `streamvision.stream.watchdog.RTCPeerConnection` resolves again.
- Processes started by the stream manager install the passed configuration before importing the host factory module.
- `render_boxes` leaves out predictions whose polygon has fewer than 3 points. Labels stay aligned with the drawn detections and the other detections keep their masks; `supervision` 0.30 on its own would remove the masks of the whole frame.
