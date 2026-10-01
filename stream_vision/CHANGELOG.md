# Changelog

This is the canonical changelog for the `streamvision` package.

## Unreleased

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
