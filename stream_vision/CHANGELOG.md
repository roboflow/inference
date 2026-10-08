# Changelog

This is the canonical changelog for the `streamvision` package.

## Unreleased

### Added

- `VideoSource.release_error` exposes the latest credential-redacted producer cleanup failure, retaining it across successful releases and restarts without changing existing status events or exception behavior.

- Camera, stream and stream-manager runtime moved here from `inference.core.interfaces`.
- `python -m streamvision --host-factory ...` launches the stream manager standalone (needs the `webrtc` extra).
- `scripts/streamvision_isolation_probe.py` verifies an installed wheel runs without `inference`.
- `nvdec` extra installs PyNvVideoCodec for NVIDIA hardware decoding of video files; messages shown when it is missing name the extra.
- `streamvision.workflows_v2` (needs the `workflows` extra) adds the Workflows 2.0 sources `video/stream@v1` and `video/stream_set@v1` over `VideoSource`, with bounded per-camera buffers, partial camera batches, member failure policies and a `VideoStatistics` resource for final per-camera counters and end reasons. `end="any"` ends a set at the first camera end, even while other cameras have frames ready. Closing fails with `VideoStopError` when a thread outlives `stop_timeout` or any selected producer fails to release. The `video_producer_factories` resource injects producer factories keyed by the exact declared reference.

### Changed

- `roboflow-workflows` moved from base dependencies to the `workflows` extra; install `streamvision[workflows]` for `InferencePipeline.init_with_workflow` and the stream manager.
- `StreamsConfiguration` manager address fields default to `None`; the manager then reads `STREAM_MANAGER_HOST`, `STREAM_MANAGER_PORT` and `STREAM_MANAGER_SOCKET_TIMEOUT`.

### Fixed

- Frames decoded after a `DROP_OLDEST` buffer eviction retain declared and measured FPS metadata, including when the rate is unknown.

- The stream manager no longer blocks forever on a command sent to a pipeline process that has exited; the caller gets a `not_found` error.
- `GstreamerCudaVideoFrameProducer` decodes container files such as MP4 and MKV; the previous pipeline never produced a frame for them.
- Declared dependencies are enough to import the workflows runtime, and the `test` extra installs `pytest-asyncio` and `pytest-timeout`.
- `streamvision.stream.watchdog.RTCPeerConnection` resolves again.
- Processes started by the stream manager install the passed configuration before importing the host factory module.
