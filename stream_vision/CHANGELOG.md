# Changelog

This is the canonical changelog for the `streamvision` package.

## Unreleased

### Added

- Camera, stream and stream-manager runtime moved here from `inference.core.interfaces`.
- `python -m streamvision --host-factory ...` launches the stream manager standalone (needs the `webrtc` extra).
- `scripts/streamvision_isolation_probe.py` verifies an installed wheel runs without `inference`.
- `nvdec` extra installs PyNvVideoCodec for NVIDIA hardware decoding of video files; messages shown when it is missing name the extra.

### Fixed

- The stream manager no longer blocks forever on a command sent to a pipeline process that has exited; the caller gets a `not_found` error.
- `GstreamerCudaVideoFrameProducer` decodes container files such as MP4 and MKV; the previous pipeline never produced a frame for them.
