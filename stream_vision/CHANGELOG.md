# Changelog

This is the canonical changelog for the `streamvision` package.

## Unreleased

### Added

- Camera, stream and stream-manager runtime moved here from `inference.core.interfaces`.
- `python -m streamvision --host-factory ...` launches the stream manager standalone (needs the `webrtc` extra).
- `scripts/streamvision_isolation_probe.py` verifies an installed wheel runs without `inference`.
