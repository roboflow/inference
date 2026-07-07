# Edge camera parameters client

Workflow-side HTTP client for roboflow-edge runtime camera register writes.

- `camera_register_catalog.json` is vendored from `roboflow/packages/shared/device/cameraRegisterCatalog.json`.
- Sync with `scripts/sync_camera_register_catalog.sh`.
- Hardware apply logic lives in roboflow-edge `stream_camera_parameters/`, not here.
