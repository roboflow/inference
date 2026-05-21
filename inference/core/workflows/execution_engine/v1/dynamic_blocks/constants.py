"""Shared constants for the Custom Python Block / Modal execution path.

Values defined here are part of the wire contract between the Modal handler
(`modal/modal_app.py`) and the worker-side HTTPS client
(`inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py`),
and are matched on by the per-frame failure classifier. Treat them as
public API: do not rename or remove without coordinating with the classifier.
"""

from typing import Final

# The `error_type` string returned by the Modal handler when the in-handler
# watchdog fires. Matched on by `ModalExecutor.execute_remote` to raise the
# typed `DynamicBlockTimeoutError`, and by the classifier to emit the
# `MODAL_FRAME_TIMEOUT` category. Must remain stable across releases.
MODAL_TIMEOUT_ERROR_TYPE: Final[str] = "CustomPythonBlockTimeout"
