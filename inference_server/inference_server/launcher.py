"""Inference stack wiring.

  launch_inprocess()     → ModelManager
      Caller loads backends directly via mm.load(..., backend="direct").
      Inference via mm.infer_sync() / mm.submit() / mm.infer_async().
      One model per ModelManager; no ZMQ involved.
"""

from __future__ import annotations

from inference_server.legacy_env import apply_legacy_env

apply_legacy_env()

from inference_model_manager.model_manager import ModelManager  # noqa: E402


def launch_inprocess() -> ModelManager:
    """Return a ModelManager for in-process use.

    Example::

        mm = launch_inprocess()
        mm.load("yolov8n-640", api_key=key)
        result = mm.infer_sync("yolov8n-640", image_bytes)
    """
    return ModelManager()
