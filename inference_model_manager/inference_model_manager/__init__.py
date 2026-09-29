"""
Public API of the ``inference-model-manager`` package.

The model manager orchestrates model lifecycle (load/unload/evict) and
dispatches inference to direct in-process backends, or to community/plugin
backends registered via the ``inference_model_manager.backends`` entry point.

Usage::

    from inference_model_manager import ModelManager

    mm = ModelManager()
    mm.load("yolov8n-640", api_key=key, backend="direct")
    result = mm.process("yolov8n-640", images=image, confidence=0.7)
    mm.shutdown()
"""

import importlib.metadata as _meta

try:
    __version__ = _meta.version(__package__ or __name__)
except _meta.PackageNotFoundError:
    __version__ = "development"

from inference_model_manager.model_manager import ModelManager
