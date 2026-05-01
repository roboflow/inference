"""
Public API of the ``inference-model-manager`` package.

The model manager orchestrates model lifecycle (load/unload/evict),
SHM-based zero-copy transport, subprocess worker management, and batched
GPU inference.

Usage::

    from inference_model_manager import ModelManager

    mm = ModelManager()
    mm.load("yolov8n-640", api_key=key, backend="subprocess")
    result = mm.infer_sync("yolov8n-640", image)
    mm.shutdown()
"""

import importlib.metadata as _meta

try:
    __version__ = _meta.version(__package__ or __name__)
except _meta.PackageNotFoundError:
    __version__ = "development"

from inference_model_manager.model_manager import ModelManager
