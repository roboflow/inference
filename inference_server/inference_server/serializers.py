"""JSON serializers for model predictions.

Uses orjson with OPT_SERIALIZE_NUMPY for zero-copy numpy→JSON.
Compact format: parallel arrays (no per-object dicts).

Benchmark (bench_serialization.py):
  5 boxes:   2.4 us (23x faster than pickle)
  50 boxes:  6.5 us (8x faster than pickle)
  200 boxes: 19 us  (3x faster than pickle)
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Any

import numpy as np
import orjson
import torch

_ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY


def _to_numpy(t: Any) -> Any:
    """Tensor/ndarray → numpy; passthrough otherwise."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if isinstance(t, np.ndarray):
        return t
    return t


def serialize_json(obj: Any) -> bytes:
    """Serialize any model output to compact JSON bytes.

    Handles:
      - Detections, InstanceDetections, KeyPoints (with xyxy/class_id/confidence)
      - ClassificationPrediction, MultiLabelClassificationPrediction
      - SemanticSegmentationResult
      - List of any of the above
      - Raw torch.Tensor / numpy.ndarray
      - Tuple (e.g. OCR returns (List[str], List[Detections]))
      - Anything with __dict__ (generic dataclass)
      - Primitives (str, int, float, list, dict)
    """
    return orjson.dumps(_convert(obj), option=_ORJSON_OPTS)


def _convert(obj: Any) -> Any:
    """Recursively convert model output to JSON-serializable structure."""
    if obj is None:
        return None

    # Tensor → numpy
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        return obj

    # List / tuple of results
    if isinstance(obj, (list, tuple)):
        converted = [_convert(item) for item in obj]
        return converted

    # Dict passthrough with conversion
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}

    # Dataclass → dict of fields
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = _convert(val)
        return result

    # Primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")

    # Fallback: try __dict__
    if hasattr(obj, "__dict__"):
        return {
            k: _convert(v) for k, v in obj.__dict__.items() if not k.startswith("_")
        }

    return str(obj)
