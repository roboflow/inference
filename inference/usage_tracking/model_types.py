"""Process-local map of model id to Roboflow ``modelType``.

Populated while a model type is being resolved for model loading, so that usage
tracking can label a row with the model architecture without ever calling the
model registry itself. Registry lookups may issue an authenticated HTTP request,
and the usage decorator runs on the inference hot path.
"""

from threading import Lock
from typing import Dict, Optional

# Servers load a bounded number of models, but the map is keyed by caller-supplied
# ids so it is capped to keep a pathological caller from growing it without end.
_MAX_TRACKED_MODELS = 1024

_lock = Lock()
_model_types: Dict[str, str] = {}


def record_model_type(model_id: Optional[str], model_type: Optional[str]) -> None:
    if not model_id or not model_type:
        return
    model_id = str(model_id)
    with _lock:
        if model_id not in _model_types and len(_model_types) >= _MAX_TRACKED_MODELS:
            return
        _model_types[model_id] = str(model_type)


def get_recorded_model_type(model_id: Optional[str]) -> Optional[str]:
    if not model_id:
        return None
    with _lock:
        return _model_types.get(str(model_id))


def clear_recorded_model_types() -> None:
    with _lock:
        _model_types.clear()
