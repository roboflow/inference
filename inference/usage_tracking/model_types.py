"""Process-local map of model id to Roboflow ``modelType``.

Populated while a model type is being resolved for model loading, so that usage
tracking can label a row with the model architecture without ever calling the
model registry itself. Registry lookups may issue an authenticated HTTP request,
and the usage decorator runs on the inference hot path.

The map is a bounded FIFO: when the cap is hit, the oldest entry is evicted so a
pathological stream of distinct ids cannot grow the process forever, and newer
ids still get a chance to be labeled.
"""

from collections import OrderedDict
from typing import Optional

# Servers load a bounded number of models, but the map is keyed by caller-supplied
# ids so it is capped to keep a pathological caller from growing it without end.
_MAX_TRACKED_MODELS = 1024

_MODEL_TYPES: OrderedDict[str, str] = OrderedDict()


def record_model_type(model_id: Optional[str], model_type: Optional[str]) -> None:
    if not model_id or not model_type:
        return
    model_id = str(model_id)
    model_type = str(model_type)
    if model_id in _MODEL_TYPES:
        _MODEL_TYPES.move_to_end(model_id)
        _MODEL_TYPES[model_id] = model_type
        return
    while len(_MODEL_TYPES) >= _MAX_TRACKED_MODELS:
        _MODEL_TYPES.popitem(last=False)
    _MODEL_TYPES[model_id] = model_type


def get_recorded_model_type(model_id: Optional[str]) -> Optional[str]:
    if not model_id:
        return None
    return _MODEL_TYPES.get(str(model_id))


def clear_recorded_model_types() -> None:
    _MODEL_TYPES.clear()
