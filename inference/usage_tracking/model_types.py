"""Process-local map of model id to Roboflow model architecture and variant.

Populated while a model type is being resolved for model loading, so that usage
tracking can label a row with the same architecture / variant pair the Roboflow
model registry stores, without ever calling the model registry itself. Registry
lookups may issue an authenticated HTTP request, and the usage decorator runs on
the inference hot path.

The map is a bounded FIFO: when the cap is hit, the oldest entry is evicted so a
pathological stream of distinct ids cannot grow the process forever, and newer
ids still get a chance to be labeled.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ModelDescriptor:
    """Labels describing which model served a usage row.

    Not a unique id: many loaded models share the same architecture / variant
    pair.

    Attributes:
        architecture: Model architecture, e.g. ``yolov8`` or ``sam2``.
        variant: Distinguishing size / task label within the architecture, e.g.
            ``yolov8-n`` or ``hiera_large``. None when the architecture is
            served in a single flavour.
    """

    architecture: str
    variant: Optional[str] = None


# Servers load a bounded number of models, but the map is keyed by caller-supplied
# ids so it is capped to keep a pathological caller from growing it without end.
_MAX_TRACKED_MODELS = 1024

_MODEL_DESCRIPTORS: OrderedDict[str, ModelDescriptor] = OrderedDict()


def record_model_descriptor(
    model_id: Optional[str],
    *,
    architecture: Optional[str],
    variant: Optional[str] = None,
) -> None:
    """Remember the architecture / variant pair resolved for a model id.

    Args:
        model_id: Id the caller asked for, or its de-aliased spelling.
        architecture: Model architecture reported by the registry.
        variant: Platform ``modelVariant``, or the coded-model variant suffix.
    """
    if not model_id or not architecture:
        return

    model_id = str(model_id)
    descriptor = ModelDescriptor(
        architecture=str(architecture),
        variant=str(variant) if variant else None,
    )
    if model_id in _MODEL_DESCRIPTORS:
        _MODEL_DESCRIPTORS.move_to_end(model_id)
        _MODEL_DESCRIPTORS[model_id] = descriptor
        return

    while len(_MODEL_DESCRIPTORS) >= _MAX_TRACKED_MODELS:
        _MODEL_DESCRIPTORS.popitem(last=False)
    _MODEL_DESCRIPTORS[model_id] = descriptor


def get_recorded_model_descriptor(
    model_id: Optional[str],
) -> Optional[ModelDescriptor]:
    """Return the descriptor recorded for ``model_id``, or None."""
    if not model_id:
        return None

    return _MODEL_DESCRIPTORS.get(str(model_id))


def bind_usage_model_descriptor(model: Any, *model_ids: Optional[str]) -> None:
    """Copy the recorded usage descriptor onto a loaded model instance.

    The map is filled during registry resolve. Storing the labels on the
    instance means later ``infer()`` calls do not need the caller to pass
    ``model_id``, and the labels survive map eviction.

    Does not set ``model.model_id``. That field is the usage ``resource_id``
    and must stay whatever the caller / request already used.

    Args:
        model: Loaded model instance to label.
        *model_ids: Candidate ids to look up, in order of preference.
    """
    if model is None:
        return

    recorded = None
    for model_id in model_ids:
        if not model_id:
            continue
        recorded = get_recorded_model_descriptor(str(model_id))
        if recorded:
            break

    if recorded:
        model.model_architecture = recorded.architecture
        model.model_variant = recorded.variant


def clear_recorded_model_descriptors() -> None:
    _MODEL_DESCRIPTORS.clear()
