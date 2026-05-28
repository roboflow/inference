from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from inference_server.framework.entities import ModelHandlerDescription


_HANDLERS: dict[tuple[str, str], ModelHandlerDescription] = {}


def _register(
    model_type: str, action: str, description: ModelHandlerDescription
) -> None:
    key = (model_type, action)
    if key in _HANDLERS:
        raise RuntimeError(f"duplicate handler registration: {key}")
    _HANDLERS[key] = description


DYNAMIC_MODELS_HANDLERS: Mapping[tuple[str, str], ModelHandlerDescription] = (
    MappingProxyType(_HANDLERS)
)


def supported_actions_for(model_type: str) -> list[str]:
    return sorted(
        action for (mt, action) in DYNAMIC_MODELS_HANDLERS if mt == model_type
    )


def has_handler_for_model_type(model_type: str) -> bool:
    return any(mt == model_type for (mt, _) in DYNAMIC_MODELS_HANDLERS)


import inference_server.handlers.object_detection.description  # noqa: E402, F401
import inference_server.handlers.classification.description  # noqa: E402, F401
import inference_server.handlers.multilabel_classification.description  # noqa: E402, F401
import inference_server.handlers.instance_segmentation.description  # noqa: E402, F401
import inference_server.handlers.semantic_segmentation.description  # noqa: E402, F401
import inference_server.handlers.keypoints.description  # noqa: E402, F401
import inference_server.handlers.depth.description  # noqa: E402, F401
