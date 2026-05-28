"""Handler registry — keyed by ``(model_type, action)``.

Built at import time. Each ``handlers/<family>/description.py`` module
calls ``_register(...)`` at its own import; ``framework/registry.py``
imports those modules so the dict is fully populated by the time the
first request arrives. Post-import the dict is read-only — the public
name is a ``MappingProxyType`` so any accidental write raises
``TypeError`` instead of corrupting the contract.

No lock on the hot path: reads only, never mutated post-import. If
hot-reload or per-tenant registration ever lands, switch readers to a
snapshot and gate writes with ``threading.RLock``.
"""

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
