"""Subprocess probe for `test_streams_public_contract.py`'s frozen contracts.

Importing `inference` at test-module import time binds several factory
defaults (`predictions_queue_size`, `decoding_buffer_size`, `Stream.api_key`,
...) from whatever environment variables happen to be set in the pytest
process - so the same test could pass or fail depending on ambient shell
state that has nothing to do with an actual contract change. This module is
run as a standalone script, once per environment a caller wants to probe, so
`inference` is only ever imported inside a fresh subprocess whose
environment the caller fully controls; nothing here touches the parent
pytest process's environment or its already-imported modules.

Merely importing this module (as opposed to running it) only defines pure
helpers and never imports `inference` - `test_streams_public_contract.py`
reuses `_stable_signature` directly for its self-contained renderer checks.
"""

import hashlib
import importlib
import inspect
import json
from enum import Enum
from typing import Callable, Dict, Tuple

# (frozen-contract name, "module.path:Qualified.attribute") - kept in the
# same order as `_FROZEN_CONTRACTS` in test_streams_public_contract.py, whose
# expected hashes this probe's output is compared against.
_CONTRACT_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "InferencePipeline.init",
        "inference.core.interfaces.stream.inference_pipeline:InferencePipeline.init",
    ),
    (
        "InferencePipeline.init_with_yolo_world",
        "inference.core.interfaces.stream.inference_pipeline:"
        "InferencePipeline.init_with_yolo_world",
    ),
    (
        "InferencePipeline.init_with_workflow",
        "inference.core.interfaces.stream.inference_pipeline:"
        "InferencePipeline.init_with_workflow",
    ),
    (
        "InferencePipeline.init_with_custom_logic",
        "inference.core.interfaces.stream.inference_pipeline:"
        "InferencePipeline.init_with_custom_logic",
    ),
    ("Stream.__init__", "inference.core.interfaces.stream.stream:Stream.__init__"),
    ("sinks.display_image", "inference.core.interfaces.stream.sinks:display_image"),
    ("sinks.render_boxes", "inference.core.interfaces.stream.sinks:render_boxes"),
    (
        "sinks.render_statistics",
        "inference.core.interfaces.stream.sinks:render_statistics",
    ),
    ("sinks.multi_sink", "inference.core.interfaces.stream.sinks:multi_sink"),
    (
        "sinks.active_learning_sink",
        "inference.core.interfaces.stream.sinks:active_learning_sink",
    ),
)

_STABLE_DEFAULT_TYPES = (type(None), bool, int, float, str, tuple, frozenset)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _stable_default_repr(default: object, fn: Callable) -> str:
    if isinstance(default, _STABLE_DEFAULT_TYPES):
        return repr(default)
    if isinstance(default, Enum):
        return f"{type(default).__qualname__}.{default.name}"
    if callable(default):
        return f"{default.__module__}.{default.__qualname__}"
    fn_module = inspect.getmodule(fn)
    if fn_module is not None:
        for attr_name in dir(fn_module):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(fn_module, attr_name)
                if attr_value is default:
                    return f"{fn_module.__name__}.{attr_name}"
            except (AttributeError, TypeError):
                pass
    raise ValueError(
        f"Cannot serialize non-literal default {default!r} of type "
        f"{type(default).__module__}.{type(default).__qualname__} - "
        f"expected enum, callable, or module-level sentinel"
    )


def _stable_signature(fn: Callable) -> str:
    sig = inspect.signature(fn)
    parts = []
    last_kind = None
    for param in sig.parameters.values():
        if (
            last_kind is inspect.Parameter.POSITIONAL_ONLY
            and param.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            parts.append("/")
        if param.kind is inspect.Parameter.KEYWORD_ONLY and last_kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            parts.append("*")
        piece = param.name
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            piece = f"*{piece}"
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            piece = f"**{piece}"
        if param.annotation is not inspect.Parameter.empty:
            piece += f": {param.annotation}"
        if param.default is not inspect.Parameter.empty:
            default_repr = _stable_default_repr(param.default, fn)
            piece += f" = {default_repr}"
        parts.append(piece)
        last_kind = param.kind
    if last_kind is inspect.Parameter.POSITIONAL_ONLY:
        parts.append("/")
    rendered = f"({', '.join(parts)})"
    if sig.return_annotation is not inspect.Signature.empty:
        rendered += f" -> {sig.return_annotation}"
    return rendered


def _resolve_contract(dotted_path: str) -> Callable:
    """Resolve `"module.path:Attr.chain"` to the live callable it names."""
    module_name, _, attr_path = dotted_path.partition(":")
    target = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target


def _capture_contract(dotted_path: str) -> Dict[str, str]:
    fn = _resolve_contract(dotted_path)
    signature = _stable_signature(fn)
    docstring = inspect.getdoc(fn) or ""

    return {
        "signature": signature,
        "sig_hash": _hash(signature),
        "doc_hash": _hash(docstring),
    }


def _capture_all() -> dict:
    """Import `inference` in this process and capture every frozen contract.

    Only called from `__main__`, after the caller has already set up this
    process's environment - never on a plain import of this module.
    """
    from inference.core.interfaces.stream.environment import (
        DEFAULT_BUFFER_SIZE,
        ENABLE_TENSOR_DATA_REPRESENTATION,
        PREDICTIONS_QUEUE_SIZE,
        PREDICTIONS_QUEUE_SIZE_EXPLICIT,
    )

    contracts = {name: _capture_contract(path) for name, path in _CONTRACT_PATHS}
    config = {
        "predictions_queue_size": PREDICTIONS_QUEUE_SIZE,
        "predictions_queue_size_explicit": PREDICTIONS_QUEUE_SIZE_EXPLICIT,
        "decoding_buffer_size": DEFAULT_BUFFER_SIZE,
        "enable_tensor_data_representation": ENABLE_TENSOR_DATA_REPRESENTATION,
    }

    return {"contracts": contracts, "config": config}


if __name__ == "__main__":
    print(json.dumps(_capture_all()))
