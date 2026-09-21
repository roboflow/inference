"""Historic identifiers for the moved Workflows package.

Public identifiers must keep their pre-move dotted form so operator-facing
config (WORKFLOW_DISABLED_BLOCK_PATTERNS), user-visible error strings and
telemetry (fully_qualified_block_class_name, logger names) do not change
when the code lives under ``roboflow_workflows.*`` instead of
``inference.core.workflows.*`` / ``inference.enterprise.workflows.enterprise_blocks.*``.

We map at read-time; ``class.__module__`` stays canonical so pickling and
``importlib.import_module(cls.__module__)`` continue to work in a standalone
``roboflow_workflows`` install where ``inference.*`` is not importable.
"""

from __future__ import annotations

import logging

_CANONICAL_ROOT = "roboflow_workflows"
_ENTERPRISE_CANONICAL = "roboflow_workflows.enterprise_blocks"
_CORE_LEGACY = "inference.core.workflows"
_ENTERPRISE_LEGACY = "inference.enterprise.workflows.enterprise_blocks"


def to_legacy_module(module_name: str) -> str:
    """Return the historic ``inference.*`` dotted form of a canonical module."""
    if module_name == _ENTERPRISE_CANONICAL or module_name.startswith(
        _ENTERPRISE_CANONICAL + "."
    ):
        return _ENTERPRISE_LEGACY + module_name[len(_ENTERPRISE_CANONICAL) :]
    if module_name == _CANONICAL_ROOT or module_name.startswith(_CANONICAL_ROOT + "."):
        return _CORE_LEGACY + module_name[len(_CANONICAL_ROOT) :]
    return module_name


def get_logger(module_name: str) -> logging.Logger:
    """``logging.getLogger`` under the historic ``inference.*`` name.

    Keeps the logger tree rooted at the ``inference`` logger so handlers /
    level configured on ``inference`` propagate as they did before the move.
    """
    return logging.getLogger(to_legacy_module(module_name))


def historic_type_display(type_obj: type) -> str:
    """Fully-qualified type name using the historic module path.

    Used by user-facing error strings that spell out ``type.__module__`` so
    the string does not change when the code moves under ``roboflow_workflows``.
    Unlike block identifiers, these errors historically included ``builtins``.
    """
    module = to_legacy_module(type_obj.__module__)
    return f"{module}.{type_obj.__qualname__}"
