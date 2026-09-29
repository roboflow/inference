"""Serialisation of buffered workflow results before they leave the manager.

Replaces the manager's use of the deprecated HTTP helper
`inference.core.interfaces.http.orjson_utils.serialise_single_workflow_result_element`
with the same behavior and no HTTP imports.
"""

from typing import Any, Callable, Dict, List, Optional

from inference.core.interfaces.stream import environment


def _resolve_wildcard_serializer() -> Callable[..., Any]:
    """Pick the wildcard serialiser at call time (not import time).

    The tensor-native path yields torch (often CUDA) tensors that must be moved
    to CPU before results cross the stream-manager process boundary via a
    multiprocessing queue: pickling a live CUDA tensor relies on CUDA IPC, which
    is unsupported on Jetson/Tegra. The tensor serialiser calls
    `.detach().cpu()`; the numpy one passes tensor-native objects through
    untouched. This is the only place results are materialised on CPU - the
    pipeline itself hands tensors to its sinks as they are.

    The flag is read from the configuration facade as a module attribute on
    every call, and the tensor serialiser (which imports torch) is only
    imported when selected.
    """
    if environment.ENABLE_TENSOR_DATA_REPRESENTATION:
        from roboflow_workflows.core_steps.common.serializers_tensor import (
            serialize_wildcard_kind,
        )
    else:
        from roboflow_workflows.core_steps.common.serializers import (
            serialize_wildcard_kind,
        )
    return serialize_wildcard_kind


def serialise_single_workflow_result_element(
    result_element: Dict[str, Any],
    excluded_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Serialise one workflow result, dropping the excluded top-level outputs.

    Args:
        result_element: Workflow outputs of a single frame, keyed by name.
        excluded_fields: Top-level output names to leave out. Nested keys with
            the same names are kept.

    Returns:
        The outputs converted by the Workflows wildcard serialiser of the
        active data representation.
    """
    if excluded_fields is None:
        excluded_fields = []
    excluded_fields = set(excluded_fields)
    serialize_wildcard_kind = _resolve_wildcard_serializer()
    serialised_result = {}
    for key, value in result_element.items():
        if key in excluded_fields:
            continue
        serialised_result[key] = serialize_wildcard_kind(value=value)

    return serialised_result
