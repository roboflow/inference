"""Compatibility seams for video-worker experiments across inference runtimes.

Variant D deliberately keeps the exact legacy inference 1.3.5 runtime while
adding only the per-job process topology. Tensor-native workflow symbols and
the freshest-frame pipeline option were added later, so importing or passing
them unconditionally makes the legacy control invalid before it starts.
"""

import importlib
import inspect

TENSOR_FLAG_NAME = "ENABLE_TENSOR_DATA_REPRESENTATION"
FRESHEST_MODE_PARAMETER = "video_processing_mode"


def resolve_workflow_serializer():
    """Return the effective tensor flag and its matching workflow serializer.

    Missing tensor support is a normal legacy-runtime capability result, not an
    instruction to infer it from the similarly named environment variable.
    This prevents a stray flag from selecting a module that inference 1.3.5
    does not contain.
    """

    inference_env = importlib.import_module("inference.core.env")
    tensor_enabled = bool(getattr(inference_env, TENSOR_FLAG_NAME, False))
    module_name = (
        "inference.core.workflows.core_steps.common.serializers_tensor"
        if tensor_enabled
        else "inference.core.workflows.core_steps.common.serializers"
    )
    serializer_module = importlib.import_module(module_name)
    return tensor_enabled, serializer_module.serialize_wildcard_kind


def pipeline_supports_freshest_mode(inference_pipeline):
    """Whether this runtime accepts the v1.4 live-video policy keyword."""

    try:
        parameters = inspect.signature(inference_pipeline.init_with_workflow).parameters
    except (TypeError, ValueError):
        return False
    return FRESHEST_MODE_PARAMETER in parameters
