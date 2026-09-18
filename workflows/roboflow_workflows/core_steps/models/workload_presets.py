"""Portable restriction presets shared by the model blocks.

These are the ``RestrictionMetadata`` counterparts of the caveats the model
blocks already publish through ``get_restrictions()``. The legacy declarations
carry a human ``note`` and are filtered against this host's environment flags;
the portable ones carry a stable ``code`` plus the condition under which the
caveat applies, so a target service can evaluate them for ITS own runtime.

Two rules hold for everything in this module:

* nothing here reads the environment or the block configuration - a preset is
  a constant (or a pure function of its arguments), so
  ``discover_portable_restrictions()`` returns the same list on every host;
* a flag-gated caveat is declared unconditionally and carries the flag value it
  applies to in ``when.configuration_equals``; it is never dropped because the
  flag happens to be enabled on the machine doing the introspection.

The stateful-video / still-image presets are re-exported from
``roboflow_workflows.prototypes.block`` so that a model block needs a single
import for all of its portable restrictions.
"""

from roboflow_workflows.execution_engine.entities.workload import (
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.prototypes.block import (  # noqa: F401
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
)

__all__ = [
    "DEPRECATED_BLOCK_ALWAYS_RAISES",
    "REQUIRES_GPU_FOR_LOCAL_EXECUTION",
    "ROBOFLOW_INTERNAL_ENDPOINT_ONLY",
    "STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION",
    "STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION",
    "UNSUPPORTED_IN_TENSOR_REPRESENTATION",
    "hosted_endpoint_disabled_by_flag",
]


REQUIRES_GPU_FOR_LOCAL_EXECUTION = RestrictionMetadata(
    code="requires_gpu_for_local_execution",
    severity=Severity.HARD,
    when=RestrictionCondition(
        runtimes=[Runtime.SELF_HOSTED_CPU],
        step_execution_modes=[StepExecutionMode.LOCAL],
    ),
)
"""Local execution loads a model that needs CUDA, so a CPU-only host fails."""


ROBOFLOW_INTERNAL_ENDPOINT_ONLY = RestrictionMetadata(
    code="roboflow_internal_endpoint_only",
    severity=Severity.HARD,
    when=RestrictionCondition(
        runtimes=[
            Runtime.SELF_HOSTED_CPU,
            Runtime.SELF_HOSTED_GPU,
            Runtime.INFERENCE_PIPELINE,
        ],
    ),
)
"""The block's only execution path is a Roboflow-internal endpoint that a
self-hosted deployment cannot reach."""


DEPRECATED_BLOCK_ALWAYS_RAISES = RestrictionMetadata(
    code="deprecated_block_always_raises",
    severity=Severity.HARD,
    when=RestrictionCondition(),
)
"""`run()` raises `FeatureDeprecatedError` before doing any work, on every
runtime and in every execution mode - the step can never produce a result."""


UNSUPPORTED_IN_TENSOR_REPRESENTATION = RestrictionMetadata(
    code="unsupported_in_tensor_representation",
    severity=Severity.HARD,
    when=RestrictionCondition(
        configuration_equals={"ENABLE_TENSOR_DATA_REPRESENTATION": True},
    ),
)
"""The tensor-native implementation of the block raises
`FeatureDeprecatedError`, so the block only works on a server running the numpy
representation."""


def hosted_endpoint_disabled_by_flag(flag_name: str) -> RestrictionMetadata:
    """Build the "hosted endpoint is not registered" caveat for one flag.

    ``flag_name`` is the server environment flag that registers the model's
    endpoint (e.g. ``MOONDREAM2_ENABLED``). The returned declaration applies
    when the target runtime has that flag set to ``False``: remote execution
    then reaches a route that was never registered and returns 404.

    The flag value of the host running the introspection is irrelevant and is
    never read here.
    """
    return RestrictionMetadata(
        code="hosted_endpoint_disabled_by_flag",
        severity=Severity.HARD,
        when=RestrictionCondition(
            runtimes=[Runtime.HOSTED_SERVERLESS],
            step_execution_modes=[StepExecutionMode.REMOTE],
            configuration_equals={flag_name: False},
        ),
    )
