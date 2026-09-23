"""Restriction presets shared by the model blocks.

These are the ``get_actual_restrictions()`` presets of the model blocks:
``RuntimeRestriction`` values carrying the human ``note``, the stable ``code``
and the condition the caveat applies under. The legacy ``get_restrictions()``
classmethod keeps publishing its own, environment-filtered list; these presets
describe the TARGET deployment, so a consumer can evaluate them for ITS own
runtime.

Two rules hold for everything in this module:

* nothing here reads the environment or the block configuration - a preset is
  a constant (or a pure function of its arguments), so a block declares the
  same list on every host;
* a flag-gated caveat is declared unconditionally and carries the flag value it
  applies to in ``applies_to_configuration``; it is never dropped because the
  flag happens to be enabled on the machine doing the introspection.
  ``get_actual_restrictions(ignore_environment_restrictions=False)`` is the
  single place where this host's flags are consulted.

The stateful-video / still-image presets are re-exported from
``roboflow_workflows.prototypes.block`` so that a model block needs a single
import for all of its restrictions.
"""

from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.prototypes.block import (  # noqa: F401
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
)

__all__ = [
    "DEPRECATED_BLOCK_ALWAYS_RAISES",
    "REQUIRES_GPU_FOR_LOCAL_EXECUTION",
    "ROBOFLOW_INTERNAL_ENDPOINT_ONLY",
    "STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION",
    "STILL_IMAGE_INPUT_SOFT_RESTRICTION",
    "UNSUPPORTED_IN_TENSOR_REPRESENTATION",
    "hosted_endpoint_disabled_by_flag",
]


REQUIRES_GPU_FOR_LOCAL_EXECUTION = RuntimeRestriction(
    code="requires_gpu_for_local_execution",
    severity=Severity.HARD,
    note="Requires a GPU; run_locally() loads a model that needs CUDA.",
    applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
    applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
)
"""Local execution loads a model that needs CUDA, so a CPU-only host fails."""


ROBOFLOW_INTERNAL_ENDPOINT_ONLY = RuntimeRestriction(
    code="roboflow_internal_endpoint_only",
    severity=Severity.HARD,
    note=(
        "The block's only execution path is a Roboflow-internal endpoint, "
        "which is reachable from Roboflow-hosted runtimes only; self-hosted "
        "deployments cannot run this block."
    ),
    applies_to_runtimes=[
        Runtime.SELF_HOSTED_CPU,
        Runtime.SELF_HOSTED_GPU,
        Runtime.INFERENCE_PIPELINE,
    ],
)
"""The block's only execution path is a Roboflow-internal endpoint that a
self-hosted deployment cannot reach."""


DEPRECATED_BLOCK_ALWAYS_RAISES = RuntimeRestriction(
    code="deprecated_block_always_raises",
    severity=Severity.HARD,
    note=(
        "Block is deprecated: run() raises FeatureDeprecatedError before doing "
        "any work, so the step can never produce a result."
    ),
)
"""`run()` raises `FeatureDeprecatedError` before doing any work, on every
runtime and in every execution mode - the step can never produce a result."""


UNSUPPORTED_IN_TENSOR_REPRESENTATION = RuntimeRestriction(
    code="unsupported_in_tensor_representation",
    severity=Severity.HARD,
    note=(
        "The tensor-native implementation of this block raises "
        "FeatureDeprecatedError, so the block only works on a server running "
        "the numpy image representation."
    ),
    applies_to_configuration={"ENABLE_TENSOR_DATA_REPRESENTATION": True},
)
"""The tensor-native implementation of the block raises
`FeatureDeprecatedError`, so the block only works on a server running the numpy
representation."""


def hosted_endpoint_disabled_by_flag(flag_name: str) -> RuntimeRestriction:
    """Build the "hosted endpoint is not registered" caveat for one flag.

    ``flag_name`` is the server environment flag that registers the model's
    endpoint (e.g. ``MOONDREAM2_ENABLED``). The returned declaration applies
    when the target runtime has that flag set to ``False``: remote execution
    then reaches a route that was never registered and returns 404.

    The flag value of the host running the introspection is irrelevant and is
    never read here.
    """
    return RuntimeRestriction(
        code="hosted_endpoint_disabled_by_flag",
        severity=Severity.HARD,
        note=(
            f"{flag_name}=False on Roboflow Hosted Serverless: the model's "
            f"endpoint is not registered, so remote execution returns 404."
        ),
        applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
        applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
        applies_to_configuration={flag_name: False},
    )
