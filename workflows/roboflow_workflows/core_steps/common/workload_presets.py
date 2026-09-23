"""Shared restriction declarations for the non-model core blocks.

These constants are the ``get_actual_restrictions()`` presets of the core
blocks: ``RuntimeRestriction`` values carrying a human ``note``, a stable
``code`` and the condition under which the caveat applies. They are the SAME
entity the legacy ``get_restrictions()`` returns, so a block never authors a
restriction twice; the workload document derives its portable
``RestrictionMetadata`` from them through ``restriction_metadata_of()``.

Two rules hold for everything in this module:

* **No environment or configuration reads.** A preset is plain data. A
  restriction that a legacy ``get_restrictions()`` emits only on one branch of
  a flag is declared here UNCONDITIONALLY, with the flag pinned in
  ``applies_to_configuration``. The declaration therefore describes the TARGET
  configuration; whether it applies to the host answering an introspection call
  is decided by ``get_actual_restrictions(ignore_environment_restrictions=False)``,
  never by the preset itself.
* **One code, one meaning.** Codes come from the coordinator-owned registry;
  a preset is reused only where the semantics are actually identical.

Presets that already have a sibling constant living next to their block
(``core_steps/cache/common.py`` and ``core_steps/sinks/onvif_movement/v1.py``)
stay next to that sibling, so the note and the code cannot drift apart.
"""

from typing import Tuple

from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
)

# The pair emitted by every block whose legacy declaration is
# ``[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION, STILL_IMAGE_INPUT_SOFT_RESTRICTION]``
# - cross-frame state in process memory, plus "a still image has no history to
# work with". Blocks that spell the first restriction out with their own note
# (frame stack, heat accumulation, trace history) share the same axes and
# therefore the same code.
STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS: Tuple[RuntimeRestriction, ...] = (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
)


# ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False`` makes the local file
# sink raise at run time.
LOCAL_STORAGE_ACCESS_DISABLED_RESTRICTION = RuntimeRestriction(
    code="local_storage_access_disabled",
    severity=Severity.HARD,
    note=(
        "Block raises RuntimeError when ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_"
        "STORAGE is False."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_configuration={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": False},
)


# The two caveats of the ENABLED branch: on a dedicated deployment the file is
# written but cannot be fetched back through the Roboflow API, and on hosted
# serverless the container disk disappears with the worker.
DEPLOYMENT_VOLUME_NOT_RETRIEVABLE_RESTRICTION = RuntimeRestriction(
    code="writes_to_deployment_volume_not_retrievable",
    severity=Severity.SOFT,
    note=(
        "Files are persisted on the deployment's volume but are not "
        "retrievable through the Roboflow API; treat as internal-only logs."
    ),
    applies_to_runtimes=[Runtime.DEDICATED_DEPLOYMENT],
    applies_to_configuration={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True},
)


EPHEMERAL_CONTAINER_DISK_RESTRICTION = RuntimeRestriction(
    code="ephemeral_container_disk_loses_writes",
    severity=Severity.SOFT,
    note=(
        "Container disk is ephemeral, so files are lost when the worker scales "
        "down; if there's more than one replica consuming workflow requests "
        "the result will be non deterministic."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
    applies_to_configuration={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True},
)


# Both flag branches together: whichever way the target deployment sets
# ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE``, exactly the entries whose
# ``applies_to_configuration`` matches it apply.
LOCAL_FILE_SINK_RESTRICTIONS: Tuple[RuntimeRestriction, ...] = (
    DEPLOYMENT_VOLUME_NOT_RETRIEVABLE_RESTRICTION,
    EPHEMERAL_CONTAINER_DISK_RESTRICTION,
    LOCAL_STORAGE_ACCESS_DISABLED_RESTRICTION,
)


ENVIRONMENT_VARIABLE_ACCESS_DISABLED_RESTRICTION = RuntimeRestriction(
    code="environment_variable_access_disabled",
    severity=Severity.HARD,
    note=(
        "Block raises RuntimeError when ALLOW_WORKFLOW_BLOCKS_ACCESSING_"
        "ENVIRONMENTAL_VARIABLES is False. Roboflow's hosted runtimes set this "
        "flag to False for security, so environment variables cannot be "
        "exposed."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_configuration={
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": False
    },
)


# Append-log mode accumulates entries in process memory before uploading the
# whole object, so it splits across stateless workers.
S3_APPEND_BUFFER_RESTRICTION = RuntimeRestriction(
    code="s3_append_buffer_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Append-log mode buffers entries in process memory before uploading "
        "the accumulated object to S3. With remote step execution on stateless "
        "or multi-replica HTTP runtimes, successive requests may be served by "
        "different worker processes, so append-log objects can reset or split "
        "across workers. Use separate_files mode, or local step execution in "
        "an InferencePipeline when each entry must be captured in a single "
        "ordered log."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
)


# Fire-and-forget writes hide delivery / persistence failures and pile up when
# the destination is slower than the video stream. Shared by the sinks that
# offer the switch (PostgreSQL, Kafka producer), so the note stays neutral
# about what the destination is.
FIRE_AND_FORGET_RESTRICTION = RuntimeRestriction(
    code="fire_and_forget_hides_persistence_failures",
    severity=Severity.SOFT,
    note=(
        "Use fire_and_forget=false to observe delivery and persistence "
        "failures and avoid accumulating background writes when the "
        "destination is slower than the stream."
    ),
    applies_to_runtimes=[Runtime.INFERENCE_PIPELINE],
)
