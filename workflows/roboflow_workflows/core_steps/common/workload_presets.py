"""Shared portable restriction declarations for the non-model core blocks.

These constants are the ``discover_portable_restrictions()`` counterparts of
the human-readable ``RuntimeRestriction`` presets in
``roboflow_workflows.prototypes.block`` and of the per-block legacy
declarations: a stable ``code`` plus a ``RestrictionCondition`` instead of a
free-text ``note``. ``get_restrictions()`` is unchanged and remains the legacy
API.

Two rules hold for everything in this module:

* **No environment or configuration reads.** A preset is plain data. A
  restriction that a legacy ``get_restrictions()`` emits only on one branch of
  a flag is declared here UNCONDITIONALLY, with the flag pinned in
  ``configuration_equals``. The portable declaration therefore describes the
  TARGET configuration and never depends on the flags of the host that is
  answering the introspection call.
* **One code, one meaning.** Codes come from the coordinator-owned registry;
  a preset is reused only where the semantics are actually identical.

Presets that already have a legacy sibling constant living next to their block
(``core_steps/cache/common.py`` and ``core_steps/sinks/onvif_movement/v1.py``)
keep their portable twin next to that sibling, so the legacy note and the
portable code cannot drift apart.
"""

from typing import Tuple

from roboflow_workflows.execution_engine.entities.workload import (
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
)

# The pair emitted by every block whose legacy declaration is
# ``[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION, STILL_IMAGE_INPUT_SOFT_RESTRICTION]``
# - cross-frame state in process memory, plus "a still image has no history to
# work with". Blocks that spell the first restriction out with their own note
# (frame stack, heat accumulation, trace history) share the same axes and
# therefore the same code.
STATEFUL_VIDEO_TEMPORAL_PORTABLE_RESTRICTIONS: Tuple[RestrictionMetadata, ...] = (
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
)


# ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False`` makes the local file
# sink raise at run time.
LOCAL_STORAGE_ACCESS_DISABLED_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="local_storage_access_disabled",
    severity=Severity.HARD,
    when=RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
        configuration_equals={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": False},
    ),
)


# The two caveats of the ENABLED branch: on a dedicated deployment the file is
# written but cannot be fetched back through the Roboflow API, and on hosted
# serverless the container disk disappears with the worker.
DEPLOYMENT_VOLUME_NOT_RETRIEVABLE_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="writes_to_deployment_volume_not_retrievable",
    severity=Severity.SOFT,
    when=RestrictionCondition(
        runtimes=[Runtime.DEDICATED_DEPLOYMENT],
        configuration_equals={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True},
    ),
)


EPHEMERAL_CONTAINER_DISK_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="ephemeral_container_disk_loses_writes",
    severity=Severity.SOFT,
    when=RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS],
        configuration_equals={"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True},
    ),
)


# Both flag branches together: whichever way the target deployment sets
# ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE``, exactly the entries whose
# ``configuration_equals`` matches it apply.
LOCAL_FILE_SINK_PORTABLE_RESTRICTIONS: Tuple[RestrictionMetadata, ...] = (
    DEPLOYMENT_VOLUME_NOT_RETRIEVABLE_PORTABLE_RESTRICTION,
    EPHEMERAL_CONTAINER_DISK_PORTABLE_RESTRICTION,
    LOCAL_STORAGE_ACCESS_DISABLED_PORTABLE_RESTRICTION,
)


ENVIRONMENT_VARIABLE_ACCESS_DISABLED_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="environment_variable_access_disabled",
    severity=Severity.HARD,
    when=RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
        configuration_equals={
            "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": False
        },
    ),
)


# Append-log mode accumulates entries in process memory before uploading the
# whole object, so it splits across stateless workers.
S3_APPEND_BUFFER_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="s3_append_buffer_resets_on_stateless_http",
    severity=Severity.SOFT,
    when=RestrictionCondition(
        runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
        step_execution_modes=[StepExecutionMode.REMOTE],
    ),
)


# Fire-and-forget database writes hide persistence failures and pile up when
# the database is slower than the video stream.
FIRE_AND_FORGET_PORTABLE_RESTRICTION = RestrictionMetadata(
    code="fire_and_forget_hides_persistence_failures",
    severity=Severity.SOFT,
    when=RestrictionCondition(runtimes=[Runtime.INFERENCE_PIPELINE]),
)
