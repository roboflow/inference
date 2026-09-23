"""Shared restriction declarations for the non-model core blocks.

These constants are the ``get_actual_restrictions()`` presets of the core
blocks: ``RuntimeRestriction`` values carrying a human ``note``, a stable
``code`` and the condition under which the caveat applies. The workload
document derives its portable ``RestrictionMetadata`` from them through
``restriction_metadata_of()``.

Both ``get_restrictions()`` and ``get_actual_restrictions()`` use the same
entity type, ``RuntimeRestriction``. The state-loss presets (stateful video,
cooldown, S3 append buffer) are intentionally declared separately from the
legacy editor declarations, to preserve the editor contract: the legacy side
keeps its historic step-execution-mode scope, while the actual presets here
drop it.

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
stay next to that sibling, so the note and the code cannot drift apart. The PLC
LAN preset below reuses the ONVIF code but not its camera note, so it lives
here instead of importing the ONVIF block.
"""

from typing import Tuple

from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
)
from roboflow_workflows.prototypes.block import STILL_IMAGE_INPUT_SOFT_RESTRICTION

# Intentionally separate from get_restrictions() to preserve the editor
# contract. Actual restrictions describe state loss independently of model
# execution mode. Legacy twin: prototypes.block.STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.
STATEFUL_VIDEO_ACTUAL_RESTRICTION = RuntimeRestriction(
    code="stateful_video_state_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Block keeps per-video state (keyed by video_metadata.video_identifier) "
        "in its workflow block instance. An HTTP workflow request that builds a "
        "fresh workflow / block instance starts from empty state, even on the "
        "same CPU worker, so tracking / counting / aggregation output is "
        "meaningless across requests. Running models locally or remotely does "
        "not change this. Stable cross-frame results need a target that "
        "preserves this step's state for the same video stream; reusing some "
        "engine or process, CPU vs GPU, or the host answering introspection "
        "does not guarantee that by itself."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_input_modes=[RuntimeInputMode.VIDEO],
)


# Intentionally separate from get_restrictions() to preserve the editor
# contract. Actual restrictions describe state loss independently of model
# execution mode. Legacy twin: prototypes.block.COOLDOWN_HTTP_SOFT_RESTRICTION.
COOLDOWN_ACTUAL_RESTRICTION = RuntimeRestriction(
    code="cooldown_timer_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Cooldown / rate-limit timer is kept in the workflow block instance. An "
        "HTTP workflow request that builds a fresh workflow / block instance "
        "starts with a fresh timer, even on the same CPU worker, so cooldown "
        "does not throttle across requests. Running models locally or remotely "
        "does not change this. Cooldown behaves as documented only when the "
        "target preserves this step's state across calls of the same stream; "
        "reusing some engine or process, CPU vs GPU, or the host answering "
        "introspection does not guarantee that by itself."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
)


# The pair emitted by every block whose legacy declaration is
# ``[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION, STILL_IMAGE_INPUT_SOFT_RESTRICTION]``
# - cross-frame state in the block instance, plus "a still image has no history
# to work with". Blocks that spell the first restriction out with their own
# note (frame stack, heat accumulation, trace history) share the same axes and
# therefore the same code.
STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS: Tuple[RuntimeRestriction, ...] = (
    STATEFUL_VIDEO_ACTUAL_RESTRICTION,
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


# Direct PLC connections (Modbus TCP, EtherNet/IP) are opened from this process
# regardless of where model steps execute, so only network reachability of the
# PLC restricts the block. Same code and axes as the ONVIF camera restriction
# (core_steps/sinks/onvif_movement/v1.py), with a PLC-specific note. No legacy
# twin: these blocks never declared it in get_restrictions().
PLC_LAN_ACCESS_ACTUAL_RESTRICTION = RuntimeRestriction(
    code="requires_lan_access_to_device",
    severity=Severity.HARD,
    note=(
        "Block connects directly to a PLC, so the process running the "
        "workflow must reach the PLC's address over the network. Hosted "
        "Serverless and Roboflow Dedicated Deployments cannot reach customer "
        "LANs."
    ),
    applies_to_runtimes=[
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    ],
)


# Append-log mode accumulates entries in the block instance before uploading
# the whole object. Intentionally separate from get_restrictions() to preserve
# the editor contract. Actual restrictions describe state loss independently of
# model execution mode.
S3_APPEND_BUFFER_RESTRICTION = RuntimeRestriction(
    code="s3_append_buffer_resets_on_stateless_http",
    severity=Severity.SOFT,
    note=(
        "Append-log mode buffers entries in the workflow block instance before "
        "uploading the accumulated object to S3. An HTTP workflow request that "
        "builds a fresh workflow / block instance starts with an empty buffer, "
        "even on the same CPU worker, so append-log objects reset or split "
        "across requests. Running models locally or remotely does not change "
        "this. Use separate_files mode, or a target that preserves this step's "
        "state for the same stream, when each entry must be captured in a "
        "single ordered log; reusing some engine or process does not guarantee "
        "that by itself."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
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
