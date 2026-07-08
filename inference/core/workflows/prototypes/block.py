from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.utils import (
    get_full_type_name,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl

BatchElementOutputs = Dict[str, Any]
BatchElementResult = Union[BatchElementOutputs, FlowControl]
BlockResult = Union[
    BatchElementResult, List[BatchElementResult], List[List[BatchElementResult]]
]


@dataclass(frozen=True)
class AirGappedAvailability:
    """Declares whether a block can operate without internet access.

    Blocks that require cloud APIs (e.g. OpenAI, Anthropic) return
    ``AirGappedAvailability(available=False, reason="requires_internet")``.
    Blocks that work fully offline return the default (available=True).
    """

    available: bool = True
    reason: Optional[str] = None


class Severity(str, Enum):
    """Severity of a runtime restriction for a workflow block in a given runtime.

    SOFT: the block runs to completion and returns the right output shape,
    but the values are degraded or meaningless (e.g. tracker IDs reset across
    requests, cooldown does not throttle, file is written to ephemeral disk).

    HARD: the block does not run / raises / cannot produce a usable output
    in this runtime. The engine should refuse to compile or fail-fast.
    """

    SOFT = "soft"
    HARD = "hard"


class Runtime(str, Enum):
    """Canonical runtimes a workflow block can be executed in.

    Runtimes not covered by ``get_restrictions()`` are considered OK.
    """

    HOSTED_SERVERLESS = "hosted_serverless"
    DEDICATED_DEPLOYMENT = "dedicated_deployment"
    SELF_HOSTED_CPU = "self_hosted_cpu"
    SELF_HOSTED_GPU = "self_hosted_gpu"
    INFERENCE_PIPELINE = "inference_pipeline"


class RuntimeInputMode(str, Enum):
    """Workflow input modes for a restriction."""

    IMAGE = "image"
    VIDEO = "video"


class StepExecutionMode(Enum):
    """How a workflow step is dispatched at runtime.

    LOCAL: the step executes in-process inside the current Python interpreter.
    REMOTE: the step delegates execution to a remote inference service / HTTP
    runtime.

    Kept in ``prototypes/block.py`` so the framework layer owns this enum and
    higher-level packages (``core_steps``, executor, compiler) depend on
    ``prototypes`` rather than the other way around.
    """

    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class RuntimeRestriction:
    """A single caveat for a workflow block.

    ``note`` is a one-line, human-readable explanation of the failure mode or
    degraded behavior. It should describe what happens (e.g. "track_ids reset
    between requests", "raises RuntimeError", "writes to ephemeral /tmp"),
    not abstract preconditions.

    ``applies_to_runtimes`` narrows the restriction to specific workflow
    runtimes. When unset, the restriction applies to all runtimes.

    ``applies_to_step_execution_modes`` narrows the restriction to specific
    workflow step execution modes. When unset, the restriction applies to all
    step execution modes.

    ``applies_to_input_modes`` narrows the restriction to specific workflow
    input modes, such as video workflows that depend on cross-frame state.
    When unset, the restriction applies to all input modes.
    """

    severity: Severity
    note: str
    applies_to_runtimes: Optional[List[Runtime]] = None
    applies_to_step_execution_modes: Optional[List[StepExecutionMode]] = None
    applies_to_input_modes: Optional[List[RuntimeInputMode]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"severity": self.severity.value, "note": self.note}
        if self.applies_to_runtimes is not None:
            result["applies_to_runtimes"] = [
                runtime.value for runtime in self.applies_to_runtimes
            ]
        if self.applies_to_step_execution_modes is not None:
            result["applies_to_step_execution_modes"] = [
                mode.value for mode in self.applies_to_step_execution_modes
            ]
        if self.applies_to_input_modes is not None:
            result["applies_to_input_modes"] = [
                mode.value for mode in self.applies_to_input_modes
            ]
        return result


# ----------------------------------------------------------------------------
# Common block-restriction presets.
#
# Many blocks share the same failure mode (e.g. all stateful video blocks
# degrade the same way on stateless HTTP runtimes). Reusing these presets
# keeps the per-block overrides tight and the wording consistent across the
# codebase.
# ----------------------------------------------------------------------------


STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION = RuntimeRestriction(
    severity=Severity.SOFT,
    note=(
        "Block keeps per-video state in process memory (keyed by "
        "video_metadata.video_identifier). With remote step execution on "
        "stateless or multi-replica HTTP runtimes, successive requests may "
        "be served by different worker processes, so the state resets "
        "between calls and the output is meaningless for tracking / "
        "counting / aggregation. Use local step execution in an "
        "InferencePipeline for stable cross-frame results."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
    applies_to_input_modes=[RuntimeInputMode.VIDEO],
)


COOLDOWN_HTTP_SOFT_RESTRICTION = RuntimeRestriction(
    severity=Severity.SOFT,
    note=(
        "Cooldown / rate-limit timer is stored in process memory. With "
        "remote step execution on stateless or multi-replica HTTP runtimes "
        "each request gets a fresh worker, so cooldown does not throttle. "
        "Cooldown only behaves as documented with local step execution inside "
        "an InferencePipeline."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
)


STILL_IMAGE_INPUT_SOFT_RESTRICTION = RuntimeRestriction(
    severity=Severity.SOFT,
    note=(
        "Block depends on temporal context from video or repeated-frame "
        "workflows. With a still image/photo, there is no meaningful history "
        "to track, compare, aggregate, or visualize, so the block provides "
        "little or no benefit."
    ),
    applies_to_input_modes=[RuntimeInputMode.IMAGE],
)


@dataclass(frozen=True)
class BlockAirGappedInfo:
    """Full air-gapped status for a block, as returned by the describe endpoint."""

    available: bool = True
    reason: Optional[str] = None
    model_id: Optional[str] = None
    compatible_task_types: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"available": self.available}
        if self.reason is not None:
            result["reason"] = self.reason
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.compatible_task_types is not None:
            result["compatible_task_types"] = self.compatible_task_types
        return result


class WorkflowBlockManifest(BaseModel, ABC):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    type: str
    name: str = Field(
        title="Step Name", description="Enter a unique identifier for this step."
    )

    @classmethod
    @abstractmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        raise BlockInterfaceError(
            public_message=f"Class method `describe_outputs()` must be implemented "
            f"for {get_full_type_name(selected_type=cls)} to be valid "
            f"`WorkflowBlockManifest`.",
            context="getting_block_outputs",
        )

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return self.describe_outputs()

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        """Declare whether this block can operate without internet access.

        Override in subclasses that require cloud APIs to return
        ``AirGappedAvailability(available=False, reason="requires_internet")``.

        The default indicates the block works offline.
        """
        return AirGappedAvailability(available=True)

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Caveats for this block.

        Return restrictions describing where the block degrades
        (``Severity.SOFT``) or fails outright (``Severity.HARD``). Each
        restriction can scope itself to runtimes, step execution modes, and/or
        input modes.
        """
        return []

    @classmethod
    def is_stateful_for_video_processing(cls) -> bool:
        """Whether the block must observe video frames strictly in stream order.

        Stateful blocks keep cross-frame state (trackers, counters,
        aggregators) or have order-sensitive side effects (notifications,
        sinks), so the stream scheduler must run them frame by frame, in
        order. Stateless blocks — pure functions of their per-frame inputs —
        may be executed ahead of stream order, which lets the stream
        scheduler overlap several frames' expensive steps (e.g. remote model
        requests).

        The default is conservative: blocks are assumed stateful unless they
        override this to return ``False``.
        """
        return True

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return model IDs whose cached weights enable this block to run offline.

        For foundation-model blocks, return the list of model variant IDs
        (e.g. ``["sam2/hiera_large", "sam2/hiera_small"]``).  The block is
        considered available if **any** variant has cached artifacts.

        Return ``None`` (the default) for blocks that do not depend on
        locally-cached model weights (pure logic blocks, cloud API blocks, etc.).
        """
        return None

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        """Return task types this block can process (e.g. ``["object-detection"]``).

        Used by the air-gapped builder to match user-trained models to
        compatible workflow blocks.  Return ``None`` (the default) for blocks
        that are not parameterised by a Roboflow model.
        """
        return None

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {}

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return None

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 0

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return (
            len(cls.get_parameters_accepting_batches()) > 0
            or len(cls.get_parameters_accepting_batches_and_scalars()) > 0
        )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return []

    @classmethod
    def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
        return []

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return False

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return None


class WorkflowBlock(ABC):

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        raise BlockInterfaceError(
            public_message="Class method `get_manifest()` must be implemented for any entity "
            "deriving from WorkflowBlockManifest.",
            context="getting_block_manifest",
        )

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs,
    ) -> BlockResult:
        pass
