from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
