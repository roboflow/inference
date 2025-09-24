from abc import ABC, abstractmethod
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
        return len(cls.get_parameters_accepting_batches()) > 0 or len(
            cls.get_parameters_accepting_batches_and_scalars()
        )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
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
