from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

from openai import BaseModel
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import FlowControl
from inference.core.workflows.errors import BlockInterfaceError
from inference.core.workflows.execution_engine.introspection.utils import (
    get_full_type_name,
)


class WorkflowBlockManifest(BaseModel, ABC):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    type: str
    name: str = Field(description="Unique name of step in workflows")

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

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def produces_batch_output(cls) -> bool:
        return True

    @abstractmethod
    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass

    async def run_remotely(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        logger.info(
            "Block has no implementation for run_remotely() method - using run_locally() instead"
        )
        return await self.run_locally(*args, **kwargs)
