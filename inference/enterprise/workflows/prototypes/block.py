from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

from openai import BaseModel
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import FlowControl


class WorkflowBlockManifest(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    type: str
    name: str = Field(description="Unique name of step in workflows")


class WorkflowBlock(ABC):

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    @abstractmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        pass

    @classmethod
    @abstractmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        pass

    @classmethod
    def get_actual_outputs(
        cls, manifest: WorkflowBlockManifest
    ) -> List[OutputDefinition]:
        return cls.describe_outputs()

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
