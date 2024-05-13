from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    WILDCARD_KIND,
    Kind,
)


class StepExecutionMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class OutputDefinition(BaseModel):
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])


class CoordinatesSystem(Enum):
    OWN = "own"
    PARENT = "parent"


class JsonField(BaseModel):
    type: Literal["JsonField"]
    name: str
    selector: str
    coordinates_system: CoordinatesSystem = Field(default=CoordinatesSystem.PARENT)

    def get_type(self) -> str:
        return self.type


class WorkflowImage(BaseModel):
    type: Literal["WorkflowImage", "InferenceImage"]
    name: str
    kind: List[Kind] = Field(default=[BATCH_OF_IMAGES_KIND])


class WorkflowParameter(BaseModel):
    type: Literal["WorkflowParameter", "InferenceParameter"]
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )


InputType = Annotated[
    Union[WorkflowImage, WorkflowParameter], Field(discriminator="type")
]
