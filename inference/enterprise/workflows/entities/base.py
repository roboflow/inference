from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from inference.enterprise.workflows.entities.types import (
    IMAGE_KIND,
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


class InferenceImage(BaseModel):
    type: Literal["InferenceImage"]
    name: str
    kind: List[Kind] = Field(default=[IMAGE_KIND])


class InferenceParameter(BaseModel):
    type: Literal["InferenceParameter"]
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )


InputType = Annotated[
    Union[InferenceImage, InferenceParameter], Field(discriminator="type")
]
