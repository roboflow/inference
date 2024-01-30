from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.base import GraphNone


class CoordinatesSystem(Enum):
    OWN = "own"
    PARENT = "parent"


class JsonField(BaseModel, GraphNone):
    type: Literal["JsonField"]
    name: str
    selector: str
    coordinates_system: CoordinatesSystem = Field(default=CoordinatesSystem.PARENT)

    def get_type(self) -> str:
        return self.type
