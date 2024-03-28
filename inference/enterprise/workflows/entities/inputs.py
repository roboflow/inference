from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.base import GraphNone
from inference.enterprise.workflows.entities.types import (
    IMAGE_KIND,
    WILDCARD_KIND,
    Kind,
)


class InferenceImage(BaseModel, GraphNone):
    type: Literal["InferenceImage"]
    name: str
    kind: List[Kind] = Field(default=[IMAGE_KIND])

    def get_type(self) -> str:
        return self.type


class InferenceParameter(BaseModel, GraphNone):
    type: Literal["InferenceParameter"]
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )

    def get_type(self) -> str:
        return self.type
