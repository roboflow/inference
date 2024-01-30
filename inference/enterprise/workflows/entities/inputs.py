from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from inference.enterprise.workflows.entities.base import GraphNone


class InferenceImage(BaseModel, GraphNone):
    type: Literal["InferenceImage"]
    name: str

    def get_type(self) -> str:
        return self.type


class InferenceParameter(BaseModel, GraphNone):
    type: Literal["InferenceParameter"]
    name: str
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )

    def get_type(self) -> str:
        return self.type
