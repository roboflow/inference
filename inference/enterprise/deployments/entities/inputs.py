from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class InferenceImage(BaseModel):
    type: Literal["InferenceImage"]
    name: str


class InferenceParameter(BaseModel):
    type: Literal["InferenceParameter"]
    name: str
    default_value: Optional[Union[float, int, str, bool]] = Field(default=None)
