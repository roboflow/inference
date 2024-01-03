from enum import Enum
from typing import Literal, Dict, Union

from pydantic import BaseModel


class CVModel(BaseModel):
    type: Literal["CVModel"]
    name: str
    inputs: Dict[str, str]


class Crop(BaseModel):
    type: Literal["Crop"]
    name: str
    inputs: Dict[str, str]


class Operator(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LOWER_THAN = "lower_than"
    GREATER_THAN = "greater_than"
    LOWER_OR_EQUAL_THAN = "lower_or_equal_than"
    GREATER_OR_EQUAL_THAN = "greater_or_equal_than"


class ConditionSpecs(BaseModel):
    left: Union["ConditionSpecs", str, int, float, bool]
    operator: Operator
    right: Union["ConditionSpecs", str, int, float, bool]


class Condition(BaseModel):
    type: Literal["Condition"]
    name: str
    inputs: Dict[str, str]
    condition: ConditionSpecs
    step_if_true: str
    step_if_false: str
