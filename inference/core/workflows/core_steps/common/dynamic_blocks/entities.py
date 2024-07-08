from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SelectorType(Enum):
    INPUT_IMAGE = "input_image"
    INPUT_PARAMETER = "input_parameter"
    STEP_OUTPUT = "step_output"


class ValueType(Enum):
    ANY = "any"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DICT = "dict"
    LIST = "list"
    STRING = "string"


class DynamicInputDefinition(BaseModel):
    default_value: Any
    is_optional: bool = Field(default=False)
    is_dimensionality_reference: bool = Field(default=False)
    dimensionality_offset: int = Field(default=0, ge=-1, le=1)
    selector_types: List[SelectorType] = Field(default_factory=list)
    selector_data_kind: Dict[SelectorType, List[str]] = Field(default_factory=dict)
    value_types: List[ValueType] = Field(default_factory=lambda: [ValueType.ANY])


class DynamicOutputDefinition(BaseModel):
    kind: List[str] = Field(default_factory=list)


class ManifestDescription(BaseModel):
    inputs: Dict[str, DynamicInputDefinition]
    outputs: Dict[str, DynamicOutputDefinition] = Field(default_factory=dict)
    output_dimensionality_offset: int = Field(default=0, ge=-1, le=1)
    accepts_batch_input: bool = Field(default=False)
    accepts_empty_values: bool = Field(default=False)
