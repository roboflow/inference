from typing import List, Optional

from pydantic import BaseModel, Field

from inference.core.workflows.execution_engine.entities.types import Kind


class OperationDescription(BaseModel):
    operation_type: str
    compound: bool
    input_kind: List[Kind]
    output_kind: List[Kind]
    nested_operation_input_kind: Optional[List[Kind]] = None
    nested_operation_output_kind: Optional[List[Kind]] = None
    description: Optional[str] = None

    property_name_options: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of possible property names. \
            Optional parameter for operations extracting property values from data. "
        ),
        examples=[
            "size",
            "height",
            "width",
            "aspect_ratio",
        ],
    )


class OperatorDescription(BaseModel):
    operator_type: str
    operands_number: int
    operands_kinds: List[List[Kind]]
    description: Optional[str] = None
