from typing import List, Optional

from pydantic import BaseModel

from inference.core.workflows.execution_engine.entities.types import Kind


class OperationDescription(BaseModel):
    operation_type: str
    compound: bool
    input_kind: List[Kind]
    output_kind: List[Kind]
    nested_operation_input_kind: Optional[List[Kind]] = None
    nested_operation_output_kind: Optional[List[Kind]] = None
    description: Optional[str] = None


class OperatorDescription(BaseModel):
    operator_type: str
    operands_number: int
    operands_kinds: List[List[Kind]]
    description: Optional[str] = None
