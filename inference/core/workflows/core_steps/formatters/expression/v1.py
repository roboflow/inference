from copy import copy
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Creates specific output based on defined input variables and configured rules - which is
useful while creating business logic in workflows.

Based on configuration, block takes input data, optionally performs operation on data, 
save it as variables and evaluate switch-case like statements to get the final result.
"""

SHORT_DESCRIPTION = (
    "Creates specific output based on defined input variables and configured rules."
)


class StaticCaseResult(BaseModel):
    type: Literal["StaticCaseResult"]
    value: Any


class DynamicCaseResult(BaseModel):
    type: Literal["DynamicCaseResult"]
    parameter_name: str
    operations: Optional[List[AllOperationsType]] = Field(default=None)


class CaseDefinition(BaseModel):
    type: Literal["CaseDefinition"]
    condition: StatementGroup
    result: Annotated[
        Union[StaticCaseResult, DynamicCaseResult], Field(discriminator="type")
    ]


class CasesDefinition(BaseModel):
    type: Literal["CasesDefinition"]
    cases: List[CaseDefinition]
    default: Annotated[
        Union[StaticCaseResult, DynamicCaseResult], Field(discriminator="type")
    ]


SWITCH_STATEMENT_EXAMPLE = {
    "type": "CasesDefinition",
    "cases": [
        {
            "type": "CaseDefinition",
            "condition": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "class_name",
                        },
                        "comparator": {"type": "=="},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "reference",
                        },
                    }
                ],
            },
            "result": {"type": "StaticCaseResult", "value": "PASS"},
        }
    ],
    "default": {"type": "StaticCaseResult", "value": "FAIL"},
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Expression",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal["roboflow_core/expression@v1", "Expression"]
    data: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References data to be used to construct results",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data "
        "before switch-case instruction",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=lambda: {},
    )
    switch: CasesDefinition = Field(
        description="Definition of switch-case statement",
        examples=[SWITCH_STATEMENT_EXAMPLE],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ExpressionBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Dict[str, Any],
        data_operations: Dict[str, List[AllOperationsType]],
        switch: CasesDefinition,
    ) -> BlockResult:
        variables = copy(data)
        for variable_name, operations in data_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            variables[variable_name] = operations_chain(
                variables[variable_name], global_parameters={}
            )
        for case in switch.cases:
            case_eval_operation = build_eval_function(definition=case.condition)
            if case_eval_operation(variables):
                return build_result(variables=variables, result_definition=case.result)
        return build_result(variables=variables, result_definition=switch.default)


def build_result(
    variables: Dict[str, Any],
    result_definition: Union[StaticCaseResult, DynamicCaseResult],
) -> Dict[str, Any]:
    if result_definition.type == "StaticCaseResult":
        return {"output": result_definition.value}
    selected_variable = variables[result_definition.parameter_name]
    if not result_definition.operations:
        return {"output": selected_variable}
    operations_chain = build_operations_chain(operations=result_definition.operations)
    result = operations_chain(selected_variable, global_parameters=variables)
    return {"output": result}
