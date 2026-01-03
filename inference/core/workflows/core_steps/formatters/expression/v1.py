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
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Create conditional logic and business rules in workflows using switch-case statements that evaluate conditions on input variables, optionally transform data with operations, and return different outputs based on which condition matches, enabling conditional execution, business logic implementation, rule-based decision making, and dynamic output generation workflows.

## How This Block Works

This block implements conditional logic similar to switch-case or if-else-if statements in programming. The block:

1. Receives input data as a dictionary of named variables from workflow steps
2. Optionally applies data transformations using operations:
   - Performs operations on data variables before condition evaluation
   - Uses the same operation system as Property Definition block
   - Transforms data (e.g., extract properties, filter, select) to prepare variables for conditions
   - Stores transformed values as variables for use in conditions
3. Evaluates switch-case statements sequentially:
   - Tests each case condition in order until one matches
   - Stops at the first matching case and returns its result
   - If no case matches, returns the default result
4. Evaluates conditions using a flexible expression system:
   - **Binary Statements**: Compare two values using operators (==, !=, >, <, >=, <=, contains, startsWith, endsWith, in, any in, all in)
   - **Unary Statements**: Test single values (Exists, DoesNotExist, is True, is False, is empty, is not empty)
   - **Statement Groups**: Combine multiple statements with AND/OR operators for complex conditions
   - Conditions can reference variables by name (DynamicOperand) or use literal values (StaticOperand)
5. Returns results based on matched case:

   **Static Results:**
   - Returns a fixed value defined in the case (e.g., "PASS", "FAIL", numeric values, strings)

   **Dynamic Results:**
   - Returns a value from a variable (can reference any input variable)
   - Optionally applies operations to transform the variable before returning
   - Enables returning computed or extracted values as output

6. Handles default case:
   - If no case condition matches, returns the default result
   - Default can be static or dynamic, just like case results

The block enables complex conditional logic by combining data transformation operations with flexible condition evaluation. Conditions can compare variables, test existence, check membership, perform string operations, and combine multiple conditions with logical operators. This makes it powerful for implementing business rules, validation logic, classification based on multiple criteria, and conditional data transformation.

## Common Use Cases

- **Business Logic Implementation**: Implement conditional business rules and validation logic (e.g., validate detection matches reference, implement quality checks, enforce business rules), enabling business logic workflows
- **Conditional Classification**: Classify data based on multiple conditions and criteria (e.g., classify detections based on properties, categorize results by conditions, implement multi-criteria classification), enabling conditional classification workflows
- **Validation and Quality Control**: Validate data or predictions against reference values or thresholds (e.g., validate predictions match expected classes, check quality thresholds, verify compliance), enabling validation workflows
- **Rule-Based Decision Making**: Make decisions based on complex rule sets (e.g., approve/reject based on multiple criteria, route data based on conditions, make decisions using rule sets), enabling rule-based decision workflows
- **Dynamic Output Generation**: Generate different outputs based on input conditions (e.g., return different values based on conditions, generate conditional outputs, create dynamic results), enabling dynamic output workflows
- **Multi-Condition Filtering**: Implement complex filtering logic with multiple conditions (e.g., filter based on multiple criteria, apply complex conditional filters, implement multi-factor filtering), enabling conditional filtering workflows

## Connecting to Other Blocks

This block receives data from workflow steps and produces conditional output:

- **After model or analytics blocks** to implement conditional logic on predictions or results (e.g., validate predictions, classify results, apply conditional rules), enabling conditional logic workflows
- **After Property Definition blocks** to use extracted properties in conditions (e.g., use extracted values in conditions, compare extracted properties, implement logic on extracted data), enabling property-to-condition workflows
- **Before logic blocks** like Continue If to provide conditional inputs (e.g., provide conditional values for filtering, supply conditional inputs for decisions), enabling expression-to-logic workflows
- **Before data storage blocks** to conditionally format or transform data for storage (e.g., conditionally format for storage, apply conditional transformations, prepare conditional outputs), enabling conditional storage workflows
- **Before notification blocks** to send conditional notifications (e.g., send conditional alerts, notify based on conditions, trigger conditional notifications), enabling conditional notification workflows
- **In workflow outputs** to provide conditional final outputs (e.g., conditional workflow outputs, dynamic result generation, conditional output formatting), enabling conditional output workflows

## Requirements

This block requires input data as a dictionary where keys are variable names and values are data from workflow steps. The switch parameter defines cases with conditions and results. Conditions support binary comparisons (==, !=, >, <, >=, <=, contains, in, etc.), unary tests (Exists, is empty, etc.), and logical combinations (AND/OR). Data operations are optional and use the same operation system as Property Definition block. The block evaluates cases in order and returns the result of the first matching case, or the default result if no cases match. Results can be static values or dynamic values from variables (optionally with operations applied).
"""

SHORT_DESCRIPTION = (
    "Create a specific output based on defined input variables and configured rules."
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
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-code",
                "blockPriority": 1,
                "inDevelopment": True,
            },
        }
    )
    type: Literal["roboflow_core/expression@v1", "Expression"]
    data: Dict[
        str,
        Union[Selector()],
    ] = Field(
        description="Dictionary of named variables containing data from workflow steps. Variable names are used in conditions and results. Keys are variable names, values are selectors referencing workflow step outputs. Variables can be referenced in conditions and dynamic results. Example: {'predictions': '$steps.model.predictions', 'reference': '$inputs.reference_class_names'} creates variables 'predictions' and 'reference'.",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Optional dictionary of operations to transform data variables before condition evaluation. Keys are variable names from data, values are lists of operations (same as Property Definition block). Operations are applied to transform variables before they are used in conditions. Useful for extracting properties, filtering, or transforming data before evaluation. Empty dictionary (default) means no transformations are applied.",
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
        title="Logic Definition",
        description="Switch-case logic definition containing cases with conditions and results. Each case has a condition (StatementGroup with binary/unary statements) and a result (static value or dynamic variable). Cases are evaluated in order - first matching case's result is returned. Default result is returned if no cases match. Supports complex conditions with AND/OR operators, comparison operators (==, !=, >, <, >=, <=), string operations (contains, startsWith, endsWith), membership tests (in, any in, all in), and existence tests (Exists, is empty).",
        examples=[SWITCH_STATEMENT_EXAMPLE],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


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
