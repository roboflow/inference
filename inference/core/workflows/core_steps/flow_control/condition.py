from typing import Any, Dict, List, Literal, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.operators import (
    OPERATORS_FUNCTIONS,
    Operator,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    FlowControl,
    StepOutputSelector,
    StepSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Create a branch of logic that runs only when a specified condition is met.

This block is the "if statement" in Workflows.

This block is responsible for flow-control in execution graph based on the condition 
defined in its body.

Right now, this block is only capable to make conditions based on output of binary 
operators that takes two operands. 

*The `Condition` block only works when a  single image is provided to the input of the 
`workflow` (or more precisely, both `left` and `right` if provided with reference, 
then the reference can only hold value for a result of operation made against single 
input). This is to prevent a situation when evaluation of condition for multiple 
images yield different execution paths.*
"""

SHORT_DESCRIPTION = "Control the flow of a workflow based on the result of a step."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    type: Literal["Condition"]
    left: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        WorkflowParameterSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
                WILDCARD_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Left operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "foo"],
    )
    operator: Operator = Field(
        description="Operator in expression `left operator right` to evaluate boolean value of condition statement",
        examples=["equal", "in"],
    )
    right: Union[
        float,
        int,
        bool,
        StepOutputSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
            ]
        ),
        WorkflowParameterSelector(
            kind=[
                FLOAT_KIND,
                INTEGER_KIND,
                BOOLEAN_KIND,
                STRING_KIND,
                LIST_OF_VALUES_KIND,
                WILDCARD_KIND,
            ]
        ),
        str,
        list,
        set,
    ] = Field(
        description="Right operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "bar"],
    )
    step_if_true: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=["$steps.on_true"],
    )
    step_if_false: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=["$steps.on_false"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ConditionBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        left: Union[float, int, bool, str, list, set],
        operator: Operator,
        right: Union[float, int, bool, str, list, set],
        step_if_true: StepSelector,
        step_if_false: StepSelector,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        evaluation_result = OPERATORS_FUNCTIONS[operator](left, right)
        next_step = step_if_true if evaluation_result else step_if_false
        flow_control = FlowControl(mode="select_step", context=next_step)
        return [], flow_control

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False
