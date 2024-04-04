from typing import Any, Dict, List, Literal, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FlowControl,
    StepOutputSelector,
    StepSelector,
)
from inference.enterprise.workflows.steps.common.operators import (
    OPERATORS_FUNCTIONS,
    Operator,
)


class BlockManifest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block is responsible for flow-control in execution graph based on the condition defined in its body. As for now, only capable to make conditions based on output of binary operators that takes two operands. IMPORTANT NOTE: `Condition` block is only capable to operate, when single image is provided to the input of the `workflow` (or more precisely, both `left` and `right` if provided with reference, then the reference can only hold value for a result of operation made against single input). This is to prevent situation when evaluation of condition for multiple images yield different execution paths.",
            "docs": None,
            "block_type": "flow_control",
        }
    )
    type: Literal["Condition"]
    name: str = Field(description="Unique name of step in workflows")
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
        str,
        list,
        set,
    ] = Field(
        description="Left operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "some"],
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
        str,
        list,
        set,
    ] = Field(
        description="Right operand of expression `left operator right` to evaluate boolean value of condition statement",
        examples=["$steps.classification.top", 3, "some"],
    )
    step_if_true: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=["$steps.on_true"],
    )
    step_if_false: StepSelector = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=["$steps.on_false"],
    )


class ConditionBlock:

    @classmethod
    def get_input_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

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
