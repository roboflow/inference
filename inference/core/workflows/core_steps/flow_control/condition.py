from typing import Any, Dict, List, Literal, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    FlowControl,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """"""

SHORT_DESCRIPTION = ""


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
    condition_statement: StatementGroup
    evaluation_parameters: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=lambda: {},
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
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        step_if_true: StepSelector,
        step_if_false: StepSelector,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        evaluation_function = build_eval_function(definition=condition_statement)
        evaluation_result = evaluation_function(evaluation_parameters)
        next_step = step_if_true if evaluation_result else step_if_false
        flow_control = FlowControl(mode="select_step", context=next_step)
        return [], flow_control

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False
