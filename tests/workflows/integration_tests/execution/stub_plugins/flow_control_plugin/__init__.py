"""
This is just example, test implementation, please do not assume it being fully functional.
"""

import random
from typing import Any, Dict, List, Literal, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ABTestManifest(WorkflowBlockManifest):
    type: Literal["ABTest"]
    name: str = Field(description="name field")
    a_step: StepSelector
    b_step: StepSelector

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ABTestBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ABTestManifest

    def run(
        self,
        a_step: StepSelector,
        b_step: StepSelector,
    ) -> BlockResult:
        choice = a_step
        if random.random() > 0.5:
            choice = b_step
        return FlowControl(mode="select_step", context=choice)


LONG_DESCRIPTION = """
Based on provided configuration, block decides which execution path to take given
data fed into condition logic.
"""

SHORT_DESCRIPTION = "Creates alternative execution branches for data"


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
    steps_if_true: List[StepSelector] = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=[["$steps.on_true"]],
    )
    steps_if_false: List[StepSelector] = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=[["$steps.on_false"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ConditionBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        steps_if_true: List[StepSelector],
        steps_if_false: List[StepSelector],
    ) -> BlockResult:
        if not steps_if_true and not steps_if_false:
            return FlowControl(mode="terminate_branch")
        evaluation_function = build_eval_function(definition=condition_statement)
        evaluation_result = evaluation_function(evaluation_parameters)
        next_steps = steps_if_true if evaluation_result else steps_if_false
        if not next_steps:
            return FlowControl(mode="terminate_branch")
        flow_control = FlowControl(mode="select_step", context=next_steps)
        return flow_control


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [ABTestBlock, ConditionBlock]
