from typing import Any, Dict, List, Literal, Type, Union

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
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Based on provided configuration, block decides if it should follow to pointed
execution path
"""

SHORT_DESCRIPTION = "Stops execution of processing branch under certain condition"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    type: Literal["ContinueIf"]
    condition_statement: StatementGroup = Field(
        description="Workflows UQL definition of conditional logic.",
    )
    evaluation_parameters: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=lambda: {},
    )
    next_steps: List[StepSelector] = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=[["$steps.on_true"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ContinueIfBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        next_steps: List[StepSelector],
    ) -> BlockResult:
        if not next_steps:
            return FlowControl(mode="terminate_branch")
        evaluation_function = build_eval_function(definition=condition_statement)
        evaluation_result = evaluation_function(evaluation_parameters)
        if evaluation_result:
            return FlowControl(mode="select_step", context=next_steps)
        return FlowControl(mode="terminate_branch")
