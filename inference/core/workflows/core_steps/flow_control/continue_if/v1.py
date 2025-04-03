from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    Selector,
    StepSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Based on provided configuration, block decides if it should follow to pointed
execution path
"""

SHORT_DESCRIPTION = "Conditionally stop execution of a branch."

CONDITION_STATEMENT_EXAMPLE = {
    "type": "StatementGroup",
    "statements": [
        {
            "type": "BinaryStatement",
            "left_operand": {
                "type": "DynamicOperand",
                "operand_name": "left",
            },
            "comparator": {"type": "(Number) =="},
            "right_operand": {"type": "StaticOperand", "value": 1},
        }
    ],
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Continue If",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "fak fa-branching",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/continue_if@v1", "ContinueIf"]
    condition_statement: StatementGroup = Field(
        title="Conditional Statement",
        description="Define the conditional logic.",
        examples=[CONDITION_STATEMENT_EXAMPLE],
    )
    evaluation_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Data to be used in the conditional logic.",
        examples=[{"left": "$inputs.some"}],
        default_factory=lambda: {},
    )
    next_steps: List[StepSelector] = Field(
        description="Steps to execute if the condition evaluates to true.",
        examples=[["$steps.on_true"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ContinueIfBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
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
