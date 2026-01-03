import time
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
    FLOAT_KIND,
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
Conditionally control workflow execution by evaluating custom logic statements and either continuing to specified next steps or terminating the current branch based on the condition result, enabling dynamic branching, conditional processing, and workflow control flow.

## How This Block Works

This block evaluates a conditional statement and controls whether the workflow branch continues execution or stops. The block:

1. Takes a conditional statement (using a query language syntax) and evaluation parameters as input
2. Builds an evaluation function from the conditional statement definition
3. Evaluates the condition using the provided evaluation parameters (which can reference workflow inputs, step outputs, or other dynamic values)
4. If the condition evaluates to `true`:
   - Continues execution to the specified `next_steps` blocks
   - If a `stop_delay` is configured, records the current time to enable delayed termination
5. If the condition evaluates to `false`:
   - Terminates the current workflow branch (stops execution of downstream blocks in this branch)
   - If `stop_delay` was previously triggered and the delay period hasn't elapsed, continues execution to `next_steps` for the remaining delay duration
6. Returns flow control directives that either continue execution to the next steps or terminate the branch

The block uses a query language system that supports binary comparisons (equality, inequality, greater than, less than, etc.) between dynamic values (from workflow data) and static values. Conditions can check numeric values, string values, or other data types. The `stop_delay` feature allows the branch to remain active for a short period after a condition becomes false, which is useful for handling transient states or maintaining execution during brief condition fluctuations (e.g., keeping a workflow active for a few seconds after a detection count drops below threshold).

## Common Use Cases

- **Conditional Processing Based on Detection Counts**: Continue processing only when the number of detected objects exceeds a threshold (e.g., process alerts only when 3+ objects are detected, skip processing when count is below threshold)
- **Dynamic Quality Control**: Evaluate image quality metrics, detection confidence scores, or model outputs and continue workflow execution only when quality criteria are met, terminating branches that don't meet standards
- **Conditional Notifications**: Send notifications or trigger actions only when specific conditions are met (e.g., continue to notification blocks when confidence scores are above 0.9, or when specific object classes are detected)
- **Branch Filtering and Routing**: Route workflow execution to different branches based on dynamic conditions, allowing one path to continue while others terminate (e.g., continue video recording branch when motion is detected, terminate when no activity)
- **Threshold-Based Actions**: Execute downstream blocks only when values meet thresholds (e.g., continue to data storage when detection count > 5, terminate otherwise; continue processing when temperature > threshold, skip when below)
- **Transient State Handling**: Use `stop_delay` to handle brief condition changes by keeping branches active for a short period after conditions become false, preventing rapid on/off toggling in response to temporary fluctuations

## Connecting to Other Blocks

This block controls workflow execution flow and can be connected:

- **After detection or analysis blocks** (e.g., Object Detection, Classification, Keypoint Detection) to evaluate detection counts, confidence scores, class names, or other prediction results and conditionally continue processing based on the analysis results
- **After data processing blocks** (e.g., Property Definition, Expression, Delta Filter) to evaluate computed values, metrics, or processed data and control whether subsequent blocks execute based on the processed results
- **Before notification blocks** (e.g., Email Notification, Slack Notification, Twilio SMS Notification) to conditionally trigger notifications only when specific conditions are met, preventing unnecessary alerts
- **Before data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to conditionally save or send data only when certain criteria are satisfied, filtering what gets stored or transmitted
- **Between workflow stages** to create conditional processing paths, where different branches execute based on dynamic conditions, enabling complex workflow logic and decision trees
- **In parallel branches** to create multiple conditional paths, allowing different parts of a workflow to continue or terminate independently based on their respective conditions
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
        description="Define the conditional logic using the query language syntax. Specifies the condition to evaluate (e.g., comparisons, equality checks, numeric comparisons). The condition is built using StatementGroup syntax with binary statements that compare dynamic operands (referenced in evaluation_parameters) against static values using comparators like (Number) ==, (Number) >, (Number) <, (String) ==, etc. Example: Compare a dynamic value 'left' against static value 1 using (Number) ==.",
        examples=[CONDITION_STATEMENT_EXAMPLE],
    )
    evaluation_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping operand names (used in condition_statement) to actual values from the workflow. These parameters provide the dynamic data that gets evaluated in the conditional statement. Keys match operand names in the condition (e.g., 'left', 'right'), and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'left': '$steps.detection.count', 'threshold': 5} where 'left' is referenced in the condition_statement.",
        examples=[{"left": "$inputs.some"}],
        default_factory=lambda: {},
    )
    next_steps: List[StepSelector] = Field(
        description="List of workflow steps to execute if the condition evaluates to true. These steps receive control flow when the condition is satisfied, allowing the workflow branch to continue execution. If empty, the branch terminates even when the condition is true. Each step selector references a block in the workflow that should execute when the condition passes.",
        examples=[["$steps.on_true"]],
    )

    stop_delay: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Stop Delay",
        description="Number of seconds to continue execution after the condition becomes false, before terminating the branch. If the condition was previously true and then becomes false, execution continues to next_steps for this delay duration before terminating. This is useful for handling transient state changes or preventing rapid on/off toggling. Must be greater than 0 to take effect. Set to 0 (default) to terminate immediately when condition becomes false.",
        gt=0,
        examples=[5],
        default=0,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ContinueIfBlockV1(WorkflowBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        next_steps: List[StepSelector],
        stop_delay: float,
    ) -> BlockResult:
        if not next_steps:
            return FlowControl(mode="terminate_branch")
        evaluation_function = build_eval_function(definition=condition_statement)
        evaluation_result = evaluation_function(evaluation_parameters)

        if evaluation_result:
            if stop_delay > 0:
                self.start_time = time.time()
            return FlowControl(mode="select_step", context=next_steps)

        if self.start_time and time.time() - self.start_time <= stop_delay:
            return FlowControl(mode="select_step", context=next_steps)
        return FlowControl(mode="terminate_branch")
