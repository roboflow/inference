from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator

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
Route workflow execution to one of several branches by matching an input value against a set of
case values, similar to a switch-case statement in programming, enabling multi-way branching,
value-based routing, and decision trees without chaining multiple Continue If blocks.

## How This Block Works

This block compares a single input value against the keys of a case mapping and directs execution
to the step associated with the first matching case. The block:

1. Takes a `value` input (typically a selector referencing a workflow input or a step output, e.g.
   a classification result) and converts it to a string
2. Looks the string up in the `cases` mapping, where each key is a case value and each value is the
   step to execute when that case matches (e.g. `{"red": "$steps.on_red", "blue": "$steps.on_blue"}`)
3. If `case_insensitive` is enabled, the comparison ignores letter case
4. If a case matches, execution continues to that case's step and all other branches terminate
5. If no case matches, execution continues to the steps listed in `default_next_steps`
6. If no case matches and `default_next_steps` is empty, the branch terminates

Because the input value is converted to a string before matching, non-string values match their
string representation: `True` matches the key `"True"`, `1.0` matches `"1.0"` (not `"1"`), and a
missing/None value matches `"None"`. Each target step may appear at most once across `cases` and
`default_next_steps` — to route several case values to the same logic, point each case at its own
step or normalize the value upstream (e.g. with an Expression block).

## Common Use Cases

- **Routing by classification result**: Send images down different processing paths based on the
  top class predicted by a classification model (e.g. "damaged" → alert branch, "ok" → logging
  branch, anything else → default review branch)
- **Mode-based pipelines**: Use a workflow input parameter (e.g. `$inputs.mode`) to select between
  alternative processing branches at runtime without editing the workflow
- **Multi-way alerting**: Route to different notification blocks (email, Slack, webhook) depending
  on a severity or category value computed earlier in the workflow
- **Replacing chained conditions**: Collapse a ladder of Continue If blocks comparing the same
  value against different constants into a single, easier-to-read block

## Connecting to Other Blocks

This block controls workflow execution flow and can be connected:

- **After classification or detection blocks** to branch on predicted classes, counts, or other
  prediction properties (often via a Property Definition or Expression block that extracts the
  value to switch on)
- **After workflow inputs** to select a branch from a runtime parameter
- **Before any downstream blocks** (models, notifications, sinks) that should only run for a
  specific case — each case target becomes the head of its own execution branch
- **With a default branch** wired via `default_next_steps` to handle unmatched values, or left
  empty to simply stop when nothing matches
"""

SHORT_DESCRIPTION = "Route execution to one of several branches based on the value of an input."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Switch Case",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-split",
                "blockPriority": 4,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/switch_case@v1", "SwitchCase"]
    value: Union[Selector(), str, int, float, bool] = Field(
        title="Value",
        description="Value to match against the case keys. Typically a selector referencing a "
        "workflow input or a step output (e.g. $inputs.mode or $steps.classifier.top). The value "
        "is converted to a string before comparison, so booleans match keys 'True'/'False', "
        "1.0 matches '1.0' and a None value matches 'None'.",
        examples=["$steps.classifier.top", "$inputs.mode", "red"],
    )
    cases: Dict[str, StepSelector] = Field(
        title="Cases",
        description="Mapping of case value to the step that should execute when `value` matches "
        "it, e.g. {\"red\": \"$steps.on_red\", \"blue\": \"$steps.on_blue\"}. Each target step "
        "may appear at most once across `cases` and `default_next_steps`.",
        examples=[{"red": "$steps.on_red", "blue": "$steps.on_blue"}],
        default_factory=dict,
    )
    case_insensitive: bool = Field(
        title="Case Insensitive",
        description="When enabled, case values are matched ignoring letter case "
        "(e.g. value 'RED' matches case key 'red').",
        default=False,
        examples=[False],
    )
    default_next_steps: List[StepSelector] = Field(
        title="Default Steps",
        description="Steps to execute when no case matches. Leave empty to terminate the branch "
        "when nothing matches.",
        examples=[["$steps.fallback"]],
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_targets_and_keys(self) -> "BlockManifest":
        targets = list(self.cases.values()) + list(self.default_next_steps)
        seen, duplicated = set(), set()
        for target in targets:
            if target in seen:
                duplicated.add(target)
            seen.add(target)
        if duplicated:
            raise ValueError(
                f"Each step may be targeted at most once across `cases` and "
                f"`default_next_steps`. Duplicated targets: {sorted(duplicated)}"
            )
        if self.case_insensitive:
            lowered_keys = [key.lower() for key in self.cases]
            if len(set(lowered_keys)) != len(lowered_keys):
                raise ValueError(
                    "`cases` contains keys that collide when case_insensitive is enabled"
                )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class SwitchCaseBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        value: Any,
        cases: Dict[str, StepSelector],
        case_insensitive: bool,
        default_next_steps: List[StepSelector],
    ) -> BlockResult:
        coerced_value = str(value)
        if case_insensitive:
            coerced_value = coerced_value.lower()
            matched_step = next(
                (
                    target
                    for case_key, target in cases.items()
                    if case_key.lower() == coerced_value
                ),
                None,
            )
        else:
            matched_step = cases.get(coerced_value)
        if matched_step is not None:
            return FlowControl(mode="select_step", context=[matched_step])
        if default_next_steps:
            return FlowControl(mode="select_step", context=default_next_steps)
        return FlowControl(mode="terminate_branch")
