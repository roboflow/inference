from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import ConfigDict, Field, model_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Run a nested workflow definition with parameters mapped from the parent workflow."
)

LONG_DESCRIPTION = """
Execute a **nested workflow** while mapping parent data into the child's inputs via `parameter_bindings`.

Provide either a full inline definition in `workflow_definition`, or resolve a saved workflow using
`workflow_workspace_id` and `workflow_id` (optional `workflow_version_id`).
Reference fields are expanded at compile time via `workflows_core.inner_workflow_spec_resolver`
(default: Roboflow API using `workflows_core.api_key`, or local definitions when workspace is
`"local"`).

At compile time the engine validates composition (acyclicity, max depth) and `parameter_bindings`,
then **inlines** the child's steps into the parent workflow graph (same execution path as ordinary
steps).

The block's `run()` method is not used at runtime; do not call it directly.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Inner Workflow",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "fak fa-diagram-nested",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/inner_workflow@v1"]
    workflow_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Full nested workflow definition (same JSON shape as a root workflow: version, inputs, "
            "steps, outputs). Required unless `workflow_workspace_id` and `workflow_id` are set; "
            "mutually exclusive with those reference fields."
        ),
    )
    workflow_workspace_id: Optional[str] = Field(
        default=None,
        description=(
            'Workspace id for a saved workflow to load (Roboflow slug or `"local"` for on-disk '
            "definitions). Use with `workflow_id`; mutually exclusive with a non-empty "
            "`workflow_definition`."
        ),
    )
    workflow_id: Optional[str] = Field(
        default=None,
        description="Saved workflow id to fetch. Use with `workflow_workspace_id`.",
    )
    workflow_version_id: Optional[str] = Field(
        default=None,
        description="Optional pinned workflow version when resolving by id.",
    )
    parameter_bindings: Dict[str, Selector()] = Field(
        description=(
            "Maps **child** workflow input names to a selector (or literal coerced by the engine) "
            "from the parent. Required for every child input except `WorkflowParameter` / "
            "`InferenceParameter` entries that declare a non-null `default_value` (those may be "
            "omitted and the child's default is used during compilation inlining)."
        ),
        json_schema_extra={
            "keys_bound_in": "parameter_bindings",
        },
    )

    @model_validator(mode="after")
    def validate_workflow_or_reference(self) -> "BlockManifest":
        has_inline = isinstance(self.workflow_definition, dict) and len(
            self.workflow_definition
        ) > 0
        ws = (self.workflow_workspace_id or "").strip()
        wf = (self.workflow_id or "").strip()
        has_ref = bool(ws and wf)

        if has_inline and has_ref:
            raise ValueError(
                "Provide either `workflow_definition` or workflow reference fields "
                "(`workflow_workspace_id` and `workflow_id`), not both."
            )
        if has_inline or has_ref:
            return self
        raise ValueError(
            "inner_workflow requires a non-empty `workflow_definition` object or both "
            "`workflow_workspace_id` and `workflow_id`."
        )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="*", kind=[WILDCARD_KIND])]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class InnerWorkflowBlockV1(WorkflowBlock):
    """Placeholder block; inner workflows are expanded at compile time and never executed as a unit."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, *args, **kwargs) -> BlockResult:
        raise RuntimeError(
            "inner_workflow steps are compiled away into ordinary steps; block.run() must not be called."
        )
