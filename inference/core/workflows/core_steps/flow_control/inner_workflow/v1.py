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

Compilation validates composition (acyclicity, max depth) and child input names; execution uses a
pluggable `InnerWorkflowRunner` (see `workflows_core.inner_workflow_runner`).

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
        description="Maps each **child** workflow input name to a selector (or literal coerced by the engine) from the parent.",
        json_schema_extra={
            "keys_bound_in": "parameter_bindings",
        },
    )
    resolved_child_outputs: Optional[List[OutputDefinition]] = Field(
        default=None,
        description="Set by the compiler when resolving child workflows.",
        repr=False,
    )
    nested_output_dimensionality_lift: int = Field(
        default=0,
        exclude=True,
        description=(
            "Compiler-only: max ``get_output_dimensionality_offset()`` among child steps "
            "referenced by the child workflow JsonField outputs (drives parent batch lineage)."
        ),
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

    def get_actual_outputs(self) -> List[OutputDefinition]:
        if self.resolved_child_outputs is not None:
            return list(self.resolved_child_outputs)
        return super().get_actual_outputs()

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class InnerWorkflowBlockV1(WorkflowBlock):
    """Placeholder block; execution engine runs inner workflows via InnerWorkflowRunner."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, *args, **kwargs) -> BlockResult:
        raise RuntimeError(
            "inner_workflow steps are executed by the execution engine via InnerWorkflowRunner; "
            "block.run() must not be called."
        )
