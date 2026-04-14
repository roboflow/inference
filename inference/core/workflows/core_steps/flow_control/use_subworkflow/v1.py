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

SHORT_DESCRIPTION = "Run an embedded workflow definition with parameters mapped from the parent workflow."

LONG_DESCRIPTION = """
Execute a **nested workflow** while mapping parent data into the child's inputs via `parameter_bindings`.

Provide either a full inline definition in `embedded_workflow`, or resolve a saved workflow using
`embedded_workflow_workspace_id` and `embedded_workflow_id` (optional `embedded_workflow_version_id`).
Reference fields are expanded at compile time via `workflows_core.inner_workflow_spec_resolver`
(legacy: `workflows_core.subworkflow_spec_resolver`; default: Roboflow API using
`workflows_core.api_key`, or local definitions when workspace is `"local"`).

Compilation validates composition (acyclicity, max depth) and child input names; execution uses a
pluggable `InnerWorkflowRunner` (see `workflows_core.inner_workflow_runner`; legacy:
`workflows_core.subworkflow_runner`).

The block's `run()` method is not used at runtime; do not call it directly.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Use Sub-workflow",
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
    type: Literal["roboflow_core/use_subworkflow@v1"]
    embedded_workflow: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Full nested workflow definition (same JSON shape as a root workflow: version, inputs, "
            "steps, outputs). Required unless `embedded_workflow_workspace_id` and "
            "`embedded_workflow_id` are set; mutually exclusive with those reference fields."
        ),
    )
    embedded_workflow_workspace_id: Optional[str] = Field(
        default=None,
        description=(
            'Workspace id for a saved workflow to embed (Roboflow slug or `"local"` for on-disk '
            "definitions). Use with `embedded_workflow_id`; mutually exclusive with a non-empty "
            "`embedded_workflow`."
        ),
    )
    embedded_workflow_id: Optional[str] = Field(
        default=None,
        description="Saved workflow id to fetch and embed. Use with `embedded_workflow_workspace_id`.",
    )
    embedded_workflow_version_id: Optional[str] = Field(
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
        description="Set by the compiler when resolving embedded workflows.",
        repr=False,
    )
    nested_output_dimensionality_lift: int = Field(
        default=0,
        exclude=True,
        description=(
            "Compiler-only: max ``get_output_dimensionality_offset()`` among child steps "
            "referenced by embedded workflow JsonField outputs (drives parent batch lineage)."
        ),
    )

    @model_validator(mode="after")
    def validate_embedded_or_reference(self) -> "BlockManifest":
        has_inline = (
            isinstance(self.embedded_workflow, dict) and len(self.embedded_workflow) > 0
        )
        ws = (self.embedded_workflow_workspace_id or "").strip()
        wf = (self.embedded_workflow_id or "").strip()
        has_ref = bool(ws and wf)

        if has_inline and has_ref:
            raise ValueError(
                "Provide either `embedded_workflow` or workflow reference fields "
                "(`embedded_workflow_workspace_id` and `embedded_workflow_id`), not both."
            )
        if has_inline or has_ref:
            return self
        raise ValueError(
            "use_subworkflow requires a non-empty `embedded_workflow` object or both "
            "`embedded_workflow_workspace_id` and `embedded_workflow_id`."
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


class UseSubworkflowBlockV1(WorkflowBlock):
    """Placeholder block; execution engine runs inner workflows via InnerWorkflowRunner."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, *args, **kwargs) -> BlockResult:
        raise RuntimeError(
            "use_subworkflow steps are executed by the execution engine via InnerWorkflowRunner; "
            "block.run() must not be called."
        )
