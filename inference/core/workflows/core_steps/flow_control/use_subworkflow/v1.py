from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    Selector,
    WILDCARD_KIND,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Run an embedded workflow definition with parameters mapped from the parent workflow."
)

LONG_DESCRIPTION = """
Execute a **nested workflow** defined inline (`embedded_workflow`) while mapping parent data into the
child's inputs via `parameter_bindings`. Compilation validates composition (acyclicity, max depth) and
child input names; execution is handled by the execution engine using a pluggable `SubworkflowRunner`
(see `workflows_core.subworkflow_runner` init parameter).

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
    embedded_workflow: Dict[str, Any] = Field(
        description="Full nested workflow definition (same JSON shape as a root workflow: version, inputs, steps, outputs).",
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
    """Placeholder block; execution engine runs nested workflows via SubworkflowRunner."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, *args, **kwargs) -> BlockResult:
        raise RuntimeError(
            "use_subworkflow steps are executed by the execution engine via SubworkflowRunner; "
            "block.run() must not be called."
        )
