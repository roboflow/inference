"""A SAM3 step with ``model_id: null`` is reported as an unknown resource.

Outside proxy execution, LOCAL and SDK REMOTE step execution both pass
``model_id`` on, so ``None`` names no model. The public workload inspection
must say the step's resources are incomplete, and the aggregate model summary
must carry the same problem. Runtime pre-loading reads the same declaration
and must skip it without crashing.

No model is loaded; no network or platform API is called.
"""

import networkx as nx
import pytest
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v1 as sam3_v1_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v1_tensor as sam3_v1_tensor_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v2 as sam3_v2_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v2_tensor as sam3_v2_tensor_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v3 as sam3_v3_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v3_tensor as sam3_v3_tensor_module,
)
from roboflow_workflows.execution_engine.entities.workload import (
    invalid_resource_identifier_problem,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
    ParsedWorkflowDefinition,
)
from roboflow_workflows.execution_engine.v1.compiler.utils import (
    deduce_blocks_dependencies,
)
from roboflow_workflows.prototypes.block import roboflow_platform_model

SAM3_MODULES = [
    sam3_v1_module,
    sam3_v2_module,
    sam3_v3_module,
    sam3_v1_tensor_module,
    sam3_v2_tensor_module,
    sam3_v3_tensor_module,
]
VARIANTS = [
    (sam3_v1_module, "roboflow_core/sam3@v1"),
    (sam3_v2_module, "roboflow_core/sam3@v2"),
    (sam3_v3_module, "roboflow_core/sam3@v3"),
    (sam3_v1_tensor_module, "roboflow_core/sam3@v1"),
    (sam3_v2_tensor_module, "roboflow_core/sam3@v2"),
    (sam3_v3_tensor_module, "roboflow_core/sam3@v3"),
]
VARIANT_IDS = ["v1", "v2", "v3", "v1_tensor", "v2_tensor", "v3_tensor"]


def _missing_model_id(step_name: str):
    return invalid_resource_identifier_problem(
        node_id=f"$steps.{step_name}",
        declaration="resources",
        field="model_id",
        resource_type="roboflow_platform_model",
    )


def _step(name: str, block_type: str, **overrides) -> dict:
    step = {"type": block_type, "name": name, "images": "$inputs.image"}
    step.update(overrides)
    return step


@pytest.fixture
def non_proxy_sam3(monkeypatch) -> None:
    # Whichever variant the blocks loader registers, it runs outside the proxy.
    for module in SAM3_MODULES:
        monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")


@pytest.mark.parametrize(
    "block_type",
    ["roboflow_core/sam3@v1", "roboflow_core/sam3@v2", "roboflow_core/sam3@v3"],
)
def test_workload_inspection_reports_a_null_sam3_model_id_as_incomplete(
    block_type: str, non_proxy_sam3
) -> None:
    # given
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            _step("missing", block_type, model_id=None),
            _step("known", block_type),
        ],
        "outputs": [],
    }

    # when
    introspection = describe_workflow_workload(workflow_definition=definition)

    # then - the step reports the missing identity ...
    (missing,) = [s for s in introspection.steps if s.node_id == "$steps.missing"]
    assert missing.resources.complete is False
    assert missing.resources.items == []
    assert missing.resources.unknown_reasons == [_missing_model_id("missing")]
    # ... and the aggregate summary carries it, next to the known model
    models = introspection.summary.models
    assert models.complete is False
    assert models.unknown_reasons == [_missing_model_id("missing")]
    assert [(m.model_id, m.used_by_steps) for m in models.items] == [
        ("sam3/sam3_final", ["$steps.known"])
    ]


@pytest.mark.parametrize("module,block_type", VARIANTS, ids=VARIANT_IDS)
def test_pre_loading_skips_a_null_sam3_model_id_without_crashing(
    module, block_type: str, non_proxy_sam3
) -> None:
    # given
    manifest_class = module.BlockManifest
    compiled_workflow = CompiledWorkflow(
        workflow_definition=ParsedWorkflowDefinition(
            version="1.0",
            inputs=[],
            steps=[
                manifest_class.model_validate(
                    _step("missing", block_type, model_id=None)
                ),
                manifest_class.model_validate(_step("known", block_type)),
            ],
            outputs=[],
        ),
        execution_graph=nx.DiGraph(),
        steps={},
        input_substitutions=[],
        workflow_json={},
        init_parameters={},
    )

    # when
    dependencies = deduce_blocks_dependencies(compiled_workflow=compiled_workflow)

    # then - only the known model reaches the pre-loader
    assert dependencies == [roboflow_platform_model(model_id="sam3/sam3_final")]
