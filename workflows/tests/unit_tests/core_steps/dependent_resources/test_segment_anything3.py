"""
Dependent-resources discovery tests for the SAM 3 image blocks
(``roboflow_core/sam3@v1|v2|v3``, numpy and tensor variants).

The model id is held directly in the Optional ``model_id`` field (default
``sam3/sam3_final``). Outside proxy execution, LOCAL and SDK REMOTE step
execution both pass ``model_id`` on, so an explicit ``None`` names no model:
the declaration is an incomplete discovery with an
``invalid_resource_identifier`` problem, never a known-empty list. Under proxy
execution (``SAM3_EXEC_MODE == "remote"``) every block declares ``[]``.
"""

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
    Discovery,
    incomplete_discovery,
    invalid_resource_identifier_problem,
)
from roboflow_workflows.prototypes.block import (
    ModelExecutionLocation,
    roboflow_platform_model,
)

VARIANTS = [
    (sam3_v1_module, "roboflow_core/sam3@v1"),
    (sam3_v2_module, "roboflow_core/sam3@v2"),
    (sam3_v3_module, "roboflow_core/sam3@v3"),
    (sam3_v1_tensor_module, "roboflow_core/sam3@v1"),
    (sam3_v2_tensor_module, "roboflow_core/sam3@v2"),
    (sam3_v3_tensor_module, "roboflow_core/sam3@v3"),
]
VARIANT_IDS = ["v1", "v2", "v3", "v1_tensor", "v2_tensor", "v3_tensor"]

MISSING_MODEL_ID = incomplete_discovery(
    [],
    [
        invalid_resource_identifier_problem(
            node_id="$steps.model",
            declaration="resources",
            field="model_id",
            resource_type="roboflow_platform_model",
        )
    ],
)

# (model_id override, expected declaration outside proxy execution)
DECLARATIONS = [
    ({}, [roboflow_platform_model(model_id="sam3/sam3_final")]),
    (
        {"model_id": "my_workspace/3"},
        [roboflow_platform_model(model_id="my_workspace/3")],
    ),
    (
        {"model_id": "$inputs.model_variant"},
        [roboflow_platform_model(model_id="$inputs.model_variant")],
    ),
    ({"model_id": None}, MISSING_MODEL_ID),
]
DECLARATION_IDS = ["default", "literal", "selector", "null"]


def _manifest(module, block_type: str, overrides: dict):
    payload = {"type": block_type, "name": "model", "images": "$inputs.image"}
    payload.update(overrides)
    return module.BlockManifest.model_validate(payload)


@pytest.mark.parametrize("module,block_type", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("overrides,expected", DECLARATIONS, ids=DECLARATION_IDS)
def test_sam3_declares_the_model_id_outside_proxy_execution(
    module, block_type, overrides, expected, monkeypatch
) -> None:
    # given
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")
    manifest = _manifest(module, block_type, overrides)

    # when
    declared = manifest.discover_dependent_resources()

    # then
    assert declared == expected


@pytest.mark.parametrize("module,block_type", VARIANTS, ids=VARIANT_IDS)
def test_sam3_null_model_id_is_unknown_not_a_known_absence(
    module, block_type, monkeypatch
) -> None:
    # given
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")
    manifest = _manifest(module, block_type, {"model_id": None})

    # when
    declared = manifest.discover_dependent_resources()

    # then - the Optional field is kept, and the answer is explicitly incomplete
    assert manifest.model_id is None
    assert isinstance(declared, Discovery)
    assert declared.complete is False
    assert declared.items == []


@pytest.mark.parametrize("module,block_type", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("overrides", [d[0] for d in DECLARATIONS], ids=DECLARATION_IDS)
def test_sam3_declares_nothing_under_proxy_execution_mode(
    module, block_type, overrides, monkeypatch
) -> None:
    # given
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "remote")
    manifest = _manifest(module, block_type, overrides)

    # when
    declared = manifest.discover_dependent_resources()

    # then - the proxy ignores the configured model id and runs its own fixed
    # SAM3 server-side, whatever model_id holds
    assert declared == []


@pytest.mark.parametrize("module,block_type", VARIANTS, ids=VARIANT_IDS)
def test_sam3_declares_environment_defined_execution_when_exec_mode_local(
    module, block_type, monkeypatch
) -> None:
    # given
    monkeypatch.setattr(module, "SAM3_EXEC_MODE", "local")
    manifest = _manifest(module, block_type, {})

    # when
    (resource,) = manifest.discover_dependent_resources()

    # then
    assert (
        resource.metadata.execution_location
        is ModelExecutionLocation.ENVIRONMENT_DEFINED
    )
