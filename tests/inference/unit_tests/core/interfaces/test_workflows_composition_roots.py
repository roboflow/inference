"""Every composition root must bind the adapter, not the raw manager.

Round-1 defect 8: a name-occurrence check passes on an unused import while all
four raw-manager assignments stay unchanged.
"""

import ast
from pathlib import Path

# tests/inference/unit_tests/core/interfaces/<file> -> parents[5] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]

from inference.core.workflows.core_steps.common.entities import StepExecutionMode

ROOTS = {
    "inference/core/interfaces/http/http_api.py": 2,
    "inference/core/interfaces/stream/inference_pipeline.py": 1,
    "inference_cli/lib/workflows/local_image_adapter.py": 1,
}
KEY = "workflows_core.model_manager"
ADAPTER = "ModelManagerModelsProvider"


def _bound_values(tree: ast.AST) -> list:
    """Every expression bound to the `workflows_core.model_manager` key.

    Covers both shapes in the tree: a dict literal entry (bare or wrapped in
    `install_workflows_platform_bindings({...})`), and
    `params["workflows_core.model_manager"] = <value>`.
    """
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == KEY:
                    values.append(value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == KEY
                ):
                    values.append(node.value)
    return values


def _is_adapter_call(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == ADAPTER
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "model_manager"
    )


def test_every_binding_wraps_the_manager_in_the_adapter() -> None:
    total = 0
    for relative, expected_engine_inits in ROOTS.items():
        path = REPO_ROOT / relative
        assert path.is_file(), path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        values = _bound_values(tree)
        assert values, f"{relative}: no `{KEY}` binding found"
        for value in values:
            assert _is_adapter_call(value), (
                f"{relative}: `{KEY}` is bound to "
                f"{ast.dump(value)[:120]}, not {ADAPTER}(model_manager)"
            )
        engine_inits = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ExecutionEngine"
        )
        assert engine_inits == expected_engine_inits, (relative, engine_inits)
        assert len(values) == engine_inits, (relative, len(values), engine_inits)
        total += len(values)
    assert total == 4, total


def test_a_step_constructed_through_the_engine_receives_the_adapter() -> None:
    """Round-1 defect 8 / round-2: stopping at `retrieve_init_parameters_values`
    proves the lookup, not the construction. This goes through
    `initialise_step` (`steps_initialiser.py:53-86`), which is what the compiler
    calls, and asserts the *constructed block instance* holds the adapter."""
    from unittest.mock import MagicMock

    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )
    from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
        BlockManifest,
        RoboflowObjectDetectionModelBlockV1,
    )
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        BlockSpecification,
    )
    from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
        initialise_step,
    )

    manager = MagicMock()
    provider = ModelManagerModelsProvider(manager)
    manifest = BlockManifest.model_construct(
        name="detection", type="roboflow_core/roboflow_object_detection_model@v1"
    )
    specification = BlockSpecification(
        block_source="workflows_core",
        identifier="roboflow_core/roboflow_object_detection_model@v1",
        block_class=RoboflowObjectDetectionModelBlockV1,
        manifest_class=BlockManifest,
    )
    initialised = initialise_step(
        step_manifest=manifest,
        block_specification=specification,
        explicit_init_parameters={
            "workflows_core.model_manager": provider,
            "workflows_core.api_key": "key",
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        initializers={},
    )
    assert initialised.step._model_manager is provider
    assert initialised.step._model_manager._model_manager is manager
