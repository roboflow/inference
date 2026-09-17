import ast
from pathlib import Path
from unittest.mock import MagicMock

from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.core_steps.common.utils import load_core_model

# tests/workflows/unit_tests/core_steps/common/<file> -> parents[5] is the repo root
UTILS = (
    Path(__file__).resolve().parents[5]
    / "inference/core/workflows/core_steps/common/utils.py"
)


def test_utils_does_not_import_server_entities() -> None:
    assert UTILS.is_file(), UTILS
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    offenders = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("inference.core.entities")
    ]
    assert offenders == []


def test_load_core_model_registers_the_derived_core_model_id() -> None:
    model_manager = MagicMock()
    assert (
        load_core_model(
            model_manager=model_manager,
            core_model="sam2",
            version_id="hiera_large",
            api_key="key",
        )
        == "sam2/hiera_large"
    )
    args, kwargs = model_manager.add_model.call_args
    assert args == ("sam2/hiera_large", "key")
    # Order-independent: Phase 9's string constant (Task 9.2) and the enum a
    # pre-Phase-9 tree passes both coerce to the CORE_MODEL member.
    assert set(kwargs) == {"endpoint_type"}
    assert ModelEndpointType(kwargs["endpoint_type"]) is ModelEndpointType.CORE_MODEL
