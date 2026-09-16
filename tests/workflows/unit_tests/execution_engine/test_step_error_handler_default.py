import ast
import os
from pathlib import Path

from inference.core.workflows.execution_engine.v1 import core as ee_core
from inference.core.workflows.execution_engine.v1 import step_error_handlers


def test_standalone_registry_only_knows_legacy() -> None:
    assert set(ee_core.REGISTERED_STEP_ERROR_HANDLERS) == {"legacy"}
    assert ee_core.DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER == os.getenv(
        "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "legacy"
    )


def test_legacy_handler_maps_nothing_on_its_own() -> None:
    # Every mapping it used to carry was for server exception classes and now
    # lives in inference.core.interfaces.workflows_step_error_handlers.
    assert (
        step_error_handlers.legacy_step_error_handler("step", RuntimeError("x")) is None
    )


def test_step_error_handlers_module_has_no_server_imports() -> None:
    # Import nodes, not source text: the new docstring names the old module.
    tree = ast.parse(Path(step_error_handlers.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {
        name
        for name in imported
        if name and name.startswith("inference.core.exceptions")
    }
