"""Modal/WebExec settings come from the injected configuration (D4).

`modal_executor.py` hides FOUR function-local `inference.core.env` imports
(`:612`, `:859`, `:1629`, `:2327`) on top of its module-level one - the exact
shape the decontamination lint's `ast.walk` was written to catch
(`test_decontamination_lint.py:79-81`).

Round-1 defect 5: `WEBEXEC_TRANSPORT` is NOT a module attribute of
`modal_executor` - it is imported inside `get_modal_executor`. The behavioural
test therefore calls that function and checks which executor class comes back.
"""

import ast
from pathlib import Path

import pytest

from inference.core.workflows import environment
from inference.core.workflows.execution_engine.v1.dynamic_blocks import modal_executor

MODULE = Path(modal_executor.__file__)

MODAL_NAMES = {
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_WORKSPACE_NAME",
    "MODAL_ALLOW_ANONYMOUS_EXECUTION",
    "MODAL_ANONYMOUS_WORKSPACE_NAME",
    "WEBEXEC_MODAL_APP_NAME",
    "WEBEXEC_JPEG_QUALITY",
    "WEBEXEC_TRANSPORT",
    "WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS",
    "WEBEXEC_WS_READ_TIMEOUT_SECONDS",
    "WEBEXEC_WS_CONNECTION_POOL_SIZE",
    "WEBEXEC_WS_FAIL_ON_SESSION_LOSS",
    "WEBEXEC_WS_IDLE_RELEASE_SECONDS",
}


def _import_statements() -> list:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    # ast.walk, not tree.body: four of the five statements live inside function
    # bodies and a top-level scan would miss every one of them.
    return [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def test_no_import_of_the_server_env_module_anywhere_in_the_file() -> None:
    assert not [n for n in _import_statements() if n.module == "inference.core.env"]


def test_every_modal_name_is_imported_from_the_workflows_facade() -> None:
    imported = {
        alias.name
        for node in _import_statements()
        if node.module == "inference.core.workflows.environment"
        for alias in node.names
    }
    assert MODAL_NAMES <= imported, sorted(MODAL_NAMES - imported)


def test_the_four_lazy_import_sites_stay_lazy() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    local = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.ImportFrom)
                and sub.module == "inference.core.workflows.environment"
            ):
                local += 1
    assert local == 4, local


def test_module_level_modal_availability_follows_the_facade() -> None:
    assert modal_executor.MODAL_AVAILABLE is bool(
        environment.MODAL_TOKEN_ID and environment.MODAL_TOKEN_SECRET
    )
    assert modal_executor.WEBEXEC_JPEG_QUALITY == environment.WEBEXEC_JPEG_QUALITY
    assert (
        modal_executor.WEBEXEC_WS_READ_TIMEOUT_SECONDS
        == environment.WEBEXEC_WS_READ_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize(
    "transport, expected_class_name",
    [
        ("http", "ModalExecutor"),
        ("websocket", "PooledWebSocketModalExecutor"),
    ],
)
def test_get_modal_executor_selects_the_transport_from_the_facade(
    monkeypatch, transport, expected_class_name
) -> None:
    """`get_modal_executor` (`modal_executor.py:2325-2331`) re-imports the
    transport on every call, so patching the FACADE - the module it now imports
    from - changes the class it returns. This is the behavioural proof that the
    lazy site really reads the configuration."""
    monkeypatch.setattr(environment, "WEBEXEC_TRANSPORT", transport)
    executor = modal_executor.get_modal_executor(workspace_id="workspace")
    assert type(executor).__name__ == expected_class_name
