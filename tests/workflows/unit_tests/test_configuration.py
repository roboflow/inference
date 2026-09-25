"""Server-parity slice retained after Phase E split.

Everything else moved to `workflows/tests/unit_tests/test_configuration.py`.
The cases here reach into `inference/core/env.py` (the differential harness
lifts its actual resolution block out by AST) or spawn a subprocess importing
`inference.core.env` — neither is a Workflows dependency.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roboflow_workflows import configuration as configuration_module
from roboflow_workflows.configuration import resolve_image_tensor_device

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / "inference" / "core" / "env.py"


# --------------------------------------------------------------------------
# resolve_image_tensor_device - DIFFERENTIAL: the COPY against the ORIGINAL
# --------------------------------------------------------------------------

# Round-2 defect 5: the round-1 tests compared the copy against `env.py`'s
# already-materialised module attribute in a subprocess, which (a) never pinned
# `USE_INFERENCE_MODELS` (`env.py:1486` ANDs it into the flag, so a nominal
# "on" case could compare None with None) and (b) could not cover missing torch
# at all, because importing `inference.core.env` needs torch transitively.
#
# The harness below lifts `env.py`'s OWN resolution statement out by AST and
# executes it beside the copy under identical, controlled import conditions.
# No `inference.core.env` import is needed, so every cell - including missing
# torch and cuda-absent - is a true copy-vs-original comparison.


def _original_resolution_block() -> ast.If:
    """`if ENABLE_TENSOR_DATA_REPRESENTATION: try: import torch ...` from env.py."""
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "ENABLE_TENSOR_DATA_REPRESENTATION"
            and any(isinstance(statement, ast.Try) for statement in node.body)
        ):
            for statement in node.body:
                if isinstance(statement, ast.Try) and any(
                    isinstance(inner, ast.Import)
                    and any(alias.name == "torch" for alias in inner.names)
                    for inner in ast.walk(statement)
                ):
                    return node
    raise AssertionError("env.py's device-resolution block was not found")


def _run_original(flag: bool, device: str | None):
    node = _original_resolution_block()
    namespace = {
        "ENABLE_TENSOR_DATA_REPRESENTATION": flag,
        "WORKFLOWS_IMAGE_TENSOR_DEVICE_STR": device,
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": None,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(ENV_PATH), "exec"),
        namespace,
    )
    return namespace["WORKFLOWS_IMAGE_TENSOR_DEVICE"]


@pytest.mark.parametrize(
    "flag, device",
    [(False, None), (False, "cuda"), (True, "cpu"), (True, None)],
    ids=["off-nodevice", "off-device", "on-cpu", "on-autodetect"],
)
def test_the_copy_matches_the_original_resolution(flag, device) -> None:
    pytest.importorskip("torch")
    assert _run_original(flag, device) == resolve_image_tensor_device(flag, device)


def test_the_copy_matches_the_original_when_cuda_is_absent(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = _run_original(True, None)
    copied = resolve_image_tensor_device(True, None)
    assert original == copied == torch.device("cpu")


def test_the_copy_matches_the_original_when_torch_is_missing() -> None:
    import builtins

    real_import = builtins.__import__

    def _refuse_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    saved = sys.modules.pop("torch", None)
    builtins.__import__ = _refuse_torch
    try:
        original = _run_original(True, None)
        copied = resolve_image_tensor_device(True, None)
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["torch"] = saved
    assert original is None
    assert copied is None


def test_the_copy_preserves_the_originals_exception_boundary() -> None:
    """`env.py:1535-1547` wraps import + autodetect + materialisation in ONE
    `try`, catching only `ImportError`. The copy must do the same, or a
    `torch.cuda.is_available()` that raises `ImportError` would propagate from
    the copy while `env.py` swallows it."""
    node = _original_resolution_block()
    try_statement = next(s for s in node.body if isinstance(s, ast.Try))
    assert len(try_statement.body) == 3, ast.dump(try_statement)
    assert [type(s).__name__ for s in try_statement.body] == ["Import", "If", "Assign"]
    assert [handler.type.id for handler in try_statement.handlers] == ["ImportError"]

    copy_source = ast.parse(
        Path(configuration_module.__file__).read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in ast.walk(copy_source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_image_tensor_device"
    )
    copied_try = next(s for s in ast.walk(function) if isinstance(s, ast.Try))
    assert [type(s).__name__ for s in copied_try.body] == ["Import", "If", "Return"]
    assert [handler.type.id for handler in copied_try.handlers] == ["ImportError"]


def test_the_copy_rejects_an_invalid_device_exactly_as_the_original_does() -> None:
    pytest.importorskip("torch")
    # Neither guards `torch.device(...)`: an invalid string raises RuntimeError.
    with pytest.raises(RuntimeError):
        _run_original(True, "nonsense")
    with pytest.raises(RuntimeError):
        resolve_image_tensor_device(True, "nonsense")


# The server-parity `test_the_resolver_agrees_with_a_real_env_import` case
# stayed with `tests/workflows/unit_tests/test_configuration.py` — it needs
# the `inference.core.env` module which is not a workflows dependency.



def test_the_resolver_agrees_with_a_real_env_import(tmp_path) -> None:
    """End-to-end cell: a subprocess imports the REAL `inference.core.env` with
    the flag forced on, `USE_INFERENCE_MODELS` pinned (env.py ANDs it in),
    and asserts the module's own effective flag before comparing devices."""
    pytest.importorskip("torch")
    child = (
        "import json, os\n"
        "from inference.core import env\n"
        "from roboflow_workflows.configuration import resolve_image_tensor_device\n"
        "raw = os.environ.get('WORKFLOWS_IMAGE_TENSOR_DEVICE')\n"
        "print(json.dumps({\n"
        "  'flag': env.ENABLE_TENSOR_DATA_REPRESENTATION,\n"
        "  'equal': resolve_image_tensor_device(env.ENABLE_TENSOR_DATA_REPRESENTATION, raw)\n"
        "           == env.WORKFLOWS_IMAGE_TENSOR_DEVICE,\n"
        "  'device': str(env.WORKFLOWS_IMAGE_TENSOR_DEVICE),\n"
        "}))\n"
    )
    child_env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "inference_models"),
        "ENABLE_TENSOR_DATA_REPRESENTATION": "True",
        "USE_INFERENCE_MODELS": "True",
        "WORKFLOWS_IMAGE_TENSOR_DEVICE": "cpu",
    }
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["flag"] is True, payload
    assert payload["device"] == "cpu", payload
    assert payload["equal"], payload
