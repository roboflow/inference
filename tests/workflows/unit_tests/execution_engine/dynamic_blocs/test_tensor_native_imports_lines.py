"""The exec'd import string must point at the workflows facade, and must run.

`block_scaffolding.py:92-100` assembles the namespace of every dynamically
assembled block by `exec`-ing these strings. `ast` sees a string, not an
import, which is why the decontamination lint has a separate textual scan
(`test_decontamination_lint.py:40-53`, `:101`) and why this row is spelled
`inference.core.env (exec'd string)` in the baseline.
"""

import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    block_scaffolding,
)


def test_no_server_import_is_smuggled_through_the_generated_source() -> None:
    for line in (
        block_scaffolding.IMPORTS_LINES + block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES
    ):
        assert "inference.core.env" not in line, line
        assert "from inference.core.utils" not in line, line


def test_the_tensor_device_comes_from_the_workflows_facade() -> None:
    assert (
        "from inference.core.workflows.environment import WORKFLOWS_IMAGE_TENSOR_DEVICE"
        in block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES
    )
    # `modal/modal_app.py:607-611` imports this constant into the Modal
    # sandbox; it must stay a plain, self-contained list of strings.
    assert isinstance(block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES, list)
    assert len(block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES) == 7
    assert all(
        isinstance(line, str) for line in block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES
    )


def test_the_generated_tensor_namespace_actually_executes() -> None:
    pytest.importorskip("torch")
    namespace = {}
    exec(
        "\n".join(
            block_scaffolding.IMPORTS_LINES
            + block_scaffolding.TENSOR_NATIVE_IMPORTS_LINES
        ),
        namespace,
    )
    assert "WORKFLOWS_IMAGE_TENSOR_DEVICE" in namespace
    assert "WorkflowImageData" in namespace
    assert "Detections" in namespace
