"""Exercise Modal startup imports and cancellation in a fresh interpreter."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "worker", ["RTCPeerConnectionModalCPU", "RTCPeerConnectionModalGPU"]
)
def test_workflow_imports_survive_cancelled_modal_input(worker):
    # Repo-root modal/ makes a bare "modal" import succeed without the SDK.
    pytest.importorskip("modal._partial_function")
    root = Path(__file__).resolve().parents[6]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import __future__
                import ast
                import importlib
                import signal
                import sys
                from pathlib import Path

                import modal
                from modal._partial_function import (
                    _find_callables_for_obj,
                    _PartialFunctionFlags,
                )
                from modal.exception import InputCancellation

                # Keep the real classes/hooks without registering an app or loading models.
                path = Path("inference/core/interfaces/webrtc_worker/modal.py")
                tree = ast.parse(path.read_text())
                classes = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                    and node.name.startswith("RTCPeerConnectionModal")
                ]
                for cls in classes:
                    cls.decorator_list = []
                namespace = {"modal": modal}
                exec(compile(
                    ast.Module(body=classes, type_ignores=[]),
                    str(path), "exec",
                    flags=__future__.annotations.compiler_flag,
                ), namespace)
                worker = namespace[sys.argv[1]]()
                hooks = _find_callables_for_obj(
                    worker, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT
                )
                hooks["_preload_workflow_dependencies"]()

                original = importlib._bootstrap._find_and_load

                def cancel(signum, frame):
                    raise InputCancellation("cancelled during session initialization")

                def interrupt_first_pandas_import(name, *args):
                    if name == "pandas._testing":
                        signal.raise_signal(signal.SIGUSR1)
                    return original(name, *args)

                signal.signal(signal.SIGUSR1, cancel)
                importlib._bootstrap._find_and_load = interrupt_first_pandas_import
                try:
                    from inference.core.workflows.execution_engine.core import ExecutionEngine
                    signal.raise_signal(signal.SIGUSR1)
                except InputCancellation:
                    pass
                finally:
                    importlib._bootstrap._find_and_load = original

                # The next input must reuse complete modules after cancellation.
                from inference.core.workflows.execution_engine.core import ExecutionEngine
                import pandas as pd

                assert pd.core is sys.modules["pandas.core"]
                assert pd.DataFrame({"value": [1]}).to_csv(index=False) == "value\\n1\\n"
                assert ExecutionEngine is sys.modules[
                    "roboflow_workflows.execution_engine.core"
                ].ExecutionEngine
                """),
            worker,
        ],
        cwd=root,
        env={**os.environ, "DISABLE_VERSION_CHECK": "True"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
