"""`roboflow_workflows.execution_engine.core` must import without `fastapi`."""

import os
import subprocess
import sys

CHILD = r"""
import sys

sys.modules["fastapi"] = None
import roboflow_workflows.execution_engine.core
"""


def test_core_imports_without_fastapi() -> None:
    child_env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    completed = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        timeout=300,
        env=child_env,
    )

    assert completed.returncode == 0, completed.stderr
