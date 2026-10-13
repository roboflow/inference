"""The offline task-model example (development/workflows-2.0/43-task-models).

The example runs in a subprocess: other example tests import their own
top-level ``run_demo`` module, so an in-process import would share that name.
"""

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "development" / "workflows-2.0").is_dir()
)
DEMO_DIR = REPOSITORY_ROOT / "development/workflows-2.0/43-task-models"


def test_offline_demo_passes_every_check() -> None:
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            str(REPOSITORY_ROOT / part)
            for part in (".", "workflows", "inference_models", "stream_vision")
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(DEMO_DIR / "run_demo.py")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    output = completed.stdout
    assert completed.returncode == 0, output + completed.stderr
    assert "[FAIL]" not in output
    assert output.count("[ok]") == 11
