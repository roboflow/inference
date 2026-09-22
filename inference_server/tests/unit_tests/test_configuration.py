import os
import subprocess
import sys


def test_offline_mode_forces_local_sam3_execution():
    code = (
        "from inference_server import configuration as c; "
        "assert c.SAM3_EXEC_MODE == 'local', c.SAM3_EXEC_MODE; "
        "assert c.SAM3_FINE_TUNED_MODELS_ENABLED is False, "
        "c.SAM3_FINE_TUNED_MODELS_ENABLED"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": "true",
            "SAM3_EXEC_MODE": "remote",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
