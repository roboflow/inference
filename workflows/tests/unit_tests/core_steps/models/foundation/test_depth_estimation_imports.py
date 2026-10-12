import os
import subprocess
import sys
import textwrap


def test_importing_depth_estimation_block_does_not_import_matplotlib_pyplot() -> None:
    # Run in a fresh interpreter so modules imported by other tests cannot mask
    # an eager import in the module under test.
    code = textwrap.dedent("""
        import sys

        import roboflow_workflows.core_steps.models.foundation.depth_estimation.v1_tensor  # noqa: E501,F401

        assert "matplotlib.pyplot" not in sys.modules, (
            "matplotlib.pyplot was imported while importing depth_estimation.v1_tensor"
        )
        print("ok")
        """)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "API_LOGGING_ENABLED": "False"},
        check=False,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().endswith("ok")
