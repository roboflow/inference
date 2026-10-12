import os
import subprocess
import sys
import textwrap


def test_importing_entities_base_does_not_import_torchvision_io() -> None:
    # Run in a fresh interpreter so modules imported by other tests cannot mask
    # an eager import in the module under test.
    code = textwrap.dedent("""
        import sys

        import roboflow_workflows.execution_engine.entities.base  # noqa: F401

        assert "torchvision.io" not in sys.modules, (
            "torchvision.io was imported while importing entities.base"
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
