import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "inference.models.depth_anything_v2.depth_anything_v2",
        "inference.models.depth_anything_v2.depth_anything_v2_inference_models",
        "inference.models.depth_anything_v3.depth_anything_v3",
        "inference.models.depth_anything_v3.depth_anything_v3_inference_models",
    ],
)
def test_importing_depth_model_does_not_import_matplotlib_pyplot(
    module_name: str,
) -> None:
    # Run in a fresh interpreter so modules imported by other tests cannot mask
    # an eager import in the module under test.
    code = textwrap.dedent(f"""
        import importlib
        import sys

        importlib.import_module({module_name!r})

        assert "matplotlib.pyplot" not in sys.modules, (
            "matplotlib.pyplot was imported while importing {module_name}"
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
