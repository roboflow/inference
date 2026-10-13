"""Offline formatter examples run in separate interpreters by file path."""

import os
import subprocess
import sys

import pytest

from tests.v2.support.paths import CHECKOUT_ROOT, DEVELOPMENT_EXAMPLES


@pytest.mark.parametrize(
    "example, checks",
    [
        (
            "48-text-and-values",
            [
                "templates use the Berlin time",
                "gate merges with the default",
                "CSV only on the last reading",
                "CSV keeps column order",
                "commas survive quoting",
                "formula text is neutralized",
                "timestamps are UTC",
            ],
        ),
        (
            "49-json-fields",
            [
                "error_status is True only for the answer that is not JSON",
                "fenced and raw JSON resize the image; the failed row resizes nothing",
                "width 'wide' fails in v2/resize",
                "['output width'] is rejected",
                "['error_status'] is rejected",
                "['height', 'height'] is rejected",
            ],
        ),
    ],
)
def test_formatter_example_passes_every_visible_check(
    example, checks, tmp_path
) -> None:
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        PYTHONPATH=os.pathsep.join(
            str(CHECKOUT_ROOT / part) for part in ("workflows", "inference_models")
        ),
    )

    completed = subprocess.run(
        [sys.executable, "-B", str(DEVELOPMENT_EXAMPLES / example / "run_demo.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    output = completed.stdout
    assert completed.returncode == 0, output + completed.stderr
    assert "[FAIL]" not in output
    assert [
        line.removeprefix("[ok] ")
        for line in output.splitlines()
        if line.startswith("[ok] ")
    ] == checks
