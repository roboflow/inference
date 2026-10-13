"""The offline merge and combine example (workflows-2.0/46-merge-and-combine).

The demo runs in its own interpreter by file path, so its generic ``run_demo``
module name never meets other examples' modules in this test process.
"""

import os
import subprocess
import sys

import pytest

from tests.v2.support.paths import CHECKOUT_ROOT, DEVELOPMENT_EXAMPLES

DEMO = DEVELOPMENT_EXAMPLES / "46-merge-and-combine" / "run_demo.py"
BOXES_CHECKS = [
    "combined rows keep their own class names despite clashing ids",
    "combined rows are new rows: fresh identity, new exported ids",
    "the duplicate car merges into one union box at the lower confidence",
    "the merged car is a new row with a fresh row identity",
]
VIEWS_CHECKS = [*BOXES_CHECKS, "views on one grid combine without upscaling"]
GATE_CHECK = "a gated input skips combine and merge, as in V1"
ERROR_CHECK = "boxes and masks do not mix, as in V1"
# The general car (20, 20, 44, 36) at 0.8 and the vehicles car, grown by 4 px,
# at 0.7 merge into their union box at the lower confidence.
MERGED_CAR = "merged    car    box (16, 16, 48, 40)  conf 0.7"


def run_demo(*arguments: str, tmp_path) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        PYTHONPATH=os.pathsep.join(
            str(CHECKOUT_ROOT / part) for part in ("workflows", "inference_models")
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(DEMO), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    return completed


def section_lines(output: str, title: str) -> list:
    """Stripped lines printed under one ``== title ==`` section."""
    section = output.split(f"\n== {title} ==\n", 1)[1].split("\n== ", 1)[0]
    lines = [line.strip() for line in section.splitlines() if line.strip()]

    return lines


def passed_checks(lines: list) -> list:
    checks = [line.removeprefix("[ok] ") for line in lines if line.startswith("[ok] ")]

    return checks


@pytest.mark.parametrize(
    "arguments, use_vehicles, merged",
    [([], True, [MERGED_CAR]), (["--no-vehicles"], False, [])],
    ids=["with-vehicles", "no-vehicles"],
)
def test_demo_passes_every_check(arguments, use_vehicles, merged, tmp_path) -> None:
    # given
    detector_checks = {
        "predictions": BOXES_CHECKS if use_vehicles else [GATE_CHECK],
        "segmentation": VIEWS_CHECKS if use_vehicles else [GATE_CHECK],
    }

    # when
    completed = run_demo(*arguments, tmp_path=tmp_path)

    # then
    output = completed.stdout
    assert completed.returncode == 0, output + completed.stderr
    assert "[FAIL]" not in output
    for output_name, checks in detector_checks.items():
        title = f"Detectors' `{output_name}`, use_vehicles={use_vehicles}"
        lines = section_lines(output, title)
        assert passed_checks(lines) == checks, title
        assert [line for line in lines if line.startswith("merged")] == merged
    assert passed_checks(section_lines(output, "Errors")) == [ERROR_CHECK]
    expected_count = sum(map(len, detector_checks.values())) + 1
    assert output.count("[ok]") == expected_count
    assert output.rstrip().endswith("\n0 failed check(s)")
