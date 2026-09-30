"""The development examples, run as tests.

* Every reference case: V1 in a child process (``reference/``) against V2 in
  this process, compared by ``comparison.compare_case``. A case passes with
  verdict ``parity`` or ``difference_confirmed`` (labelled, asserted V2 values).
* Every V2 capability example of ``02-sequential-parity/capabilities.py``.
* The migrated ``01-passive-foundation`` demo, through its own command.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "development" / "workflows-2.0").is_dir()
)
EXAMPLES_ROOT = REPOSITORY_ROOT / "development" / "workflows-2.0"
PARITY_DIR = EXAMPLES_ROOT / "02-sequential-parity"
if str(PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(PARITY_DIR))

import capabilities  # noqa: E402
from comparison import compare_case  # noqa: E402
from reference import REFERENCE_CASES, collect_v1_observations  # noqa: E402
from v2_runner import observe_v2_case  # noqa: E402


@pytest.fixture(scope="module")
def v1_observations() -> Dict[str, Dict[str, Any]]:
    report = collect_v1_observations()
    assert report["summary"]["pin_failures"] == []
    assert report["network_attempts"] == []

    return {case["case_id"]: case for case in report["cases"]}


@pytest.mark.parametrize("case_id", [case.case_id for case in REFERENCE_CASES])
def test_reference_case_matches_v1_or_confirms_labelled_difference(
    case_id: str, v1_observations: Dict[str, Dict[str, Any]]
) -> None:
    # given
    case = next(case for case in REFERENCE_CASES if case.case_id == case_id)

    # when
    comparison = compare_case(case, v1_observations[case_id], observe_v2_case(case))

    # then
    assert comparison["verdict"] in ("parity", "difference_confirmed"), comparison[
        "failed_checks"
    ]


@pytest.mark.parametrize("name", list(capabilities.EXAMPLES))
def test_capability_example(name: str, tmp_path: Path) -> None:
    # given
    report = capabilities.ExampleReport(name=name)
    example = capabilities.EXAMPLES[name]

    # when
    if name == "inspection_without_execution":
        example(report, output_dir=tmp_path)
    else:
        example(report)

    # then
    failed = [check for check in report.checks if not check["passed"]]
    assert report.checks and not failed, failed


def test_passive_foundation_demo(tmp_path: Path) -> None:
    # when
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(EXAMPLES_ROOT / "01-passive-foundation" / "run_demo.py"),
            "--scenario",
            "all",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REPOSITORY_ROOT / "workflows"),
            "PATH": "",
        },
        check=False,
    )

    # then
    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-3000:]
    )
    reports = [
        json.loads((tmp_path / name / "report.json").read_text())
        for name in (
            "nested",
            "filtered",
            "invalid-bindings",
            "author-block",
            "metadata-cost",
        )
    ]
    assert sum(len(report["checks"]) for report in reports) == 45
    assert all(check["passed"] for report in reports for check in report["checks"])
