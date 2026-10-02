"""Collect V1 reference observations from a separate Python process.

A V2 demo imports this module instead of V1. The V1 plugin bootstrap, fixture
discovery and engine state therefore never enter the caller's process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

RUNNER = Path(__file__).resolve().parent / "run_v1_reference.py"


class V1ReferenceError(RuntimeError):
    """The V1 reference process failed before producing a report."""


def collect_v1_observations(
    case_ids: Optional[Sequence[str]] = None,
    *,
    python: Optional[str] = None,
) -> Dict[str, Any]:
    """Run reference cases on V1 in a child process and return its JSON report.

    Args:
        case_ids: Case identifiers to run; ``None`` runs every case.
        python: Interpreter for the child process; defaults to the current one.

    Returns:
        The runner report: ``engine``, ``network_attempts``, ``cases`` and
        ``summary``. Check ``summary["pin_failures"]`` before trusting pins.

    Raises:
        V1ReferenceError: If the child process crashed or printed no report.
    """
    command = [python or sys.executable, "-B", str(RUNNER)]
    for case_id in case_ids or []:
        command.extend(["--case", case_id])
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise V1ReferenceError(
            f"V1 reference runner exited with {completed.returncode} without a "
            f"JSON report. stderr tail:\n{completed.stderr[-2000:]}"
        ) from error

    return report
