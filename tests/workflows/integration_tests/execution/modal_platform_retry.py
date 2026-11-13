"""Rerun a Modal test's workflow run when Modal's platform fails the request.

The Modal tests run custom Python blocks on the deployed webexec app. Modal's
own HTTP layer sometimes fails one request (status 408, 502, 503 or 504, a 500
from ``modal-http``, or a dropped or timed-out connection) and the same request
passes when sent again. Those errors say nothing about the code under test, so
a run that fails with one is run again. Any other failure, including a 500 from
the webexec app itself, fails the test on the first attempt.

When a run still fails after every attempt, Modal is taken to be down and later
runs are not retried, so an outage costs one test's retries, not every test's.
"""

import re
import subprocess
import time
import warnings
from typing import Any, List, Optional

ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0

# Matched against the traceback the workflow subprocess prints on stderr; the
# messages are the ones ModalExecutor raises.
MODAL_PLATFORM_ERROR = re.compile(
    r"DynamicBlockError: ((?:"
    r"Modal endpoint returned status (?:408|502|503|504)\b"
    r"|Modal endpoint returned status 500: modal-http: "
    r"|Failed to connect to Modal endpoint: "
    r").*)"
)

_modal_is_down = False


def find_modal_platform_error(stderr: Optional[str]) -> Optional[str]:
    """Find a Modal platform error in a workflow subprocess's stderr.

    Args:
        stderr: Output of the workflow subprocess, ``None`` if not captured.

    Returns:
        The error message, or ``None`` when stderr holds no Modal platform error.
    """
    match = MODAL_PLATFORM_ERROR.search(stderr or "")
    if match is None:
        return None

    error = match.group(1)
    return error


def run_with_modal_platform_retries(
    args: List[str], **kwargs: Any
) -> subprocess.CompletedProcess:
    """Run a workflow subprocess, rerunning it after a Modal platform error.

    Up to ``ATTEMPTS`` runs, ``RETRY_DELAY_SECONDS`` times the attempt number
    apart. Once a run fails every attempt, later calls run only once.

    Args:
        args: Command, as for ``subprocess.run``.
        **kwargs: Passed to ``subprocess.run``; include ``capture_output=True``
            and ``text=True``, since the error is read from stderr.

    Returns:
        The last run's result.
    """
    global _modal_is_down
    attempt = 1
    while True:
        result = subprocess.run(args, **kwargs)
        if result.returncode == 0 or _modal_is_down:
            return result

        error = find_modal_platform_error(result.stderr)
        if error is None:
            return result

        if attempt == ATTEMPTS:
            _modal_is_down = True
            return result

        warnings.warn(
            f"Retrying the Modal run after attempt {attempt} of {ATTEMPTS} "
            f"failed with: {error}"
        )
        time.sleep(RETRY_DELAY_SECONDS * attempt)
        attempt += 1
