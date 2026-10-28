"""Run a GitHub-generated test script without exposing its output in public logs."""

import subprocess
import sys

if __name__ == "__main__":
    try:
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", sys.argv[1]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        status = result.returncode
    except Exception:
        status = 1

    if status:
        print("::error::Private CI test step failed; output withheld")

    raise SystemExit(status if status > 0 else (1 if status < 0 else 0))
