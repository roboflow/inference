"""Compatibility launcher for the standalone Workflows isolation probe.

The real probe lives at `workflows/scripts/workflows_isolation_probe.py` and
requires a pre-built `roboflow-workflows` wheel. This launcher keeps the old
entry point working for developers: it builds the wheel from `workflows/`
into a temp directory and delegates. CLI is passthrough after `--wheel` is
resolved.

Usage:
    python scripts/workflows_isolation_probe.py --tensor-mode both
    python scripts/workflows_isolation_probe.py --tensor-mode on --keep-tree
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_PROJECT = REPO_ROOT / "workflows"
PROBE = WORKFLOWS_PROJECT / "scripts" / "workflows_isolation_probe.py"


def _build_wheel(dist_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=WORKFLOWS_PROJECT,
        check=True,
    )
    wheels = sorted(dist_dir.glob("roboflow_workflows-*.whl"))
    assert wheels, f"no wheel produced in {dist_dir}"
    return wheels[-1]


def main() -> int:
    assert PROBE.is_file(), f"standalone probe missing: {PROBE}"
    with tempfile.TemporaryDirectory(prefix="workflows-wheel-") as dist:
        wheel = _build_wheel(Path(dist))
        return subprocess.run(
            [sys.executable, str(PROBE), "--wheel", str(wheel), *sys.argv[1:]],
            env=os.environ,
        ).returncode


if __name__ == "__main__":
    sys.exit(main())
