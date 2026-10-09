import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
SCRIPT = REPO_ROOT / "scripts" / "openapi_paths.py"
SNAPSHOT = REPO_ROOT / "scripts" / "legacy_openapi_paths.json"


def test_legacy_route_snapshot_is_current(tmp_path: Path) -> None:
    output = tmp_path / "legacy_openapi_paths.json"
    environment = dict(os.environ)
    python_path = [str(REPO_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "dump",
            "--app",
            "legacy",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == SNAPSHOT.read_text(), (
        "scripts/legacy_openapi_paths.json is out of date. Regenerate it from the "
        "repository root with: PYTHONPATH=workflows:stream_vision:. python "
        "scripts/openapi_paths.py dump --app legacy "
        "--output scripts/legacy_openapi_paths.json"
    )


def test_snapshot_holds_routes_hidden_from_the_schema() -> None:
    paths = json.loads(SNAPSHOT.read_text())["paths"]

    assert paths["/build/"] == ["GET"]
    assert paths["/build/{workflow_id}"] == ["GET"]
