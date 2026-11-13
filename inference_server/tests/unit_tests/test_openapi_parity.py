import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
TOOL = SCRIPTS / "openapi_paths.py"
LEGACY_SNAPSHOT = SCRIPTS / "legacy_openapi_paths.json"
ALLOWLIST = SCRIPTS / "openapi_allowlist.txt"


def _run_compare(legacy: Path, server: Path, allowlist: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "compare",
            "--legacy",
            str(legacy),
            "--server",
            str(server),
            "--allowlist",
            str(allowlist),
        ],
        capture_output=True,
        text=True,
    )

    return result


def test_server_exposes_every_legacy_route_or_allowlists_it(tmp_path: Path) -> None:
    server_dump = tmp_path / "server_openapi_paths.json"

    dump = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "dump",
            "--app",
            "server",
            "--output",
            str(server_dump),
        ],
        capture_output=True,
        text=True,
    )
    assert dump.returncode == 0, dump.stderr
    comparison = _run_compare(LEGACY_SNAPSHOT, server_dump, ALLOWLIST)

    assert comparison.returncode == 0, comparison.stdout + comparison.stderr


def test_compare_fails_on_an_empty_legacy_inventory(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"paths": {}}))
    populated = tmp_path / "populated.json"
    populated.write_text(json.dumps({"paths": {"/info": ["GET"]}}))
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("")

    result = _run_compare(empty, populated, allowlist)

    assert result.returncode != 0
    assert "legacy inventory is empty" in result.stderr


def test_compare_fails_on_an_empty_server_inventory(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"paths": {}}))
    populated = tmp_path / "populated.json"
    populated.write_text(json.dumps({"paths": {"/info": ["GET"]}}))
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("")

    result = _run_compare(populated, empty, allowlist)

    assert result.returncode != 0
    assert "server inventory is empty" in result.stderr
