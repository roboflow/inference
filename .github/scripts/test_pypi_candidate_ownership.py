"""Exercise the candidate launcher across real Linux user/permission boundaries.

Run inside the release test container as root. Only Docker transport is stubbed:
the real launcher selects its user/environment, then the actual candidate CLI
runs with those Linux credentials on the container's native filesystem. This
avoids Docker Desktop bind mounts hiding Linux ownership failures.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_UID = 1001
RUNNER_GID = 1001


def _run(argv, *, workspace, env=None, uid=RUNNER_UID, gid=RUNNER_GID):
    result = subprocess.run(
        argv,
        cwd=workspace,
        env=env,
        user=uid,
        group=gid,
        extra_groups=[],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    return result


def _candidate(workspace, *, operation, attempt):
    environment = {
        "PATH": f"{workspace / 'bin'}:{os.environ['PATH']}",
        "HOME": str(workspace),
        "GITHUB_RUN_ID": "100",
        "GITHUB_RUN_ATTEMPT": attempt,
        "GITHUB_REPOSITORY": "roboflow/inference",
        "DOCKER_CAPTURE": str(workspace / "docker-argv.json"),
    }
    _run(
        [
            "bash",
            str(ROOT / ".release/pypi/candidate.sh"),
            operation,
            "--profile",
            "runtime",
        ],
        workspace=workspace,
        env=environment,
    )
    argv = json.loads((workspace / "docker-argv.json").read_text())
    assert argv[0] == "run"
    assert argv[argv.index("-v") + 1] == f"{workspace}:/work"
    assert argv[argv.index("-w") + 1] == "/work"
    assert argv[-4:] == ["/candidate.py", operation, "--profile", "runtime"]

    # Docker defaults to root when --user is absent. Execute the captured
    # command under that actual UID/GID, rather than mocking permission checks.
    identity = argv[argv.index("--user") + 1] if "--user" in argv else "0:0"
    uid, gid = map(int, identity.split(":"))
    container_env = {"PATH": os.environ["PATH"], "HOME": "/root"}
    for index, argument in enumerate(argv):
        if argument == "-e":
            name, separator, value = argv[index + 1].partition("=")
            if separator or name in environment:
                container_env[name] = value if separator else environment[name]

    result = _run(
        [
            sys.executable,
            str(ROOT / ".release/pypi/candidate.py"),
            operation,
            "--profile",
            "runtime",
        ],
        workspace=workspace,
        env=container_env,
        uid=uid,
        gid=gid,
    )

    return result


@pytest.fixture
def linux_workspace():
    assert (
        sys.platform == "linux" and os.geteuid() == 0
    ), "Run this permission regression inside the root Linux release test container"
    # Use the container's filesystem, not a potentially UID-remapped host mount.
    with tempfile.TemporaryDirectory(prefix="pypi-ownership-") as directory:
        workspace = Path(directory)
        os.chown(workspace, RUNNER_UID, RUNNER_GID)
        _run(["git", "init", "-q"], workspace=workspace)
        _run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "fixture",
            ],
            workspace=workspace,
        )
        fake_bin = workspace / "bin"
        fake_bin.mkdir(mode=0o755)
        docker = fake_bin / "docker"
        docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] == 'image':\n"
            "    print('sha256:fixture')\n"
            "elif sys.argv[1] == 'run':\n"
            "    Path(os.environ['DOCKER_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "elif sys.argv[1] != 'build':\n"
            "    raise AssertionError(sys.argv)\n"
        )
        docker.chmod(0o755)

        # Build outputs may originally belong to root. Sealing must still work
        # without changing their bytes or requiring a recursive host chown.
        wheel = workspace / "dist/pypi/runtime.whl"
        wheel.parent.mkdir(parents=True, mode=0o755)
        wheel.write_bytes(b"exact tested wheel bytes")
        _candidate(workspace, operation="seal", attempt="1")
        shutil.rmtree(workspace / "dist")

        yield workspace


@pytest.mark.parametrize("attempt", ["1", "2"])
def test_restored_runtime_accepts_runner_dependency_plan(linux_workspace, attempt):
    workspace = linux_workspace
    _candidate(workspace, operation="restore", attempt=attempt)
    _candidate(workspace, operation="verify", attempt=attempt)
    plan = '{"packages": {}}\n'

    if attempt == "1":
        # Same permission boundary as download-artifact extracting into dist.
        archive = workspace / "dependency-plan.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("dependency-plan.json", plan)
        _run(
            [
                sys.executable,
                "-c",
                "import zipfile; zipfile.ZipFile('dependency-plan.zip').extractall('dist')",
            ],
            workspace=workspace,
        )
    else:
        # The coordinator retry restores a retained plan, then copies it on
        # the host into the separately restored runtime directory.
        source = workspace / "release-plan/dependency-plan.json"
        source.parent.mkdir(mode=0o755)
        source.write_text(plan)
        _run(
            ["cp", "release-plan/dependency-plan.json", "dist/dependency-plan.json"],
            workspace=workspace,
        )

    assert (workspace / "dist/dependency-plan.json").read_text() == plan
    assert (
        workspace / "dist/pypi/runtime.whl"
    ).read_bytes() == b"exact tested wheel bytes"
    assert (workspace / "dist").stat().st_uid == RUNNER_UID
    assert (workspace / "dist").stat().st_gid == RUNNER_GID
