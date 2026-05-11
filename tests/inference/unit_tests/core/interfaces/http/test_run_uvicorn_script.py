import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[6]
SCRIPT_PATH = REPO_ROOT / "docker" / "entrypoint" / "run_uvicorn.sh"


def _fake_uvicorn_dir(tmp_path: Path) -> Path:
    """Create a directory containing a fake `uvicorn` that just echoes its args."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uvicorn"
    fake.write_text("#!/bin/sh\nfor arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n")
    fake.chmod(0o755)
    return bin_dir


def _run(env: dict, args: list, tmp_path: Path) -> subprocess.CompletedProcess:
    bin_dir = _fake_uvicorn_dir(tmp_path)
    full_env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        **env,
    }
    return subprocess.run(
        ["/bin/sh", str(SCRIPT_PATH), *args],
        env=full_env,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="entrypoint script not present")
def test_script_invokes_uvicorn_without_ssl_when_disabled(tmp_path: Path) -> None:
    result = _run({}, ["cpu_http:app"], tmp_path)
    assert result.returncode == 0, result.stderr
    forwarded = result.stdout.splitlines()
    assert forwarded[0] == "cpu_http:app"
    assert "--workers" in forwarded
    assert "--host" in forwarded
    assert "--port" in forwarded
    assert "--ssl-certfile" not in forwarded
    assert "--ssl-keyfile" not in forwarded


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="entrypoint script not present")
def test_script_appends_ssl_flags_when_enabled(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("dummy")
    key.write_text("dummy")
    env = {
        "ENABLE_HTTPS": "true",
        "SSL_CERTFILE": str(cert),
        "SSL_KEYFILE": str(key),
    }
    result = _run(env, ["cpu_http:app"], tmp_path)
    assert result.returncode == 0, result.stderr
    forwarded = result.stdout.splitlines()
    assert "--ssl-certfile" in forwarded
    assert str(cert) in forwarded
    assert "--ssl-keyfile" in forwarded
    assert str(key) in forwarded


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="entrypoint script not present")
def test_script_fails_when_ssl_files_missing(tmp_path: Path) -> None:
    env = {
        "ENABLE_HTTPS": "1",
        "SSL_CERTFILE": str(tmp_path / "missing-cert.pem"),
        "SSL_KEYFILE": str(tmp_path / "missing-key.pem"),
    }
    result = _run(env, ["cpu_http:app"], tmp_path)
    assert result.returncode != 0
    assert "ENABLE_HTTPS" in result.stderr


@pytest.mark.skipif(not SCRIPT_PATH.exists(), reason="entrypoint script not present")
def test_script_requires_app_argument(tmp_path: Path) -> None:
    result = _run({}, [], tmp_path)
    assert result.returncode != 0
    assert "missing required app module" in result.stderr


@pytest.mark.skipif(
    not SCRIPT_PATH.exists() or shutil.which("/bin/sh") is None,
    reason="entrypoint script or /bin/sh not available",
)
def test_script_passes_through_extra_args(tmp_path: Path) -> None:
    result = _run({}, ["gpu_http:app", "--proxy-headers"], tmp_path)
    assert result.returncode == 0, result.stderr
    forwarded = result.stdout.splitlines()
    assert "--proxy-headers" in forwarded
