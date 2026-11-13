"""Exercise Python discovery and fail-fast behavior without compiling OpenCV."""

import json
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docker/scripts/build_opencv.sh"


def _run_build(tmp_path, *, enabled="ON", configured="ON", missing_library=False):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("curl", "tar", "sha256sum"):
        tool = tools / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)

    cmake = tools / "cmake"
    cmake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['TEST_ROOT'])\n"
        "if sys.argv[1] == '--build':\n"
        "    (root / 'compilation-started').touch()\n"
        "    sys.exit(43)\n"
        "(root / 'cmake-args.json').write_text(json.dumps(sys.argv[1:]))\n"
        "build = Path(os.environ['OPENCV_BUILD_DIR'])\n"
        "build.mkdir()\n"
        "(build / 'CMakeCache.txt').write_text(\n"
        f"    'HAVE_opencv_python3:INTERNAL={configured}\\n')\n"
    )
    cmake.chmod(0o755)
    python = tmp_path / "selected-python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    if missing_library:
        (tmp_path / "sitecustomize.py").write_text(
            "import sysconfig\n"
            "sysconfig.get_config_vars()['LIBDIR'] = '/missing-python-library'\n"
        )

    env = dict(
        os.environ,
        PATH=f"{tools}:{os.environ['PATH']}",
        PYTHONPATH=str(tmp_path),
        TEST_ROOT=str(tmp_path),
        OPENCV_SHA256="unused-download-stub",
        OPENCV_CONTRIB_SHA256="unused-download-stub",
        OPENCV_PYTHON_EXECUTABLE=str(python),
        OPENCV_BUILD_PYTHON3=enabled,
        OPENCV_WITH_CUDA="OFF",
        OPENCV_BUILD_DIR=str(tmp_path / "build"),
        OPENCV_SOURCE_DIR=str(tmp_path / "source"),
        OPENCV_CONTRIB_SOURCE_DIR=str(tmp_path / "contrib"),
    )
    result = subprocess.run(
        ["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30
    )
    return result


def test_passes_selected_interpreter_library(tmp_path):
    """Pass the selected interpreter's library to CMake before compilation.

    Args:
        tmp_path (Path): Isolated build and tool directory.
    """
    result = _run_build(tmp_path)
    assert result.returncode == 43, result.stderr
    args = json.loads((tmp_path / "cmake-args.json").read_text())
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    assert f"PYTHON3_LIBRARY={library}" in args
    assert (tmp_path / "compilation-started").exists()


def test_missing_library_fails_before_configuration(tmp_path):
    """Reject an interpreter whose reported library does not exist.

    Args:
        tmp_path (Path): Isolated build and tool directory.
    """
    result = _run_build(tmp_path, missing_library=True)
    assert result.returncode != 0
    assert "Python library not found for" in result.stderr
    assert not (tmp_path / "cmake-args.json").exists()
    assert not (tmp_path / "compilation-started").exists()


@pytest.mark.parametrize("configured", ["OFF", ""])
def test_missing_bindings_fail_before_compilation(tmp_path, configured):
    """Reject CMake success when the requested bindings are unavailable.

    Args:
        tmp_path (Path): Isolated build and tool directory.
        configured (str): Simulated Python module availability.
    """
    result = _run_build(tmp_path, configured=configured)
    assert result.returncode == 2
    assert "Python 3 bindings were requested but CMake disabled them" in result.stderr
    assert not (tmp_path / "compilation-started").exists()


def test_explicitly_disabled_bindings_do_not_require_library(tmp_path):
    """Preserve native-only consumers even when libpython is unavailable.

    Args:
        tmp_path (Path): Isolated build and tool directory.
    """
    result = _run_build(tmp_path, enabled="OFF", configured="OFF", missing_library=True)
    assert result.returncode == 43, result.stderr
    assert (tmp_path / "compilation-started").exists()
