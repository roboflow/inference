"""CPU-only checks of standalone verifier order and the actual export guard."""

import ast
import hashlib
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_optional_format_failure_does_not_prevent_primary_jetson_checks():
    path = ROOT / "docker/scripts/verify_jetson_tensor_runtime.py"
    module = ast.parse(path.read_text())
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called = []
    namespace = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        "tempfile": tempfile,
        "Path": Path,
        "os": SimpleNamespace(getenv=lambda name: None),
    }
    for name in (
        "_prepare_h26x_fixture",
        "_prepare_bundled_fixture",
        "_create_jpeg",
        "_validate_source",
        "_validate_numpy_source",
        "_validate_threaded_retrieve",
        "_validate_repeated_grab_advances",
        "_validate_dimension_change_requires_reconnect",
        "_validate_interrupt_unblocks_pull",
    ):
        namespace[name] = lambda *args, _name=name, **kwargs: called.append(_name)

    def jpeg_failure():
        called.append("jpeg")
        raise RuntimeError("unsupported progressive capability")

    namespace["_validate_torchvision_cuda_jpeg"] = jpeg_failure
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    with pytest.raises(RuntimeError, match="progressive"):
        namespace["main"]()
    assert called[-1] == "jpeg"
    assert called.count("_validate_source") == 3
    assert "_validate_repeated_grab_advances" in called
    assert "_validate_interrupt_unblocks_pull" in called


def test_pinned_nvjpeg_export_guard_rejects_later_overwrite(tmp_path):
    dockerfile = (ROOT / "docker/dockerfiles/Dockerfile.media.jetson.7.2.0").read_text()
    # Exercise the guard actually shipped after the CUDA export loop.
    start = dockerfile.index("    test -s /opt/cuda-runtime/lib/libnvjpeg.so.13")
    end = dockerfile.index('    test "$(du -sm', start)
    guard = dockerfile[start:end].replace("\\\n", "").strip().removesuffix("&&").strip()
    library = tmp_path / "lib/libnvjpeg.so.13"
    library.parent.mkdir()
    (tmp_path / "share").mkdir()
    library.write_bytes(b"pinned nvjpeg artifact")
    (tmp_path / "share/libnvjpeg.sha256").write_text(
        hashlib.sha256(library.read_bytes()).hexdigest() + "  " + str(library) + "\n"
    )
    guard = guard.replace("/opt/cuda-runtime", str(tmp_path))
    assert subprocess.run(["sh", "-c", guard], capture_output=True).returncode == 0
    library.write_bytes(b"older overwritten artifact")
    assert subprocess.run(["sh", "-c", guard], capture_output=True).returncode != 0
