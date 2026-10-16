"""Regressions for native collection and build-flavor selection (no fake NMS)."""

import importlib.util
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

WINDOWS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "packaging_checks", WINDOWS / "packaging_checks.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)

install_spec = importlib.util.spec_from_file_location(
    "install_dependencies", WINDOWS / "install_dependencies.py"
)
installer = importlib.util.module_from_spec(install_spec)
# Resolve the sibling module through pytest.ini pythonpath.
install_spec.loader.exec_module(installer)


@pytest.mark.parametrize("names", [("_C", "image"), ("_C_stable", "image_stable")])
def test_hook_collects_indirect_extensions_and_nested_dlls(tmp_path, names):
    package = tmp_path / "torchvision"
    (package / "libs").mkdir(parents=True)
    # Collection must not import torchvision: that can fail before binaries load.
    (package / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    for name in names:
        (package / f"{name}.pyd").write_bytes(b"native extension fixture")
    (package / "libs" / "jpeg.dll").write_bytes(b"dependent DLL fixture")
    (package / "unrelated.txt").write_text("not a binary")
    code = (
        "import runpy,json; "
        f"hook=runpy.run_path({str(WINDOWS / 'hooks/hook-torchvision.py')!r}); "
        "print(json.dumps(hook['binaries']))"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    binaries = json.loads(result.stdout)
    collected = {
        (Path(source).name, Path(destination).as_posix())
        for source, destination in binaries
    }
    assert collected == {
        (f"{names[0]}.pyd", "torchvision"),
        (f"{names[1]}.pyd", "torchvision"),
        ("jpeg.dll", "torchvision/libs"),
    }


@pytest.mark.parametrize(
    "torch_version,vision_version,cuda_version,distributions",
    [
        ("2.14.1+cpu", "0.29.1+cpu", None, ("2.14.1+cpu", "0.29.1+cpu")),
        ("2.14.1+cu126", "0.26.0+cpu", "12.6", ("2.14.1+cu126", "0.26.0+cpu")),
        ("2.14.1+cu126", "0.29.1+cu126", None, ("2.14.1+cu126", "0.29.1+cu126")),
        ("2.14.1+cu126", "0.29.1+cu126", "12.6", ("2.14.1", "0.29.1")),
    ],
)
def test_gpu_gate_rejects_cpu_wheels_wrong_abi_or_stale_metadata(
    torch_version, vision_version, cuda_version, distributions
):
    with pytest.raises(RuntimeError, match="Wrong gpu torch build"):
        checks.check_build_identity(
            "gpu",
            torch_version=torch_version,
            vision_version=vision_version,
            cuda_version=cuda_version,
            distributions=distributions,
        )


def _wheel(directory, name, version):
    info = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            info + "/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        wheel.writestr(
            info + "/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr(info + "/RECORD", "")
    return path


@pytest.mark.parametrize("flavor", ["cpu", "gpu"])
def test_explicit_pair_replaces_preinstalled_newer_cpu_packages(tmp_path, flavor):
    # Real pip, isolated environment and metadata-only wheels: test selection,
    # not operator execution. Newer public versions reproduce the GPU no-op.
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip = [str(python), "-m", "pip", "install", "--no-index", "--no-deps"]
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    newer = [_wheel(wheels, name, "9.0.0") for name in ("torch", "torchvision")]
    subprocess.run(pip + list(map(str, newer)), check=True, capture_output=True)
    desired = checks.TORCH_BUILDS[flavor][:2]
    selected = tmp_path / "selected"
    selected.mkdir()
    for name, version in zip(("torch", "torchvision"), desired):
        wheel = _wheel(selected, name, version)
        project_index = selected / name
        project_index.mkdir()
        (project_index / "index.html").write_text(
            f'<a href="../{wheel.name}">{wheel.name}</a>'
        )
    pins = installer._install_torch_pair(
        str(python), flavor, index_url=selected.as_uri()
    )
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m,json; "
            "print(json.dumps([m.version('torch'),m.version('torchvision')]))",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert tuple(json.loads(result.stdout)) == desired
    # Later application requirements cannot silently upgrade the constrained pair.
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("\n".join(pins) + "\n")
    subprocess.run(
        pip
        + [
            "-U",
            "--find-links",
            str(wheels),
            "-c",
            str(constraints),
            "torch",
            "torchvision",
        ],
        check=True,
        capture_output=True,
    )
    after = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m,json; "
            "print(json.dumps([m.version('torch'),m.version('torchvision')]))",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert tuple(json.loads(after.stdout)) == desired


def test_smoke_environment_isolates_caches_and_drops_credentials(tmp_path, monkeypatch):
    import smoke_test

    monkeypatch.setenv("ROBOFLOW_API_KEY", "must-not-inherit")
    monkeypatch.setenv("PYTHONPATH", "/build/site-packages")
    monkeypatch.setenv("CUDA_PATH", "/build/cuda")
    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid")
    env = smoke_test._environment(tmp_path, port=9001)
    for key in ("ROBOFLOW_API_KEY", "PYTHONPATH", "CUDA_PATH", "HTTP_PROXY"):
        assert key not in env
    # Independently name the library contracts: merely setting profile env vars
    # does not redirect Windows KnownFolder lookup or all third-party caches.
    for key in (
        "TORCH_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "TRANSFORMERS_CACHE",
        "TLD_EXTRACT_CACHE",
        "MPLCONFIGDIR",
        "MATPLOTLIBCONFIGDIR",
        "YOLO_CONFIG_DIR",
        "MODEL_CACHE_DIR",
        "WIN_PD_OVERRIDE_LOCAL_APPDATA",
    ):
        assert Path(env[key]).is_relative_to(tmp_path)
