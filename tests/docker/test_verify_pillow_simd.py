"""Regression tests for the standalone Pillow-SIMD isolation smoke check."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "docker/scripts/verify_pillow_simd.py"
SPEC = importlib.util.spec_from_file_location("pillow_build_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _package(root, *, version):
    package_dir = root / "PIL"
    package_dir.mkdir(parents=True)
    for filename in ("__init__.py", "Image.py", "_imaging.so"):
        (package_dir / filename).touch()

    image = SimpleNamespace(
        __file__=str(package_dir / "Image.py"),
        core=SimpleNamespace(__file__=str(package_dir / "_imaging.so")),
        fromarray=Image.fromarray,
        Resampling=Image.Resampling,
    )
    package = SimpleNamespace(
        __file__=str(package_dir / "__init__.py"), __version__=version, Image=image
    )
    return package


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """Provide isolated module paths without distribution provenance metadata.

    Args:
        tmp_path (Path): Temporary installation root.
        monkeypatch (pytest.MonkeyPatch): Replace native loading and build policy.

    Returns:
        SimpleNamespace: Standard and SIMD packages for fault injection.
    """
    package = _package(tmp_path / "standard", version="12.3.0")
    root = tmp_path / "simd"
    simd = _package(root, version="12.3.0.post0")
    requirements = tmp_path / "build/requirements"
    requirements.mkdir(parents=True)
    policy = requirements / "_requirements.txt"
    policy.write_text("Pillow>=12.3.0,<13.0.0\n")
    monkeypatch.setattr(verifier, "REPOSITORY_ROOT", requirements.parent)
    monkeypatch.setattr(verifier, "PIL", package)
    monkeypatch.setattr(verifier, "_load_simd", lambda path: (simd, simd.Image))
    result = SimpleNamespace(root=root, package=package, simd=simd, policy=policy)
    return result


def test_valid_installation(installation, capsys):
    """Accept isolated packages without RECORD or Git metadata.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        capsys (pytest.CaptureFixture): Capture the success message.
    """
    verifier._verify(installation.root)
    assert "isolation and resize smoke check ok" in capsys.readouterr().out


def test_standard_version_policy_comes_from_requirements(installation):
    """Read the Pillow compatibility bounds from the repository policy.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
    """
    installation.policy.write_text("Pillow>=13,<14\n")
    with pytest.raises(RuntimeError, match="does not satisfy"):
        verifier._verify(installation.root)


@pytest.mark.parametrize("target", ["standard", "simd"])
@pytest.mark.parametrize("component", ["package", "image", "core"])
def test_rejects_wrong_module_location(installation, target, component):
    """Reject standard modules inside SIMD or SIMD modules outside its directory.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        target (str): Package whose location is incorrect.
        component (str): Module to redirect.
    """
    package = installation.package if target == "standard" else installation.simd
    other = installation.simd if target == "standard" else installation.package
    modules = {"package": package, "image": package.Image, "core": package.Image.core}
    modules[component].__file__ = other.__file__
    with pytest.raises(RuntimeError, match="loads"):
        verifier._verify(installation.root)


def test_rejects_shared_extension_file(installation):
    """Reject different module objects backed by the same physical extension.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
    """
    path = Path(installation.simd.Image.core.__file__)
    path.unlink()
    path.hardlink_to(installation.package.Image.core.__file__)
    with pytest.raises(RuntimeError, match="shares standard Pillow"):
        verifier._verify(installation.root)


@pytest.mark.parametrize("target", ["module", "image", "core"])
@pytest.mark.parametrize("when", ["load", "resize"])
def test_rejects_standard_pillow_mutation(installation, monkeypatch, target, when):
    """Detect replacement of standard modules during loading or resizing.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        monkeypatch (pytest.MonkeyPatch): Restore modified module references.
        target (str): Module reference to replace.
        when (str): Operation during which replacement occurs.
    """

    def _mutate():
        if target == "module":
            monkeypatch.setitem(sys.modules, "PIL.Image", SimpleNamespace())
        elif target == "image":
            monkeypatch.setattr(installation.package, "Image", SimpleNamespace())
        else:
            monkeypatch.setattr(installation.package.Image, "core", SimpleNamespace())

    if when == "load":

        def _load(path):
            _mutate()
            return installation.simd, installation.simd.Image

        monkeypatch.setattr(verifier, "_load_simd", _load)
    else:

        def _fromarray(source):
            _mutate()
            image = Image.fromarray(source)
            return image

        installation.simd.Image.fromarray = _fromarray

    with pytest.raises(RuntimeError, match="replaced standard Pillow"):
        verifier._verify(installation.root)


@pytest.mark.parametrize("fault", ["shape", "dtype", "values"])
def test_resize_smoke_check_rejects_bad_output(installation, fault):
    """Reject invalid shapes, dtypes, or excessive resize differences.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        fault (str): Resize failure to inject.
    """

    def _fromarray(source):
        def _resize(size, resample):
            output = np.asarray(Image.fromarray(source).resize(size, resample)).copy()
            if fault == "shape":
                output = output[:1]
            elif fault == "dtype":
                output = output.astype(np.float32)
            else:
                output.flat[0] = int(output.flat[0]) ^ 128
            return output

        image = SimpleNamespace(resize=_resize)
        return image

    installation.simd.Image.fromarray = _fromarray
    with pytest.raises(RuntimeError, match="resize"):
        verifier._verify(installation.root)


def test_failed_import_cleans_up_alias(installation, monkeypatch):
    """Remove partially loaded aliases after native loading fails.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        monkeypatch (pytest.MonkeyPatch): Inject an import failure.
    """

    def _load(path):
        monkeypatch.setitem(sys.modules, verifier.SIMD_ALIAS, SimpleNamespace())
        monkeypatch.setitem(
            sys.modules, verifier.SIMD_ALIAS + ".Image", SimpleNamespace()
        )
        raise ImportError("native load failed")

    monkeypatch.setattr(verifier, "_load_simd", _load)
    with pytest.raises(ImportError, match="native load failed"):
        verifier._verify(installation.root)
    assert verifier.SIMD_ALIAS not in sys.modules
    assert verifier.SIMD_ALIAS + ".Image" not in sys.modules


def test_existing_alias_is_not_removed(installation, monkeypatch):
    """Refuse an occupied alias without removing its original module.

    Args:
        installation (SimpleNamespace): Isolated module fixture.
        monkeypatch (pytest.MonkeyPatch): Restore the occupied alias.
    """
    original = SimpleNamespace()
    monkeypatch.setitem(sys.modules, verifier.SIMD_ALIAS, original)
    with pytest.raises(RuntimeError, match="alias already in use"):
        verifier._verify(installation.root)
    assert sys.modules[verifier.SIMD_ALIAS] is original


def test_checks_survive_python_optimization():
    """Keep build checks active when Python assertions are disabled."""
    code = f"import runpy; runpy.run_path({str(SCRIPT)!r})['_require'](False, 'guard active')"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "guard active" in result.stderr


def test_dockerfiles_share_the_install_pin():
    """Keep all five Docker installs on the shared requirements file."""
    for suffix in ("cpu", "cpu.dev", "gpu", "gpu.dev", "cu13.gpu"):
        dockerfile = SCRIPT.parents[2] / f"docker/dockerfiles/Dockerfile.onnx.{suffix}"
        contents = dockerfile.read_text()
        assert (
            "--target /opt/pillow_simd -r requirements/requirements.pillow-simd.txt"
            in contents
        )
        assert "python3 docker/scripts/verify_pillow_simd.py" in contents
        assert "git+https://github.com/uploadcare/pillow-simd" not in contents
