"""Pillow-SIMD isolation, compatibility, and explicit numerical tolerance."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from inference_models.models.common import pillow_simd as loader
from inference_models.models.optimization.contracts import (
    ExecutionContext,
    OptimizationStage,
)
from inference_models.models.rfdetr.optimization.backend_path import RFDetrBackendPath
from inference_models.models.rfdetr.optimization.catalog import (
    build_rfdetr_implementation_registry,
)
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.optimization.preprocessors import pillow_simd
from inference_models.models.rfdetr.pre_processing import pre_process_network_input

from .test_backend_execution_plan import config


def _fake_package(tmp_path, *, version="12.3.0.post0", broken=False):
    package = tmp_path / "PIL"
    package.mkdir()
    (package / "__init__.py").write_text(f"__version__ = {version!r}\n")
    (package / "Image.py").write_text(
        "raise RuntimeError('broken native extension')\n"
        if broken
        else "from PIL.Image import fromarray, BILINEAR\ncore = object()\n"
    )
    root = str(tmp_path)

    return root


def test_loader_preserves_standard_pillow(tmp_path):
    """Load an isolated module without replacing standard Pillow.

    Args:
        tmp_path (Path): Temporary native-package fixture root.
    """
    image = loader._load_image(_fake_package(tmp_path))
    assert image is not Image
    assert image.core is not Image.core
    assert image.__name__.startswith("PILSIMD_")
    from PIL import Image as standard

    assert standard is Image


@pytest.mark.parametrize("version", ["12.2.0.post0", "12.3.0"])
def test_loader_rejects_old_or_non_simd_package(tmp_path, version):
    """Reject packages that do not meet the SIMD version contract.

    Args:
        tmp_path (Path): Temporary package root.
        version (str): Unsupported package version to simulate.
    """
    with pytest.raises(ImportError, match="required"):
        loader._load_image(_fake_package(tmp_path, version=version))


def test_broken_native_build_has_import_error(tmp_path):
    """Convert native import failures into actionable compatibility errors.

    Args:
        tmp_path (Path): Temporary package root.
    """
    with pytest.raises(ImportError, match="broken native extension"):
        loader._load_image(_fake_package(tmp_path, broken=True))


def test_arm_selection_does_not_attempt_native_import(monkeypatch):
    """Reject ARM via metadata without loading an x86 extension.

    Args:
        monkeypatch (pytest.MonkeyPatch): Native import replacement fixture.
    """

    def _unexpected():
        pytest.fail("ARM must not import the x86 native extension")

    monkeypatch.setattr(pillow_simd, "load_pillow_simd_image", _unexpected)
    registry = build_rfdetr_implementation_registry(
        device=torch.device("cpu"), preprocessor_max_workers=1, backend="torch"
    )
    selection = registry.resolve_selection(
        stage=OptimizationStage.PREPROCESS,
        requested_id="pillow-simd-v1",
        context=ExecutionContext("cpu", "cpu", host_architecture="aarch64"),
        allow_fallback=True,
    )
    assert selection.effective_id == "base"
    assert "architecture" in selection.fallback_reason


@pytest.mark.parametrize("reason", ["disabled", "absent", "broken native extension"])
def test_missing_build_uses_observable_base(monkeypatch, reason):
    """Expose native-build failures through base-fallback metadata.

    Args:
        monkeypatch (pytest.MonkeyPatch): Host and loader replacement fixture.
        reason (str): Native loader failure to simulate.
    """

    def _unavailable():
        raise ImportError(reason)

    monkeypatch.setattr(pillow_simd, "load_pillow_simd_image", _unavailable)
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    path = RFDetrBackendPath(
        device=torch.device("cpu"),
        inference_config=config(),
        backend="torch",
        execution_plan=RFDetrExecutionPlan(preprocessor_id="pillow-simd-v1"),
    )
    assert path.plan.preprocessor_id == "base"
    assert (
        reason
        in path.runtime_metadata["model_selection"]["preprocessor"]["fallback_reason"]
    )


def test_selected_simd_is_separate_from_reference_and_reports_numerics(monkeypatch):
    """Verify explicit SIMD selection, numerical metadata, and input fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): Native module and host replacement fixture.
    """
    calls = []

    def _fromarray(array):
        calls.append(array.shape)
        image = Image.fromarray(array)

        return image

    monkeypatch.setattr(
        pillow_simd,
        "load_pillow_simd_image",
        lambda: SimpleNamespace(fromarray=_fromarray, BILINEAR=Image.BILINEAR),
    )
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    cfg = config()
    path = RFDetrBackendPath(
        device=torch.device("cpu"),
        inference_config=cfg,
        backend="onnx",
        execution_plan=RFDetrExecutionPlan(preprocessor_id="pillow-simd-v1"),
    )
    image = np.random.default_rng(7).integers(0, 256, (97, 83, 3), dtype=np.uint8)
    actual, _ = path.preprocess(image)
    expected, _ = pre_process_network_input(
        image, cfg.image_pre_processing, cfg.network_input, torch.device("cpu")
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [(97, 83, 3)]
    metadata = path.runtime_metadata
    assert metadata["preprocessor"]["changes_numerics"] is True
    assert (
        metadata["last_execution"]["preprocessor"]["effective_id"] == "pillow-simd-v1"
    )
    json.dumps(metadata)
    # Floating tensors have a different reference contract and must not be cast
    # to uint8 just to use SIMD. Selection records the declared fallback.
    path.preprocess(torch.zeros((3, 97, 83)))
    assert (
        path.runtime_metadata["last_execution"]["preprocessor"]["effective_id"]
        == "base"
    )
    assert calls == [(97, 83, 3)]


@pytest.mark.parametrize("shape", [(32, 32, 3), (160, 200, 3), (480, 640, 3)])
def test_real_simd_resize_tolerance_when_installed(shape):
    """Check native resize tolerance and exact no-op behavior when available.

    Args:
        shape (tuple[int, int, int]): Source image shape.
    """
    try:
        simd = loader.load_pillow_simd_image()
    except ImportError as error:
        pytest.skip(str(error))

    source = np.random.default_rng(23).integers(0, 256, shape, dtype=np.uint8)
    standard = np.asarray(
        Image.fromarray(source).resize((32, 32), Image.BILINEAR)
    ).astype(np.int16)
    actual = np.asarray(simd.fromarray(source).resize((32, 32), simd.BILINEAR)).astype(
        np.int16
    )
    assert np.abs(actual - standard).max() <= 1
    if shape[:2] == (32, 32):
        np.testing.assert_array_equal(actual, standard)
