import importlib
import os
import sys
import textwrap

import numpy as np
import PIL.Image
import pytest
import torch

from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)

LOADER = "inference_models.models.common.pillow_simd"
PREPROCESSOR = "inference_models.models.rfdetr.pre_processing"
PATH_ENV = "INFERENCE_MODELS_PILLOW_SIMD_PATH"
ALIAS = "PILSIMD"


def _forget_loaded_build():
    sys.modules.pop(LOADER, None)
    for name in [m for m in sys.modules if m.split(".")[0] == ALIAS]:
        del sys.modules[name]


def _import_loader_fresh():
    _forget_loaded_build()
    return importlib.import_module(LOADER)


def _reload_preprocessor():
    _forget_loaded_build()
    return importlib.reload(importlib.import_module(PREPROCESSOR))


@pytest.fixture(autouse=True)
def _restore_modules(monkeypatch):
    yield
    # Leave the preprocessor bound to standard Pillow for the rest of the session.
    monkeypatch.setenv(PATH_ENV, "")
    _reload_preprocessor()


def _write_fake_pillow(root, version="12.2.0.post0", broken=False):
    package_dir = os.path.join(root, "PIL")
    os.makedirs(package_dir)
    with open(os.path.join(package_dir, "__init__.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    with open(os.path.join(package_dir, "Image.py"), "w") as f:
        body = "raise ImportError('no _imaging')\n" if broken else textwrap.dedent("""
                import PIL.Image as _std
                BILINEAR = _std.BILINEAR
                FROMARRAY_CALLS = []

                def fromarray(array):
                    FROMARRAY_CALLS.append(array.shape)
                    return _std.fromarray(array)
                """)
        f.write(body)


def test_import_raises_module_not_found_when_path_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv(PATH_ENV, str(tmp_path / "missing"))

    with pytest.raises(ModuleNotFoundError):
        _import_loader_fresh()
    assert ALIAS not in sys.modules


def test_import_raises_module_not_found_when_path_is_empty(monkeypatch):
    monkeypatch.setenv(PATH_ENV, "")

    with pytest.raises(ModuleNotFoundError):
        _import_loader_fresh()


def test_import_raises_import_error_when_the_build_does_not_load(monkeypatch, tmp_path):
    _write_fake_pillow(tmp_path, broken=True)
    monkeypatch.setenv(PATH_ENV, str(tmp_path))

    with pytest.raises(ImportError):
        _import_loader_fresh()
    assert ALIAS not in sys.modules


def test_import_loads_the_alias_beside_standard_pillow(monkeypatch, tmp_path):
    _write_fake_pillow(tmp_path)
    monkeypatch.setenv(PATH_ENV, str(tmp_path))

    loader = _import_loader_fresh()

    assert loader.Image is not PIL.Image
    assert loader.Image.__name__ == f"{ALIAS}.Image"
    assert sys.modules[ALIAS].__version__ == "12.2.0.post0"
    assert PIL.Image.__name__ == "PIL.Image"


def test_rfdetr_preprocessor_binds_to_the_build_and_falls_back(monkeypatch, tmp_path):
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255,
        normalization=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    )
    rgb = np.random.default_rng(3).integers(0, 256, (160, 200, 3), dtype=np.uint8)

    def run(preprocessor):
        tensor, _ = preprocessor.pre_process_network_input(
            images=rgb[:, :, ::-1].copy(),
            image_pre_processing=ImagePreProcessing(),
            network_input=network_input,
            target_device=torch.device("cpu"),
            input_color_format="bgr",
        )
        return tensor

    _write_fake_pillow(tmp_path)
    monkeypatch.setenv(PATH_ENV, str(tmp_path))
    with_build = _reload_preprocessor()
    assert with_build.Image.__name__ == f"{ALIAS}.Image"
    tensor_with_build = run(with_build)
    assert with_build.Image.FROMARRAY_CALLS == [(160, 200, 3)]

    monkeypatch.setenv(PATH_ENV, "")
    with_pillow = _reload_preprocessor()
    assert with_pillow.Image is PIL.Image
    torch.testing.assert_close(tensor_with_build, run(with_pillow), atol=0, rtol=0)
