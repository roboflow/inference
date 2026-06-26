import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.errors import ModelInputError, ModelRuntimeError
from inference_models.models.rfdetr import triton_preprocess
from inference_models.models.rfdetr.triton_preprocess import (
    build_resample_tables,
    resolve_two_pass_launch_config,
    triton_preprocess_rfdetr_stretch,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PREPROC_ENV_VARS = (
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_BLOCK_H",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_BLOCK_W",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_HORIZONTAL_BLOCK_H",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_HORIZONTAL_BLOCK_W",
)


def _reference_preprocess(image_rgb: np.ndarray, target_h: int, target_w: int):
    resized = TF.resize(
        Image.fromarray(image_rgb),
        (target_h, target_w),
        antialias=True,
    )
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(
        tensor,
        mean=list(_IMAGENET_MEAN),
        std=list(_IMAGENET_STD),
    )
    return tensor.unsqueeze(0)


def test_build_resample_tables_shapes_on_cpu() -> None:
    tables = build_resample_tables(
        src_h=11,
        src_w=13,
        target_h=7,
        target_w=5,
        device=torch.device("cpu"),
    )

    assert tuple(tables.ymin_gpu.shape) == (7,)
    assert tuple(tables.xmin_gpu.shape) == (5,)
    assert tuple(tables.wy_gpu.shape) == (7 * tables.ksize_y,)
    assert tuple(tables.wx_gpu.shape) == (5 * tables.ksize_x,)
    assert tables.ymin_gpu.dtype == torch.int32
    assert tables.wx_gpu.dtype == torch.int32


def test_resolve_launch_config_rejects_non_power_of_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in _PREPROC_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_BLOCK_W", "96")

    with pytest.raises(ModelRuntimeError, match="positive power of two"):
        resolve_two_pass_launch_config()


@pytest.mark.skipif(
    not triton_preprocess.TRITON_AVAILABLE,
    reason="Triton is required for runtime validation",
)
def test_triton_preprocess_rejects_cpu_source_tensor() -> None:
    source = torch.zeros((8, 8, 3), dtype=torch.uint8)
    tables = build_resample_tables(
        src_h=8,
        src_w=8,
        target_h=8,
        target_w=8,
        device=torch.device("cpu"),
    )

    with pytest.raises(ModelInputError, match="expected CUDA src tensor"):
        triton_preprocess_rfdetr_stretch(
            src=source,
            tables=tables,
            target_h=8,
            target_w=8,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_preprocess.TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_triton_preprocess_matches_pil_for_rgb_numpy() -> None:
    rng = np.random.default_rng(seed=17)
    image_rgb = rng.integers(0, 256, size=(77, 51, 3), dtype=np.uint8)
    target_h, target_w = 48, 64
    device = torch.device("cuda")
    tables = build_resample_tables(
        src_h=image_rgb.shape[0],
        src_w=image_rgb.shape[1],
        target_h=target_h,
        target_w=target_w,
        device=device,
    )

    actual = triton_preprocess_rfdetr_stretch(
        src=torch.from_numpy(image_rgb).to(device=device),
        tables=tables,
        target_h=target_h,
        target_w=target_w,
        means=_IMAGENET_MEAN,
        stds=_IMAGENET_STD,
        swap_rb=False,
    )
    torch.cuda.synchronize()

    expected = _reference_preprocess(image_rgb, target_h=target_h, target_w=target_w)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_preprocess.TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_triton_preprocess_matches_pil_for_bgr_numpy_with_preallocated_buffers() -> (
    None
):
    rng = np.random.default_rng(seed=23)
    image_rgb = rng.integers(0, 256, size=(63, 85, 3), dtype=np.uint8)
    image_bgr = image_rgb[:, :, ::-1].copy()
    target_h, target_w = 64, 64
    device = torch.device("cuda")
    tables = build_resample_tables(
        src_h=image_bgr.shape[0],
        src_w=image_bgr.shape[1],
        target_h=target_h,
        target_w=target_w,
        device=device,
    )
    out = torch.empty((1, 3, target_h, target_w), dtype=torch.float32, device=device)
    tmp = torch.empty(
        (3, image_bgr.shape[0], target_w),
        dtype=torch.uint8,
        device=device,
    )

    actual = triton_preprocess_rfdetr_stretch(
        src=torch.from_numpy(image_bgr).to(device=device),
        tables=tables,
        target_h=target_h,
        target_w=target_w,
        means=_IMAGENET_MEAN,
        stds=_IMAGENET_STD,
        swap_rb=True,
        out=out,
        tmp=tmp,
    )
    torch.cuda.synchronize()

    expected = _reference_preprocess(image_rgb, target_h=target_h, target_w=target_w)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=0)
