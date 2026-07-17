import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.threshold.v1 import (
    ImageThresholdBlockV1,
    ImageThresholdManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_threshold_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/threshold@v1",
        "name": "threshold1",
        images_field_alias: "$inputs.image",
        "threshold_type": "binary",
        "thresh_value": 210,
        "max_value": 255,
    }

    # when
    result = ImageThresholdManifest.model_validate(data)

    # then
    assert result == ImageThresholdManifest(
        type="roboflow_core/threshold@v1",
        name="threshold1",
        image="$inputs.image",
        threshold_type="binary",
        thresh_value=210,
        max_value=255,
    )


def test_threshold_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/threshold@v1",
        "name": "threshold1",
        "image": "invalid",
        "threshold_type": "binary",
        "thresh_value": 210,
        "max_value": 255,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ImageThresholdManifest.model_validate(data)


def test_threshold_block(dogs_image: np.ndarray) -> None:
    # given
    block = ImageThresholdBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        threshold_type="binary",
        thresh_value=210,
        max_value=255,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output must be 3 dimensional
    assert output.get("image").numpy_image.shape == dogs_image.shape
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, dogs_image)


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: outputs match the numpy block bit-exactly on every path.


def _tensor_threshold_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.threshold.v1_tensor import (
        ImageThresholdBlockV1 as TensorImageThresholdBlockV1,
    )

    return torch, TensorImageThresholdBlockV1


def _paired_images(torch, bgr: np.ndarray):
    numpy_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=bgr,
    )
    if bgr.ndim == 2:
        chw = torch.from_numpy(bgr.copy()).unsqueeze(0)
    else:
        chw = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=chw,
    )
    return numpy_born, tensor_born


def _run_threshold_pair(
    block_cls, image: WorkflowImageData, threshold_type: str, thresh_value, max_value
):
    return block_cls().run(
        image=image,
        threshold_type=threshold_type,
        thresh_value=thresh_value,
        max_value=max_value,
    )["image"]


FIXED_THRESHOLD_TYPES = ["binary", "binary_inv", "trunc", "tozero", "tozero_inv"]


@pytest.mark.parametrize("threshold_type", FIXED_THRESHOLD_TYPES)
@pytest.mark.parametrize("thresh_value", [0, 127, 254, 255])
@pytest.mark.parametrize("layout", ["grayscale", "color"])
def test_tensor_threshold_fixed_types_bit_exact_parity(
    threshold_type, thresh_value, layout
) -> None:
    # given - same pixels numpy-born (BGR) and tensor-born (RGB CHW); half the
    # pixels sit in {t-1, t, t+1} to exercise cv2's strict `>` comparison
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    rng = np.random.default_rng(31)
    shape = (24, 32) if layout == "grayscale" else (24, 32, 3)
    noise = rng.integers(0, 256, size=shape)
    band = rng.integers(
        max(0, thresh_value - 1), min(255, thresh_value + 1) + 1, size=shape
    )
    pixels = np.where(rng.random(shape) < 0.5, band, noise).astype(np.uint8)
    numpy_born, tensor_born = _paired_images(torch, pixels)

    # when
    numpy_result = _run_threshold_pair(
        ImageThresholdBlockV1, numpy_born, threshold_type, thresh_value, 255
    ).numpy_image
    tensor_result = _run_threshold_pair(
        TensorImageThresholdBlockV1, tensor_born, threshold_type, thresh_value, 255
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result)


@pytest.mark.parametrize("threshold_type", FIXED_THRESHOLD_TYPES)
@pytest.mark.parametrize(
    "thresh_value,max_value",
    [
        (127, 300),  # maxval saturates to 255
        (127, 254.5),  # cvRound is half-to-even -> 254
        (254, 300),
        (300, 255),  # degenerate: threshold above the uint8 range
        (-5, 255),  # degenerate: negative threshold
    ],
)
def test_tensor_threshold_fixed_types_non_standard_params_parity(
    threshold_type, thresh_value, max_value
) -> None:
    # given - selectors can feed values outside the documented 0-255 ints
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, size=(16, 20), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, pixels)

    # when
    numpy_result = _run_threshold_pair(
        ImageThresholdBlockV1, numpy_born, threshold_type, thresh_value, max_value
    ).numpy_image
    tensor_result = _run_threshold_pair(
        TensorImageThresholdBlockV1,
        tensor_born,
        threshold_type,
        thresh_value,
        max_value,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result)


def _otsu_case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if case == "bimodal":
        dark = np.clip(rng.normal(80, 12, size=(16, 32)), 0, 255)
        bright = np.clip(rng.normal(170, 12, size=(16, 32)), 0, 255)
        return np.concatenate([dark, bright], axis=0).astype(np.uint8)
    if case == "uniform_noise":
        return rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    if case == "flat_zero":
        return np.zeros((16, 20), dtype=np.uint8)
    if case == "flat_mid":
        return np.full((16, 20), 130, dtype=np.uint8)
    if case == "two_adjacent":
        image = np.full((16, 16), 100, dtype=np.uint8)
        image[::2, ::2] = 101
        return image
    raise ValueError(case)


@pytest.mark.parametrize(
    "case", ["bimodal", "uniform_noise", "flat_zero", "flat_mid", "two_adjacent"]
)
def test_tensor_threshold_otsu_bit_exact_parity(case) -> None:
    # given
    import cv2

    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    from inference.core.workflows.core_steps.classical_cv.threshold.v1_tensor import (
        _otsu_threshold_from_counts,
    )

    gray = _otsu_case_image(case)
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = _run_threshold_pair(
        ImageThresholdBlockV1, numpy_born, "otsu", 0, 255
    ).numpy_image
    tensor_result = _run_threshold_pair(
        TensorImageThresholdBlockV1, tensor_born, "otsu", 0, 255
    )

    # then
    expected_threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    counts = np.bincount(gray.reshape(-1), minlength=256).tolist()
    assert _otsu_threshold_from_counts(counts) == int(expected_threshold)
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result)


def test_tensor_threshold_otsu_three_channel_tensor_raises_like_v1() -> None:
    # given - cv2 rejects multi-channel otsu; delegation surfaces the identical error
    import cv2

    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when / then
    with pytest.raises(cv2.error):
        _run_threshold_pair(ImageThresholdBlockV1, numpy_born, "otsu", 0, 255)
    with pytest.raises(cv2.error):
        _run_threshold_pair(TensorImageThresholdBlockV1, tensor_born, "otsu", 0, 255)


def _adaptive_case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(17)
    if case == "noise":
        return rng.integers(0, 256, size=(24, 32), dtype=np.uint8)
    if case == "gradient":
        return np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    if case == "odd_dims":
        return rng.integers(0, 256, size=(37, 53), dtype=np.uint8)
    if case == "tiny":
        # smaller than the 11x11 window - cv2's replicate border still works
        return rng.integers(0, 256, size=(3, 5), dtype=np.uint8)
    if case == "boundary_hugging":
        # src within 2 of the local mean - off-by-one mean errors flip the decision
        base = np.full((32, 32), 100, dtype=np.int32)
        return np.clip(base + rng.integers(-2, 3, size=base.shape), 0, 255).astype(
            np.uint8
        )
    raise ValueError(case)


@pytest.mark.parametrize(
    "case", ["noise", "gradient", "odd_dims", "tiny", "boundary_hugging"]
)
@pytest.mark.parametrize("max_value", [255, 300])
def test_tensor_threshold_adaptive_mean_bit_exact_parity(case, max_value) -> None:
    # given
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    gray = _adaptive_case_image(case)
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = _run_threshold_pair(
        ImageThresholdBlockV1, numpy_born, "adaptive_mean", 127, max_value
    ).numpy_image
    tensor_result = _run_threshold_pair(
        TensorImageThresholdBlockV1, tensor_born, "adaptive_mean", 127, max_value
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result)


def test_tensor_threshold_adaptive_gaussian_delegates_to_numpy_math() -> None:
    # given - cv2's float32 Gaussian mean is SIMD-dispatch dependent, so it delegates
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    gray = _adaptive_case_image("noise")
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = _run_threshold_pair(
        ImageThresholdBlockV1, numpy_born, "adaptive_gaussian", 127, 255
    ).numpy_image
    tensor_result = _run_threshold_pair(
        TensorImageThresholdBlockV1, tensor_born, "adaptive_gaussian", 127, 255
    )

    # then
    assert np.array_equal(tensor_result.numpy_image, numpy_result)


def test_tensor_threshold_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    rng = np.random.default_rng(23)
    gray = rng.integers(0, 256, size=(24, 32), dtype=np.uint8)
    numpy_born, _ = _paired_images(torch, gray)
    reference = _run_threshold_pair(
        ImageThresholdBlockV1,
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="parent"),
            numpy_image=gray,
        ),
        "binary",
        127,
        255,
    ).numpy_image

    # when
    result = _run_threshold_pair(
        TensorImageThresholdBlockV1, numpy_born, "binary", 127, 255
    )

    # then
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_threshold_unknown_type_raises_on_tensor_path() -> None:
    # given
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros((1, 8, 8), dtype=torch.uint8),
    )

    # when / then
    with pytest.raises(ValueError):
        _run_threshold_pair(
            TensorImageThresholdBlockV1, tensor_born, "unknown_type", 127, 255
        )


def test_tensor_threshold_on_mps_device(monkeypatch) -> None:
    # given - patch the global device so tensor-born images materialise on MPS
    torch, TensorImageThresholdBlockV1 = _tensor_threshold_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    gray = _otsu_case_image("bimodal")
    numpy_born, _ = _paired_images(torch, gray)
    references = {
        threshold_type: _run_threshold_pair(
            ImageThresholdBlockV1, numpy_born, threshold_type, 127, 255
        ).numpy_image
        for threshold_type in ["binary", "otsu"]
    }
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, gray)
    assert tensor_born.tensor_image.device.type == "mps"

    for threshold_type, reference in references.items():
        # when
        result = _run_threshold_pair(
            TensorImageThresholdBlockV1, tensor_born, threshold_type, 127, 255
        )

        # then
        assert result.tensor_image.device.type == "mps"
        assert np.array_equal(result.numpy_image, reference)
