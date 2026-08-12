import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.contrast_equalization.v1 import (
    ContrastEqualizationBlockV1,
    ContrastEqualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

ALL_METHODS = [
    "Contrast Stretching",
    "Histogram Equalization",
    "Adaptive Equalization",
]


class TestContrastEqualizationManifest:
    def test_contrast_equalization_validation_when_valid_manifest_is_given(self):
        manifest = ContrastEqualizationManifest.model_validate(
            {
                "type": "roboflow_core/contrast_equalization@v1",
                "name": "contrast_equalization",
                "image": "$inputs.image",
                "equalization_type": "Contrast Stretching",
            }
        )

        assert manifest.type == "roboflow_core/contrast_equalization@v1"
        assert manifest.name == "contrast_equalization"
        assert manifest.equalization_type == "Contrast Stretching"

    def test_contrast_equalization_validation_when_image_is_missing(self):
        with pytest.raises(ValidationError):
            ContrastEqualizationManifest.model_validate(
                {
                    "type": "roboflow_core/contrast_equalization@v1",
                    "name": "contrast_equalization",
                }
            )

    def test_contrast_equalization_manifest_outputs(self):
        outputs = ContrastEqualizationManifest.describe_outputs()

        assert len(outputs) == 1
        assert outputs[0].name == "image"


@pytest.mark.parametrize("equalization_type", ALL_METHODS)
def test_contrast_equalization_block_with_color_image(equalization_type) -> None:
    # given
    rng = np.random.default_rng(11)
    image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=rng.integers(60, 190, size=(48, 64, 3), dtype=np.uint8),
    )

    # when
    result = ContrastEqualizationBlockV1().run(
        image=image_data, equalization_type=equalization_type
    )

    # then
    assert "image" in result
    equalized = result["image"].numpy_image
    assert equalized.shape == (48, 64, 3)
    assert equalized.dtype == np.uint8


@pytest.mark.parametrize("equalization_type", ALL_METHODS)
def test_contrast_equalization_block_with_grayscale_image(equalization_type) -> None:
    # given
    rng = np.random.default_rng(12)
    image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=rng.integers(90, 150, size=(48, 64), dtype=np.uint8),
    )

    # when
    result = ContrastEqualizationBlockV1().run(
        image=image_data, equalization_type=equalization_type
    )

    # then
    assert "image" in result
    equalized = result["image"].numpy_image
    assert equalized.shape == (48, 64)
    assert equalized.dtype == np.uint8


def test_contrast_equalization_block_with_unknown_method() -> None:
    # given
    image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    # when / then
    with pytest.raises(ValueError):
        ContrastEqualizationBlockV1().run(
            image=image_data, equalization_type="Unknown Method"
        )


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: for the two LUT methods a tensor-born image must produce
# bit-identical pixels to the numpy block on the equivalent BGR image; CLAHE
# and numpy-born images delegate to the numpy implementation.


def _tensor_equalization_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.contrast_equalization.v1_tensor import (
        ContrastEqualizationBlockV1 as TensorContrastEqualizationBlockV1,
    )

    return torch, TensorContrastEqualizationBlockV1


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


def _case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if case == "noise":
        return rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    if case == "narrow_range":
        return rng.integers(100, 141, size=(24, 32, 3), dtype=np.uint8)
    if case == "gradient":
        ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
        return np.stack([ramp, ramp[:, ::-1], ramp], axis=-1)
    if case == "skewed":
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        image[rng.random(image.shape) < 0.99] = 77  # p2 == p98 == 77
        return image
    raise ValueError(case)


@pytest.mark.parametrize(
    "equalization_type", ["Contrast Stretching", "Histogram Equalization"]
)
@pytest.mark.parametrize("case", ["noise", "narrow_range", "gradient", "skewed"])
def test_tensor_contrast_equalization_bit_exact_parity(equalization_type, case) -> None:
    # given - the same pixels as numpy-born BGR and tensor-born RGB CHW
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    bgr = _case_image(case)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = (
        ContrastEqualizationBlockV1()
        .run(image=numpy_born, equalization_type=equalization_type)["image"]
        .numpy_image
    )
    tensor_result_image = TensorContrastEqualizationBlockV1().run(
        image=tensor_born, equalization_type=equalization_type
    )["image"]

    # then - bit-exact parity, and the output stays tensor-born
    assert tensor_result_image.is_tensor_materialised()
    assert np.array_equal(tensor_result_image.numpy_image, numpy_result)


@pytest.mark.parametrize(
    "equalization_type", ["Contrast Stretching", "Histogram Equalization"]
)
@pytest.mark.parametrize("value", [0, 130, 255])
def test_tensor_contrast_equalization_flat_grayscale_parity(
    equalization_type, value
) -> None:
    # given - degenerate histogram: single distinct value
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    gray = np.full((16, 20), value, dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = (
        ContrastEqualizationBlockV1()
        .run(image=numpy_born, equalization_type=equalization_type)["image"]
        .numpy_image
    )
    tensor_result = (
        TensorContrastEqualizationBlockV1()
        .run(image=tensor_born, equalization_type=equalization_type)["image"]
        .numpy_image
    )

    # then - tensor path emits (1, H, W) whose numpy view is the (H, W) shape
    assert tensor_result.shape == numpy_result.shape == (16, 20)
    assert np.array_equal(tensor_result, numpy_result)


def test_tensor_contrast_equalization_grayscale_parity() -> None:
    # given - a low-contrast grayscale ramp
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    gray = np.tile(np.linspace(90, 160, 32, dtype=np.uint8), (24, 1))
    numpy_born, tensor_born = _paired_images(torch, gray)

    for equalization_type in ["Contrast Stretching", "Histogram Equalization"]:
        # when
        numpy_result = (
            ContrastEqualizationBlockV1()
            .run(image=numpy_born, equalization_type=equalization_type)["image"]
            .numpy_image
        )
        tensor_result = (
            TensorContrastEqualizationBlockV1()
            .run(image=tensor_born, equalization_type=equalization_type)["image"]
            .numpy_image
        )

        # then
        assert tensor_result.shape == numpy_result.shape == (24, 32)
        assert np.array_equal(tensor_result, numpy_result)


def test_tensor_contrast_equalization_adaptive_delegates_to_numpy_math() -> None:
    # given - CLAHE's mapping is tile-local, so the tensor block delegates to
    # the numpy implementation even for tensor-born images
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    bgr = _case_image("noise")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = (
        ContrastEqualizationBlockV1()
        .run(image=numpy_born, equalization_type="Adaptive Equalization")["image"]
        .numpy_image
    )
    tensor_result = (
        TensorContrastEqualizationBlockV1()
        .run(image=tensor_born, equalization_type="Adaptive Equalization")["image"]
        .numpy_image
    )

    # then
    assert np.array_equal(tensor_result, numpy_result)


def test_tensor_contrast_equalization_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    bgr = _case_image("narrow_range")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ContrastEqualizationBlockV1()
        .run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="parent"),
                numpy_image=bgr,
            ),
            equalization_type="Histogram Equalization",
        )["image"]
        .numpy_image
    )

    # when
    result = TensorContrastEqualizationBlockV1().run(
        image=numpy_born, equalization_type="Histogram Equalization"
    )["image"]

    # then - identical output via the numpy delegate, and no forced H2D
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_contrast_equalization_unknown_method_raises() -> None:
    # given
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros((3, 8, 8), dtype=torch.uint8),
    )

    # when / then
    with pytest.raises(ValueError):
        TensorContrastEqualizationBlockV1().run(
            image=tensor_born, equalization_type="Unknown Method"
        )


def test_tensor_contrast_equalization_on_mps_device(monkeypatch) -> None:
    # given - tensor images land on the globally configured device; simulate an
    # MPS deployment by patching the global
    torch, TensorContrastEqualizationBlockV1 = _tensor_equalization_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    bgr = _case_image("noise")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ContrastEqualizationBlockV1()
        .run(image=numpy_born, equalization_type="Contrast Stretching")["image"]
        .numpy_image
    )
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    result = TensorContrastEqualizationBlockV1().run(
        image=tensor_born, equalization_type="Contrast Stretching"
    )["image"]

    # then - stays on device, and matches the numpy block bit-exactly
    assert result.tensor_image.device.type == "mps"
    assert np.array_equal(result.numpy_image, reference)
