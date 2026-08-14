import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.image_blur.v1 import (
    ImageBlurBlockV1,
    ImageBlurManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_image_blur_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/image_blur@v1",
        "name": "blur1",
        images_field_alias: "$inputs.image",
        "blur_type": "gaussian",
        "kernel_size": 5,
    }

    # when
    result = ImageBlurManifest.model_validate(data)
    print(result)

    # then
    assert result == ImageBlurManifest(
        type="roboflow_core/image_blur@v1",
        name="blur1",
        image="$inputs.image",
        blur_type="gaussian",
        kernel_size=5,
    )


def test_image_blur_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/image_blur@v1",
        "name": "image_blur1",
        "image": "invalid",
        "blur_type": "gaussian",
        "kernel_size": 5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = ImageBlurManifest.model_validate(data)


def test_image_blur_block() -> None:
    # given
    block = ImageBlurBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        blur_type="gaussian",
        kernel_size=5,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: outputs match the numpy block bit-exactly on every path.


def _tensor_blur_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.image_blur.v1_tensor import (
        ImageBlurBlockV1 as TensorImageBlurBlockV1,
    )

    return torch, TensorImageBlurBlockV1


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


def _blur_case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if case == "noise_color":
        return rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    if case == "noise_gray":
        return rng.integers(0, 256, size=(41, 29), dtype=np.uint8)
    if case == "extremes":
        return rng.choice([0, 255], size=(32, 24, 3)).astype(np.uint8)
    if case == "gradient":
        ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
        return np.stack([ramp, ramp[:, ::-1], 255 - ramp], axis=-1)
    raise ValueError(case)


def _run_both_blur_blocks(
    torch, tensor_block_class, bgr: np.ndarray, blur_type: str, kernel_size: int
):
    numpy_born, tensor_born = _paired_images(torch, bgr)
    numpy_result = ImageBlurBlockV1().run(
        image=numpy_born, blur_type=blur_type, kernel_size=kernel_size
    )["image"]
    tensor_result = tensor_block_class().run(
        image=tensor_born, blur_type=blur_type, kernel_size=kernel_size
    )["image"]
    return numpy_result, tensor_result


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 9, 15])
@pytest.mark.parametrize("case", ["noise_color", "noise_gray", "extremes"])
def test_tensor_image_blur_average_bit_exact_parity(kernel_size, case) -> None:
    # given - the same pixels as numpy-born BGR and tensor-born RGB CHW
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image(case)

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "average", kernel_size
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


@pytest.mark.parametrize("kernel_size", [1, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("case", ["noise_color", "noise_gray", "gradient"])
def test_tensor_image_blur_gaussian_bit_exact_parity(kernel_size, case) -> None:
    # given - even sizes exercise v1's _to_positive_odd coercion (4 -> 5, 6 -> 7)
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image(case)

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "gaussian", kernel_size
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


@pytest.mark.parametrize("kernel_size", [2, 3, 5, 9, 15])
@pytest.mark.parametrize("case", ["noise_color", "noise_gray", "extremes"])
def test_tensor_image_blur_median_bit_exact_parity(kernel_size, case) -> None:
    # given - kernel_size=2 exercises v1's _to_positive_odd coercion (2 -> 3)
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image(case)

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "median", kernel_size
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


@pytest.mark.parametrize("kernel_size", [2, 4])
def test_tensor_image_blur_average_even_kernel_ties_delegate(kernel_size) -> None:
    # given - 0/1 checkerboard: every window mean is an exact .5 rounding tie;
    # cv2's tie rounding is build-specific, so even kernels delegate
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    yy, xx = np.mgrid[0:16, 0:20]
    checker = ((yy + xx) % 2).astype(np.uint8)

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, checker, "average", kernel_size
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_image_blur_gaussian_large_kernel_delegates() -> None:
    # given - kernel 9 is beyond the small_gaussian_tab regime, so gaussian delegates
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image("noise_color")

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "gaussian", 9
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_image_blur_median_large_kernel_delegates() -> None:
    # given - median kernels above 15 delegate
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image("noise_color")

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "median", 17
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_image_blur_bilateral_delegates() -> None:
    # given - bilateral always delegates (data-dependent float weights)
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image("noise_color")

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "bilateral", 5
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_image_blur_oversized_kernel_regime_delegates() -> None:
    # given - kernel pad 7 reaches the 6-pixel image height, so this regime delegates
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, size=(6, 9, 3), dtype=np.uint8)

    # when
    numpy_result, tensor_result = _run_both_blur_blocks(
        torch, TensorImageBlurBlockV1, bgr, "average", 15
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


@pytest.mark.parametrize("blur_type", ["average", "gaussian", "median", "bilateral"])
def test_tensor_image_blur_delegates_for_numpy_born_images(blur_type) -> None:
    # given
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    bgr = _blur_case_image("noise_color")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ImageBlurBlockV1()
        .run(image=numpy_born, blur_type=blur_type, kernel_size=5)["image"]
        .numpy_image
    )

    # when
    result = TensorImageBlurBlockV1().run(
        image=numpy_born, blur_type=blur_type, kernel_size=5
    )["image"]

    # then
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_image_blur_unknown_blur_type_raises() -> None:
    # given
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=torch.zeros((3, 8, 8), dtype=torch.uint8),
    )

    # when / then
    with pytest.raises(ValueError, match="Unknown blur type"):
        TensorImageBlurBlockV1().run(
            image=tensor_born, blur_type="motion", kernel_size=5
        )


def test_tensor_image_blur_on_mps_device(monkeypatch) -> None:
    # given - patch the global device so tensor-born images materialise on MPS
    torch, TensorImageBlurBlockV1 = _tensor_blur_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    bgr = _blur_case_image("noise_color")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ImageBlurBlockV1()
        .run(image=numpy_born, blur_type="average", kernel_size=5)["image"]
        .numpy_image
    )
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    result = TensorImageBlurBlockV1().run(
        image=tensor_born, blur_type="average", kernel_size=5
    )["image"]

    # then
    assert result.tensor_image.device.type == "mps"
    assert np.array_equal(result.numpy_image, reference)
