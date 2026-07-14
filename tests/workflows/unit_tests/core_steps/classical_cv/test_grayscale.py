import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1 import (
    ConvertGrayscaleBlockV1,
    ConvertGrayscaleManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_convert_grayscale_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/convert_grayscale@v1",
        "name": "grayscale1",
        images_field_alias: "$inputs.image",
    }

    # when
    result = ConvertGrayscaleManifest.model_validate(data)

    # then
    assert result == ConvertGrayscaleManifest(
        type="roboflow_core/convert_grayscale@v1",
        name="grayscale1",
        image="$inputs.image",
    )


def test_convert_grayscale_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/convert_grayscale@v1",
        "name": "grayscale1",
        "image": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = ConvertGrayscaleManifest.model_validate(data)


def test_convert_grayscale_block() -> None:
    # given
    block = ConvertGrayscaleBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output must be 1 dimensional
    assert output.get("image").numpy_image.shape == (1000, 1000)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: a tensor-born image through the v1_tensor block must produce
# BIT-IDENTICAL pixels to the numpy block on the equivalent BGR image. OpenCV's
# uint8 BGR2GRAY is the per-pixel fixed-point map
# (9798*R + 19235*G + 3735*B + 2^14) >> 15 (verified exhaustively over all 2^24
# RGB triples against the installed cv2), so the tensor path mirrors it
# device-side; numpy-born images and tensor-born single-channel images delegate
# to the numpy implementation.


def _tensor_grayscale_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1_tensor import (
        ConvertGrayscaleBlockV1 as TensorConvertGrayscaleBlockV1,
    )

    return torch, TensorConvertGrayscaleBlockV1


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


@pytest.mark.parametrize(
    "height,width", [(24, 32), (17, 23), (1, 1), (3, 641), (127, 129)]
)
def test_tensor_convert_grayscale_bit_exact_parity(height, width) -> None:
    # given - the same random pixels as numpy-born BGR and tensor-born RGB CHW
    torch, TensorConvertGrayscaleBlockV1 = _tensor_grayscale_imports()
    rng = np.random.default_rng(height * 1000 + width)
    bgr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = ConvertGrayscaleBlockV1().run(image=numpy_born)["image"].numpy_image
    tensor_result_image = TensorConvertGrayscaleBlockV1().run(image=tensor_born)[
        "image"
    ]

    # then - the fixed-point map is a pure function of (R, G, B): bit-exact
    # parity, tensor-born output of shape (1, H, W) whose numpy view is (H, W)
    assert tensor_result_image.is_tensor_materialised()
    assert tuple(tensor_result_image.tensor_image.shape) == (1, height, width)
    assert tensor_result_image.numpy_image.shape == (height, width)
    assert np.array_equal(tensor_result_image.numpy_image, numpy_result)


def test_tensor_convert_grayscale_rounding_boundary_pixels() -> None:
    # given - (B, G, R) triples whose weighted sum lands EXACTLY on the .5
    # luminance boundary (exhaustive sweep: 9798*R + 19235*G + 3735*B == 2^14
    # mod 2^15), where round-half-up and truncation diverge, plus channel
    # extremes; expected grays come straight from cv2 on this exact data
    torch, TensorConvertGrayscaleBlockV1 = _tensor_grayscale_imports()
    boundary_bgr_triples = [
        (4, 12, 0),
        (36, 44, 32),
        (60, 52, 64),
        (147, 251, 95),
        (92, 20, 128),
        (116, 28, 160),
        (203, 227, 191),
        (227, 235, 223),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]
    expected_gray = [8, 40, 57, 193, 61, 78, 214, 231, 29, 150, 76, 255, 0]
    bgr = np.array([boundary_bgr_triples], dtype=np.uint8)  # (1, 13, 3)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = ConvertGrayscaleBlockV1().run(image=numpy_born)["image"].numpy_image
    tensor_result = (
        TensorConvertGrayscaleBlockV1().run(image=tensor_born)["image"].numpy_image
    )

    # then - both paths agree with cv2's round-half-up on the exact boundary
    assert np.array_equal(tensor_result, numpy_result)
    assert np.array_equal(tensor_result, np.array([expected_gray], dtype=np.uint8))


def test_tensor_convert_grayscale_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorConvertGrayscaleBlockV1 = _tensor_grayscale_imports()
    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 256, size=(19, 27, 3), dtype=np.uint8)
    numpy_born, _ = _paired_images(torch, bgr)
    reference = (
        ConvertGrayscaleBlockV1()
        .run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="parent"),
                numpy_image=bgr,
            )
        )["image"]
        .numpy_image
    )

    # when
    result = TensorConvertGrayscaleBlockV1().run(image=numpy_born)["image"]

    # then - identical output via the numpy delegate, and no forced H2D
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_convert_grayscale_raises_on_single_channel_input_like_v1() -> None:
    # given - grayscale input: v1 feeds a 2-D numpy image to cv2.cvtColor, which
    # rejects it; the tensor sibling must fail with the same exception type
    import cv2

    torch, TensorConvertGrayscaleBlockV1 = _tensor_grayscale_imports()
    gray = np.full((8, 8), 77, dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when / then
    with pytest.raises(cv2.error):
        ConvertGrayscaleBlockV1().run(image=numpy_born)
    with pytest.raises(cv2.error):
        TensorConvertGrayscaleBlockV1().run(image=tensor_born)


def test_tensor_convert_grayscale_on_mps_device(monkeypatch) -> None:
    # given - tensor images live on the globally configured device; simulate an
    # MPS deployment by patching the global, then check math runs on-device
    torch, TensorConvertGrayscaleBlockV1 = _tensor_grayscale_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    rng = np.random.default_rng(9)
    bgr = rng.integers(0, 256, size=(33, 47, 3), dtype=np.uint8)
    numpy_born, _ = _paired_images(torch, bgr)
    reference = ConvertGrayscaleBlockV1().run(image=numpy_born)["image"].numpy_image
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    result = TensorConvertGrayscaleBlockV1().run(image=tensor_born)["image"]

    # then - stays on device, and matches the numpy block bit-exactly
    assert result.tensor_image.device.type == "mps"
    assert np.array_equal(result.numpy_image, reference)
