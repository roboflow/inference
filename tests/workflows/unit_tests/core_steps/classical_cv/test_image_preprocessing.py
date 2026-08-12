import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1 import (
    ImagePreprocessingBlockV1,
    ImagePreprocessingManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


class TestImagePreprocessingManifest:
    def test_image_preprocessing_validation_when_valid_manifest_is_given(self):
        manifest = ImagePreprocessingManifest.model_validate(
            {
                "type": "roboflow_core/image_preprocessing@v1",
                "name": "image_preprocessing",
                "image": "$inputs.image",
                "task_type": "resize",
                "width": 320,
                "height": 240,
            }
        )

        assert manifest.type == "roboflow_core/image_preprocessing@v1"
        assert manifest.name == "image_preprocessing"
        assert manifest.task_type == "resize"
        assert manifest.width == 320
        assert manifest.height == 240

    def test_image_preprocessing_validation_when_image_is_missing(self):
        with pytest.raises(ValidationError):
            ImagePreprocessingManifest.model_validate(
                {
                    "type": "roboflow_core/image_preprocessing@v1",
                    "name": "image_preprocessing",
                    "task_type": "flip",
                }
            )

    def test_image_preprocessing_manifest_outputs(self):
        outputs = ImagePreprocessingManifest.describe_outputs()

        assert len(outputs) == 1
        assert outputs[0].name == "image"


def _numpy_born_image(bgr: np.ndarray) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=bgr,
    )


def _run_v1(image: WorkflowImageData, **overrides) -> WorkflowImageData:
    kwargs = {
        "task_type": "flip",
        "width": None,
        "height": None,
        "rotation_degrees": None,
        "flip_type": None,
    }
    kwargs.update(overrides)
    return ImagePreprocessingBlockV1().run(image=image, **kwargs)["image"]


def test_image_preprocessing_resize_to_exact_dimensions() -> None:
    # given
    rng = np.random.default_rng(3)
    image = _numpy_born_image(rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8))

    # when
    result = _run_v1(image, task_type="resize", width=32, height=24)

    # then
    assert result.numpy_image.shape == (24, 32, 3)
    assert result.numpy_image.dtype == np.uint8


def test_image_preprocessing_rotate_by_right_angle() -> None:
    # given
    rng = np.random.default_rng(4)
    image = _numpy_born_image(rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8))

    # when
    result = _run_v1(image, task_type="rotate", rotation_degrees=90)

    # then - canvas swaps dimensions for a 90 degrees rotation
    assert result.numpy_image.shape == (64, 48, 3)


def test_image_preprocessing_flip_vertical() -> None:
    # given
    rng = np.random.default_rng(5)
    bgr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    image = _numpy_born_image(bgr)

    # when
    result = _run_v1(image, task_type="flip", flip_type="vertical")

    # then
    assert np.array_equal(result.numpy_image, bgr[::-1])


def test_image_preprocessing_invalid_task_type_raises() -> None:
    # given
    image = _numpy_born_image(np.zeros((10, 10, 3), dtype=np.uint8))

    # when / then
    with pytest.raises(ValueError):
        _run_v1(image, task_type="unknown")


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: outputs match the numpy block bit-exactly on every path.


def _tensor_preprocessing_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1_tensor import (
        ImagePreprocessingBlockV1 as TensorImagePreprocessingBlockV1,
    )

    return torch, TensorImagePreprocessingBlockV1


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


def _run_tensor(
    block_class, image: WorkflowImageData, **overrides
) -> WorkflowImageData:
    kwargs = {
        "task_type": "flip",
        "width": None,
        "height": None,
        "rotation_degrees": None,
        "flip_type": None,
    }
    kwargs.update(overrides)
    return block_class().run(image=image, **kwargs)["image"]


def _case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if case == "color_odd":
        return rng.integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
    if case == "gray_odd":
        return rng.integers(0, 256, size=(7, 9), dtype=np.uint8)
    if case == "color_even":
        return rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    if case == "color_mixed":
        return rng.integers(0, 256, size=(8, 7, 3), dtype=np.uint8)
    if case == "color_larger":
        return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    if case == "checker":
        # 0/1 checkerboard: every 2-tap half-pixel sum is an exact rounding tie
        checker = ((np.indices((7, 9)).sum(axis=0)) % 2).astype(np.uint8)
        return np.stack([checker, 1 - checker, checker], axis=-1)
    raise ValueError(case)


@pytest.mark.parametrize("flip_type", ["vertical", "horizontal", "both"])
@pytest.mark.parametrize("case", ["color_odd", "gray_odd"])
def test_tensor_flip_bit_exact_parity(flip_type, case) -> None:
    # given - the same pixels as numpy-born BGR and tensor-born RGB CHW
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image(case)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="flip", flip_type=flip_type)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="flip",
        flip_type=flip_type,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_flip_unrecognized_type_passes_image_through() -> None:
    # given - flip_type=None is the only run()-reachable pass-through value in v1
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_odd")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="flip", flip_type=None)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1, tensor_born, task_type="flip", flip_type=None
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(numpy_result.numpy_image, bgr)
    assert np.array_equal(tensor_result.numpy_image, bgr)


@pytest.mark.parametrize("rotation_degrees", [90, 180, 270, -90, -180, -270, 360, -360])
@pytest.mark.parametrize("case", ["color_odd", "color_even", "color_mixed", "checker"])
def test_tensor_rotate_right_angles_bit_exact_parity(rotation_degrees, case) -> None:
    # given - the warpAffine mapping differs per axis parity, hence odd/even/mixed dims
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image(case)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(
        numpy_born, task_type="rotate", rotation_degrees=rotation_degrees
    )
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="rotate",
        rotation_degrees=rotation_degrees,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert tensor_result.numpy_image.shape == numpy_result.numpy_image.shape
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


@pytest.mark.parametrize("rotation_degrees", [90, -90, 180, 360])
def test_tensor_rotate_right_angles_grayscale_parity(rotation_degrees) -> None:
    # given
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    gray = _case_image("gray_odd")
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when
    numpy_result = _run_v1(
        numpy_born, task_type="rotate", rotation_degrees=rotation_degrees
    )
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="rotate",
        rotation_degrees=rotation_degrees,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_rotate_arbitrary_angle_delegates_to_numpy_math() -> None:
    # given - arbitrary angles are a real bilinear warp, so the block delegates
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_larger")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="rotate", rotation_degrees=33)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="rotate",
        rotation_degrees=33,
    )

    # then
    assert not tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_rotate_zero_degrees_returns_tensor_copy() -> None:
    # given - v1 early-returns a copy for rotation_degrees=0
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_odd")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="rotate", rotation_degrees=0)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="rotate",
        rotation_degrees=0,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)
    assert np.array_equal(tensor_result.numpy_image, bgr)


@pytest.mark.parametrize(
    "width,height",
    [
        (32, 24),  # downscale to exact dims
        (96, 128),  # upscale to exact dims
        (30, None),  # height derived from aspect ratio: int(48 * 30 / 64) = 22
        (None, 25),  # width derived from aspect ratio: int(64 * 25 / 48) = 33
    ],
)
def test_tensor_resize_delegates_with_identical_output(width, height) -> None:
    # given - INTER_AREA is not bit-exactly replicable in torch, so resizes delegate
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_larger")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="resize", width=width, height=height)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="resize",
        width=width,
        height=height,
    )

    # then
    assert tensor_result.numpy_image.shape == numpy_result.numpy_image.shape
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)


def test_tensor_resize_without_target_dims_returns_tensor_copy() -> None:
    # given - v1 early-returns a copy for width=height=None
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_odd")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_result = _run_v1(numpy_born, task_type="resize", width=None, height=None)
    tensor_result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="resize",
        width=None,
        height=None,
    )

    # then
    assert tensor_result.is_tensor_materialised()
    assert np.array_equal(tensor_result.numpy_image, numpy_result.numpy_image)
    assert np.array_equal(tensor_result.numpy_image, bgr)


@pytest.mark.parametrize(
    "overrides",
    [
        {"task_type": "resize", "width": 0, "height": 24},
        {"task_type": "resize", "width": -5, "height": 24},
        {"task_type": "resize", "width": 32, "height": 0},
        {"task_type": "rotate", "rotation_degrees": 361},
        {"task_type": "rotate", "rotation_degrees": -361},
        {"task_type": "flip", "flip_type": "diagonal"},
        {"task_type": "unknown"},
    ],
)
def test_tensor_validation_errors_match_v1(overrides) -> None:
    # given
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_odd")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when / then
    with pytest.raises(ValueError):
        _run_v1(numpy_born, **overrides)
    with pytest.raises(ValueError):
        _run_tensor(TensorImagePreprocessingBlockV1, tensor_born, **overrides)


def test_tensor_block_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    bgr = _case_image("color_odd")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = _run_v1(
        _numpy_born_image(bgr), task_type="flip", flip_type="vertical"
    ).numpy_image

    # when
    result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        numpy_born,
        task_type="flip",
        flip_type="vertical",
    )

    # then
    assert np.array_equal(result.numpy_image, reference)
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_flip_on_mps_device(monkeypatch) -> None:
    # given - patch the global device so tensor-born images materialise on MPS
    torch, TensorImagePreprocessingBlockV1 = _tensor_preprocessing_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    bgr = _case_image("color_odd")
    numpy_born, _ = _paired_images(torch, bgr)
    reference = _run_v1(
        numpy_born, task_type="flip", flip_type="horizontal"
    ).numpy_image
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    result = _run_tensor(
        TensorImagePreprocessingBlockV1,
        tensor_born,
        task_type="flip",
        flip_type="horizontal",
    )

    # then
    assert result.tensor_image.device.type == "mps"
    assert np.array_equal(result.numpy_image, reference)
