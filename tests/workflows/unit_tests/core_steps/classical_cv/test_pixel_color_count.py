import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    ColorPixelCountManifest,
    PixelationCountBlockV1,
    convert_color_to_bgr_tuple,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_pixelation_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        images_field_alias: "$inputs.image",
        "target_color": (255, 0, 0),  # Add target_color field
    }

    # when
    result = ColorPixelCountManifest.model_validate(data)

    # then
    assert result == ColorPixelCountManifest(
        type="roboflow_core/pixel_color_count@v1",
        name="pixelation1",
        images="$inputs.image",
        target_color=(255, 0, 0),
    )


def test_pixelation_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/pixel_color_count@v1",  # Correct type
        "name": "pixelation1",
        "images": "invalid",
        "target_color": (255, 0, 0),  # Add target_color field
    }

    # when
    with pytest.raises(ValidationError):
        _ = ColorPixelCountManifest.model_validate(data)


def test_pixelation_block() -> None:
    # given
    block = PixelationCountBlockV1()
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[0:100, 0:100] = (0, 0, 245)
    image[0:10, 0:10] = (0, 0, 255)

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        target_color=(255, 0, 0),
        tolerance=10,
    )

    assert output is not None
    assert output["matching_pixels_count"] == 100 * 100, (
        "Expected 100*100 square to be matched, as 100 pixels match dominant color, "
        "and remaining are within margin"
    )


def test_convert_color_to_bgr_tuple_when_valid_tuple_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color=(255, 0, 0))

    # then
    assert result == (0, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_tuple_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color=(256, 0, 0, 0))


def test_convert_color_to_bgr_tuple_when_valid_hex_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="#ff000A")

    # then
    assert result == (10, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_valid_short_hex_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="#f0A")

    # then
    assert result == (170, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_hex_string_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="#invalid")


def test_convert_color_to_bgr_tuple_when_tuple_string_given() -> None:
    # when
    result = convert_color_to_bgr_tuple(color="(255, 0, 128)")

    # then
    assert result == (128, 0, 255), "Expected RGB to be converted into BGR"


def test_convert_color_to_bgr_tuple_when_invalid_tuple_string_given() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="(255, 0, a)")


def test_convert_color_to_bgr_tuple_when_invalid_value() -> None:
    # when
    with pytest.raises(ValueError):
        _ = convert_color_to_bgr_tuple(color="invalid")


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: for tensor-born 3-channel images the v1_tensor block must
# return EXACTLY the count v1 produces on the equivalent BGR numpy image.
# cv2.inRange is an inclusive per-channel integer range test (bounds cvRound-ed
# but NOT saturated to uint8), so the torch mirror is exact integer math and
# only the final scalar count crosses device->host - never the frame.


def _tensor_pixel_count_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1_tensor import (
        PixelationCountBlockV1 as TensorPixelationCountBlockV1,
    )

    return torch, TensorPixelationCountBlockV1


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


def _image_with_planted_target() -> np.ndarray:
    # random background with a planted block of the exact target colour
    # RGB (68, 17, 34) - i.e. BGR (34, 17, 68) - plus pixels at +-10 offsets
    rng = np.random.default_rng(42)
    bgr = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    bgr[0:4, 0:5] = (34, 17, 68)
    bgr[10, 10] = (24, 7, 58)
    bgr[10, 11] = (44, 27, 78)
    return bgr


@pytest.mark.parametrize(
    "target_color",
    ["#441122", "#412", "(68, 17, 34)", (68, 17, 34)],
    ids=["hex_6_digit", "hex_3_digit", "tuple_string", "rgb_tuple"],
)
@pytest.mark.parametrize("tolerance", [0, 10, 64])
def test_tensor_pixel_color_count_exact_parity_across_formats(
    target_color, tolerance
) -> None:
    # given - every target format spells the same colour, RGB (68, 17, 34),
    # which is planted in the image so counts are non-trivial
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    numpy_born, tensor_born = _paired_images(torch, _image_with_planted_target())

    # when
    numpy_count = PixelationCountBlockV1().run(
        image=numpy_born, target_color=target_color, tolerance=tolerance
    )["matching_pixels_count"]
    tensor_count = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=target_color, tolerance=tolerance
    )["matching_pixels_count"]

    # then - exact count parity, and the frame never crossed device->host
    assert isinstance(tensor_count, int)
    assert tensor_count == numpy_count
    assert tensor_count >= 20, "planted 4x5 block must match at any tolerance"
    assert tensor_born.is_tensor_materialised()
    assert tensor_born._numpy_image is None, "tensor path must not materialise numpy"


def test_tensor_pixel_color_count_boundary_inclusivity() -> None:
    # given - target RGB (100, 150, 200) i.e. BGR (200, 150, 100), tolerance
    # 10: pixels exactly AT the lower and upper bounds are inclusive matches;
    # one unit beyond either bound is not
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[0, 0] = (190, 140, 90)  # exactly at lower bound -> match
    bgr[0, 1] = (210, 160, 110)  # exactly at upper bound -> match
    bgr[0, 2] = (200, 150, 100)  # the target itself -> match
    bgr[0, 3] = (189, 140, 90)  # one below lower on one channel -> no match
    bgr[1, 0] = (211, 160, 110)  # one above upper on one channel -> no match
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_count = PixelationCountBlockV1().run(
        image=numpy_born, target_color=(100, 150, 200), tolerance=10
    )["matching_pixels_count"]
    tensor_count = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=(100, 150, 200), tolerance=10
    )["matching_pixels_count"]

    # then - pins the inclusive semantics on both paths
    assert tensor_count == numpy_count == 3


@pytest.mark.parametrize(
    "target_rgb, planted_matching, planted_non_matching",
    [
        # target at 0 with tolerance 10: lower bound -10 is below uint8 range
        # and acts as "no lower limit" (v1 does not clip its bounds)
        ((0, 0, 0), [(0, 0, 0), (10, 10, 10)], [(11, 11, 11)]),
        # target at 255 with tolerance 10: upper bound 265 acts as "no upper limit"
        ((255, 255, 255), [(255, 255, 255), (245, 245, 245)], [(244, 244, 244)]),
    ],
    ids=["lower_bound_below_zero", "upper_bound_above_255"],
)
def test_tensor_pixel_color_count_bounds_beyond_uint8_range(
    target_rgb, planted_matching, planted_non_matching
) -> None:
    # given - a mid-grey background that never matches, plus planted pixels
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    bgr = np.full((6, 6, 3), 128, dtype=np.uint8)
    planted = planted_matching + planted_non_matching
    for index, pixel in enumerate(planted):
        bgr[0, index] = pixel
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_count = PixelationCountBlockV1().run(
        image=numpy_born, target_color=target_rgb, tolerance=10
    )["matching_pixels_count"]
    tensor_count = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=target_rgb, tolerance=10
    )["matching_pixels_count"]

    # then
    assert tensor_count == numpy_count == len(planted_matching)


def test_tensor_pixel_color_count_tolerance_255_matches_everything() -> None:
    # given - tolerance 255 makes every per-channel range cover all of [0, 255]
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    rng = np.random.default_rng(11)
    bgr = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_count = PixelationCountBlockV1().run(
        image=numpy_born, target_color=(200, 3, 254), tolerance=255
    )["matching_pixels_count"]
    tensor_count = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=(200, 3, 254), tolerance=255
    )["matching_pixels_count"]

    # then
    assert tensor_count == numpy_count == 17 * 23


def test_tensor_pixel_color_count_zero_matches() -> None:
    # given - pixel values capped at 200, so a target red channel of 255 with
    # tolerance 10 (range [245, 265]) can never match
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 200, size=(16, 16, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    numpy_count = PixelationCountBlockV1().run(
        image=numpy_born, target_color=(255, 0, 255), tolerance=10
    )["matching_pixels_count"]
    tensor_count = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=(255, 0, 255), tolerance=10
    )["matching_pixels_count"]

    # then
    assert tensor_count == numpy_count == 0


def test_tensor_pixel_color_count_delegates_for_numpy_born_images() -> None:
    # given
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    bgr = _image_with_planted_target()
    numpy_born, _ = _paired_images(torch, bgr)
    reference = PixelationCountBlockV1().run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="parent"),
            numpy_image=bgr,
        ),
        target_color="#441122",
        tolerance=10,
    )["matching_pixels_count"]

    # when
    result = TensorPixelationCountBlockV1().run(
        image=numpy_born, target_color="#441122", tolerance=10
    )["matching_pixels_count"]

    # then - identical count via the numpy delegate, and no forced H2D
    assert result == reference
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


@pytest.mark.parametrize(
    "invalid_color",
    ["invalid", "#zzz", "(1, 2, a)", (255, 0, 0, 0)],
    ids=["plain_string", "bad_hex", "bad_tuple_string", "four_element_tuple"],
)
def test_tensor_pixel_color_count_invalid_color_raises(invalid_color) -> None:
    # given
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when / then - the shared converter raises the identical ValueError on
    # both paths, before any image work
    with pytest.raises(ValueError):
        PixelationCountBlockV1().run(
            image=numpy_born, target_color=invalid_color, tolerance=10
        )
    with pytest.raises(ValueError):
        TensorPixelationCountBlockV1().run(
            image=tensor_born, target_color=invalid_color, tolerance=10
        )
    assert tensor_born._numpy_image is None, "failed run must not materialise numpy"


def test_tensor_pixel_color_count_grayscale_matches_v1_error() -> None:
    # given - v1 on a (H, W) grayscale image raises cv2.error (3-element
    # bounds against a 1-channel image); the sibling delegates tensor-born
    # (1, H, W) images to the numpy path so the failure mode is identical
    import cv2

    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    gray = np.full((8, 9), 128, dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, gray)

    # when / then
    with pytest.raises(cv2.error):
        PixelationCountBlockV1().run(
            image=numpy_born, target_color=(128, 128, 128), tolerance=10
        )
    with pytest.raises(cv2.error):
        TensorPixelationCountBlockV1().run(
            image=tensor_born, target_color=(128, 128, 128), tolerance=10
        )


def test_tensor_pixel_color_count_on_mps_device(monkeypatch) -> None:
    # given - tensor images live on the globally configured device; simulate an
    # MPS deployment by patching the global, then check math runs on-device
    torch, TensorPixelationCountBlockV1 = _tensor_pixel_count_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    bgr = _image_with_planted_target()
    numpy_born, _ = _paired_images(torch, bgr)
    reference = PixelationCountBlockV1().run(
        image=numpy_born, target_color=(68, 17, 34), tolerance=10
    )["matching_pixels_count"]
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    result = TensorPixelationCountBlockV1().run(
        image=tensor_born, target_color=(68, 17, 34), tolerance=10
    )["matching_pixels_count"]

    # then - exact parity with the numpy block, computed on the MPS device
    assert result == reference
    assert tensor_born._numpy_image is None, "tensor path must not materialise numpy"
