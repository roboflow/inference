import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.classical_cv.camera_focus import v2 as camera_focus
from roboflow_workflows.core_steps.classical_cv.camera_focus.v2 import (
    CameraFocusBlockV2,
    CameraFocusManifest,
    _hud_geometry,
    _tenengrad,
    _to_gray,
)
from roboflow_workflows.core_steps.common.tensor_native import (
    HOST_CLASS_ID_KEY,
    HOST_CONFIDENCE_KEY,
    HOST_XYXY_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _reference_zebra_overlay(
    image: np.ndarray,
    gray: np.ndarray,
    under_thresh: int = 16,
    over_thresh: int = 239,
    opacity: float = 0.5,
) -> np.ndarray:
    """The original overlay arithmetic, kept as a pixel-parity reference."""
    output = image.copy()
    y, x = np.ogrid[: gray.shape[0], : gray.shape[1]]
    zebra = ((x + y) // 8) % 2 == 0
    for mask, color in (
        ((gray < under_thresh) & zebra, (255, 0, 0)),
        ((gray > over_thresh) & zebra, (0, 0, 255)),
    ):
        output[mask] = (
            output[mask] * (1 - opacity) + np.array(color, dtype=np.uint8) * opacity
        ).astype(np.uint8)
    return output


@pytest.mark.parametrize("opacity", [0, 0.3, 0.5, 0.6, 1])
@pytest.mark.parametrize("thresholds", [(16, 239), (200, 100), (256, -1), (0, 255)])
@pytest.mark.parametrize("strided", [False, True])
def test_zebra_overlay_preserves_pixels_and_inputs(
    opacity: float, thresholds: tuple, strided: bool
) -> None:
    values = np.tile(np.arange(256, dtype=np.uint8), (17, 1))
    image = np.stack([values, np.roll(values, 1), np.roll(values, 2)], axis=-1)
    gray = values.copy()
    if strided:
        image, gray = image[::2, ::-1], gray[::2, ::-1]
    before_image, before_gray = image.copy(), gray.copy()
    image.flags.writeable = False
    gray.flags.writeable = False
    under_thresh, over_thresh = thresholds

    actual = camera_focus._apply_zebra_warnings(
        image,
        gray=gray,
        under_thresh=under_thresh,
        over_thresh=over_thresh,
        opacity=opacity,
    )
    expected = _reference_zebra_overlay(
        image,
        gray=gray,
        under_thresh=under_thresh,
        over_thresh=over_thresh,
        opacity=opacity,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(gray, before_gray)
    assert actual.dtype == image.dtype
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize(
    "dtype, opacity, shape",
    [
        (np.float32, 0.3, (17, 29, 3)),
        (np.uint16, 0.5, (17, 29, 3)),
        (np.uint8, np.float32(0.3), (17, 29, 3)),
        (np.uint8, -0.5, (17, 29, 3)),
        (np.uint8, 1.5, (17, 29, 3)),
        (np.uint8, 0.5, (0, 3, 3)),
    ],
)
def test_zebra_overlay_preserves_other_input_arithmetic(
    dtype: type, opacity: float, shape: tuple
) -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, shape, dtype=np.uint8).astype(dtype)
    gray = rng.integers(0, 256, shape[:2], dtype=np.uint8)

    actual = camera_focus._apply_zebra_warnings(image, gray=gray, opacity=opacity)
    expected = _reference_zebra_overlay(image, gray=gray, opacity=opacity)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == image.dtype


@pytest.mark.parametrize("spacing", [1, 3, 8, 16])
def test_zebra_mask_preserves_phase_beyond_byte_range(spacing: int) -> None:
    shape = (517, 1027)
    y, x = np.ogrid[: shape[0], : shape[1]]

    actual = camera_focus._create_zebra_mask(shape, spacing=spacing)

    np.testing.assert_array_equal(actual, ((x + y) // spacing) % 2 == 0)


@pytest.fixture
def isolated_zebra_caches():
    camera_focus._cached_zebra_mask.cache_clear()
    camera_focus._zebra_blend_tables.cache_clear()
    yield
    camera_focus._cached_zebra_mask.cache_clear()
    camera_focus._zebra_blend_tables.cache_clear()


def test_zebra_caches_are_bounded_and_read_only(isolated_zebra_caches) -> None:
    mask = camera_focus._cached_zebra_mask((17, 29))
    assert camera_focus._cached_zebra_mask((17, 29)) is mask
    assert not mask.flags.writeable
    camera_focus._cached_zebra_mask((29, 17))
    assert camera_focus._cached_zebra_mask.cache_info().currsize == 1

    for opacity in np.linspace(0, 1, 10):
        for table in camera_focus._zebra_blend_tables(float(opacity)):
            assert not table.flags.writeable
    assert camera_focus._zebra_blend_tables.cache_info().currsize == 8


def test_zebra_overlay_does_not_cache_oversized_masks(
    isolated_zebra_caches, monkeypatch
) -> None:
    monkeypatch.setattr(camera_focus, "_MAX_CACHED_ZEBRA_MASK_BYTES", 100)
    image = np.zeros((11, 10, 3), dtype=np.uint8)
    gray = np.zeros((11, 10), dtype=np.uint8)

    actual = camera_focus._apply_zebra_warnings(image, gray=gray)

    np.testing.assert_array_equal(actual, _reference_zebra_overlay(image, gray=gray))
    assert camera_focus._cached_zebra_mask.cache_info().currsize == 0


@pytest.mark.parametrize("grayscale", [False, True])
def test_camera_focus_v2_visualization_matches_original_overlay(
    grayscale: bool, monkeypatch
) -> None:
    rng = np.random.default_rng(7)
    shape = (53, 71) if grayscale else (53, 71, 3)
    pixels = rng.integers(0, 256, shape, dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=pixels.copy(),
    )
    detections = sv.Detections(xyxy=np.array([[3, 4, 30, 40]]))
    parameters = {
        **_ALL_VISUALIZATIONS,
        "underexposed_threshold_percent": 80.0,
        "overexposed_threshold_percent": 20.0,
    }

    actual = CameraFocusBlockV2().run(image=image, detections=detections, **parameters)
    monkeypatch.setattr(camera_focus, "_apply_zebra_warnings", _reference_zebra_overlay)
    expected = CameraFocusBlockV2().run(
        image=image, detections=detections, **parameters
    )

    assert actual.keys() == expected.keys()
    np.testing.assert_array_equal(
        actual["image"].numpy_image, expected["image"].numpy_image
    )
    np.testing.assert_array_equal(image.numpy_image, pixels)
    assert actual["focus_measure"] == expected["focus_measure"]
    assert actual["bbox_focus_measures"] == expected["bbox_focus_measures"]
    assert actual["image"].parent_metadata == expected["image"].parent_metadata


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_camera_focus_v2_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    data = {
        "type": "roboflow_core/camera_focus@v2",
        "name": "camera_focus",
        images_field_alias: "$inputs.image",
    }

    result = CameraFocusManifest.model_validate(data)

    assert result == CameraFocusManifest(
        type="roboflow_core/camera_focus@v2",
        name="camera_focus",
        image="$inputs.image",
    )


def test_camera_focus_v2_validation_when_invalid_image_is_given() -> None:
    data = {
        "type": "roboflow_core/camera_focus@v2",
        "name": "image_contours",
        "image": "invalid",
    }

    with pytest.raises(ValidationError):
        _ = CameraFocusManifest.model_validate(data)


def test_camera_focus_v2_block(dogs_image: np.ndarray) -> None:
    block = CameraFocusBlockV2()

    start_image = dogs_image

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=None,
    )

    assert output is not None
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert "bbox_focus_measures" in output
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_detections(dogs_image: np.ndarray) -> None:
    block = CameraFocusBlockV2()

    start_image = dogs_image

    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 10, 100, 100],
                [150, 150, 300, 300],
            ]
        ),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=detections,
    )

    assert output is not None
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert "bbox_focus_measures" in output
    assert len(output["bbox_focus_measures"]) == 2
    assert all(fm >= 0 for fm in output["bbox_focus_measures"])


def test_camera_focus_v2_block_returns_same_image_when_all_visualizations_disabled(
    dogs_image: np.ndarray,
) -> None:
    block = CameraFocusBlockV2()

    input_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=dogs_image,
    )

    output = block.run(
        image=input_image,
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=None,
    )

    assert output is not None
    assert output["image"] is input_image
    assert "focus_measure" in output
    assert output["focus_measure"] >= 0
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_grayscale_image() -> None:
    block = CameraFocusBlockV2()
    gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=gray_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=None,
    )

    assert output["focus_measure"] >= 0
    assert output["image"].numpy_image.shape == (100, 100, 3)


def test_camera_focus_v2_block_with_small_image() -> None:
    block = CameraFocusBlockV2()
    small_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=small_image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
        detections=None,
    )

    assert output["focus_measure"] >= 0
    assert output["bbox_focus_measures"] == []


def test_camera_focus_v2_block_with_out_of_bounds_detections() -> None:
    block = CameraFocusBlockV2()
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array(
            [
                [-50, -50, 50, 50],
                [80, 80, 200, 200],
            ]
        ),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=detections,
    )

    assert output["focus_measure"] >= 0
    assert len(output["bbox_focus_measures"]) == 2
    assert all(fm >= 0 for fm in output["bbox_focus_measures"] if fm is not None)


def test_camera_focus_v2_block_with_completely_out_of_bounds_detections() -> None:
    block = CameraFocusBlockV2()
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    detections = sv.Detections(
        xyxy=np.array(
            [
                [200, 200, 300, 300],
                [-100, -100, -50, -50],
                [10, 10, 50, 50],
            ]
        ),
    )

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=image,
        ),
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=False,
        grid_overlay="None",
        show_hud=False,
        show_focus_peaking=False,
        show_center_marker=False,
        detections=detections,
    )

    assert output["focus_measure"] >= 0
    assert len(output["bbox_focus_measures"]) == 3
    assert output["bbox_focus_measures"][0] is None
    assert output["bbox_focus_measures"][1] is None
    assert output["bbox_focus_measures"][2] is not None
    assert output["bbox_focus_measures"][2] >= 0


# --- device path -------------------------------------------------------------
# `run()` picks it up automatically for a tensor-materialised image, so these
# assert it against the numpy path on the same pixels.

_ALL_VISUALIZATIONS = {
    "underexposed_threshold_percent": 3.0,
    "overexposed_threshold_percent": 97.0,
    "show_zebra_warnings": True,
    "grid_overlay": "3x3",
    "show_hud": True,
    "show_focus_peaking": True,
    "show_center_marker": True,
}
_NO_VISUALIZATIONS = {
    **_ALL_VISUALIZATIONS,
    "show_zebra_warnings": False,
    "grid_overlay": "None",
    "show_hud": False,
    "show_focus_peaking": False,
    "show_center_marker": False,
}


def _paired_images(image: np.ndarray):
    """The same pixels as a numpy-born BGR image and a tensor-born RGB CHW one."""
    parent_metadata = ImageParentMetadata(parent_id="some")
    chw = (
        torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1)
        if image.ndim == 3
        else torch.from_numpy(image.copy()).unsqueeze(0)
    )
    return (
        WorkflowImageData(parent_metadata=parent_metadata, numpy_image=image.copy()),
        WorkflowImageData(parent_metadata=parent_metadata, tensor_image=chw),
    )


class _NativeDetections:
    """Stand-in for an `inference_models` prediction: `xyxy` is a torch tensor."""

    def __init__(self, xyxy: np.ndarray, bboxes_metadata=None):
        self.xyxy = torch.from_numpy(xyxy)
        self.bboxes_metadata = bboxes_metadata
        self._rows = int(xyxy.shape[0])

    def __len__(self) -> int:
        return self._rows


def _assert_parity(numpy_output: dict, tensor_output: dict) -> None:
    """Overlays must be bit-exact, except for the HUD's own `%.1f` readout.

    The device path accumulates the mean more widely than numpy's float32
    `.mean()`, so the focus value can differ in its last ulp. A value on a
    `%.1f` rounding boundary then renders a different string, which can also
    re-lay the panel out since its width feeds the panel width. When that
    happens, everything outside the union of the two panel rects must still
    match exactly.
    """
    numpy_focus, tensor_focus = (
        numpy_output["focus_measure"],
        tensor_output["focus_measure"],
    )
    assert tensor_focus == pytest.approx(numpy_focus, rel=1e-5)
    numpy_image = numpy_output["image"].numpy_image
    tensor_image = tensor_output["image"].numpy_image
    assert numpy_image.shape == tensor_image.shape
    if f"{numpy_focus:.1f}" == f"{tensor_focus:.1f}":
        assert np.array_equal(numpy_image, tensor_image)
        return
    height, width = numpy_image.shape[:2]
    comparable = np.ones((height, width), dtype=bool)
    for focus_value in (numpy_focus, tensor_focus):
        x, y, panel_width, panel_height = _hud_geometry(height, width, focus_value)[:4]
        comparable[
            max(0, y - 1) : y + panel_height + 2, max(0, x - 1) : x + panel_width + 2
        ] = False
    assert np.array_equal(numpy_image[comparable], tensor_image[comparable])


@pytest.mark.parametrize("visualizations", [_ALL_VISUALIZATIONS, _NO_VISUALIZATIONS])
@pytest.mark.parametrize(
    "shape", [(427, 640, 3), (100, 100), (10, 10, 3), (1, 50, 3), (50, 1, 3)]
)
def test_camera_focus_v2_device_path_matches_numpy(
    shape: tuple, visualizations: dict
) -> None:
    # given - colour, grayscale, and the sub-2px frames that fall back to numpy
    image = np.random.RandomState(0).randint(0, 256, shape, dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(image)

    # when
    numpy_output = CameraFocusBlockV2().run(
        image=numpy_born, detections=None, **visualizations
    )
    tensor_output = CameraFocusBlockV2().run(
        image=tensor_born, detections=None, **visualizations
    )

    # then
    _assert_parity(numpy_output, tensor_output)


def test_camera_focus_v2_device_focus_measure_is_bit_exact_with_cv2_sobel(
    dogs_image: np.ndarray,
) -> None:
    # given - the load-bearing parity claim: integral kernels over uint8 stay
    # exactly representable in float32
    _, tensor_born = _paired_images(dogs_image)
    expected_gray = cv2.cvtColor(dogs_image, cv2.COLOR_BGR2GRAY)
    expected = cv2.Sobel(expected_gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 + (
        cv2.Sobel(expected_gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    )

    # when
    gray = _to_gray(tensor_born.tensor_image)

    # then
    assert np.array_equal(gray.cpu().numpy(), expected_gray)
    assert np.array_equal(_tenengrad(gray).cpu().numpy(), expected)


def test_camera_focus_v2_device_path_keeps_frame_on_device(
    dogs_image: np.ndarray,
) -> None:
    # given
    _, tensor_born = _paired_images(dogs_image)

    # when
    with_overlays = CameraFocusBlockV2().run(
        image=tensor_born, detections=None, **_ALL_VISUALIZATIONS
    )
    without_overlays = CameraFocusBlockV2().run(
        image=tensor_born, detections=None, **_NO_VISUALIZATIONS
    )

    # then - no full-frame device->host materialisation on either side, and the
    # no-overlay case still hands back the very same object
    assert tensor_born._numpy_image is None
    assert with_overlays["image"]._numpy_image is None
    assert with_overlays["image"]._tensor_image is not None
    assert without_overlays["image"] is tensor_born


@pytest.mark.parametrize("mirrored", [False, True])
def test_camera_focus_v2_device_bbox_focus_parity(
    dogs_image: np.ndarray, mirrored: bool
) -> None:
    # given - in-bounds, partially out-of-bounds, fully out-of-bounds and
    # degenerate boxes, as sv for the numpy path and native for the device path.
    # `mirrored` feeds the per-box host mirror instead of the device `xyxy`.
    boxes = np.array(
        [
            [10, 10, 100, 100],
            [-50, -50, 50, 50],
            [2000, 2000, 3000, 3000],
            [5, 5, 5, 20],
        ],
        dtype=np.float32,
    )
    numpy_born, tensor_born = _paired_images(dogs_image)
    if mirrored:
        native = _NativeDetections(
            boxes.copy(),
            bboxes_metadata=[
                {
                    HOST_XYXY_KEY: [float(value) for value in box],
                    HOST_CLASS_ID_KEY: 0,
                    HOST_CONFIDENCE_KEY: 0.9,
                }
                for box in boxes
            ],
        )
        native.xyxy = None  # any device read would now raise
    else:
        native = _NativeDetections(boxes.copy())

    # when
    numpy_output = CameraFocusBlockV2().run(
        image=numpy_born,
        detections=sv.Detections(xyxy=boxes.copy()),
        **_NO_VISUALIZATIONS,
    )
    tensor_output = CameraFocusBlockV2().run(
        image=tensor_born, detections=native, **_NO_VISUALIZATIONS
    )

    # then - `None` entries stay in position and the rest match
    assert len(tensor_output["bbox_focus_measures"]) == len(boxes)
    for expected, actual in zip(
        numpy_output["bbox_focus_measures"], tensor_output["bbox_focus_measures"]
    ):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-5)


def test_camera_focus_v2_device_path_on_mps(
    dogs_image: np.ndarray, monkeypatch
) -> None:
    # given - MPS has no float64, so the focus reduction picks its dtype by device
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import roboflow_workflows.execution_engine.entities.base as base_module

    numpy_born, _ = _paired_images(dogs_image)
    numpy_output = CameraFocusBlockV2().run(
        image=numpy_born, detections=None, **_ALL_VISUALIZATIONS
    )
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(dogs_image)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    tensor_output = CameraFocusBlockV2().run(
        image=tensor_born, detections=None, **_ALL_VISUALIZATIONS
    )

    # then
    assert tensor_output["image"].tensor_image.device.type == "mps"
    _assert_parity(numpy_output, tensor_output)
