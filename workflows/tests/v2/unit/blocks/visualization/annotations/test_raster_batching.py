"""One table upload per call in blur, pixelate and crop visualization.

=====================================  ==========================================
check                                  oracle
=====================================  ==========================================
batched tables vs one resize per call  ``resize_nearest`` / ``resize_linear``
                                       (cv2-tested in ``test_helpers``): exact
many overlapping rows                  the latest V1 block: exact or <= 1 level,
                                       as the single-pass tests promise
uploads per call                       ``raster._to_device`` calls: 1 for any
                                       number of resampled rows, 0 without
input immutability                     the image and predictions before the run
parameter bounds                       the same value fails compile and run
=====================================  ==========================================
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.blur.v1 import (
    BlurVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.crop.v1 import (
    CropVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    CropVisualizationBlock,
    PixelateVisualizationBlock,
    raster,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import (
    apply_linear,
    apply_nearest,
    elliptic_kernel,
    grow_mask,
    linear_tables,
    nearest_tables,
    resize_linear,
    resize_nearest,
    upload_tables,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.instance_segmentation import InstanceDetections

from .support import (
    V1_COLOR_DEFAULTS,
    detections,
    image_of,
    pixels_of,
    run_v1,
    run_v2,
    scene,
    v1_detections,
)

HEIGHT, WIDTH = 72, 96
Row = Tuple[float, float, float, float, int, float]


def overlapping_rows(count: int, *, seed: int = 0) -> List[Row]:
    """``count`` overlapping boxes, some crossing the frame edges."""
    generator = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        x1 = float(generator.uniform(-10, WIDTH - 20))
        y1 = float(generator.uniform(-10, HEIGHT - 16))
        width, height = generator.uniform(14, 45), generator.uniform(12, 35)
        rows.append((x1, y1, x1 + width, y1 + height, index % 3, 0.9))

    return rows


def ellipse_masks(rows: List[Row]) -> np.ndarray:
    """One bool ellipse inside each row's box; the last row's mask is empty."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    masks = np.zeros((len(rows), HEIGHT, WIDTH), dtype=bool)
    for index, (x1, y1, x2, y2, _, _) in enumerate(rows[:-1]):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = max((x2 - x1) / 2, 1.0), max((y2 - y1) / 2, 1.0)
        masks[index] = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1

    return masks


def instances_of(
    image: ImageData, rows: List[Row], *, device: str = "cpu"
) -> InstanceDetections:
    boxes = detections(image, rows, device=device)
    instances = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=torch.from_numpy(ellipse_masks(rows)).to(device),
        image_metadata=boxes.image_metadata,
    )

    return instances


@pytest.fixture
def pixels() -> np.ndarray:
    return scene(HEIGHT, WIDTH, seed=3)


@pytest.fixture
def image(pixels: np.ndarray) -> ImageData:
    return image_of(pixels)


@pytest.fixture
def uploads(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    """Byte counts of every ``raster`` host-to-device copy during the test."""
    copies: List[int] = []
    original = raster._to_device

    def counted(host: np.ndarray, *, device: torch.device) -> torch.Tensor:
        copies.append(int(host.nbytes))
        return original(host, device=device)

    monkeypatch.setattr(raster, "_to_device", counted)

    return copies


def level_difference(output: np.ndarray, expected: np.ndarray) -> int:
    difference = np.abs(output.astype(np.int16) - expected.astype(np.int16))

    return int(difference.max())


# --- tables ------------------------------------------------------------------


def test_upload_tables_returns_every_array_with_its_dtype_from_one_copy(
    uploads: List[int],
) -> None:
    host = [
        linear_tables((7, 13), (3, 5), scale_xy=(13 / 5, 7 / 3)),
        nearest_tables((3, 5), (7, 13)),
        linear_tables((1, 1), (4, 9), scale_xy=(1 / 9, 1 / 4)),
    ]

    uploaded = upload_tables(host, device=torch.device("cpu"))

    assert len(uploads) == 1
    assert [type(table) for table in uploaded] == [type(table) for table in host]
    for host_table, table in zip(host, uploaded):
        for host_array, array in zip(host_table, table):
            assert array.dtype == torch.from_numpy(host_array).dtype
            np.testing.assert_array_equal(array.numpy(), host_array)


def test_upload_tables_of_nothing_makes_no_copy(uploads: List[int]) -> None:
    uploaded = upload_tables([], device=torch.device("cpu"))

    assert uploaded == []
    assert uploads == []


def test_upload_size_is_the_tables_bytes_plus_alignment(uploads: List[int]) -> None:
    # Linear: 8 + 8 + 4 bytes per output row and column; nearest: 8 + 8.
    host = [linear_tables((40, 60), (5, 7), scale_xy=(8.0, 8.0))] * 3 + [
        nearest_tables((5, 7), (40, 60))
    ] * 3

    upload_tables(host, device=torch.device("cpu"))

    table_bytes = 3 * 20 * (5 + 7) + 3 * 8 * (40 + 60)
    assert table_bytes <= uploads[0] <= table_bytes + 7 * 6 * 3


@pytest.mark.parametrize(
    "source_hw, size_hw", [((9, 14), (3, 5)), ((5, 7), (40, 60)), ((1, 3), (6, 2))]
)
def test_batched_resizes_equal_one_resize_per_call(
    source_hw: Tuple[int, int], size_hw: Tuple[int, int]
) -> None:
    generator = np.random.default_rng(1)
    regions = [
        torch.from_numpy(generator.integers(0, 256, (3, *source_hw), dtype=np.uint8))
        for _ in range(3)
    ]
    scale_xy = (source_hw[1] / size_hw[1], source_hw[0] / size_hw[0])
    host = [linear_tables(source_hw, size_hw, scale_xy=scale_xy)] * 3 + [
        nearest_tables(source_hw, size_hw)
    ] * 3

    uploaded = upload_tables(host, device=torch.device("cpu"))

    for region, linear, nearest in zip(regions, uploaded[:3], uploaded[3:]):
        assert torch.equal(
            apply_linear(region, linear),
            resize_linear(region, size_hw, scale_xy=scale_xy),
        )
        assert torch.equal(
            apply_nearest(region, nearest), resize_nearest(region, size_hw)
        )


@pytest.mark.parametrize("padding", [1, 4])
def test_grow_mask_with_a_shared_kernel_equals_its_own_kernel(padding: int) -> None:
    masks = torch.from_numpy(ellipse_masks(overlapping_rows(4)))
    kernel = elliptic_kernel(padding, device=masks.device)

    for mask in masks:
        assert torch.equal(
            grow_mask(mask, padding, kernel=kernel), grow_mask(mask, padding)
        )


# --- blocks: uploads per call ------------------------------------------------


@pytest.mark.parametrize("count", [1, 6, 30])
def test_pixelate_uploads_tables_once_for_any_number_of_rows(
    image: ImageData, uploads: List[int], count: int
) -> None:
    run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image, overlapping_rows(count)),
        pixel_size=5,
    )

    assert len(uploads) == 1


def test_pixelate_with_only_mean_filled_rows_uploads_nothing(
    image: ImageData, uploads: List[int]
) -> None:
    run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image, overlapping_rows(6)),
        pixel_size=60,
    )

    assert uploads == []


@pytest.mark.parametrize("count", [1, 6, 30])
def test_crop_uploads_tables_once_for_any_number_of_rows(
    image: ImageData, uploads: List[int], count: int
) -> None:
    run_v2(
        CropVisualizationBlock,
        image=image,
        predictions=detections(image, overlapping_rows(count)),
        scale_factor=1.5,
    )

    assert len(uploads) == 1


@pytest.mark.parametrize("count", [2, 6, 30])
def test_blur_uploads_one_growth_kernel_for_any_number_of_masks(
    image: ImageData, uploads: List[int], count: int
) -> None:
    run_v2(
        BlurVisualizationBlock,
        image=image,
        predictions=instances_of(image, overlapping_rows(count)),
        kernel_size=5,
        padding=2,
    )

    assert len(uploads) == 1


def test_blur_without_padding_or_masks_uploads_nothing(
    image: ImageData, uploads: List[int]
) -> None:
    rows = overlapping_rows(6)

    run_v2(
        BlurVisualizationBlock,
        image=image,
        predictions=instances_of(image, rows),
        kernel_size=5,
        padding=0,
    )
    run_v2(
        BlurVisualizationBlock,
        image=image,
        predictions=detections(image, rows),
        kernel_size=5,
        padding=3,
    )

    assert uploads == []


# --- blocks: pixels and inputs -----------------------------------------------


# 20: rows smaller than a cell get mean fills between resampled rows.
@pytest.mark.parametrize("pixel_size", [3, 7, 20])
def test_pixelate_many_overlapping_rows_are_within_one_level_of_v1(
    pixels: np.ndarray, image: ImageData, pixel_size: int
) -> None:
    rows = overlapping_rows(30, seed=pixel_size)

    output = run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image, rows),
        pixel_size=pixel_size,
    )

    expected = run_v1(
        PixelateVisualizationBlockV1,
        pixels,
        predictions=v1_detections(rows),
        pixel_size=pixel_size,
    )
    assert level_difference(pixels_of(output), expected) <= 1
    assert (pixels_of(output) != pixels).any()


def test_pixelate_identity_cells_equal_v1_exactly_for_many_rows(
    pixels: np.ndarray, image: ImageData
) -> None:
    rows = overlapping_rows(30, seed=11)

    output = run_v2(
        PixelateVisualizationBlock,
        image=image,
        predictions=detections(image, rows),
        pixel_size=1,
    )

    expected = run_v1(
        PixelateVisualizationBlockV1,
        pixels,
        predictions=v1_detections(rows),
        pixel_size=1,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


@pytest.mark.parametrize("position", ["TOP_CENTER", "CENTER"])
def test_crop_many_rows_at_scale_one_equal_v1(
    pixels: np.ndarray, image: ImageData, position: str
) -> None:
    # Rows inside the frame: V1 mishandles boxes crossing the left/top edge.
    rows = [
        (max(x1, 0.0), max(y1, 0.0), x2, y2, class_id, confidence)
        for x1, y1, x2, y2, class_id, confidence in overlapping_rows(20, seed=5)
    ]
    parameters: Dict[str, Any] = dict(
        position=position, scale_factor=1.0, border_thickness=1, **V1_COLOR_DEFAULTS
    )

    output = run_v2(
        CropVisualizationBlock,
        image=image,
        predictions=detections(image, rows),
        **parameters,
    )

    expected = run_v1(
        CropVisualizationBlockV1, pixels, predictions=v1_detections(rows), **parameters
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


def test_blur_many_masks_with_padding_equal_v1(
    pixels: np.ndarray, image: ImageData
) -> None:
    rows = overlapping_rows(12, seed=7)

    output = run_v2(
        BlurVisualizationBlock,
        image=image,
        predictions=instances_of(image, rows),
        kernel_size=7,
        padding=3,
    )

    expected = run_v1(
        BlurVisualizationBlockV1,
        pixels,
        predictions=v1_detections(rows, masks=ellipse_masks(rows)),
        kernel_size=7,
        padding=3,
    )
    np.testing.assert_array_equal(pixels_of(output), expected)


@pytest.mark.parametrize(
    "block, predictions_of, parameters",
    [
        (PixelateVisualizationBlock, detections, {"pixel_size": 4}),
        (CropVisualizationBlock, detections, {"scale_factor": 2.0}),
        (BlurVisualizationBlock, instances_of, {"kernel_size": 5, "padding": 2}),
    ],
)
def test_inputs_are_unchanged(
    image: ImageData,
    block: type,
    predictions_of: Callable[..., Any],
    parameters: Dict[str, Any],
) -> None:
    predictions = predictions_of(image, overlapping_rows(12))
    pixels_before = image.tensor_image.clone()
    boxes_before = predictions.xyxy.clone()

    output = run_v2(block, image=image, predictions=predictions, **parameters)

    assert torch.equal(image.tensor_image, pixels_before)
    assert torch.equal(predictions.xyxy, boxes_before)
    assert not torch.equal(output.tensor_image, pixels_before)


# --- parameter bounds ----------------------------------------------------------


@pytest.mark.parametrize(
    "block, name, value",
    [
        (PixelateVisualizationBlock, "pixel_size", 0),
        (BlurVisualizationBlock, "kernel_size", 0),
        (BlurVisualizationBlock, "padding", -1),
        (CropVisualizationBlock, "scale_factor", 0.0),
        (CropVisualizationBlock, "border_thickness", 0),
    ],
)
def test_a_bound_rejects_the_same_value_at_compile_and_run_time(
    image: ImageData, block: type, name: str, value: Any
) -> None:
    with pytest.raises(ValueError) as compiled:
        spec_of(block).validate_params(
            {"image": "$inputs.image", "predictions": "$inputs.p", name: value}
        )
    defaults = dict(
        spec_of(block).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.p",
                name: f"$inputs.{name}",
            }
        )
    )

    with pytest.raises(ValueError) as ran:
        block().run(
            **{
                **defaults,
                "image": image,
                "predictions": detections(image, overlapping_rows(2)),
                name: value,
            }
        )

    assert str(ran.value) in str(compiled.value)
