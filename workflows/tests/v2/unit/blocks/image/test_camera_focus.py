"""Unit tests of ``v2/camera_focus@v2``, called directly.

The Tenengrad score of a step edge is worked out by hand. Overlays, scores
and box scores are compared with V1's ``visualize_tenengrad_measure`` on V1's
BGR layout: pixels are expected to be identical and scores equal to float
rounding.
"""

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.classical_cv.camera_focus.v2 import (
    visualize_tenengrad_measure,
)
from roboflow_workflows.v2.blocks.image.camera_focus import CameraFocusBlock
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import RowRef

from inference_models.models.base.object_detection import Detections

OVERLAYS_OFF = dict(
    show_zebra_warnings=False,
    grid_overlay="None",
    show_hud=False,
    show_focus_peaking=False,
    show_center_marker=False,
)
# Two inside the image, one outside it, one clipped at the bottom right.
BOXES = [[10.7, 5.2, 120.9, 80.0], [-5, -5, 0, 0], [30, 20, 9999, 9999]]


def random_image(channels: int, height: int, width: int, seed: int = 1) -> ImageData:
    generator = np.random.default_rng(seed)
    pixels = generator.integers(0, 256, (channels, height, width), dtype=np.uint8)
    pixels[:, : height // 7] = 2
    pixels[:, -height // 9 :] = 253
    image = ImageData.from_tensor(torch.from_numpy(pixels))

    return image


def detections_of(image: ImageData, xyxy) -> Detections:
    detections = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(xyxy), dtype=torch.int64),
        confidence=torch.ones(len(xyxy)),
        image_metadata=image.prediction_metadata(),
    )

    return detections


def measure(image: ImageData, detections=None, **settings) -> dict:
    values = dict(
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        show_zebra_warnings=True,
        grid_overlay="3x3",
        show_hud=True,
        show_focus_peaking=True,
        show_center_marker=True,
    )
    values.update(settings)
    result = CameraFocusBlock().run(image=image, detections=detections, **values)

    return result


@pytest.mark.parametrize(
    ("shape", "settings"),
    [
        ((3, 360, 500), {}),
        ((1, 200, 300), {}),
        ((3, 1100, 1500), {"grid_overlay": "5x5"}),
        ((3, 1, 40), {}),
        ((3, 3, 2), {"show_hud": False}),
        ((3, 240, 320), {"show_zebra_warnings": False, "grid_overlay": "2x2"}),
        ((3, 240, 320), {"show_focus_peaking": False, "show_center_marker": False}),
    ],
)
def test_overlays_and_scores_match_v1(shape, settings) -> None:
    image = random_image(*shape)
    pixels = image.tensor_image.permute(1, 2, 0).numpy()
    bgr = np.ascontiguousarray(pixels[..., ::-1] if shape[0] == 3 else pixels[..., 0])
    v1_settings = {
        name: settings.get(name, default)
        for name, default in [
            ("show_zebra_warnings", True),
            ("grid_overlay", "3x3"),
            ("show_hud", True),
            ("show_focus_peaking", True),
            ("show_center_marker", True),
        ]
    }
    v1_image, v1_focus, v1_boxes = visualize_tenengrad_measure(
        bgr,
        underexposed_threshold=7,
        overexposed_threshold=247,
        detections=sv.Detections(xyxy=np.array(BOXES, dtype=np.float32)),
        **v1_settings,
    )

    result = measure(image, detections_of(image, BOXES), **settings)

    drawn = result["image"].tensor_image.permute(1, 2, 0).numpy()[..., ::-1]
    assert np.array_equal(drawn, v1_image)
    assert result["focus_measure"] == pytest.approx(v1_focus, rel=1e-6)
    assert [value is None for value in result["bbox_focus_measures"]] == [
        value is None for value in v1_boxes
    ]
    for value, v1_value in zip(result["bbox_focus_measures"], v1_boxes):
        if value is not None:
            assert value == pytest.approx(v1_value, rel=1e-6)


def test_step_edge_score_is_worked_out_by_hand() -> None:
    # Sobel_x across a 0 -> 100 step is (1 + 2 + 1) * 100 = 400 on the two
    # columns beside the step and 0 elsewhere; Sobel_y is 0 everywhere.
    pixels = torch.zeros((1, 10, 20), dtype=torch.uint8)
    pixels[:, :, 8:] = 100

    result = measure(ImageData.from_tensor(pixels), **OVERLAYS_OFF)

    assert result["focus_measure"] == pytest.approx(2 * 10 * 400**2 / (10 * 20))


def test_sharper_images_score_higher_and_flat_images_score_zero() -> None:
    ys, xs = np.mgrid[0:120, 0:160]
    board = (((xs // 8) + (ys // 8)) % 2 * 255).astype(np.uint8)
    sharp = torch.from_numpy(np.stack([board] * 3))
    kernel = torch.ones((3, 1, 7, 7)) / 49
    blurred = (
        torch.nn.functional.conv2d(sharp.float()[None], kernel, padding=3, groups=3)[0]
        .round()
        .to(torch.uint8)
    )
    flat = torch.full((3, 120, 160), 90, dtype=torch.uint8)

    scores = [
        measure(ImageData.from_tensor(pixels), **OVERLAYS_OFF)["focus_measure"]
        for pixels in (sharp, blurred, flat)
    ]

    assert scores[0] > 3 * scores[1] > 0
    assert scores[2] == 0.0


def test_without_overlays_the_input_image_is_returned() -> None:
    image = random_image(3, 60, 80)

    result = measure(image, detections_of(image, []), **OVERLAYS_OFF)

    assert result["image"] is image
    assert result["focus_measure"] > 0
    assert result["bbox_focus_measures"] == []


def test_overlays_keep_identity_and_geometry_and_leave_the_input_alone() -> None:
    image = random_image(1, 60, 80).with_parent_row(RowRef("frames", 4))
    before = image.tensor_image.clone()

    drawn = measure(image)["image"]

    assert drawn.image_id == image.image_id
    assert drawn.parent == image.parent and drawn.root == image.root
    assert drawn.parent_row == RowRef("frames", 4)
    assert drawn.tensor_image.shape == (3, 60, 80)
    assert torch.equal(image.tensor_image, before)


def test_box_scores_follow_row_order_and_mark_boxes_outside_the_image() -> None:
    image = random_image(3, 60, 80)

    values = measure(
        image, detections_of(image, [[0, 0, 80, 60], [90, 70, 99, 79]]), **OVERLAYS_OFF
    )

    whole, outside = values["bbox_focus_measures"]
    assert whole == pytest.approx(values["focus_measure"])
    assert outside is None


def test_detections_of_another_image_are_refused() -> None:
    other = random_image(3, 60, 80, seed=2)

    with pytest.raises(ContractError, match="same image"):
        measure(random_image(3, 60, 80), detections_of(other, [[1, 1, 5, 5]]))
