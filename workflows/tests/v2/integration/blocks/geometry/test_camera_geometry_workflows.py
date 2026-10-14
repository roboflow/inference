"""Compiled workflows over the four V2 camera and geometry blocks.

These run the real compiler, session and executor. The blocks are added to
the catalogue here because the shared catalogue does not list them yet.
Images paint numbered markers that ``test/marker_detector@v1`` finds exactly,
so a detector run on a warped image can be compared with predictions moved
into it by ``roboflow_core/perspective_correction@v1``. ``test/marker_corner_key_points@v1``
(defined here) gives the markers' corners as keypoints with covariance.

Warped, rotated and undistorted images are new root frames: predictions made
on them stay in their frame and are not mapped back to the input's pixels.
"""

from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.camera_calibration.v1 import CameraCalibrationBlock
from roboflow_workflows.v2.blocks.geometry.perspective_correction.v1 import PerspectiveCorrectionBlock
from roboflow_workflows.v2.blocks.image.auto_rotate_on_edges.v1 import AutoRotateOnEdgesBlock
from roboflow_workflows.v2.blocks.image.camera_focus.v2 import CameraFocusBlock
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import KEYPOINT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.keypoints_detection import KeyPoints
from tests.v2.integration.blocks.geometry.deterministic import (
    MarkerDetectorBlock,
    painted_image,
    regions_catalogue,
)


class MarkerCornerKeyPointsBlock(Block):
    """Top-left and bottom-right corner of every painted marker as keypoints.

    Each point has covariance ``diag(1, 4)`` in input pixels, as a keypoint
    model with uncertainty would report.
    """

    type = "test/marker_corner_key_points@v1"
    outputs = {
        "predictions": Output(KEYPOINT_DETECTION_PREDICTION_KIND, source="image")
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to search.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        (detections,) = MarkerDetectorBlock().run(image=image).values()
        corners = detections.xyxy.reshape(-1, 2, 2)
        key_points = KeyPoints(
            xy=corners,
            class_id=detections.class_id,
            confidence=torch.ones(corners.shape[:2]),
            # The row record stays on the Detections only.
            image_metadata=FrameRef.from_image(image).prediction_metadata(),
            covariance=torch.diag(torch.tensor([1.0, 4.0])).expand(
                *corners.shape[:2], 2, 2
            ),
        )

        return {"predictions": (key_points, detections)}


CATALOGUE = regions_catalogue().with_blocks(
    [
        AutoRotateOnEdgesBlock,
        CameraFocusBlock,
        PerspectiveCorrectionBlock,
        CameraCalibrationBlock,
        MarkerCornerKeyPointsBlock,
    ]
)
MODES = ["run", "phases"]
IMAGES_INPUT = {"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}
# Maps to the 400 x 300 rectangle by the shift (-100, -100).
SHIFT_POLYGON = [[100, 399], [100, 100], [499, 100], [499, 399]]
# Two markers inside the polygon, one outside it.
MARKERS = [
    (150, 150, 200, 250, 1, 0),
    (300, 200, 340, 260, 2, 1),
    (20, 20, 60, 60, 3, 0),
]
DETECT = {
    "type": "test/marker_detector@v1",
    "name": "detect",
    "image": "$inputs.images",
}
PERSPECTIVE = {
    "type": "roboflow_core/perspective_correction@v1",
    "name": "perspective",
    "images": "$inputs.images",
    "predictions": "$steps.detect.predictions",
    "perspective_polygons": "$inputs.polygon",
    "transformed_rect_width": 400,
    "transformed_rect_height": 300,
    "warp_image": True,
}
FOCUS_OFF = {
    "show_zebra_warnings": False,
    "grid_overlay": "None",
    "show_hud": False,
    "show_focus_peaking": False,
    "show_center_marker": False,
}
CAMERA = {"fx": 300.0, "fy": 310.0, "cx": 280.0, "cy": 210.0, "k1": -0.05}


def _definition(steps: List[dict], outputs: Dict[str, str], *parameters: str) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [
            IMAGES_INPUT,
            *({"type": "WorkflowParameter", "name": name} for name in parameters),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def _compile(definition: dict, mode: str = "run", **options: Any):
    plan = compile_workflow(
        definition,
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode, **options),
    )

    return plan


def _scene() -> torch.Tensor:
    pixels = painted_image((420, 560), MARKERS)
    scene = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))

    return scene


def _ruled_page(tilt: float) -> torch.Tensor:
    """A 300 x 200 page of horizontal lines rotated counter-clockwise by ``tilt``."""
    canvas = np.full((400, 500, 3), 235, np.uint8)
    for y in range(20, 400, 16):
        cv2.line(canvas, (0, y), (500, y), (30, 30, 30), 2)
    matrix = cv2.getRotationMatrix2D((250, 200), tilt, 1)
    canvas = cv2.warpAffine(canvas, matrix, (500, 400), borderValue=(235, 235, 235))
    page = np.ascontiguousarray(canvas[100:300, 100:400].transpose(2, 0, 1))

    return torch.from_numpy(page)


def _boxes_by_marker(detections) -> Dict[int, List[float]]:
    boxes = {
        row["marker"]: box
        for row, box in zip(detections.bboxes_metadata, detections.xyxy.tolist())
    }

    return boxes


@pytest.mark.parametrize("mode", MODES)
def test_detections_on_the_warped_view_match_the_corrected_ones(mode: str) -> None:
    # given
    steps = [
        DETECT,
        PERSPECTIVE,
        {
            "type": "test/marker_detector@v1",
            "name": "detect_top",
            "image": "$steps.perspective.warped_image",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "crop",
            "image": "$steps.perspective.warped_image",
            "predictions": "$steps.detect_top.predictions",
        },
    ]
    outputs = {
        "corrected": "$steps.perspective.corrected_coordinates",
        "warped": "$steps.perspective.warped_image",
        "top": "$steps.detect_top.predictions",
        "crops": "$steps.crop.crops",
    }
    plan = _compile(_definition(steps, outputs, "polygon"), mode)
    scene = _scene()

    # when
    (row,) = (
        plan.create_session().run({"images": [scene], "polygon": SHIFT_POLYGON}).rows()
    )

    # then
    warped = row["warped"]
    assert torch.equal(warped.tensor_image, scene[:, 100:400, 100:500])
    assert warped.root == FrameMapping.identity(warped.image_id, (300, 400))
    corrected = _boxes_by_marker(row["corrected"])
    assert corrected == {
        1: [50, 50, 100, 150],
        2: [200, 100, 240, 160],
        3: [-80, -80, -40, -40],
    }
    assert _boxes_by_marker(row["top"]) == {1: corrected[1], 2: corrected[2]}
    assert frame_of(row["corrected"]).image_id == warped.image_id
    assert frame_of(row["top"]).image_id == warped.image_id
    assert [crop.root.frame_id for crop in row["crops"]] == [warped.image_id] * 2
    assert [crop.size_hw for crop in row["crops"]] == [(100, 50), (60, 40)]


@pytest.mark.parametrize("mode", MODES)
def test_key_point_covariance_is_scaled_into_the_rectangle(mode: str) -> None:
    # given: the shift polygon onto 799 x 898 pixels scales x by 2 and y by 3.
    steps = [
        {
            "type": "test/marker_corner_key_points@v1",
            "name": "corners",
            "image": "$inputs.images",
        },
        PERSPECTIVE
        | {
            "predictions": "$steps.corners.predictions",
            "transformed_rect_width": 799,
            "transformed_rect_height": 898,
        },
    ]
    outputs = {
        "input": "$steps.corners.predictions",
        "corrected": "$steps.perspective.corrected_coordinates",
        "warped": "$steps.perspective.warped_image",
    }
    plan = _compile(_definition(steps, outputs, "polygon"), mode)

    # when
    (row,) = (
        plan.create_session()
        .run({"images": [_scene()], "polygon": SHIFT_POLYGON})
        .rows()
    )

    # then
    (key_points, boxes), (points_in, _) = row["corrected"], row["input"]
    assert key_points.xy[0].tolist() == [[100.0, 150.0], [200.0, 450.0]]
    # cv2 solves the homography numerically: off-diagonals are ~1e-17, not 0.
    assert torch.allclose(
        key_points.covariance,
        torch.diag(torch.tensor([4.0, 36.0])).expand(3, 2, 2, 2),
        atol=1e-6,
    )
    assert points_in.covariance.tolist() == [[[[1.0, 0.0], [0.0, 4.0]]] * 2] * 3
    assert torch.equal(key_points.class_id, boxes.class_id)
    assert frame_of(key_points).image_id == row["warped"].image_id
    assert frame_of(boxes).image_id == row["warped"].image_id


@pytest.mark.parametrize("mode", MODES)
def test_top_down_predictions_do_not_map_back_to_the_input(mode: str) -> None:
    # given
    steps = [
        DETECT,
        {**PERSPECTIVE, "warp_image": False},
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.images",
            "predictions": "$steps.perspective.corrected_coordinates",
        },
    ]
    plan = _compile(_definition(steps, {"crops": "$steps.crop.crops"}, "polygon"), mode)

    # when
    with pytest.raises(StepExecutionError) as error:
        plan.create_session().run({"images": [_scene()], "polygon": SHIFT_POLYGON})

    # then
    assert isinstance(error.value.__cause__, ContractError)
    assert "perspective-" in str(error.value.__cause__)


@pytest.mark.parametrize("mode", MODES)
def test_undistort_warp_and_score_focus_per_box(mode: str) -> None:
    # given
    steps = [
        {
            "type": "roboflow_core/camera-calibration@v1",
            "name": "undistort",
            "image": "$inputs.images",
            **{name: f"$inputs.{name}" for name in CAMERA},
        },
        DETECT | {"image": "$steps.undistort.calibrated_image"},
        PERSPECTIVE
        | {
            "images": "$steps.undistort.calibrated_image",
            "perspective_polygons": SHIFT_POLYGON,
        },
        {
            "type": "test/marker_detector@v1",
            "name": "detect_top",
            "image": "$steps.perspective.warped_image",
        },
        {
            "type": "roboflow_core/camera_focus@v2",
            "name": "focus",
            "image": "$steps.perspective.warped_image",
            "detections": "$steps.detect_top.predictions",
            "show_hud": True,
        },
    ]
    outputs = {
        "undistorted": "$steps.undistort.calibrated_image",
        "warped": "$steps.perspective.warped_image",
        "focus_image": "$steps.focus.image",
        "focus": "$steps.focus.focus_measure",
        "box_focus": "$steps.focus.bbox_focus_measures",
    }
    plan = _compile(_definition(steps, outputs, *CAMERA), mode)
    scene = ImageData.from_tensor(_scene(), image_id="camera")

    # when
    (row,) = plan.create_session().run({"images": [scene], **CAMERA}).rows()

    # then
    undistorted, warped = row["undistorted"], row["warped"]
    assert len({scene.image_id, undistorted.image_id, warped.image_id}) == 3
    assert undistorted.root.frame_id == undistorted.image_id
    assert row["focus_image"].image_id == warped.image_id
    assert row["focus_image"].root == warped.root
    direct = CameraFocusBlock().run(
        image=warped,
        underexposed_threshold_percent=3.0,
        overexposed_threshold_percent=97.0,
        detections=None,
        **FOCUS_OFF,
    )
    assert row["focus"] == direct["focus_measure"] > 0
    assert len(row["box_focus"]) == 2
    assert all(value > 0 for value in row["box_focus"])


@pytest.mark.parametrize("mode", MODES)
def test_auto_rotate_feeds_a_new_root_to_the_next_step(mode: str) -> None:
    # given
    steps = [
        {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "rotate",
            "image": "$inputs.images",
            "target_orientation": "$inputs.orientation",
        },
        {
            "type": "roboflow_core/camera_focus@v2",
            "name": "focus",
            "image": "$steps.rotate.image",
            **FOCUS_OFF,
        },
    ]
    outputs = {
        "rotated": "$steps.rotate.image",
        "angle": "$steps.rotate.angle",
        "focus_image": "$steps.focus.image",
    }
    plan = _compile(_definition(steps, outputs, "orientation"), mode)
    tilted = ImageData.from_tensor(_ruled_page(tilt=8.0))
    level = ImageData.from_tensor(_ruled_page(tilt=0.0))

    # when
    rotated_row, level_row = (
        plan.create_session()
        .run({"images": [tilted, level], "orientation": "horizontal"})
        .rows()
    )

    # then
    rotated = rotated_row["rotated"]
    assert rotated_row["angle"] == pytest.approx(-8.0, abs=1.0)
    assert rotated.image_id != tilted.image_id
    assert rotated.root == FrameMapping.identity(rotated.image_id, rotated.size_hw)
    assert rotated_row["focus_image"] is rotated
    assert level_row["angle"] == 0.0
    assert level_row["rotated"].image_id == level.image_id


def test_a_new_polygon_is_used_on_the_next_run_of_one_plan() -> None:
    # given
    plan = _compile(
        _definition(
            [DETECT, PERSPECTIVE],
            {"corrected": "$steps.perspective.corrected_coordinates"},
            "polygon",
        )
    )
    session = plan.create_session()
    shifted_more = [[x + 50, y] for x, y in SHIFT_POLYGON]

    # when
    first, second = (
        session.run({"images": [_scene()], "polygon": polygon}).rows()[0]
        for polygon in (SHIFT_POLYGON, shifted_more)
    )

    # then
    assert _boxes_by_marker(first["corrected"])[1] == [50, 50, 100, 150]
    assert _boxes_by_marker(second["corrected"])[1] == [0, 50, 50, 150]


@pytest.mark.parametrize(
    ("step", "inputs", "cause", "message"),
    [
        (
            PERSPECTIVE
            | {"extend_perspective_polygon_by_detections_anchor": "$inputs.value"},
            {"value": "MIDDLE"},
            ValueError,
            "MIDDLE",
        ),
        (
            PERSPECTIVE,
            {"polygon": [[0, 0], [10, 10], [20, 20], [0, 30]]},
            ValueError,
            "degenerate",
        ),
        (
            PERSPECTIVE | {"transformed_rect_width": "$inputs.value"},
            {"value": 1},
            ContractError,
            "transformed_rect_width",
        ),
        (
            {
                "type": "roboflow_core/auto_rotate_on_edges@v1",
                "name": "rotate",
                "image": "$inputs.images",
                "target_orientation": "$inputs.value",
            },
            {"value": "diagonal"},
            ValueError,
            "target_orientation",
        ),
        (
            {
                "type": "roboflow_core/camera-calibration@v1",
                "name": "undistort",
                "image": "$inputs.images",
                **CAMERA,
                "k1": "$inputs.value",
            },
            {"value": float("nan")},
            ValueError,
            "k1 must be a finite number",
        ),
    ],
)
def test_invalid_selected_values_fail_the_step(step, inputs, cause, message) -> None:
    # given
    steps = [DETECT, step]
    outputs = {"out": f"$steps.{step['name']}.*"}
    session = _compile(_definition(steps, outputs, "polygon", "value")).create_session()

    # when
    with pytest.raises(StepExecutionError) as error:
        session.run(
            {"images": [_scene()], "polygon": SHIFT_POLYGON, "value": None, **inputs}
        )

    # then
    assert isinstance(error.value.__cause__, cause)
    assert message in str(error.value.__cause__)


@pytest.mark.parametrize(
    "step",
    [
        {"type": "roboflow_core/camera-calibration@v1", **CAMERA, "fx": 0.0},
        {"type": "roboflow_core/camera-calibration@v1", "fy": 1.0},
        {"type": "roboflow_core/camera_focus@v2", "grid_overlay": "6x6"},
        {"type": "roboflow_core/camera_focus@v2", "overexposed_threshold_percent": 101},
        {"type": "roboflow_core/auto_rotate_on_edges@v1", "internal_resolution": 0},
        {"type": "roboflow_core/auto_rotate_on_edges@v1", "target_orientation": "diagonal"},
        {
            "type": "roboflow_core/perspective_correction@v1",
            "perspective_polygons": SHIFT_POLYGON,
            "transformed_rect_width": 0,
        },
        {
            "type": "roboflow_core/perspective_correction@v1",
            "perspective_polygons": SHIFT_POLYGON,
            "transformed_rect_height": 1,
        },
        {
            "type": "roboflow_core/perspective_correction@v1",
            "perspective_polygons": SHIFT_POLYGON,
            "extend_perspective_polygon_by_detections_anchor": "MIDDLE",
        },
    ],
)
def test_invalid_literal_parameters_fail_compilation(step: dict) -> None:
    # given
    step = {"name": "step", "image": "$inputs.images", **step}
    definition = _definition([step], {"out": "$steps.step.*"})

    # when / then
    with pytest.raises(ParamsValidationError):
        _compile(definition)


@pytest.mark.parametrize("mode", MODES)
def test_requested_outputs_drop_undemanded_steps(mode: str) -> None:
    # given
    steps = [
        DETECT,
        PERSPECTIVE,
        {
            "type": "roboflow_core/camera_focus@v2",
            "name": "focus",
            "image": "$inputs.images",
            **FOCUS_OFF,
        },
    ]
    outputs = {
        "corrected": "$steps.perspective.corrected_coordinates",
        "focus": "$steps.focus.focus_measure",
    }
    definition = _definition(steps, outputs, "polygon")

    # when
    full = _compile(definition, mode)
    demanded = _compile(definition, mode, requested_outputs=("focus",))
    (row,) = (
        demanded.create_session()
        .run({"images": [_scene()], "polygon": SHIFT_POLYGON})
        .rows()
    )

    # then: the test detector is not prunable, so only perspective is dropped.
    assert ("perspective",) in [step.path for step in full.steps]
    assert [step.path for step in demanded.steps] == [("detect",), ("focus",)]
    assert set(row) == {"focus"}


@pytest.mark.parametrize("mode", MODES)
def test_perspective_without_predictions_and_empty_batches(mode: str) -> None:
    # given
    step = {key: value for key, value in PERSPECTIVE.items() if key != "predictions"}
    outputs = {
        "corrected": "$steps.perspective.corrected_coordinates",
        "warped": "$steps.perspective.warped_image",
        "width": "$steps.perspective.extended_transformed_rect_width",
    }
    plan = _compile(_definition([step], outputs, "polygon"), mode)
    session = plan.create_session()

    # when
    (row,) = session.run({"images": [_scene()], "polygon": SHIFT_POLYGON}).rows()
    empty = session.run({"images": [], "polygon": SHIFT_POLYGON}).rows()

    # then
    assert row["corrected"] is None
    assert row["warped"].size_hw == (300, 400)
    assert row["width"] == 400
    assert empty == []
