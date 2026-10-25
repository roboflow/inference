"""Classical-CV blocks that draw on their output image declare VISUALIZATION.

Camera Focus v1 always writes the Brenner score onto a copy of the image and
Contours v1 always draws the detected contours. Camera Focus v2 draws only the
overlays that are switched on; with every overlay off it returns the input
image unchanged. The v2 cases below are checked against the block's own
numpy run: a declared VISUALIZATION must coincide with a changed output image.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
from roboflow_workflows.core_steps.classical_cv.camera_focus.v1 import (
    CameraFocusManifest as CameraFocusV1Manifest,
)
from roboflow_workflows.core_steps.classical_cv.camera_focus.v2 import (
    CameraFocusBlockV2,
)
from roboflow_workflows.core_steps.classical_cv.camera_focus.v2 import (
    CameraFocusManifest as CameraFocusV2Manifest,
)
from roboflow_workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

from tests.unit_tests.workload_declaration_helpers import (
    portable_restrictions_discovery,
)

ANALYSIS_ONLY = [WorkOperation.IMAGE_ANALYSIS]
ANALYSIS_AND_DRAWING = [WorkOperation.IMAGE_ANALYSIS, WorkOperation.VISUALIZATION]
ALL_OVERLAYS_OFF: Dict[str, Any] = {
    "show_zebra_warnings": False,
    "grid_overlay": "None",
    "show_hud": False,
    "show_focus_peaking": False,
    "show_center_marker": False,
}
SINGLE_OVERLAY_CASES = [
    {"show_zebra_warnings": True},
    {"show_hud": True},
    {"show_focus_peaking": True},
    {"show_center_marker": True},
    {"grid_overlay": "2x2"},
    {"grid_overlay": "3x3"},
    {"grid_overlay": "4x4"},
    {"grid_overlay": "5x5"},
]


def _camera_focus_v2(**overlays: Any) -> CameraFocusV2Manifest:
    return CameraFocusV2Manifest(
        type="roboflow_core/camera_focus@v2",
        name="focus",
        image="$inputs.image",
        **overlays,
    )


def _test_image() -> WorkflowImageData:
    # Noise gives edges for focus peaking; the black and white bands give the
    # zebra warnings under- and overexposed pixels to mark.
    rng = np.random.default_rng(seed=0)
    pixels = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    pixels[:24] = 0
    pixels[-24:] = 255

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=pixels,
    )

    return image


def _run_v2(manifest: CameraFocusV2Manifest) -> Dict[str, Any]:
    image = _test_image()
    result = CameraFocusBlockV2().run(
        image=image,
        underexposed_threshold_percent=manifest.underexposed_threshold_percent,
        overexposed_threshold_percent=manifest.overexposed_threshold_percent,
        show_zebra_warnings=manifest.show_zebra_warnings,
        grid_overlay=manifest.grid_overlay,
        show_hud=manifest.show_hud,
        show_focus_peaking=manifest.show_focus_peaking,
        show_center_marker=manifest.show_center_marker,
        detections=None,
    )

    return {"input": image, "output": result["image"]}


def _workflow(step: Dict[str, Any], output_property: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [step],
        "outputs": [
            {
                "type": "JsonField",
                "name": "out",
                "selector": f"$steps.{step['name']}.{output_property}",
            }
        ],
    }


def _reported_step(step: Dict[str, Any], output_property: str) -> Any:
    introspection = describe_workflow_workload(_workflow(step, output_property))
    assert len(introspection.steps) == 1

    return introspection.steps[0]


def test_camera_focus_v1_declares_its_score_drawing() -> None:
    # given
    manifest = CameraFocusV1Manifest(
        type="roboflow_core/camera_focus@v1", name="focus", image="$inputs.image"
    )

    # then
    assert manifest.discover_work_operations() == ANALYSIS_AND_DRAWING
    assert portable_restrictions_discovery(manifest).items == []


def test_contours_v1_declares_its_contour_drawing() -> None:
    # given
    manifest = ImageContoursDetectionManifest(
        type="roboflow_core/contours_detection@v1",
        name="contours",
        image="$inputs.image",
    )

    # then
    assert manifest.discover_work_operations() == ANALYSIS_AND_DRAWING
    assert portable_restrictions_discovery(manifest).items == []


@pytest.mark.parametrize(
    "step, output_property",
    [
        (
            {
                "type": "roboflow_core/camera_focus@v1",
                "name": "focus",
                "image": "$inputs.image",
            },
            "focus_measure",
        ),
        (
            {
                "type": "roboflow_core/contours_detection@v1",
                "name": "contours",
                "image": "$inputs.image",
            },
            "number_contours",
        ),
    ],
    ids=["camera_focus_v1", "contours_v1"],
)
def test_always_drawing_blocks_report_visualization_through_the_public_api(
    step: Dict[str, Any], output_property: str
) -> None:
    # when
    reported = _reported_step(step, output_property)

    # then
    assert reported.operations.complete is True
    assert reported.operations.items == ANALYSIS_AND_DRAWING
    assert reported.restrictions.items == []
    assert reported.restrictions.complete is True


def test_camera_focus_v2_defaults_declare_drawing() -> None:
    # given
    manifest = _camera_focus_v2()

    # when
    run = _run_v2(manifest)

    # then
    assert manifest.discover_work_operations() == ANALYSIS_AND_DRAWING
    assert run["output"] is not run["input"]
    assert not np.array_equal(run["output"].numpy_image, run["input"].numpy_image)


@pytest.mark.parametrize(
    "enabled",
    SINGLE_OVERLAY_CASES,
    ids=[
        f"{key}={value}" for case in SINGLE_OVERLAY_CASES for key, value in case.items()
    ],
)
def test_camera_focus_v2_any_single_overlay_declares_drawing(
    enabled: Dict[str, Any],
) -> None:
    # given
    manifest = _camera_focus_v2(**{**ALL_OVERLAYS_OFF, **enabled})

    # when
    run = _run_v2(manifest)

    # then
    assert manifest.discover_work_operations() == ANALYSIS_AND_DRAWING
    assert not np.array_equal(run["output"].numpy_image, run["input"].numpy_image)


def test_camera_focus_v2_with_every_overlay_off_declares_analysis_only() -> None:
    # given
    manifest = _camera_focus_v2(**ALL_OVERLAYS_OFF)

    # when
    run = _run_v2(manifest)

    # then
    assert manifest.discover_work_operations() == ANALYSIS_ONLY
    assert run["output"] is run["input"]
    assert portable_restrictions_discovery(manifest).items == []


@pytest.mark.parametrize(
    "overlays, expected",
    [
        ({}, ANALYSIS_AND_DRAWING),
        (ALL_OVERLAYS_OFF, ANALYSIS_ONLY),
        ({**ALL_OVERLAYS_OFF, "grid_overlay": "5x5"}, ANALYSIS_AND_DRAWING),
    ],
    ids=["defaults", "all_off", "grid_only"],
)
def test_camera_focus_v2_reports_drawing_through_the_public_api(
    overlays: Dict[str, Any], expected: List[WorkOperation]
) -> None:
    # given
    step = {
        "type": "roboflow_core/camera_focus@v2",
        "name": "focus",
        "image": "$inputs.image",
        **overlays,
    }

    # when
    reported = _reported_step(step, "focus_measure")

    # then
    assert reported.operations.complete is True
    assert reported.operations.items == expected
    assert reported.restrictions.items == []
    assert reported.restrictions.complete is True
