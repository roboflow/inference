"""Compiled workflows over ``v2/stitch_images@v1`` and
``v2/stitch_ocr_detections@v2``.

These run the real compiler, session and executor. The catalogue is
``create_catalogue()`` plus these two blocks only while it lacks them; once
they are registered the shim adds nothing. Panorama flows also use the
region blocks and the painted-marker test detector.
"""

from typing import Any, Dict, List, Optional

import cv2 as cv
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.codes.ocr_stitch import StitchOCRDetectionsBlock
from roboflow_workflows.v2.blocks.image.stitch_images import StitchImagesBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
    WorkflowExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions, RunResult

from inference_models.models.base.object_detection import Detections
from tests.v2.integration.blocks.geometry.deterministic import Marker, regions_catalogue

IMAGE_PAIR_INPUTS = [
    {"type": "WorkflowBatchInput", "name": name, "kind": ["image"]}
    for name in ("left", "right")
]
PREDICTIONS_INPUT = {
    "type": "WorkflowBatchInput",
    "name": "predictions",
    "kind": ["object_detection_prediction"],
}
# Two lines, "AB" over "CD", of wide boxes; TALL is the same layout transposed.
WIDE = {
    "A": (0, 0, 20, 10),
    "B": (24, 0, 44, 10),
    "C": (0, 30, 20, 40),
    "D": (24, 30, 44, 40),
}
TALL = {text: (y0, x0, y1, x1) for text, (x0, y0, x1, y1) in WIDE.items()}
# Painted markers in scene columns: left view only, shared, right view only.
MARKERS: List[Marker] = [
    (30, 40, 60, 80, 1, 0),
    (150, 60, 175, 100, 2, 1),
    (260, 50, 300, 90, 3, 0),
]


def _catalogue(base: Optional[Catalogue] = None) -> Catalogue:
    catalogue = create_catalogue() if base is None else base
    missing = [
        block
        for block in (StitchImagesBlock, StitchOCRDetectionsBlock)
        if block.type not in catalogue.block_types
    ]
    extended = catalogue.with_blocks(missing)

    return extended


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _run(definition: dict, inputs: Dict[str, Any]) -> RunResult:
    session = compile_workflow(definition, catalogue=_catalogue()).create_session()
    try:
        result = session.run(inputs)
    finally:
        session.close()

    return result


def _scene(seed: int) -> np.ndarray:
    # (160, 320, 3) RGB: smooth noise plus filled circles, rich in SIFT corners.
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(40, 80, 3), dtype=np.uint8)
    pixels = cv.resize(noise, (320, 160), interpolation=cv.INTER_CUBIC)
    for _ in range(25):
        x, y = int(rng.integers(0, 310)), int(rng.integers(0, 150))
        color = tuple(int(value) for value in rng.integers(0, 256, size=3))
        cv.circle(pixels, (x, y), int(rng.integers(3, 12)), color, -1)

    return pixels


def _marked_scene() -> np.ndarray:
    # The marker detector looks for red 255; the scene keeps red below it.
    pixels = _scene(6)
    pixels[..., 0] = np.minimum(pixels[..., 0], 254)
    for x0, y0, x1, y1, number, class_id in MARKERS:
        pixels[y0:y1, x0:x1] = (255, number, class_id)

    return pixels


def _panorama_detections_definition(**output_settings: Any) -> dict:
    # stitch -> detect on the panorama -> crop the panorama by its detections
    definition = {
        "version": "2.0",
        "inputs": list(IMAGE_PAIR_INPUTS),
        "steps": [
            {
                "type": StitchImagesBlock.type,
                "name": "stitch",
                "image1": "$inputs.left",
                "image2": "$inputs.right",
            },
            {
                "type": "test/marker_detector@v1",
                "name": "detect",
                "image": "$steps.stitch.stitched_image",
            },
            {
                "type": "v2/dynamic_crop@v1",
                "name": "crop",
                "image": "$steps.stitch.stitched_image",
                "predictions": "$steps.detect.predictions",
            },
        ],
        "outputs": [
            {**_output("detections", "$steps.detect.predictions"), **output_settings},
            _output("crops", "$steps.crop.crops"),
        ],
    }

    return definition


def _run_panorama_detections(definition: dict, *, mode: str) -> RunResult:
    scene = _marked_scene()
    plan = compile_workflow(
        definition,
        catalogue=_catalogue(regions_catalogue()),
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()
    try:
        result = session.run({"left": [scene[:, :200]], "right": [scene[:, 120:]]})
    finally:
        session.close()

    return result


def _ocr_detections(layout: Dict[str, tuple]) -> Detections:
    texts = list(layout)
    predictions = Detections(
        xyxy=torch.tensor(
            [layout[text] for text in texts], dtype=torch.float32
        ).reshape(-1, 4),
        class_id=torch.arange(len(texts)),
        confidence=torch.ones(len(texts)),
        image_metadata={"class_names": dict(enumerate(texts))},
        bboxes_metadata=[{} for _ in texts],
    )
    identified = with_row_identity(predictions)

    return identified


def _stitch_definition(**params: Any) -> dict:
    definition = {
        "version": "2.0",
        "inputs": list(IMAGE_PAIR_INPUTS),
        "steps": [
            {
                "type": StitchImagesBlock.type,
                "name": "stitch",
                "image1": "$inputs.left",
                "image2": "$inputs.right",
                **params,
            },
            {
                "type": "v2/invert",
                "name": "invert",
                "image": "$steps.stitch.stitched_image",
            },
        ],
        "outputs": [
            _output("stitched", "$steps.stitch.stitched_image"),
            _output("inverted", "$steps.invert.image"),
        ],
    }

    return definition


def _ocr_definition(**params: Any) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [PREDICTIONS_INPUT],
        "steps": [
            {
                "type": StitchOCRDetectionsBlock.type,
                "name": "text",
                "predictions": "$inputs.predictions",
                **params,
            }
        ],
        "outputs": [_output("text", "$steps.text.ocr_text")],
    }

    return definition


def test_stitched_pair_and_failed_pair_in_one_batch() -> None:
    # given: row 0 overlaps by 80 columns; row 1 is flat and cannot stitch
    scene = _scene(6)
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    inputs = {"left": [scene[:, :200], flat], "right": [scene[:, 120:], flat]}

    # when
    rows = _run(_stitch_definition(), inputs).rows()

    # then: row 0 is a composite listing only its anchor tile; row 1 gives
    # None and the downstream step produces nothing for it
    stitched = rows[0]["stitched"]
    assert isinstance(stitched, ImageData)
    assert stitched.size_hw == (160, 320)
    assert [tile.index for tile in stitched.composite_sources] == [(1,)]
    inverted = rows[0]["inverted"].tensor_image
    assert torch.equal(inverted, 255 - stitched.tensor_image)
    assert rows[1]["stitched"] is None
    assert rows[1]["inverted"] is None


def test_max_canvas_pixels_from_a_reference_is_enforced() -> None:
    # given
    scene = _scene(6)
    definition = _stitch_definition(max_canvas_pixels="$inputs.limit")
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "limit", "kind": ["integer"]}
    )
    inputs = {"left": [scene[:, :200]], "right": [scene[:, 120:]], "limit": 10_000}

    # when / then
    with pytest.raises(StepExecutionError, match="max_canvas_pixels=10000"):
        _run(definition, inputs)


@pytest.mark.parametrize(
    "params, field",
    [
        ({"max_canvas_pixels": 0}, "max_canvas_pixels"),
        ({"count_of_best_matches_per_query_descriptor": 1}, "count_of_best"),
        ({"max_allowed_reprojection_error": 0.0}, "max_allowed_reprojection_error"),
    ],
)
def test_invalid_stitch_literals_fail_compilation(params: dict, field: str) -> None:
    # when
    with pytest.raises(ParamsValidationError) as error:
        compile_workflow(_stitch_definition(**params), catalogue=_catalogue())

    # then
    assert error.value.step_path == ("stitch",)
    assert error.value.field_path[0].startswith(field)


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_panorama_detections_output_in_own_coordinates(mode: str) -> None:
    # given: the JsonField asks for the coordinates of the predicted image
    definition = _panorama_detections_definition(coordinates_system="own")

    # when
    row = _run_panorama_detections(definition, mode=mode).rows()[0]

    # then: boxes are in panorama pixels, which here coincide with the scene
    detections = row["detections"]
    expected = torch.tensor([marker[:4] for marker in MARKERS], dtype=torch.float32)
    assert torch.allclose(detections.xyxy, expected, atol=1.0)
    assert [crop.size_hw for crop in row["crops"]] == [
        (y1 - y0, x1 - x0) for x0, y0, x1, y1 in detections.xyxy.int().tolist()
    ]


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_panorama_detections_in_default_parent_coordinates_are_refused(
    mode: str,
) -> None:
    # given: default parent coordinates; the panorama has no single input,
    # so its predictions cannot be mapped into one (deliberate V1 difference)
    definition = _panorama_detections_definition()
    result = _run_panorama_detections(definition, mode=mode)

    # when / then: outputs are converted when rows are read
    with pytest.raises(WorkflowExecutionError, match=r"coordinates_system\W+own"):
        result.rows()


def test_ocr_text_per_image_with_auto_direction_and_empty_image() -> None:
    # given: wide boxes, tall boxes and no boxes in one batch
    empty = _ocr_detections({})
    predictions = [_ocr_detections(WIDE), _ocr_detections(TALL), empty]

    # when
    rows = _run(
        _ocr_definition(reading_direction="auto"), {"predictions": predictions}
    ).rows()

    # then: auto is decided per image
    assert [row["text"] for row in rows] == ["AB\nCD", "AB CD", ""]


@pytest.mark.parametrize("algorithm", ["tolerance", "otsu", "collimate"])
def test_ocr_algorithm_and_direction_from_references(algorithm: str) -> None:
    # given
    definition = _ocr_definition(
        stitching_algorithm="$inputs.algorithm",
        reading_direction="$inputs.direction",
    )
    definition["inputs"] += [
        {"type": "WorkflowParameter", "name": name, "kind": ["string"]}
        for name in ("algorithm", "direction")
    ]
    inputs = {
        "predictions": [_ocr_detections(WIDE)],
        "algorithm": algorithm,
        "direction": "right_to_left",
    }

    # when
    rows = _run(definition, inputs).rows()

    # then
    assert rows[0]["text"] == "BA\nDC"


def test_ocr_unknown_direction_from_a_reference_raises() -> None:
    # given
    definition = _ocr_definition(reading_direction="$inputs.direction")
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "direction", "kind": ["string"]}
    )
    inputs = {"predictions": [_ocr_detections(WIDE)], "direction": "diagonal"}

    # when / then
    with pytest.raises(StepExecutionError, match="reading_direction must be one of"):
        _run(definition, inputs)


def test_ocr_zero_tolerance_from_a_reference_raises() -> None:
    # given
    definition = _ocr_definition(
        reading_direction="left_to_right", tolerance="$inputs.tolerance"
    )
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "tolerance", "kind": ["integer"]}
    )
    inputs = {"predictions": [_ocr_detections(WIDE)], "tolerance": 0}

    # when / then
    with pytest.raises(StepExecutionError, match="tolerance is invalid.*than 0"):
        _run(definition, inputs)


@pytest.mark.parametrize(
    "params, field",
    [
        ({"stitching_algorithm": "fastest"}, "stitching_algorithm"),
        ({"reading_direction": "diagonal"}, "reading_direction"),
        ({"tolerance": 0}, "tolerance"),
        ({"collimate_tolerance": -1}, "collimate_tolerance"),
    ],
)
def test_invalid_ocr_literals_fail_compilation(params: dict, field: str) -> None:
    # when
    with pytest.raises(ParamsValidationError) as error:
        compile_workflow(
            _ocr_definition(**{"reading_direction": "left_to_right", **params}),
            catalogue=_catalogue(),
        )

    # then
    assert error.value.step_path == ("text",)
    assert error.value.field_path[0].startswith(field)


def test_ocr_omitted_reading_direction_fails_compilation() -> None:
    # The field is required, as in V1; its former V2 default "auto" read
    # horizontal character boxes as columns.
    with pytest.raises(ParamsValidationError) as error:
        compile_workflow(_ocr_definition(), catalogue=_catalogue())

    # then
    assert error.value.step_path == ("text",)
    assert error.value.field_path[0] == "reading_direction"


def test_ocr_rows_without_text_fail_the_step() -> None:
    # given: class id 1 has no name and the rows carry no text
    predictions = _ocr_detections(WIDE)
    predictions.image_metadata["class_names"].pop(1)

    # when / then
    with pytest.raises(StepExecutionError, match="row 1 has no text"):
        _run(
            _ocr_definition(reading_direction="left_to_right"),
            {"predictions": [predictions]},
        )
