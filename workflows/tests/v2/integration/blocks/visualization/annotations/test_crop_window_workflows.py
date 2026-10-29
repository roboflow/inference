"""Compiled crop visualization of parent rows on crops of the parent.

The real instance segmentation and dynamic crop blocks feed the crop
visualization; every parent row is drawn on every crop::

    image ─▶ seg ─▶ dynamic crop ─▶ crop visualization (seg rows on each crop)

Parent 96x64, objects and their boxes::

    RED    ellipse  (14,  8, 46, 40)
    BLUE   triangle (30, 24, 66, 60)   its crop is 36x36
    YELLOW ellipse  ( 2, 40, 24, 56)

On BLUE's crop, RED maps to (-16, -16, 16, 16): it crosses the left and top
edges, and its visible part (0, 0, 16, 16) is copied to the BOTTOM_RIGHT anchor
(16, 16). YELLOW maps to (-28, 16, -6, 32), entirely left of the crop: it
draws nothing. Every other row on every crop lies outside or anchors outside.
"""

import cv2
import numpy as np
import pytest
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations.scene import (
    BLUE,
    RED,
    YELLOW,
    SceneLoader,
    Shape,
    annotation_catalogue,
    pixels_of,
    scene_image,
    scene_pixels,
)

SIZE_HW = (64, 96)
SHAPES = (
    Shape("ellipse", (14, 8, 46, 40), RED),
    Shape("triangle", (30, 24, 66, 60), BLUE),
    Shape("ellipse", (2, 40, 24, 56), YELLOW),
)
BORDER = (255, 0, 255)
DEFINITION = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
            "name": "seg",
            "image": "$inputs.image",
            "model_id": "shapes-seg",
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.seg.predictions",
        },
        {
            "type": "roboflow_core/crop_visualization@v1",
            "name": "magnified",
            "image": "$steps.crop.crops",
            "predictions": "$steps.seg.predictions",
            "position": "BOTTOM_RIGHT",
            "scale_factor": 1.0,
            "border_thickness": 1,
            "color_palette": "CUSTOM",
            "custom_colors": ["#FF00FF"],
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "crops", "selector": "$steps.crop.crops"},
        {"type": "JsonField", "name": "drawn", "selector": "$steps.magnified.image"},
        {"type": "JsonField", "name": "rows", "selector": "$steps.seg.predictions"},
    ],
}


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_parent_rows_crossing_or_missing_a_crop_draw_only_visible_pixels(
    mode: str,
) -> None:
    # given
    image = scene_image("lot", SIZE_HW, SHAPES, road=False)
    parent = scene_pixels(SIZE_HW, SHAPES, road=False)
    plan = compile_workflow(
        DEFINITION,
        catalogue=annotation_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    # when
    (row,) = session.run({"image": [image]}).rows()

    # then: the model found the three boxes and cut one crop per box
    assert row["rows"].xyxy.tolist() == [list(map(float, s.xyxy)) for s in SHAPES]
    for crop, shape in zip(row["crops"], SHAPES):
        x0, y0, x1, y1 = shape.xyxy
        np.testing.assert_array_equal(pixels_of(crop), parent[y0:y1, x0:x1])

    # RED's and YELLOW's crops: no row has a visible copy on them
    red, blue, yellow = zip(row["crops"], row["drawn"])
    for crop, drawn in (red, yellow):
        assert drawn.image_id == crop.image_id
        np.testing.assert_array_equal(pixels_of(drawn), pixels_of(crop))

    # BLUE's crop: parent pixels (30..46, 24..40), RED's visible part, copied
    # at (16, 16) with its border; YELLOW's wrapped pixels must not appear
    crop, drawn = blue
    expected = pixels_of(crop).copy()
    expected[16:32, 16:32] = parent[24:40, 30:46]
    cv2.rectangle(expected, (16, 16), (32, 32), BORDER, 1)
    assert drawn.image_id == crop.image_id
    np.testing.assert_array_equal(pixels_of(drawn), expected)

    # inputs unchanged
    np.testing.assert_array_equal(pixels_of(image), parent)
    assert row["rows"].xyxy.tolist() == [list(map(float, s.xyxy)) for s in SHAPES]
