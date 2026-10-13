"""Dynamic zones, contours and dominant colors through the real V2 compiler.

Flows:
* Segment painted markers -> dynamic zone, from predictions and from a
  segmentation view, with the vertex count bound at run time.
* Fixed boxes (one off the image) -> dynamic crop -> dominant color: the
  empty crop keeps its slot, so colors stay aligned with rows.
* Dominant color ``hex_color`` -> a drawing block's ``color`` field.
* Static crop -> contours, in the crop's own pixels and frame.
Segmentation and detection are the deterministic stand-ins of
``deterministic.py``; expected values follow from the painted markers.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.geometry.contours import ContoursDetectionBlock
from roboflow_workflows.v2.blocks.geometry.dynamic_zone import DynamicZoneBlock
from roboflow_workflows.v2.blocks.image.dominant_color import DominantColorBlock
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections
from tests.v2.integration.blocks.geometry.deterministic import (
    painted_image,
    regions_catalogue,
)

REGION_COLOR_BLOCKS = (DynamicZoneBlock, ContoursDetectionBlock, DominantColorBlock)
# The middle box lies outside the 60 x 80 image, so its crop is empty.
FIXED_BOXES = [
    [5.0, 5.0, 25.0, 25.0],
    [200.0, 200.0, 220.0, 220.0],
    [30.0, 10.0, 60.0, 40.0],
]


class FixedBoxesBlock(Block):
    """Emit ``FIXED_BOXES`` for every image."""

    type = "test/fixed_boxes@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image the boxes belong to.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        count = len(FIXED_BOXES)
        detections = Detections(
            xyxy=torch.tensor(FIXED_BOXES, dtype=torch.float32),
            class_id=torch.zeros(count, dtype=torch.int64),
            confidence=torch.full((count,), 0.9),
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                "class_names": {0: "box"},
            },
        )

        return {"predictions": with_row_identity(detections)}


# (x0, y0, x1, y1, marker number, class id); boxes exclusive at x1 / y1.
MARKERS = [(10, 10, 30, 20, 1, 0), (40, 25, 60, 45, 2, 1)]
# Zone corners run through the outer pixel centres of each marker.
CORNERS = {
    1: [(10, 10), (10, 19), (29, 10), (29, 19)],
    2: [(40, 25), (40, 44), (59, 25), (59, 44)],
}


def run(steps, outputs, *, inputs, extra_inputs=()):
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *({"type": "WorkflowParameter", "name": name} for name in extra_inputs),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = regions_catalogue().with_blocks([*REGION_COLOR_BLOCKS, FixedBoxesBlock])
    plan = compile_workflow(workflow, catalogue=catalogue)

    return plan.create_session().run(inputs)


def zone_steps(source):
    return [
        {
            "type": "test/masked_marker_detector@v1",
            "name": "segment",
            "image": "$inputs.image",
        },
        {
            "type": "v2/dynamic_zone@v1",
            "name": "zone",
            "predictions": f"$steps.segment.{source}",
            "required_number_of_vertices": "$inputs.vertices",
        },
    ]


ZONE_OUTPUTS = {
    "zones": "$steps.zone.zones",
    "converged": "$steps.zone.simplification_converged",
    "predictions": "$steps.zone.predictions",
}


@pytest.mark.parametrize("source", ["predictions", "segmentation"])
def test_zones_follow_their_marker_rows(source):
    image = painted_image((60, 80), MARKERS)

    result = run(
        zone_steps(source),
        ZONE_OUTPUTS,
        inputs={"image": [image], "vertices": 4},
        extra_inputs=["vertices"],
    )

    row = result.rows()[0]
    assert row["converged"] is True
    markers = [meta["marker"] for meta in row["predictions"].bboxes_metadata]
    zones = [sorted(map(tuple, zone)) for zone in row["zones"]]
    assert zones == [CORNERS[marker] for marker in markers]
    painted = torch.from_numpy(image[..., 0] == 255)
    assert torch.equal(row["predictions"].mask.any(dim=0), painted)


def test_runtime_vertex_count_is_checked():
    with pytest.raises(Exception, match="required_number_of_vertices"):
        run(
            zone_steps("predictions"),
            ZONE_OUTPUTS,
            inputs={"image": [painted_image((60, 80), MARKERS)], "vertices": 2},
            extra_inputs=["vertices"],
        )


def test_image_without_markers_has_no_zones():
    result = run(
        zone_steps("segmentation"),
        ZONE_OUTPUTS,
        inputs={"image": [painted_image((60, 80), [])], "vertices": 4},
        extra_inputs=["vertices"],
    )

    row = result.rows()[0]
    assert row["zones"] == [] and row["converged"] is True


def test_dominant_color_keeps_the_slot_of_an_empty_crop():
    steps = [
        {"type": "test/fixed_boxes@v1", "name": "boxes", "image": "$inputs.image"},
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.boxes.predictions",
        },
        {
            "type": "v2/dominant_color@v1",
            "name": "color",
            "image": "$steps.crop.crops",
            "color_clusters": 2,
        },
    ]
    # Markers fill the first and last boxes; marker n is painted (255, n, class).
    markers = [(5, 5, 25, 25, 1, 0), (30, 10, 60, 40, 2, 1)]

    result = run(
        steps,
        {"colors": "$steps.color.hex_color", "crops": "$steps.crop.crops"},
        inputs={"image": [painted_image((60, 80), markers)]},
    )

    (row,) = result.rows()
    assert row["crops"][1] is None
    assert row["colors"] == ["#FF0100", None, "#FF0201"]


def label_step(name, text_color):
    return {
        "type": "v2/classification_label_visualization@v1",
        "name": name,
        "image": "$inputs.image",
        "predictions": "$steps.classify.predictions",
        "text_color": text_color,
    }


def test_hex_color_drives_a_drawing_color():
    # 3/4 of the image is (30, 160, 90) = #1EA05A, so that color dominates.
    pixels = np.zeros((60, 160, 3), dtype=np.uint8)
    pixels[:, 40:] = (30, 160, 90)
    steps = [
        {"type": "v2/dominant_color@v1", "name": "color", "image": "$inputs.image"},
        {
            "type": "test/color_classifier@v1",
            "name": "classify",
            "image": "$inputs.image",
        },
        label_step("from_hex", "$steps.color.hex_color"),
        label_step("literal", "#1EA05A"),
        label_step("white", "WHITE"),
    ]

    result = run(
        steps,
        {
            "hex": "$steps.color.hex_color",
            "from_hex": "$steps.from_hex.image",
            "literal": "$steps.literal.image",
            "white": "$steps.white.image",
        },
        inputs={"image": [pixels]},
    )

    (row,) = result.rows()
    assert row["hex"] == "#1EA05A"
    drawn = row["from_hex"].tensor_image
    assert torch.equal(drawn, row["literal"].tensor_image)
    assert not torch.equal(drawn, row["white"].tensor_image)


def test_catalogue_lists_name_section_and_v1_origin():
    described = {
        entry["type"]: entry["metadata"]
        for entry in regions_catalogue()
        .with_blocks(list(REGION_COLOR_BLOCKS))
        .describe()["blocks"]
    }

    assert [described[block.type] for block in REGION_COLOR_BLOCKS] == [
        {
            "name": "Dynamic Zone",
            "port_of": "roboflow_core/dynamic_zone@v1",
            "section": "advanced",
        },
        {
            "name": "Image Contours",
            "port_of": "roboflow_core/contours_detection@v1",
            "section": "classical_cv",
        },
        {
            "name": "Dominant Color",
            "port_of": "roboflow_core/dominant_color@v1",
            "section": "classical_cv",
        },
    ]


def test_contours_of_a_static_crop_are_in_crop_pixels():
    steps = [
        {
            "type": "v2/absolute_static_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "x_center": 50,
            "y_center": 35,
            "width": 40,
            "height": 30,
        },
        {
            "type": "v2/contours_detection@v1",
            "name": "trace",
            "image": "$steps.crop.crops",
        },
    ]

    result = run(
        steps,
        {
            "count": "$steps.trace.number_contours",
            "contours": "$steps.trace.contours",
            "drawn": "$steps.trace.image",
            "crop": "$steps.crop.crops",
        },
        inputs={"image": [painted_image((60, 80), MARKERS)]},
    )

    data = {name: value[0] for name, value in result.outputs.data.items()}
    # The crop spans x 30..69, y 20..49: marker 2 only, shifted by (-30, -20).
    assert data["count"] == 1
    (contour,) = data["contours"]
    assert sorted(map(tuple, contour.tolist())) == [
        (10, 5),
        (10, 24),
        (29, 5),
        (29, 24),
    ]
    assert data["drawn"].root == data["crop"].root
    assert data["drawn"].image_id == data["crop"].image_id
