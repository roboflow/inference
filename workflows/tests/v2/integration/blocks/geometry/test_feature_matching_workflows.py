"""Feature-matching blocks composed through the real V2 compiler and session.

Flows:
* template matching -> nearest neighbor -> geotag, on a batch of two images.
* template matching -> dynamic crop -> template matching per crop -> geotag:
  crop boxes geotag at the same place as the full-image boxes.
* SIFT on two images -> SIFT comparison of the descriptor tensors.
Inputs are native ``ImageData``. Expected values are worked out from the
paste positions, not read from a previous run.
"""

import math

import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.geometry.geotag import GeotagDetectionBlock
from roboflow_workflows.v2.blocks.geometry.nearest_neighbor import (
    DetectionsNearestNeighborBlock,
)
from roboflow_workflows.v2.blocks.geometry.sift import SiftBlock
from roboflow_workflows.v2.blocks.geometry.sift_comparison import SiftComparisonBlock
from roboflow_workflows.v2.blocks.geometry.template_matching import (
    TemplateMatchingBlock,
)
from roboflow_workflows.v2.blocks.image.dynamic_crop import DynamicCropBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow

# Camera: 100 m up, 90 degree field of view over a 160 px wide image, so one
# pixel is 200 / 160 = 1.25 m. Image centre (80, 50) is at latitude 0, longitude 0.
METERS_PER_PIXEL = 1.25
DEGREES_PER_METER = 1 / 111320
CAMERA = {"latitude": 0.0, "longitude": 0.0, "altitude": 100.0, "horizontal_fov": 90.0}


def texture(shape, *, block, seed):
    channels, height, width = shape
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randint(
        0,
        256,
        (channels, height // block, width // block),
        dtype=torch.uint8,
        generator=generator,
    )

    return cells.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)


def template_pixels():
    return texture((3, 12, 20), block=4, seed=3)


def scene(corners, image_id):
    # Fine pixel noise, so only the pasted copies correlate with the template.
    pixels = texture((3, 100, 160), block=1, seed=4)
    for x, y in corners:
        pixels[:, y : y + 12, x : x + 20] = template_pixels()

    return ImageData.from_tensor(pixels, image_id=image_id)


def template():
    return ImageData.from_tensor(template_pixels(), image_id="template")


def run(steps, outputs, *, inputs, images=("image",)):
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": name} for name in images],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = create_catalogue().with_blocks(
        [
            TemplateMatchingBlock,
            DetectionsNearestNeighborBlock,
            GeotagDetectionBlock,
            SiftBlock,
            SiftComparisonBlock,
            DynamicCropBlock,
        ]
    )
    result = (
        compile_workflow(workflow, catalogue=catalogue).create_session().run(inputs)
    )

    return result


def template_step(image="$inputs.image"):
    return {
        "type": "v2/template_matching@v1",
        "name": "match",
        "image": image,
        "template": "$inputs.template",
        "matching_threshold": 0.9,
    }


def geotag_step(image, predictions):
    return {
        "type": "v2/geotag_detection@v1",
        "name": "geotag",
        "image": image,
        "predictions": predictions,
        **CAMERA,
    }


def expected_position(x, y):
    # Box centre (x, y) -> (lat, lon) for the camera above, heading 0.
    east = (x - 80) * METERS_PER_PIXEL
    north = -(y - 50) * METERS_PER_PIXEL
    return round(north * DEGREES_PER_METER, 7), round(east * DEGREES_PER_METER, 7)


def test_matches_pair_up_and_geotag_per_image_of_a_batch():
    corners = [[(37, 21), (90, 50)], [(10, 10), (130, 70)]]
    steps = [
        template_step(),
        {
            "type": "v2/detections_nearest_neighbor@v1",
            "name": "pairs",
            "query_predictions": "$steps.match.predictions",
            "target_predictions": "$steps.match.predictions",
        },
        geotag_step("$inputs.image", "$steps.pairs.query_predictions"),
    ]

    result = run(
        steps,
        {
            "query": "$steps.pairs.query_predictions",
            "geo": "$steps.geotag.geo_detections",
        },
        inputs={
            "image": [scene(corners[0], "a"), scene(corners[1], "b")],
            "template": [template(), template()],
        },
        images=("image", "template"),
    )

    for row, image_corners in zip(result.rows(), corners):
        (x1, y1), (x2, y2) = image_corners
        distance = math.hypot(x2 - x1, y2 - y1)
        query = row["query"]
        assert sorted(query.xyxy[:, :2].tolist()) == sorted(
            [[float(x), float(y)] for x, y in image_corners]
        )
        assert query.bboxes_metadata[0]["nearest_target_distance"] == pytest.approx(
            distance, abs=1e-4
        )
        centres = sorted((x + 10, y + 6) for x, y in image_corners)
        assert sorted((r["pixel_x"], r["pixel_y"]) for r in row["geo"]) == centres
        assert sorted((r["lat"], r["lon"]) for r in row["geo"]) == sorted(
            expected_position(x, y) for x, y in centres
        )


def test_matches_found_in_crops_geotag_like_full_image_matches():
    corners = [(37, 21), (90, 50)]
    steps = [
        template_step(),
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.match.predictions",
        },
        {**template_step("$steps.crop.crops"), "name": "match_in_crop"},
        geotag_step("$steps.crop.crops", "$steps.match_in_crop.predictions"),
    ]

    result = run(
        steps,
        {
            "crop_boxes": "$steps.match_in_crop.predictions",
            "geo": "$steps.geotag.geo_detections",
        },
        inputs={"image": [scene(corners, "scene")], "template": [template()]},
        images=("image", "template"),
    )

    row = result.rows()[0]
    # Each crop holds one whole copy at its origin; the workflow output maps
    # it back into the input image (default parent coordinates, as V1).
    assert sorted(boxes.xyxy.tolist() for boxes in row["crop_boxes"]) == [
        [[37.0, 21.0, 57.0, 33.0]],
        [[90.0, 50.0, 110.0, 62.0]],
    ]
    records = [geo[0] for geo in row["geo"]]
    centres = sorted((x + 10, y + 6) for x, y in corners)
    assert sorted((r["pixel_x"], r["pixel_y"]) for r in records) == centres
    assert sorted((r["lat"], r["lon"]) for r in records) == sorted(
        expected_position(x, y) for x, y in centres
    )


@pytest.mark.parametrize("seed_2, expected_match", [(0, True), (9, False)])
def test_sift_descriptors_compare_through_the_engine(seed_2, expected_match):
    texture_1 = texture((3, 200, 200), block=8, seed=0)
    # Same texture moved 24 px right (a match), or an unrelated texture.
    texture_2 = torch.roll(texture((3, 200, 200), block=8, seed=seed_2), 24, dims=2)
    steps = [
        {"type": "v2/sift@v1", "name": "sift_1", "image": "$inputs.image"},
        {"type": "v2/sift@v1", "name": "sift_2", "image": "$inputs.other"},
        {
            "type": "v2/sift_comparison@v2",
            "name": "compare",
            "input_1": "$steps.sift_1.descriptors",
            "input_2": "$steps.sift_2.descriptors",
            "good_matches_threshold": 50,
        },
    ]

    result = run(
        steps,
        {
            "match": "$steps.compare.images_match",
            "count": "$steps.compare.good_matches_count",
            "keypoints": "$steps.compare.keypoints_1",
            "descriptors": "$steps.sift_1.descriptors",
        },
        inputs={
            "image": [ImageData.from_tensor(texture_1, image_id="one")],
            "other": [ImageData.from_tensor(texture_2, image_id="two")],
        },
        images=("image", "other"),
    )

    row = result.rows()[0]
    assert row["match"] is expected_match
    assert (row["count"] >= 50) is expected_match
    assert row["descriptors"].shape[1] == 128
    assert row["keypoints"] is None
