"""Unit tests of ``roboflow_core/template_matching@v1`` on pasted synthetic templates."""

import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.template_matching.v1 import TemplateMatchingBlock
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import row_record

# Template corners (x, y) in the scene; the template is 20 wide, 12 high.
CORNERS = [(37, 21), (90, 50)]


def template_pixels():
    generator = torch.Generator().manual_seed(3)
    cells = torch.randint(0, 256, (3, 3, 5), dtype=torch.uint8, generator=generator)

    return cells.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)


def scene_pixels(corners=CORNERS):
    # Fine pixel noise, so only the pasted copies correlate with the template.
    generator = torch.Generator().manual_seed(4)
    scene = torch.randint(0, 256, (3, 100, 160), dtype=torch.uint8, generator=generator)
    for x, y in corners:
        scene[:, y : y + 12, x : x + 20] = template_pixels()

    return scene


def match(scene, template, **options):
    arguments = {"matching_threshold": 0.8, "apply_nms": True, "nms_threshold": 0.5}
    arguments.update(options)

    return TemplateMatchingBlock().run(image=scene, template=template, **arguments)


def scene_image(**options):
    return ImageData.from_tensor(scene_pixels(), image_id="scene", **options)


def template_image():
    return ImageData.from_tensor(template_pixels(), image_id="template")


def test_two_pasted_copies_give_two_boxes_at_their_corners():
    result = match(scene_image(), template_image())

    predictions = result["predictions"]
    assert result["number_of_matches"] == 2
    boxes = sorted(predictions.xyxy.tolist())
    assert boxes == [[37.0, 21.0, 57.0, 33.0], [90.0, 50.0, 110.0, 62.0]]
    assert torch.allclose(predictions.confidence, torch.ones(2), atol=1e-4)
    assert predictions.class_id.tolist() == [0, 0]
    assert predictions.image_metadata["class_names"] == {0: "template_match"}
    assert row_record(predictions) is not None


def test_nms_keeps_the_best_score_among_overlapping_matches():
    # Without NMS the neighbours of each exact position also pass a low
    # threshold; NMS must keep the exact positions (V1 kept any one of them).
    raw = match(
        scene_image(), template_image(), apply_nms=False, matching_threshold=0.3
    )
    kept = match(scene_image(), template_image(), matching_threshold=0.3)

    raw_corners = raw["predictions"].xyxy[:, :2].tolist()
    kept_corners = kept["predictions"].xyxy[:, :2].tolist()
    assert [38.0, 21.0] in raw_corners and [90.0, 51.0] in raw_corners
    assert [37.0, 21.0] in kept_corners and [90.0, 50.0] in kept_corners
    assert [38.0, 21.0] not in kept_corners and [90.0, 51.0] not in kept_corners


def test_threshold_above_every_score_gives_empty_detections_in_the_frame():
    result = match(scene_image(), template_image(), matching_threshold=1.01)

    assert result["number_of_matches"] == 0
    assert result["predictions"].xyxy.shape == (0, 4)
    assert frame_of(result["predictions"]).image_id == "scene"


def test_boxes_on_a_crop_are_in_the_crop_frame():
    crop = scene_image().crop((30, 10, 130, 90), image_id="crop")

    predictions = match(crop, template_image())["predictions"]

    assert sorted(predictions.xyxy.tolist()) == [
        [7.0, 11.0, 27.0, 23.0],
        [60.0, 40.0, 80.0, 52.0],
    ]
    frame = frame_of(predictions)
    assert frame.image_id == "crop" and frame.size_hw == (80, 100)
    assert frame.root.map_xy(7.0, 11.0) == (37.0, 21.0)


def test_one_channel_inputs_match_like_rgb():
    gray_scene = ImageData.from_tensor(scene_pixels()[:1].contiguous(), image_id="g")
    gray_template = ImageData.from_tensor(template_pixels()[:1].contiguous())

    result = match(gray_scene, gray_template)

    assert sorted(result["predictions"].xyxy[:, :2].tolist()) == [
        [37.0, 21.0],
        [90.0, 50.0],
    ]


def test_template_larger_than_image_is_a_contract_error():
    small = ImageData.from_tensor(torch.zeros((3, 10, 30), dtype=torch.uint8))

    with pytest.raises(ContractError, match="does not fit"):
        match(small, template_image())


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_nms_threshold_outside_zero_to_one_is_rejected(value):
    with pytest.raises(Exception, match="nms_threshold"):
        TemplateMatchingBlock.Params.model_validate(
            {"image": "$inputs.image", "template": "$inputs.t", "nms_threshold": value}
        )
