"""Unit tests of ``roboflow_core/sift_comparison@v2`` against OpenCV references."""

import cv2
import pytest
import torch
from roboflow_workflows.v2.blocks.geometry.sift.v1 import SiftBlock
from roboflow_workflows.v2.blocks.geometry.sift_comparison.v2 import SiftComparisonBlock
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData


def texture(size=200, block=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cells = torch.randint(
        0,
        256,
        (3, size // block, size // block),
        dtype=torch.uint8,
        generator=generator,
    )

    return cells.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)


def image(pixels, image_id):
    return ImageData.from_tensor(pixels.contiguous(), image_id=image_id)


def compare(input_1, input_2, **options):
    arguments = {
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
        "matcher": "FlannBasedMatcher",
        "visualize": False,
    }
    arguments.update(options)

    return SiftComparisonBlock().run(input_1=input_1, input_2=input_2, **arguments)


def descriptors(pixels):
    return SiftBlock().run(image=image(pixels, "d"))["descriptors"]


def opencv_good_matches(descriptors_1, descriptors_2, ratio):
    # V1's BFMatcher path on the same descriptors.
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors_1.numpy(), descriptors_2.numpy(), k=2
    )

    return sum(1 for m, n in pairs if m.distance < ratio * n.distance)


@pytest.mark.parametrize("matcher", ["FlannBasedMatcher", "BFMatcher"])
def test_translated_and_rotated_views_match_noise_does_not(matcher):
    base = texture()
    reference = image(base[:, :160, :160], "reference")
    shifted = image(base[:, 9:169, 14:174], "shifted")
    rotated = image(torch.rot90(base[:, :160, :160], 1, dims=(1, 2)), "rotated")
    noise = image(texture(seed=5)[:, :160, :160], "noise")

    assert compare(reference, shifted, matcher=matcher)["images_match"]
    assert compare(reference, rotated, matcher=matcher)["images_match"]
    unrelated = compare(reference, noise, matcher=matcher)
    assert not unrelated["images_match"]
    assert unrelated["good_matches_count"] < 50


def test_bf_count_equals_opencv_ratio_test_and_threshold_is_inclusive():
    base = texture()
    descriptors_1 = descriptors(base[:, :160, :160])
    descriptors_2 = descriptors(base[:, 9:169, 14:174])
    expected = opencv_good_matches(descriptors_1, descriptors_2, ratio=0.7)

    at_threshold = compare(
        descriptors_1,
        descriptors_2,
        matcher="BFMatcher",
        good_matches_threshold=expected,
    )
    above = compare(
        descriptors_1,
        descriptors_2,
        matcher="BFMatcher",
        good_matches_threshold=expected + 1,
    )

    assert expected > 50
    assert at_threshold["good_matches_count"] == expected
    assert at_threshold["images_match"] and not above["images_match"]


def test_descriptor_inputs_pass_through_without_keypoints_or_drawings():
    base = texture()
    descriptors_1 = descriptors(base[:, :160, :160])

    result = compare(descriptors_1, image(base[:, :160, :160], "frame"), visualize=True)

    assert result["descriptors_1"] is descriptors_1
    assert result["keypoints_1"] is None and result["visualization_1"] is None
    assert result["keypoints_2"].shape[1] == 7
    assert result["visualization_2"].image_id == "frame"
    assert result["visualization_matches"] is None


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_descriptors_give_no_matches(count):
    few = torch.zeros((count, 128), dtype=torch.float32)

    result = compare(few, descriptors(texture()), good_matches_threshold=1)

    assert result["good_matches_count"] == 0 and result["images_match"] is False


def test_flat_image_side_gives_no_matches_but_keeps_its_drawing():
    flat = image(torch.full((3, 50, 50), 128, dtype=torch.uint8), "flat")

    result = compare(flat, image(texture(), "rich"), visualize=True)

    assert result["good_matches_count"] == 0
    assert result["keypoints_1"].shape == (0, 7)
    assert result["visualization_1"].image_id == "flat"
    assert result["visualization_matches"] is None


def test_match_drawing_is_a_composite_of_both_inputs():
    base = texture()
    left = image(base[:, :160, :120], "left")
    right = image(base[:, 5:105, 7:107], "right")

    drawing = compare(left, right, visualize=True)["visualization_matches"]

    assert drawing.size_hw == (160, 220) and drawing.channels == 3
    tiles = drawing.composite_sources
    assert [tile.image_id for tile in tiles] == ["left", "right"]
    assert tiles[0].canvas_xyxy == (0, 0, 120, 160)
    assert tiles[1].canvas_xyxy == (120, 0, 220, 100)
    assert tiles[1].root.map_xy(130.0, 10.0) == (10.0, 10.0)


@pytest.mark.parametrize(
    "bad, message",
    [
        (torch.zeros((4, 64)), r"shape \(N, 128\)"),
        (torch.zeros(128), r"shape \(N, 128\)"),
        (torch.zeros((4, 128), dtype=torch.int32), "floating"),
    ],
)
def test_malformed_descriptors_are_contract_errors(bad, message):
    with pytest.raises(ContractError, match=message):
        compare(bad, torch.zeros((4, 128)))


def test_unknown_matcher_is_rejected_and_selectors_are_accepted():
    with pytest.raises(Exception, match="unknown matcher 'Brute'"):
        SiftComparisonBlock.Params.model_validate(
            {"input_1": "$inputs.a", "input_2": "$inputs.b", "matcher": "Brute"}
        )

    params = SiftComparisonBlock.Params.model_validate(
        {"input_1": "$inputs.a", "input_2": "$inputs.b", "matcher": "$inputs.matcher"}
    )
    assert params.matcher == "$inputs.matcher"


@pytest.mark.parametrize(
    "field, value",
    [
        ("ratio_threshold", 1.5),
        ("ratio_threshold", -0.1),
        ("good_matches_threshold", 0),
    ],
)
def test_out_of_range_thresholds_are_rejected(field, value):
    with pytest.raises(Exception, match=field):
        SiftComparisonBlock.Params.model_validate(
            {"input_1": "$inputs.a", "input_2": "$inputs.b", field: value}
        )
