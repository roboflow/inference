import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.classical_cv.sift_comparison.v1 import (
    SIFTComparisonBlockManifest,
    SIFTComparisonBlockV1,
)


@pytest.fixture(autouse=True)
def _seed_opencv_rng() -> None:
    # FlannBasedMatcher builds randomized kd-trees from OpenCV's global RNG, so
    # the match count otherwise depends on how many FLANN calls already ran in
    # this process (443 or 444 here). A fixed seed makes every test independent
    # of test order and of which xdist worker runs it.
    cv2.setRNGSeed(0)


def test_sift_comparison_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/sift_comparison@v1",  # Correct type
        "name": "sift_comparison1",
        "descriptor_1": "$steps.sift.descriptors",
        "descriptor_2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
    }

    # when
    result = SIFTComparisonBlockManifest.model_validate(data)

    # then
    assert result == SIFTComparisonBlockManifest(
        type="roboflow_core/sift_comparison@v1",
        name="sift_comparison1",
        descriptor_1="$steps.sift.descriptors",
        descriptor_2="$steps.sift.descriptors",
        good_matches_threshold=50,
        ratio_threshold=0.7,
    )


def test_sift_comparison_validation_when_invalid_descriptor_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/sift_comparison@v1",  # Correct type
        "name": "sift_comparison1",
        "descriptor_1": "invalid",
        "descriptor_2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
    }

    # when
    with pytest.raises(ValidationError):
        _ = SIFTComparisonBlockManifest.model_validate(data)


def test_sift_comparison_block(dogs_image: np.ndarray) -> None:
    # given
    block = SIFTComparisonBlockV1()
    sift = cv2.SIFT_create()
    _, descriptor_1 = sift.detectAndCompute(dogs_image, None)
    _, descriptor_2 = sift.detectAndCompute(dogs_image[::-1, ::-1], None)

    # when
    output = block.run(
        descriptor_1=descriptor_1,
        descriptor_2=descriptor_2,
        good_matches_threshold=50,
        ratio_threshold=0.7,
    )

    # then
    assert output["good_matches_count"] == 443
    assert output["images_match"] is True
