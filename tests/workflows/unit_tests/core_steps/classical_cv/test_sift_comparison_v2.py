import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.sift_comparison.v2 import (
    SIFTComparisonBlockManifest,
    SIFTComparisonBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_sift_comparison_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/sift_comparison@v2",  # Correct type
        "name": "sift_comparison1",
        "input_1": "$steps.sift.descriptors",
        "input_2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
        "matcher": "FlannBasedMatcher",
        "visualize": True,
    }

    # when
    result = SIFTComparisonBlockManifest.model_validate(data)

    # then
    assert result == SIFTComparisonBlockManifest(
        type="roboflow_core/sift_comparison@v2",
        name="sift_comparison1",
        input_1="$steps.sift.descriptors",
        input_2="$steps.sift.descriptors",
        good_matches_threshold=50,
        ratio_threshold=0.7,
        matcher="FlannBasedMatcher",
        visualize=True,
    )


def test_sift_comparison_validation_when_invalid_descriptor_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/sift_comparison@v2",  # Correct type
        "name": "sift_comparison1",
        "descriptor_1": "invalid",
        "descriptor_2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
    }

    # when
    with pytest.raises(ValidationError):
        _ = SIFTComparisonBlockManifest.model_validate(data)


def test_sift_comparison_block_with_descriptors(dogs_image: np.ndarray) -> None:
    # given
    block = SIFTComparisonBlockV2()
    sift = cv2.SIFT_create()
    _, descriptor_1 = sift.detectAndCompute(dogs_image, None)
    _, descriptor_2 = sift.detectAndCompute(dogs_image[::-1, ::-1], None)

    # when
    output = block.run(
        input_1=descriptor_1,
        input_2=descriptor_2,
        good_matches_threshold=50,
        ratio_threshold=0.7,
        visualize=False,
    )

    # then
    assert output["good_matches_count"] == 443
    assert output["images_match"] is True
    assert output["visualization_1"] is None
    assert output["visualization_2"] is None
    assert output["visualization_matches"] is None


def test_sift_comparison_block_with_images(dogs_image: np.ndarray) -> None:
    # given
    block = SIFTComparisonBlockV2()

    # when
    output = block.run(
        input_1=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        input_2=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image[::-1, ::-1],
        ),
        good_matches_threshold=50,
        ratio_threshold=0.7,
        visualize=False,
    )

    # then
    assert output["good_matches_count"] == 443
    assert output["images_match"] is True
    assert output["visualization_1"] is None
    assert output["visualization_2"] is None
    assert output["visualization_matches"] is None


def test_sift_comparison_block_with_visualization(dogs_image: np.ndarray) -> None:
    # given
    block = SIFTComparisonBlockV2()

    # when
    output = block.run(
        input_1=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image,
        ),
        input_2=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=dogs_image[::-1, ::-1],
        ),
        good_matches_threshold=50,
        ratio_threshold=0.7,
        visualize=True,
    )

    # then
    assert output["good_matches_count"] == 443
    assert output["images_match"] is True
    assert output["visualization_1"] is not None
    assert output["visualization_2"] is not None
    assert output["visualization_matches"] is not None
