import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.traditional.siftComparison.v1 import (
    SIFTComparisonBlockManifest,
    SIFTComparisonBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("descriptors_field_alias", ["descriptor1", "descriptor2"])
def test_sift_comparison_validation_when_valid_manifest_is_given(
    descriptors_field_alias: str,
) -> None:
    # given
    data = {
        "type": "SIFTComparison",  # Correct type
        "name": "sift_comparison1",
        "descriptor1": "$steps.sift.descriptors",
        "descriptor2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
    }

    # when
    result = SIFTComparisonBlockManifest.model_validate(data)

    # then
    assert result == SIFTComparisonBlockManifest(
        type="SIFTComparison",
        name="sift_comparison1",
        descriptor1="$steps.sift.descriptors",
        descriptor2="$steps.sift.descriptors",
        good_matches_threshold=50,
        ratio_threshold=0.7,
    )


def test_sift_comparison_validation_when_invalid_descriptor_is_given() -> None:
    # given
    data = {
        "type": "SIFTComparison",  # Correct type
        "name": "sift_comparison1",
        "descriptor1": "invalid",
        "descriptor2": "$steps.sift.descriptors",
        "good_matches_threshold": 50,
        "ratio_threshold": 0.7,
    }

    # when
    with pytest.raises(ValidationError):
        _ = SIFTComparisonBlockManifest.model_validate(data)


@pytest.mark.asyncio
async def test_sift_comparison_block() -> None:
    # given
    block = SIFTComparisonBlockV1()

    descriptor1 = np.random.rand(100, 128).astype(
        np.float32
    )  # Random descriptors for testing
    descriptor2 = np.random.rand(100, 128).astype(
        np.float32
    )  # Random descriptors for testing

    output = block.run(
        descriptor1=descriptor1,
        descriptor2=descriptor2,
        good_matches_threshold=50,
        ratio_threshold=0.7,
    )

    assert output is not None
    assert "good_matches_count" in output
    assert isinstance(output["good_matches_count"], int)
    assert output["good_matches_count"] >= 0
    assert "images_match" in output
    assert isinstance(output["images_match"], bool)
