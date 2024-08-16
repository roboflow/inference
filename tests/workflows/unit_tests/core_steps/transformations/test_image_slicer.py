import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.image_slicer.v1 import (
    BlockManifest,
    ImageSlicerBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


@pytest.mark.parametrize("image_alias", ["image", "images"])
def test_manifest_v1_parsing_when_valid_input_given(image_alias: str) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/image_slicer@v1",
        "name": "slicer",
        image_alias: "$inputs.image",
        "slice_width": 100,
        "slice_height": 200,
        "overlap_ratio_width": 0.2,
        "overlap_ratio_height": 0.3,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        name="slicer",
        type="roboflow_core/image_slicer@v1",
        image="$inputs.image",
        slice_width=100,
        slice_height=200,
        overlap_ratio_width=0.2,
        overlap_ratio_height=0.3,
    )


@pytest.mark.parametrize("field_to_delete", ["image", "type", "name"])
def test_manifest_v1_parsing_when_required_field_missing(field_to_delete: str) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/image_slicer@v1",
        "name": "slicer",
        "image": "$inputs.image",
        "slice_width": 100,
        "slice_height": 200,
        "overlap_ratio_width": 0.2,
        "overlap_ratio_height": 0.3,
    }
    del raw_manifest[field_to_delete]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_manifest_v1_parsing_when_slice_width_outside_of_range() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/image_slicer@v1",
        "name": "slicer",
        "image": "$inputs.image",
        "slice_width": -1,
        "slice_height": 200,
        "overlap_ratio_width": 0.2,
        "overlap_ratio_height": 0.3,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_manifest_v1_parsing_when_slice_height_outside_of_range() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/image_slicer@v1",
        "name": "slicer",
        "image": "$inputs.image",
        "slice_width": 200,
        "slice_height": -1,
        "overlap_ratio_width": 0.2,
        "overlap_ratio_height": 0.3,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


@pytest.mark.parametrize(
    "overlap_property", ["overlap_ratio_width", "overlap_ratio_height"]
)
@pytest.mark.parametrize("invalid_value", [-0.1, 1.0, 1.0])
def test_manifest_v1_parsing_when_overlap_outside_of_range(
    overlap_property: str, invalid_value: float
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/image_slicer@v1",
        "name": "slicer",
        "image": "$inputs.image",
        "slice_width": 200,
        "slice_height": 200,
        overlap_property: invalid_value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_running_block() -> None:
    # given
    image = WorkflowImageData(
        numpy_image=np.zeros((256, 512, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    block = ImageSlicerBlockV1()

    # when
    result = block.run(
        image=image,
        slice_width=256,
        slice_height=128,
        overlap_ratio_width=0.0,
        overlap_ratio_height=0.0,
    )

    # then
    assert len(result) == 4, "Expected exactly 4 crops"
    for i in range(4):
        assert result[i]["slices"].parent_metadata.parent_id.startswith(
            "image_slicer."
        ), f"Expected parent to be set properly for {i}th crop"
    assert result[0][
        "slices"
    ].workflow_root_ancestor_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=0, left_top_y=0, origin_width=512, origin_height=256
    ), "Expected first crop to have the following coordinates regarding root"
    assert result[1][
        "slices"
    ].workflow_root_ancestor_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=256, left_top_y=0, origin_width=512, origin_height=256
    ), "Expected second crop to have the following coordinates regarding root"
    assert result[2][
        "slices"
    ].workflow_root_ancestor_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=0, left_top_y=128, origin_width=512, origin_height=256
    ), "Expected third crop to have the following coordinates regarding root"
    assert result[3][
        "slices"
    ].workflow_root_ancestor_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=256, left_top_y=128, origin_width=512, origin_height=256
    ), "Expected forth crop to have the following coordinates regarding root"
