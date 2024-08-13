import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    BlockManifest,
    crop_image,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_crop_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "some",
        images_field_alias: "$inputs.image",
        "predictions": "$steps.detection.predictions",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="some",
        images="$inputs.image",
        predictions="$steps.detection.predictions",
    )


def test_crop_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "Crop",
        "name": "some",
        "images": "invalid",
        "predictions": "$steps.detection.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_crop_image() -> None:
    # given
    np_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    np_image[0:20, 0:20] = 39
    np_image[80:120, 80:120] = 49
    np_image[450:550, 450:550] = 59
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin_image"),
        numpy_image=np_image,
    )
    detections = sv.Detections(
        xyxy=np.array(
            [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]], dtype=np.float64
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        data={
            "detection_id": np.array(["one", "two", "three"]),
            "class_name": np.array(["cat", "cat", "cat"]),
        },
    )

    # when
    result = crop_image(image=image, detections=detections)

    # then
    assert len(result) == 3, "Expected 3 crops to be created"
    assert (
        result[0]["crops"].numpy_image == (np.ones((20, 20, 3), dtype=np.uint8) * 39)
    ).all(), "Image must have expected size and color"
    assert (
        result[0]["crops"].parent_metadata.parent_id == "one"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[0][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=0,
        left_top_y=0,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
    assert (
        result[1]["crops"].numpy_image == (np.ones((40, 40, 3), dtype=np.uint8) * 49)
    ).all(), "Image must have expected size and color"
    assert (
        result[1]["crops"].parent_metadata.parent_id == "two"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[1][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=80,
        left_top_y=80,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
    assert (
        result[2]["crops"].numpy_image == (np.ones((100, 100, 3), dtype=np.uint8) * 59)
    ).all(), "Image must have expected size and color"
    assert (
        result[2]["crops"].parent_metadata.parent_id == "three"
    ), "Appropriate parent id (from detection id) must be attached"
    assert result[2][
        "crops"
    ].parent_metadata.origin_coordinates == OriginCoordinatesSystem(
        left_top_x=450,
        left_top_y=450,
        origin_width=1000,
        origin_height=1000,
    ), "Appropriate origin coordinates must be attached"
