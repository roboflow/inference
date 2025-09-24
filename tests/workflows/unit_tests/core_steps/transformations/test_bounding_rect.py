import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.bounding_rect.v1 import (
    BoundingRectBlockV1,
    BoundingRectManifest,
    calculate_minimum_bounding_rectangle,
)


def test_calculate_minimum_bounding_rectangle():
    # given
    polygon = np.array([[10, 10], [10, 1], [20, 1], [20, 10], [15, 5]])
    mask = sv.polygon_to_mask(
        polygon=polygon, resolution_wh=(np.max(polygon, axis=0) + 10)
    )

    # when
    box, width, height, angle = calculate_minimum_bounding_rectangle(mask=mask)

    # then
    expected_box = np.array([[10, 1], [20, 1], [20, 10], [10, 10]])
    assert np.allclose(
        box, expected_box
    ), f"Expected bounding box to be {expected_box}, but got {box}"
    assert np.isclose(width, 9), f"Expected width to be 9, but got {width}"
    assert np.isclose(height, 10), f"Expected height to be 10, but got {height}"
    assert (
        angle == 90 or angle == -90
    ), f"Expected angle to be 90 or -90, but got {angle}"


@pytest.mark.parametrize("type_alias", ["roboflow_core/bounding_rect@v1"])
def test_bounding_box_validation_when_valid_manifest_is_given(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "bounding_box",
        "predictions": "$steps.od_model.predictions",
    }

    # when
    result = BoundingRectManifest.model_validate(data)

    # then
    assert result == BoundingRectManifest(
        type=type_alias, name="bounding_box", predictions="$steps.od_model.predictions"
    )


def test_bounding_box_block() -> None:
    # given
    block = BoundingRectBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 100, 100]]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    polygon=np.array([[10, 10], [10, 100], [100, 100], [100, 10]]),
                    resolution_wh=(1000, 1000),
                )
            ]
        ),
    )

    output = block.run(
        predictions=detections,
    )

    assert isinstance(output, dict)
    assert "detections_with_rect" in output
    assert output["detections_with_rect"].data["height"][0] == 90
    assert output["detections_with_rect"].data["width"][0] == 90
    assert output["detections_with_rect"].data["angle"][0] == 90
    np.allclose(
        np.array([[10, 10], [10, 100], [100, 100], [100, 10]]),
        output["detections_with_rect"].data["rect"][0],
    )
    # check if the image is modified
    assert detections != output["detections_with_rect"]
