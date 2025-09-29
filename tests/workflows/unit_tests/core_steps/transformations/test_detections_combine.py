import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.detections_combine.v1 import (
    BlockManifest,
    DetectionsCombineBlockV1,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/detections_combine@v1"])
def test_detections_combine_validation_when_valid_manifest_is_given(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "detections_combine",
        "prediction_one": "$steps.od_model.predictions",
        "prediction_two": "$steps.od_model.predictions",
        "class_name": "custom_merged",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="detections_combine",
        prediction_one="$steps.od_model.predictions",
        prediction_two="$steps.od_model.predictions",
        class_name="custom_merged",
    )


def test_detections_combine_block() -> None:
    # given
    block = DetectionsCombineBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 1]),
        data={
            "class_name": np.array(["person", "person"]),
        },
    )

    # when
    output = block.run(prediction_one=detections, prediction_two=detections)

    # then
    assert isinstance(output, dict)
    assert "predictions" in output
    assert len(output["predictions"]) == 4
    assert np.allclose(
        output["predictions"].xyxy, np.array([[10, 10, 20, 20], [15, 15, 25, 25]] * 2)
    )
    assert np.allclose(output["predictions"].confidence, np.array([0.9, 0.8] * 2))
    assert np.allclose(output["predictions"].class_id, np.array([[1, 1] * 2]))
    assert all([c == "person" for c in output["predictions"].data["class_name"]])
