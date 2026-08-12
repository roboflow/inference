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


def test_bounding_box_block_drops_stale_host_mirror_tensor_native() -> None:
    # given - the block rewrites xyxy, so the per-box host mirror carried in
    # bboxes_metadata must NOT survive into the output (visualization blocks
    # prefer the mirror over the tensor and would draw the original boxes)
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common.tensor_native import (
        HOST_CLASS_ID_KEY,
        HOST_CONFIDENCE_KEY,
        HOST_MIRROR_KEYS,
        HOST_XYXY_KEY,
    )
    from inference.core.workflows.core_steps.transformations.bounding_rect.v1_tensor import (
        BoundingRectBlockV1 as TensorBoundingRectBlockV1,
    )
    from inference_models.models.base.instance_segmentation import InstanceDetections

    mask = sv.polygon_to_mask(
        polygon=np.array([[10, 10], [10, 100], [100, 100], [100, 10]]),
        resolution_wh=(1000, 1000),
    )
    input_meta = {
        "detection_id": "abc",
        HOST_XYXY_KEY: [0.0, 0.0, 5.0, 5.0],
        HOST_CLASS_ID_KEY: 1,
        HOST_CONFIDENCE_KEY: 0.5,
    }
    predictions = InstanceDetections(
        xyxy=torch.tensor([[10.0, 10.0, 100.0, 100.0]]),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.5]),
        mask=torch.from_numpy(np.array([mask])).bool(),
        image_metadata={"class_names": {1: "a"}, "image_dimensions": [1000, 1000]},
        bboxes_metadata=[input_meta],
    )
    block = TensorBoundingRectBlockV1()

    # when
    output = block.run(predictions=predictions)

    # then
    result_meta = output["detections_with_rect"].bboxes_metadata[0]
    assert not any(
        key in result_meta for key in HOST_MIRROR_KEYS
    ), f"Stale host-mirror keys must be dropped, got: {sorted(result_meta)}"
    assert result_meta["detection_id"] == "abc", "Non-mirror keys must be kept"
    assert result_meta["height"] == 90
    assert result_meta["width"] == 90
    assert result_meta["angle"] == 90
    assert output["detections_with_rect"].xyxy[0].tolist() == [10, 10, 100, 100]
    # the caller-shared input dict must not be mutated by the strip
    assert all(key in input_meta for key in HOST_MIRROR_KEYS)
