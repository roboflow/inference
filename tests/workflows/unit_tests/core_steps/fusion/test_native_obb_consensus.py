import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.fusion.detections_consensus.v2 import (
    AggregationMode,
    MaskAggregationMode,
    calculate_iou,
    merge_detections,
)


def _detections():
    corners = np.array(
        [
            [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
            [[2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0], [2.0, -1.0]],
        ]
    )
    data = {
        "xyxyxyxy": corners,
        "native_obb": np.array([True, True]),
        "heading": np.array([True, True]),
        "class_name": np.array(["plane", "plane"]),
        "detection_id": np.array(["first", "second"]),
    }
    return sv.Detections(
        xyxy=np.array([[-2.0, -1.0, 2.0, 1.0]] * 2),
        class_id=np.array([7, 7]),
        confidence=np.array([0.6, 0.9]),
        data=data,
    )


@pytest.mark.parametrize("aggregation", list(AggregationMode))
def test_native_consensus_preserves_one_contributors_shape_and_opposite_front(
    aggregation,
):
    detections = _detections()
    result = merge_detections(
        detections,
        confidence_aggregation_mode=aggregation,
        boxes_aggregation_mode=AggregationMode.AVERAGE,
        mask_aggregation_mode=MaskAggregationMode.UNION,
    )
    np.testing.assert_array_equal(
        result.data["xyxyxyxy"], detections.data["xyxyxyxy"][[1]]
    )
    assert result.data["heading"].tolist() == [True]
    assert result.data["detection_id"].tolist() == ["second"]
    assert result.data["prediction_type"].tolist() == ["oriented-object-detection"]
    assert result.confidence[0] == pytest.approx(
        {
            AggregationMode.AVERAGE: 0.75,
            AggregationMode.MAX: 0.9,
            AggregationMode.MIN: 0.6,
        }[aggregation]
    )


def test_native_localization_vote_uses_rotated_iou_independent_of_front():
    detections = _detections()
    assert calculate_iou(detections[0], detections[1]) == pytest.approx(1.0)


def test_native_geometry_tie_keeps_source_order():
    detections = _detections()
    detections.confidence[:] = 0.8
    result = merge_detections(
        detections,
        AggregationMode.AVERAGE,
        AggregationMode.AVERAGE,
        MaskAggregationMode.UNION,
    )
    assert result.data["detection_id"].tolist() == ["first"]


def test_native_class_heading_conflict_is_an_error():
    detections = _detections()
    detections.data["heading"][1] = False
    with pytest.raises(ValueError, match="heading capabilities"):
        merge_detections(
            detections,
            AggregationMode.AVERAGE,
            AggregationMode.AVERAGE,
            MaskAggregationMode.UNION,
        )


def test_native_and_ordinary_mix_is_an_error():
    detections = _detections()
    detections.data["native_obb"][1] = False
    with pytest.raises(ValueError, match="mix"):
        merge_detections(
            detections,
            AggregationMode.AVERAGE,
            AggregationMode.AVERAGE,
            MaskAggregationMode.UNION,
        )


def test_native_overlap_never_falls_back_when_corners_are_missing():
    detections = _detections()
    first = detections[0]
    del first.data["xyxyxyxy"]
    with pytest.raises(ValueError, match="corners and heading"):
        calculate_iou(first, detections[1])
