from typing import Optional

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.transformations.detections_filter.v1 import (
    DetectionsFilterBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_filter.v2 import (
    BlockManifest,
    DetectionsFilterBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def _classification_filter_operations(
    threshold_operand: Optional[dict] = None,
) -> list:
    if threshold_operand is None:
        threshold_operand = {
            "type": "DynamicOperand",
            "operand_name": "threshold",
        }
    return [
        {
            "type": "ClassificationFilter",
            "filter_operation": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operations": [
                                {
                                    "type": "ExtractClassificationPredictionProperty",
                                    "property_name": "confidence",
                                }
                            ],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": threshold_operand,
                    }
                ],
            },
        }
    ]


def _detections_filter_operations() -> list:
    return [
        {
            "type": "DetectionsFilter",
            "filter_operation": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operations": [
                                {
                                    "type": "ExtractDetectionProperty",
                                    "property_name": "confidence",
                                }
                            ],
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "threshold",
                        },
                    }
                ],
            },
        }
    ]


def _classification_property_filter_operations(
    property_name: str,
    value: object,
) -> list:
    return [
        {
            "type": "ClassificationFilter",
            "filter_operation": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operations": [
                                {
                                    "type": "ExtractClassificationPredictionProperty",
                                    "property_name": property_name,
                                }
                            ],
                        },
                        "comparator": {"type": "=="},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": value,
                        },
                    }
                ],
            },
        }
    ]


def _parse_operations(operations: list) -> list:
    return OperationsChain.model_validate({"operations": operations}).operations


def _single_label_prediction() -> dict:
    return {
        "image": {"width": 100, "height": 80},
        "predictions": [
            {"class": "cat", "class_id": 0, "confidence": 0.9},
            {"class": "dog", "class_id": 1, "confidence": 0.4},
        ],
        "top": "cat",
        "confidence": 0.9,
        "parent_id": "parent",
        "root_parent_id": "root",
        "inference_id": "inference",
        "prediction_type": "classification",
    }


def _multi_label_prediction() -> dict:
    return {
        "image": {"width": 100, "height": 80},
        "predictions": {
            "cat": {"class_id": 0, "confidence": 0.9},
            "dog": {"class_id": 1, "confidence": 0.4},
            "bird": {"class_id": 2, "confidence": 0.8},
        },
        "predicted_classes": ["cat", "dog"],
        "parent_id": "parent",
        "root_parent_id": "root",
        "inference_id": "inference",
        "prediction_type": "classification",
    }


def test_manifest_accepts_v1_detection_configuration_with_v2_type() -> None:
    raw_manifest = {
        "type": "roboflow_core/detections_filter@v2",
        "name": "filter",
        "predictions": "$steps.model.predictions",
        "operations": _detections_filter_operations(),
        "operations_parameters": {"threshold": "$inputs.threshold"},
    }

    result = BlockManifest.model_validate(raw_manifest)

    assert result.type == "roboflow_core/detections_filter@v2"
    assert result.operations[0].type == "DetectionsFilter"
    assert {kind.name for kind in result.get_actual_outputs()[0].kind} == {
        "object_detection_prediction",
        "instance_segmentation_prediction",
        "keypoint_detection_prediction",
    }


def test_manifest_does_not_claim_legacy_v1_alias() -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(
            {
                "type": "DetectionsFilter",
                "name": "filter",
                "predictions": "$steps.model.predictions",
                "operations": _detections_filter_operations(),
            }
        )


def test_manifest_reports_classification_output_for_classification_filter() -> None:
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/detections_filter@v2",
            "name": "filter",
            "predictions": "$steps.model.predictions",
            "operations": _classification_filter_operations(),
        }
    )

    assert [kind.name for kind in manifest.get_actual_outputs()[0].kind] == [
        "classification_prediction"
    ]


def test_manifest_rejects_mixed_detection_and_classification_filters() -> None:
    with pytest.raises(ValidationError, match="cannot mix"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/detections_filter@v2",
                "name": "filter",
                "predictions": "$steps.model.predictions",
                "operations": (
                    _detections_filter_operations()
                    + _classification_filter_operations()
                ),
            }
        )


def test_v2_preserves_v1_detection_filter_behavior() -> None:
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=float),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.4]),
        data={"class_name": np.array(["cat", "dog"])},
    )
    predictions = Batch(content=[detections], indices=[(0,)])
    operations = _parse_operations(_detections_filter_operations())

    v1_result = DetectionsFilterBlockV1().run(
        predictions=predictions,
        operations=operations,
        operations_parameters={"threshold": 0.5},
    )
    v2_result = DetectionsFilterBlockV2().run(
        predictions=predictions,
        operations=operations,
        operations_parameters={"threshold": 0.5},
    )

    v1_predictions = v1_result[0]["predictions"]
    v2_predictions = v2_result[0]["predictions"]
    np.testing.assert_array_equal(v2_predictions.xyxy, v1_predictions.xyxy)
    np.testing.assert_array_equal(v2_predictions.class_id, v1_predictions.class_id)
    np.testing.assert_array_equal(v2_predictions.confidence, v1_predictions.confidence)
    np.testing.assert_array_equal(
        v2_predictions.data["class_name"],
        v1_predictions.data["class_name"],
    )


def test_detection_filter_returns_empty_detections_when_all_are_removed() -> None:
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
        data={"class_name": np.array(["cat"])},
    )

    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[detections], indices=[(0,)]),
        operations=_parse_operations(_detections_filter_operations()),
        operations_parameters={"threshold": 0.95},
    )

    assert isinstance(result[0]["predictions"], sv.Detections)
    assert len(result[0]["predictions"]) == 0


def test_single_label_classification_filter_recomputes_summary_and_metadata() -> None:
    prediction = _single_label_prediction()

    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[prediction], indices=[(0,)]),
        operations=_parse_operations(_classification_filter_operations()),
        operations_parameters={"threshold": 0.5},
    )

    filtered = result[0]["predictions"]
    assert filtered["predictions"] == [
        {"class": "cat", "class_id": 0, "confidence": 0.9}
    ]
    assert filtered["top"] == "cat"
    assert filtered["confidence"] == 0.9
    assert filtered["parent_id"] == "parent"
    assert filtered["root_parent_id"] == "root"
    assert filtered["inference_id"] == "inference"
    assert prediction["predictions"][1]["class"] == "dog"


@pytest.mark.parametrize(
    ("property_name", "value"),
    [
        ("class_name", "dog"),
        ("class_id", 1),
    ],
)
def test_classification_filter_supports_class_properties(
    property_name: str,
    value: object,
) -> None:
    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[_single_label_prediction()], indices=[(0,)]),
        operations=_parse_operations(
            _classification_property_filter_operations(
                property_name=property_name,
                value=value,
            )
        ),
        operations_parameters={},
    )

    filtered = result[0]["predictions"]
    assert filtered["predictions"] == [
        {"class": "dog", "class_id": 1, "confidence": 0.4}
    ]
    assert filtered["top"] == "dog"
    assert filtered["confidence"] == 0.4


def test_single_label_classification_filter_returns_canonical_empty_result() -> None:
    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[_single_label_prediction()], indices=[(0,)]),
        operations=_parse_operations(_classification_filter_operations()),
        operations_parameters={"threshold": 0.95},
    )

    filtered = result[0]["predictions"]
    assert filtered["predictions"] == []
    assert filtered["top"] == ""
    assert filtered["confidence"] == 0.0


def test_multi_label_classification_filter_preserves_score_map_and_selection() -> None:
    prediction = _multi_label_prediction()

    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[prediction], indices=[(0,)]),
        operations=_parse_operations(_classification_filter_operations()),
        operations_parameters={"threshold": 0.5},
    )

    filtered = result[0]["predictions"]
    assert filtered["predicted_classes"] == ["cat"]
    assert filtered["predictions"] == prediction["predictions"]
    assert "bird" not in filtered["predicted_classes"]
    assert filtered["parent_id"] == "parent"
    assert filtered["root_parent_id"] == "root"


def test_multi_label_classification_filter_returns_canonical_empty_selection() -> None:
    prediction = _multi_label_prediction()

    result = DetectionsFilterBlockV2().run(
        predictions=Batch(content=[prediction], indices=[(0,)]),
        operations=_parse_operations(_classification_filter_operations()),
        operations_parameters={"threshold": 0.95},
    )

    filtered = result[0]["predictions"]
    assert filtered["predicted_classes"] == []
    assert filtered["predictions"] == prediction["predictions"]


def test_classification_filter_supports_batch_aligned_thresholds() -> None:
    result = DetectionsFilterBlockV2().run(
        predictions=Batch(
            content=[_single_label_prediction(), _single_label_prediction()],
            indices=[(0,), (1,)],
        ),
        operations=_parse_operations(_classification_filter_operations()),
        operations_parameters={
            "threshold": Batch(
                content=[0.95, 0.5],
                indices=[(0,), (1,)],
            )
        },
    )

    assert result[0]["predictions"]["predictions"] == []
    assert [
        prediction["class"] for prediction in result[1]["predictions"]["predictions"]
    ] == ["cat"]


def test_classification_input_requires_classification_filter() -> None:
    with pytest.raises(
        ValueError,
        match="Classification predictions require a `ClassificationFilter`",
    ):
        DetectionsFilterBlockV2().run(
            predictions=Batch(content=[_single_label_prediction()], indices=[(0,)]),
            operations=[],
            operations_parameters={},
        )
