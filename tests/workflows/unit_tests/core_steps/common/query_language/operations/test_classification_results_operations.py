import pytest

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
    MultiLabelClassificationInferenceResponse,
    MultiLabelClassificationPrediction,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


def test_classification_result_extraction_when_data_is_empty() -> None:
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value=None, operations=operations)


def test_classification_result_extraction_of_top_class_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == "cat"


def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_no_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == []


def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_classes_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog"]


def test_classification_result_extraction_of_top_class_confidence_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert abs(result - 0.6) < 1e-5


def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_no_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == []


def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.4]


def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == 0.6


def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_no_classes_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == 0.0


def test_classification_result_extraction_of_all_classes_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_classes",
        }
    ]
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog"]


def test_classification_result_extraction_of_all_classes_for_multi_label_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_classes",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
            "animal": MultiLabelClassificationPrediction(class_id=3, confidence=0.0),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog", "animal"]


def test_classification_result_extraction_of_all_confidences_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_confidences",
        }
    ]
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.6, 0.4]


def test_classification_result_extraction_of_all_confidences_for_multi_label_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_confidences",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "animal": MultiLabelClassificationPrediction(class_id=3, confidence=0.0),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.6, 0.4, 0.0]
