import numpy as np
import pytest
import supervision as sv

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
    MultiLabelClassificationInferenceResponse,
    MultiLabelClassificationPrediction,
)
from inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1 import (
    DetectionsClassesReplacementBlockV1,
    extract_leading_class_from_prediction,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def test_classes_replacement_when_object_detection_object_is_none() -> None:
    # given
    step = DetectionsClassesReplacementBlockV1()

    # when
    result = step.run(
        object_detection_predictions=None,
        classification_predictions=None,
    )

    # then
    assert result == {
        "predictions": None
    }, "object_detection_predictions is superior object so lack of value means lack of output"


def test_classes_replacement_when_there_are_no_predictions_is_none() -> None:
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40]]),
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=None,
    )

    # then
    assert result == {
        "predictions": sv.Detections.empty()
    }, "classification_predictions is inferior object so lack of value means empty output"


def test_classes_replacement_when_replacement_to_happen_without_filtering_for_multi_label_results() -> (
    None
):
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ]
        ),
        class_id=np.array([7, 7]),
        confidence=np.array([0.36, 0.91]),
        data={
            "class_name": np.array(["animal", "animal"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    first_cls_prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)
    second_cls_prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.2),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)
    first_cls_prediction["parent_id"] = "zero"
    second_cls_prediction["parent_id"] = "one"
    classification_predictions = Batch(
        content=[
            first_cls_prediction,
            second_cls_prediction,
        ],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
    )

    # then
    assert np.allclose(
        result["predictions"].xyxy, np.array([[10, 20, 30, 40], [11, 21, 31, 41]])
    ), "Expected coordinates not to be touched"
    assert np.allclose(
        result["predictions"].confidence, np.array([0.6, 0.4])
    ), "Expected to choose [cat, dog] confidences"
    assert np.allclose(
        result["predictions"].class_id, np.array([0, 1])
    ), "Expected to choose [cat, dog] class ids"
    assert result["predictions"].data["class_name"].tolist() == [
        "cat",
        "dog",
    ], "Expected cat class to be assigned"
    assert result["predictions"].data["detection_id"].tolist() != [
        "zero",
        "one",
    ], "Expected to generate new detection id"


def test_classes_replacement_when_replacement_to_happen_without_filtering_for_multi_class_results() -> (
    None
):
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ]
        ),
        class_id=np.array([7, 7]),
        confidence=np.array([0.36, 0.91]),
        data={
            "class_name": np.array(["animal", "animal"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    first_cls_prediction = ClassificationInferenceResponse(
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
    second_cls_prediction = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.4}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.6}
            ),
        ],
        top="dog",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)
    first_cls_prediction["parent_id"] = "zero"
    second_cls_prediction["parent_id"] = "one"
    classification_predictions = Batch(
        content=[
            first_cls_prediction,
            second_cls_prediction,
        ],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
    )

    # then
    assert np.allclose(
        result["predictions"].xyxy, np.array([[10, 20, 30, 40], [11, 21, 31, 41]])
    ), "Expected coordinates not to be touched"
    assert np.allclose(
        result["predictions"].confidence, np.array([0.6, 0.6])
    ), "Expected to choose [cat, dog] confidences"
    assert np.allclose(
        result["predictions"].class_id, np.array([0, 1])
    ), "Expected to choose [cat, dog] class ids"
    assert result["predictions"].data["class_name"].tolist() == [
        "cat",
        "dog",
    ], "Expected cat class to be assigned"
    assert result["predictions"].data["detection_id"].tolist() != [
        "zero",
        "one",
    ], "Expected to generate new detection id"


def test_classes_replacement_when_replacement_to_happen_and_one_result_to_be_filtered_out() -> (
    None
):
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ]
        ),
        class_id=np.array([7, 7]),
        confidence=np.array([0.36, 0.91]),
        data={
            "class_name": np.array(["animal", "animal"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    first_cls_prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)
    first_cls_prediction["parent_id"] = "zero"
    classification_predictions = Batch(
        content=[
            first_cls_prediction,
            None,
        ],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
    )

    # then
    assert (
        len(result["predictions"]) == 1
    ), "Expected only one bbox left, as there was mo cls result for second bbox"
    assert np.allclose(
        result["predictions"].xyxy, np.array([[10, 20, 30, 40]])
    ), "Expected first bbox to be left"
    assert np.allclose(
        result["predictions"].confidence, np.array([0.6])
    ), "Expected to choose cat confidence"
    assert np.allclose(
        result["predictions"].class_id, np.array([0])
    ), "Expected to choose cat class id"
    assert result["predictions"].data["class_name"].tolist() == [
        "cat"
    ], "Expected cat class to be assigned"
    assert (
        len(result["predictions"].data["detection_id"]) == 1
    ), "Expected only single detection_id"
    assert result["predictions"].data["detection_id"].tolist() != [
        "zero"
    ], "Expected to generate new detection id"


def test_classes_replacement_when_empty_classification_predictions():
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ]
        ),
        class_id=np.array([7, 7]),
        confidence=np.array([0.36, 0.91]),
        data={
            "class_name": np.array(["animal", "animal"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    first_cls_prediction = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[],
    ).dict(by_alias=True, exclude_none=True)
    first_cls_prediction["parent_id"] = "zero"
    classification_predictions = Batch(
        content=[
            first_cls_prediction,
            None,
        ],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
    )

    # then
    assert (
        len(result["predictions"]) == 0
    ), "Expected sv.Detections.empty(), as empty classification was passed"


def test_extract_leading_class_from_prediction_when_prediction_is_multi_label() -> None:
    # given
    prediction = ClassificationInferenceResponse(
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
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result == ("cat", 0, 0.6)


def test_extract_leading_class_from_prediction_when_prediction_is_faulty_multi_label() -> (
    None
):
    # given
    prediction = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 1, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)

    # when
    with pytest.raises(ValueError):
        _ = extract_leading_class_from_prediction(prediction=prediction)


def test_extract_leading_class_from_prediction_when_prediction_is_multi_class_with_predicted_classes() -> (
    None
):
    # given
    prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result == ("cat", 0, 0.6)


def test_extract_leading_class_from_prediction_when_prediction_is_multi_class_without_predicted_classes() -> (
    None
):
    # given
    prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result is None


def test_extract_leading_class_from_prediction_when_prediction_is_multi_class_without_classes_defined() -> (
    None
):
    # given
    prediction = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={},
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result is None
