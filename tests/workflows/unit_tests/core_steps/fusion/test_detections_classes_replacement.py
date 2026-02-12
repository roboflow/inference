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
        fallback_class_name=None,
        fallback_class_id=None,
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
        fallback_class_name=None,
        fallback_class_id=None,
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
        fallback_class_name=None,
        fallback_class_id=None,
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
        fallback_class_name=None,
        fallback_class_id=None,
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
        fallback_class_name=None,
        fallback_class_id=None,
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


def test_classes_replacement_when_empty_classification_predictions_no_fallback_class():
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
        fallback_class_name=None,
        fallback_class_id=None,
    )

    # then
    assert (
        len(result["predictions"]) == 0
    ), "Expected sv.Detections.empty(), as empty classification was passed"


def test_classes_replacement_when_empty_classification_predictions_fallback_class_provided():
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
    first_cls_prediction["parent_id"] = "zero"
    second_cls_prediction = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)
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
        fallback_class_name="unknown",
        fallback_class_id=123,
    )

    # then
    assert (
        len(result["predictions"]) == 2
    ), "Expected sv.Detections.empty(), as empty classification was passed"
    detections = result["predictions"]
    assert (
        detections.confidence[1] == 0
    ), "Fallback class confidence expected to be set to 0"
    assert (
        detections.class_id[1] == 123
    ), "class id expected to be set to value passed with fallback_class_id parameter"
    assert (
        detections.data["class_name"][1] == "unknown"
    ), "class name expected to be set to value passed with fallback_class_name parameter"


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


def test_extract_leading_class_from_prediction_when_prediction_is_string() -> None:
    # given
    prediction = "K619879"

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result == ("K619879", 0, 1.0)


def test_extract_leading_class_from_prediction_when_prediction_is_list_of_strings() -> (
    None
):
    # given
    prediction = ["K619879"]

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result == ("K619879", 0, 1.0)


def test_extract_leading_class_from_prediction_when_prediction_is_empty_list() -> None:
    # given
    prediction = []

    # when
    result = extract_leading_class_from_prediction(prediction=prediction)

    # then
    assert result is None


def test_extract_leading_class_from_prediction_when_prediction_is_empty_list_with_fallback() -> (
    None
):
    # given
    prediction = []

    # when
    result = extract_leading_class_from_prediction(
        prediction=prediction,
        fallback_class_name="unknown",
        fallback_class_id=99,
    )

    # then
    assert result == ("unknown", 99, 0.0)


def test_classes_replacement_with_list_of_strings_gemini_style() -> None:
    """Test the user's exact use case: Gemini outputs nested arrays like
    [["K619879"], ["C98191P"], ["K657648"]] which become a Batch of List[str].
    """
    # given
    step = DetectionsClassesReplacementBlockV1()
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10, 10, 50, 50],
                [60, 60, 100, 100],
                [110, 110, 150, 150],
            ]
        ),
        class_id=np.array([1, 1, 1]),
        confidence=np.array([0.9, 0.8, 0.85]),
        data={
            "class_name": np.array(
                ["license_plate", "license_plate", "license_plate"]
            ),
            "detection_id": np.array(["id1", "id2", "id3"]),
        },
    )
    # Gemini output: [["K619879"], ["C98191P"], ["K657648"]]
    classification_predictions = Batch(
        content=[
            ["K619879"],
            ["C98191P"],
            ["K657648"],
        ],
        indices=[(0, 0), (0, 1), (0, 2)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
        fallback_class_name=None,
        fallback_class_id=None,
    )

    # then
    assert len(result["predictions"]) == 3, "Expected all 3 detections to remain"
    assert np.allclose(
        result["predictions"].xyxy,
        np.array([[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]]),
    ), "Expected coordinates not to be touched"
    assert result["predictions"].data["class_name"].tolist() == [
        "K619879",
        "C98191P",
        "K657648",
    ], "Expected LP numbers as class names"
    assert np.allclose(
        result["predictions"].confidence, np.array([1.0, 1.0, 1.0])
    ), "Expected confidence to be 1.0 for string predictions"
    assert np.allclose(
        result["predictions"].class_id, np.array([0, 0, 0])
    ), "Expected class_id to be 0 for string predictions"
    assert result["predictions"].data["detection_id"].tolist() != [
        "id1",
        "id2",
        "id3",
    ], "Expected new detection ids"


def test_classes_replacement_with_plain_strings() -> None:
    """Test with plain string predictions (not wrapped in lists)."""
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
    classification_predictions = Batch(
        content=["golden_retriever", "labrador"],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
        fallback_class_name=None,
        fallback_class_id=None,
    )

    # then
    assert len(result["predictions"]) == 2
    assert result["predictions"].data["class_name"].tolist() == [
        "golden_retriever",
        "labrador",
    ]
    assert np.allclose(result["predictions"].confidence, np.array([1.0, 1.0]))
    assert np.allclose(result["predictions"].class_id, np.array([0, 0]))


def test_classes_replacement_with_strings_and_none_no_fallback() -> None:
    """Test that None predictions are filtered out when no fallback is set."""
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
            "class_name": np.array(["plate", "plate"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    classification_predictions = Batch(
        content=["K619879", None],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
        fallback_class_name=None,
        fallback_class_id=None,
    )

    # then
    assert len(result["predictions"]) == 1, "Expected only one detection (second had None)"
    assert result["predictions"].data["class_name"].tolist() == ["K619879"]


def test_classes_replacement_with_strings_and_none_with_fallback() -> None:
    """Test that None predictions use fallback class when provided."""
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
            "class_name": np.array(["plate", "plate"]),
            "detection_id": np.array(["zero", "one"]),
        },
    )
    classification_predictions = Batch(
        content=["K619879", None],
        indices=[(0, 0), (0, 1)],
    )

    # when
    result = step.run(
        object_detection_predictions=detections,
        classification_predictions=classification_predictions,
        fallback_class_name="unreadable",
        fallback_class_id=99,
    )

    # then
    assert len(result["predictions"]) == 2
    assert result["predictions"].data["class_name"].tolist() == [
        "K619879",
        "unreadable",
    ]
    assert result["predictions"].confidence[1] == 0.0
    assert result["predictions"].class_id[1] == 99

