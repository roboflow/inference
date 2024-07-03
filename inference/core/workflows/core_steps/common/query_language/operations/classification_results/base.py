from typing import Any, List, Union

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    ClassificationProperty,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def extract_top_class(prediction: dict) -> Union[str, List[str]]:
    if "top" in prediction:
        return prediction["top"]
    return prediction.get("predicted_classes", [])


def extract_top_class_confidence(prediction: dict) -> Union[float, List[float]]:
    if "confidence" in prediction:
        return prediction["confidence"]
    predicted_classes = prediction.get("predicted_classes", [])
    return [
        prediction["predictions"][class_name]["confidence"]
        for class_name in predicted_classes
    ]


def extract_all_class_names(prediction: dict) -> List[str]:
    predictions = prediction["predictions"]
    if isinstance(predictions, list):
        return [p["class"] for p in predictions]
    class_id2_class_name = {
        value["class_id"]: name for name, value in predictions.items()
    }
    sorted_ids = sorted(class_id2_class_name.keys())
    return [class_id2_class_name[class_id] for class_id in sorted_ids]


def extract_all_classes_confidence(prediction: dict) -> List[float]:
    predictions = prediction["predictions"]
    if isinstance(predictions, list):
        return [p["confidence"] for p in predictions]
    class_id2_class_confidence = {
        value["class_id"]: value["confidence"] for value in predictions.values()
    }
    sorted_ids = sorted(class_id2_class_confidence.keys())
    return [class_id2_class_confidence[class_id] for class_id in sorted_ids]


CLASSIFICATION_PROPERTY_EXTRACTORS = {
    ClassificationProperty.TOP_CLASS: extract_top_class,
    ClassificationProperty.TOP_CLASS_CONFIDENCE: extract_top_class_confidence,
    ClassificationProperty.ALL_CLASSES: extract_all_class_names,
    ClassificationProperty.ALL_CONFIDENCES: extract_all_classes_confidence,
}


def extract_classification_property(
    value: Any, property_name: ClassificationProperty, **kwargs
) -> Union[str, float, list]:
    if not isinstance(value, dict):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_classification_property(...), expected classification results object, "
            f"got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    if property_name not in CLASSIFICATION_PROPERTY_EXTRACTORS:
        raise InvalidInputTypeError(
            public_message=f"Executing extract_classification_property(...), expected property_name to be one of "
            f"{CLASSIFICATION_PROPERTY_EXTRACTORS.values()}, got {property_name}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    return CLASSIFICATION_PROPERTY_EXTRACTORS[property_name](value)
