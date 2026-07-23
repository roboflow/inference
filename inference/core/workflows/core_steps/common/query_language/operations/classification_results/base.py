from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Union

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    ClassificationPredictionProperty,
    ClassificationProperty,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DEFAULT_OPERAND_NAME,
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


def extract_top_class_confidence_single(prediction: dict) -> Union[float, List[float]]:
    if "confidence" in prediction:
        return prediction["confidence"]
    predicted_classes = prediction.get("predicted_classes", [])
    predicted_confidences = [
        prediction["predictions"][class_name]["confidence"]
        for class_name in predicted_classes
    ]
    if not predicted_confidences:
        return 0.0
    return max(predicted_confidences)


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
    ClassificationProperty.TOP_CLASS_CONFIDENCE_SINGLE: extract_top_class_confidence_single,
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


CLASSIFICATION_PREDICTION_PROPERTY_EXTRACTORS = {
    ClassificationPredictionProperty.CLASS_NAME: lambda prediction: prediction.get(
        "class", prediction.get("class_name")
    ),
    ClassificationPredictionProperty.CLASS_ID: lambda prediction: prediction[
        "class_id"
    ],
    ClassificationPredictionProperty.CONFIDENCE: lambda prediction: prediction[
        "confidence"
    ],
}


def extract_classification_prediction_property(
    value: Any,
    property_name: ClassificationPredictionProperty,
    execution_context: str,
    **kwargs,
) -> Union[str, int, float]:
    if not isinstance(value, dict):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_classification_prediction_property(...) in context "
            f"{execution_context}, expected classification prediction dictionary as value, "
            f"got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        result = CLASSIFICATION_PREDICTION_PROPERTY_EXTRACTORS[property_name](value)
    except (KeyError, TypeError) as error:
        raise InvalidInputTypeError(
            public_message=f"Executing extract_classification_prediction_property(...) in context "
            f"{execution_context}, prediction does not contain a valid `{property_name.value}`.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        ) from error
    if result is None:
        raise InvalidInputTypeError(
            public_message=f"Executing extract_classification_prediction_property(...) in context "
            f"{execution_context}, prediction does not contain a valid `{property_name.value}`.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return result


def filter_classification_predictions(
    value: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> dict:
    if not isinstance(value, dict) or not isinstance(
        value.get("predictions"), (list, dict)
    ):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message="Executing filter_classification_predictions(...), expected a "
            f"classification prediction dictionary, got {value_as_str} of type {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
        )

    result = deepcopy(value)
    predictions = value["predictions"]
    if isinstance(predictions, list):
        filtered_predictions = [
            prediction
            for prediction in predictions
            if _evaluate_classification_prediction(
                prediction=prediction,
                filtering_fun=filtering_fun,
                global_parameters=global_parameters,
            )
        ]
        result["predictions"] = filtered_predictions
        if not filtered_predictions:
            result["top"] = ""
            result["confidence"] = 0.0
            return result
        top_prediction = max(
            filtered_predictions,
            key=lambda prediction: prediction["confidence"],
        )
        result["top"] = top_prediction.get(
            "class", top_prediction.get("class_name", "")
        )
        result["confidence"] = top_prediction["confidence"]
        return result

    predicted_classes = value.get("predicted_classes")
    if not isinstance(predicted_classes, list):
        raise InvalidInputTypeError(
            public_message="Executing filter_classification_predictions(...), expected "
            "`predicted_classes` list for multi-label classification predictions.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    filtered_classes = []
    for class_name in predicted_classes:
        if class_name not in predictions or not isinstance(
            predictions[class_name], dict
        ):
            raise InvalidInputTypeError(
                public_message="Executing filter_classification_predictions(...), "
                f"`predicted_classes` references missing class `{class_name}`.",
                context="step_execution | roboflow_query_language_evaluation",
            )
        prediction = copy(predictions[class_name])
        prediction["class"] = class_name
        if _evaluate_classification_prediction(
            prediction=prediction,
            filtering_fun=filtering_fun,
            global_parameters=global_parameters,
        ):
            filtered_classes.append(class_name)
    result["predicted_classes"] = filtered_classes
    return result


def _evaluate_classification_prediction(
    prediction: Any,
    filtering_fun: Callable[[Dict[str, Any]], bool],
    global_parameters: Dict[str, Any],
) -> bool:
    if not isinstance(prediction, dict):
        raise InvalidInputTypeError(
            public_message="Executing filter_classification_predictions(...), expected each "
            "classification prediction to be a dictionary.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    local_parameters = copy(global_parameters)
    local_parameters[DEFAULT_OPERAND_NAME] = prediction
    return filtering_fun(local_parameters)
