from typing import Any, List, Union

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    ClassificationProperty,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference_models import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


def extract_top_class(prediction: dict) -> Union[str, List[str]]:
    if "top" in prediction:
        return prediction["top"]
    return prediction.get("predicted_classes", [])


def extract_top_class_tensor_native(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> Union[str, List[str]]:
    if isinstance(prediction, ClassificationPrediction):
        if prediction.images_metadata is None:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction`, but `images_metadata` "
                    "is missing — class-id-to-name lookup requires the producer block "
                    "to attach per-image metadata."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        if prediction.class_id.shape[0] != 1:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction` with batch size "
                    f"{prediction.class_id.shape[0]} — expected a single-image slice. "
                    "Batch must be unpacked before invoking this extractor."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_names = prediction.images_metadata[0].get(CLASS_NAMES_KEY)
        if class_names is None:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction`, but "
                    f"`images_metadata[0]['{CLASS_NAMES_KEY}']` is missing — the "
                    "producer block must attach the class_id → name mapping."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_id = int(prediction.class_id[0])
        class_name = class_names.get(class_id)
        if class_name is None:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction`, predicted "
                    f"class_id={class_id} is missing from the class_names mapping "
                    f"(keys present: {sorted(class_names.keys())})."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        return class_name
    if isinstance(prediction, MultiLabelClassificationPrediction):
        if prediction.image_metadata is None:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.MultiLabelClassificationPrediction`, but "
                    "`image_metadata` is missing — class-id-to-name lookup requires "
                    "the producer block to attach metadata."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_names = prediction.image_metadata.get(CLASS_NAMES_KEY)
        if class_names is None:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_tensor_native(...) on "
                    "`inference_models.MultiLabelClassificationPrediction`, but "
                    f"`image_metadata['{CLASS_NAMES_KEY}']` is missing — the producer "
                    "block must attach the class_id → name mapping."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        result: List[str] = []
        for class_id_scalar in prediction.class_ids.tolist():
            class_id = int(class_id_scalar)
            class_name = class_names.get(class_id)
            if class_name is None:
                raise OperationError(
                    public_message=(
                        "Executing extract_top_class_tensor_native(...) on "
                        "`inference_models.MultiLabelClassificationPrediction`, "
                        f"predicted class_id={class_id} is missing from the "
                        f"class_names mapping (keys present: "
                        f"{sorted(class_names.keys())})."
                    ),
                    context="step_execution | roboflow_query_language_evaluation",
                )
            result.append(class_name)
        return result
    raise InvalidInputTypeError(
        public_message=(
            "While executing extract_top_class_tensor_native(...) operation it was "
            "expected to get `inference_models.ClassificationPrediction` or "
            "`inference_models.MultiLabelClassificationPrediction`, but got instance "
            f"of type {type(prediction)}"
        ),
        context="step_execution | roboflow_query_language_evaluation",
    )


def extract_top_class_confidence(prediction: dict) -> Union[float, List[float]]:
    if "confidence" in prediction:
        return prediction["confidence"]
    predicted_classes = prediction.get("predicted_classes", [])
    return [
        prediction["predictions"][class_name]["confidence"]
        for class_name in predicted_classes
    ]


def extract_top_class_confidence_tensor_native(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> Union[float, List[float]]:
    if isinstance(prediction, ClassificationPrediction):
        if prediction.class_id.shape[0] != 1:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_confidence_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction` with batch size "
                    f"{prediction.class_id.shape[0]} — expected a single-image slice. "
                    "Batch must be unpacked before invoking this extractor."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_id = int(prediction.class_id[0])
        return float(prediction.confidence[0, class_id])
    if isinstance(prediction, MultiLabelClassificationPrediction):
        return [float(c) for c in prediction.confidence[prediction.class_ids].tolist()]
    raise InvalidInputTypeError(
        public_message=(
            "While executing extract_top_class_confidence_tensor_native(...) operation "
            "it was expected to get `inference_models.ClassificationPrediction` or "
            "`inference_models.MultiLabelClassificationPrediction`, but got instance "
            f"of type {type(prediction)}"
        ),
        context="step_execution | roboflow_query_language_evaluation",
    )


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


def extract_top_class_confidence_single_tensor_native(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> float:
    if isinstance(prediction, ClassificationPrediction):
        if prediction.class_id.shape[0] != 1:
            raise OperationError(
                public_message=(
                    "Executing extract_top_class_confidence_single_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction` with batch size "
                    f"{prediction.class_id.shape[0]} — expected a single-image slice. "
                    "Batch must be unpacked before invoking this extractor."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_id = int(prediction.class_id[0])
        return float(prediction.confidence[0, class_id])
    if isinstance(prediction, MultiLabelClassificationPrediction):
        if prediction.class_ids.shape[0] == 0:
            return 0.0
        return float(prediction.confidence[prediction.class_ids].max())
    raise InvalidInputTypeError(
        public_message=(
            "While executing extract_top_class_confidence_single_tensor_native(...) "
            "operation it was expected to get `inference_models.ClassificationPrediction` "
            "or `inference_models.MultiLabelClassificationPrediction`, but got instance "
            f"of type {type(prediction)}"
        ),
        context="step_execution | roboflow_query_language_evaluation",
    )


def extract_all_class_names(prediction: dict) -> List[str]:
    predictions = prediction["predictions"]
    if isinstance(predictions, list):
        return [p["class"] for p in predictions]
    class_id2_class_name = {
        value["class_id"]: name for name, value in predictions.items()
    }
    sorted_ids = sorted(class_id2_class_name.keys())
    return [class_id2_class_name[class_id] for class_id in sorted_ids]


def extract_all_class_names_tensor_native(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> List[str]:
    if isinstance(prediction, ClassificationPrediction):
        if prediction.images_metadata is None:
            raise OperationError(
                public_message=(
                    "Executing extract_all_class_names_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction`, but `images_metadata` "
                    "is missing — class-id-to-name lookup requires the producer block "
                    "to attach per-image metadata."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        if prediction.class_id.shape[0] != 1:
            raise OperationError(
                public_message=(
                    "Executing extract_all_class_names_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction` with batch size "
                    f"{prediction.class_id.shape[0]} — expected a single-image slice. "
                    "Batch must be unpacked before invoking this extractor."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_names = prediction.images_metadata[0].get(CLASS_NAMES_KEY)
        carrier_path = f"images_metadata[0]['{CLASS_NAMES_KEY}']"
    elif isinstance(prediction, MultiLabelClassificationPrediction):
        if prediction.image_metadata is None:
            raise OperationError(
                public_message=(
                    "Executing extract_all_class_names_tensor_native(...) on "
                    "`inference_models.MultiLabelClassificationPrediction`, but "
                    "`image_metadata` is missing — class-id-to-name lookup requires "
                    "the producer block to attach metadata."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        class_names = prediction.image_metadata.get(CLASS_NAMES_KEY)
        carrier_path = f"image_metadata['{CLASS_NAMES_KEY}']"
    else:
        raise InvalidInputTypeError(
            public_message=(
                "While executing extract_all_class_names_tensor_native(...) operation "
                "it was expected to get `inference_models.ClassificationPrediction` or "
                "`inference_models.MultiLabelClassificationPrediction`, but got "
                f"instance of type {type(prediction)}"
            ),
            context="step_execution | roboflow_query_language_evaluation",
        )
    if class_names is None:
        raise OperationError(
            public_message=(
                f"Executing extract_all_class_names_tensor_native(...), but "
                f"`{carrier_path}` is missing — the producer block must attach the "
                "class_id → name mapping."
            ),
            context="step_execution | roboflow_query_language_evaluation",
        )
    return [class_names[class_id] for class_id in sorted(class_names)]


def extract_all_classes_confidence(prediction: dict) -> List[float]:
    predictions = prediction["predictions"]
    if isinstance(predictions, list):
        return [p["confidence"] for p in predictions]
    class_id2_class_confidence = {
        value["class_id"]: value["confidence"] for value in predictions.values()
    }
    sorted_ids = sorted(class_id2_class_confidence.keys())
    return [class_id2_class_confidence[class_id] for class_id in sorted_ids]


def extract_all_classes_confidence_tensor_native(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> List[float]:
    if isinstance(prediction, ClassificationPrediction):
        if prediction.confidence.shape[0] != 1:
            raise OperationError(
                public_message=(
                    "Executing extract_all_classes_confidence_tensor_native(...) on "
                    "`inference_models.ClassificationPrediction` with batch size "
                    f"{prediction.confidence.shape[0]} — expected a single-image slice. "
                    "Batch must be unpacked before invoking this extractor."
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
        return [float(c) for c in prediction.confidence[0].tolist()]
    if isinstance(prediction, MultiLabelClassificationPrediction):
        return [float(c) for c in prediction.confidence.tolist()]
    raise InvalidInputTypeError(
        public_message=(
            "While executing extract_all_classes_confidence_tensor_native(...) operation "
            "it was expected to get `inference_models.ClassificationPrediction` or "
            "`inference_models.MultiLabelClassificationPrediction`, but got instance "
            f"of type {type(prediction)}"
        ),
        context="step_execution | roboflow_query_language_evaluation",
    )


CLASSIFICATION_PROPERTY_EXTRACTORS = {
    ClassificationProperty.TOP_CLASS: (
        extract_top_class
        if not ENABLE_TENSOR_DATA_REPRESENTATION
        else extract_top_class_tensor_native
    ),
    ClassificationProperty.TOP_CLASS_CONFIDENCE: (
        extract_top_class_confidence
        if not ENABLE_TENSOR_DATA_REPRESENTATION
        else extract_top_class_confidence_tensor_native
    ),
    ClassificationProperty.TOP_CLASS_CONFIDENCE_SINGLE: (
        extract_top_class_confidence_single
        if not ENABLE_TENSOR_DATA_REPRESENTATION
        else extract_top_class_confidence_single_tensor_native
    ),
    ClassificationProperty.ALL_CLASSES: (
        extract_all_class_names
        if not ENABLE_TENSOR_DATA_REPRESENTATION
        else extract_all_class_names_tensor_native
    ),
    ClassificationProperty.ALL_CONFIDENCES: (
        extract_all_classes_confidence
        if not ENABLE_TENSOR_DATA_REPRESENTATION
        else extract_all_classes_confidence_tensor_native
    ),
}


def extract_classification_property(
    value: Any, property_name: ClassificationProperty, **kwargs
) -> Union[str, float, list]:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        # Native-only under the flag: the *_tensor_native extractors operate on
        # `inference_models` prediction dataclasses, not on serialised dicts.
        if not isinstance(
            value, (ClassificationPrediction, MultiLabelClassificationPrediction)
        ):
            value_as_str = safe_stringify(value=value)
            raise InvalidInputTypeError(
                public_message=(
                    "Executing extract_classification_property(...) under "
                    "ENABLE_TENSOR_DATA_REPRESENTATION, expected "
                    "`inference_models.ClassificationPrediction` or "
                    "`inference_models.MultiLabelClassificationPrediction`, got "
                    f"{value_as_str} of type {type(value)}"
                ),
                context="step_execution | roboflow_query_language_evaluation",
            )
    elif not isinstance(value, dict):
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
