"""Turn a VLM classification answer into the workflow prediction dict.

The output shape is a verbatim port of the ``vlm_as_classifier`` formatter
block and MUST stay byte-for-byte identical to it: the tensor-pipeline
serializers reproduce this exact dict (see the "D4 / formatter shape" notes
in ``common/serializers_tensor.py``).
"""

from typing import List, Optional, Tuple

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_json,
)
from inference.core.workflows.core_steps.common.vlm_decoding.utils import (
    create_classes_index,
    scale_confidence,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def decode_classification(
    raw_output: str,
    image: WorkflowImageData,
    classes: List[str],
    inference_id: str,
) -> Tuple[bool, Optional[dict]]:
    """Decode a raw VLM answer into a classification prediction.

    Single-class and multi-label answers are told apart by their keys:
    ``class_name`` + ``confidence`` for single-class, ``predicted_classes``
    for multi-label. Never raises.

    Args:
        raw_output: Raw string produced by the model.
        image: Workflow image the prediction refers to.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to the prediction.

    Returns:
        Tuple of ``(error_status, predictions)``; ``predictions`` is ``None``
        when ``error_status`` is ``True``.
    """
    error_status, parsed_data = extract_json(raw_output)
    if error_status:
        return True, None
    if not isinstance(parsed_data, dict):
        logger.warning(
            "Could not decode VLM classification output - unexpected JSON root "
            "type: %s.",
            type(parsed_data).__name__,
        )
        return True, None
    if "class_name" in parsed_data and "confidence" in parsed_data:
        return parse_multi_class_classification_results(
            image=image,
            results=parsed_data,
            classes=classes,
            inference_id=inference_id,
        )
    if "predicted_classes" in parsed_data:
        return parse_multi_label_classification_results(
            image=image,
            results=parsed_data,
            classes=classes,
            inference_id=inference_id,
        )
    return True, None


def parse_multi_class_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> Tuple[bool, Optional[dict]]:
    """Build the single-class classification prediction dict.

    Args:
        image: Workflow image the prediction refers to.
        results: Parsed model answer.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to the prediction.

    Returns:
        Tuple of ``(error_status, predictions)``.
    """
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image._read_shape_without_materialization()
        top_class = results["class_name"]
        confidences = {top_class: scale_confidence(results["confidence"])}
        predictions = []
        if top_class not in class2id_mapping:
            predictions.append(
                {
                    "class": top_class,
                    "class_id": -1,
                    "confidence": confidences.get(top_class, 0.0),
                }
            )
        for class_name, class_id in class2id_mapping.items():
            predictions.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidences.get(class_name, 0.0),
                }
            )
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "top": top_class,
            "confidence": confidences[top_class],
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return False, parsed_prediction
    except Exception as error:
        logger.warning(
            "Could not decode multi-class VLM classification output. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, None


def parse_multi_label_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> Tuple[bool, Optional[dict]]:
    """Build the multi-label classification prediction dict.

    Args:
        image: Workflow image the prediction refers to.
        results: Parsed model answer.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to the prediction.

    Returns:
        Tuple of ``(error_status, predictions)``.
    """
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image._read_shape_without_materialization()
        predicted_classes_confidences = {}
        for prediction in results["predicted_classes"]:
            if prediction["class"] not in class2id_mapping:
                class2id_mapping[prediction["class"]] = -1
            if prediction["class"] in predicted_classes_confidences:
                old_confidence = predicted_classes_confidences[prediction["class"]]
                new_confidence = scale_confidence(value=prediction["confidence"])
                predicted_classes_confidences[prediction["class"]] = max(
                    old_confidence, new_confidence
                )
            else:
                predicted_classes_confidences[prediction["class"]] = scale_confidence(
                    value=prediction["confidence"]
                )
        predictions = {
            class_name: {
                "confidence": predicted_classes_confidences.get(class_name, 0.0),
                "class_id": class_id,
            }
            for class_name, class_id in class2id_mapping.items()
        }
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "predicted_classes": list(predicted_classes_confidences.keys()),
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return False, parsed_prediction
    except Exception as error:
        logger.warning(
            "Could not decode multi-label VLM classification output. "
            "Error type: %s. Details: %s",
            error.__class__.__name__,
            error,
        )
        return True, None
