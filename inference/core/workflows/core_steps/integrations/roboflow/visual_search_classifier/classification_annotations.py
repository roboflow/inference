from typing import Any, Dict, List, Optional


def parse_classification_annotation(
    classification: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(classification, dict):
        return None
    if _has_single_label_annotation(classification=classification):
        return {
            "type": "single_label",
            "classes": [_build_class_entry(classification=classification)],
        }
    return _parse_multi_label_annotation(classification=classification)


def _has_single_label_annotation(classification: Dict[str, Any]) -> bool:
    return (
        isinstance(classification.get("class"), str)
        and len(classification["class"]) > 0
    )


def _parse_multi_label_annotation(
    classification: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    predicted_classes = classification.get("predicted_classes")
    if not isinstance(predicted_classes, list) or not predicted_classes:
        return None
    predictions = classification.get("predictions")
    if not isinstance(predictions, dict):
        predictions = {}
    classes = []
    seen_class_names = set()
    for class_name in predicted_classes:
        if (
            not isinstance(class_name, str)
            or not class_name
            or class_name in seen_class_names
        ):
            continue
        prediction = predictions.get(class_name) or {}
        classes.append(
            {
                "class": class_name,
                "class_id": _normalise_class_id(prediction.get("class_id")),
            }
        )
        seen_class_names.add(class_name)
    if not classes:
        return None
    return {
        "type": "multi_label",
        "classes": classes,
    }


def _build_class_entry(classification: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class": classification["class"],
        "class_id": _normalise_class_id(classification.get("class_id")),
    }


def _normalise_class_id(class_id: Any) -> int:
    if class_id is None:
        return -1
    try:
        return int(class_id)
    except (TypeError, ValueError):
        return -1
