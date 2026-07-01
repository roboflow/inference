from typing import Any, Dict, List, Optional


def parse_visual_search_classification(
    candidate: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    classification = parse_classification_annotation(
        classification=candidate.get("classification")
    )
    if classification is not None:
        return classification
    classification = parse_labels(labels=candidate.get("labels"))
    if classification is not None:
        return classification
    return parse_annotations(annotations=candidate.get("annotations"))


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


def parse_labels(labels: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(labels, list):
        return None
    class_entries = []
    for label in labels:
        if isinstance(label, str):
            class_entries.append({"class": label})
        elif isinstance(label, dict):
            class_entries.append(
                {
                    "class": label.get("class"),
                    "class_id": label.get("class_id"),
                }
            )
    return _build_annotation_from_class_entries(class_entries=class_entries)


def parse_annotations(annotations: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(annotations, list):
        return None
    class_names = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        classes = annotation.get("classes")
        if isinstance(classes, list):
            class_names.extend(classes)
    return _build_annotation_from_class_entries(
        class_entries=[{"class": class_name} for class_name in class_names]
    )


def _has_single_label_annotation(classification: Dict[str, Any]) -> bool:
    return (
        isinstance(classification.get("class"), str)
        and len(classification["class"]) > 0
    )


def _build_annotation_from_class_entries(
    class_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    classes = []
    seen_class_names = set()
    used_class_ids = set()
    for fallback_class_id, class_entry in enumerate(class_entries):
        class_name = class_entry.get("class")
        if (
            not isinstance(class_name, str)
            or not class_name
            or class_name in seen_class_names
        ):
            continue
        class_id = _normalise_class_id(class_entry.get("class_id"))
        if class_id < 0 or class_id in used_class_ids:
            class_id = _next_available_class_id(
                preferred_class_id=fallback_class_id,
                used_class_ids=used_class_ids,
            )
        classes.append(
            {
                "class": class_name,
                "class_id": class_id,
            }
        )
        seen_class_names.add(class_name)
        used_class_ids.add(class_id)
    if not classes:
        return None
    if len(classes) == 1:
        return {
            "type": "single_label",
            "classes": classes,
        }
    return {
        "type": "multi_label",
        "classes": classes,
    }


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
    used_class_ids = set()
    for fallback_class_id, class_name in enumerate(predicted_classes):
        if (
            not isinstance(class_name, str)
            or not class_name
            or class_name in seen_class_names
        ):
            continue
        prediction = predictions.get(class_name) or {}
        if not isinstance(prediction, dict):
            prediction = {}
        class_id = _normalise_class_id(prediction.get("class_id"))
        if class_id < 0 or class_id in used_class_ids:
            class_id = _next_available_class_id(
                preferred_class_id=fallback_class_id,
                used_class_ids=used_class_ids,
            )
        classes.append(
            {
                "class": class_name,
                "class_id": class_id,
            }
        )
        seen_class_names.add(class_name)
        used_class_ids.add(class_id)
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


def _next_available_class_id(
    preferred_class_id: int,
    used_class_ids: set,
) -> int:
    while preferred_class_id in used_class_ids:
        preferred_class_id += 1
    return preferred_class_id
