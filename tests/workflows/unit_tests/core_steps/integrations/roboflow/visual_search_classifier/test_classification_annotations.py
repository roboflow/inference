from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier.classification_annotations import (
    parse_classification_annotation,
    parse_visual_search_classification,
)


def test_parse_single_label_annotation() -> None:
    result = parse_classification_annotation(
        classification={"class": "widget-a", "class_id": "7"}
    )

    assert result == {
        "type": "single_label",
        "classes": [{"class": "widget-a", "class_id": 7}],
    }


def test_parse_single_label_annotation_defaults_missing_class_id() -> None:
    result = parse_classification_annotation(classification={"class": "widget-a"})

    assert result == {
        "type": "single_label",
        "classes": [{"class": "widget-a", "class_id": -1}],
    }


def test_parse_multi_label_annotation() -> None:
    result = parse_classification_annotation(
        classification={
            "predicted_classes": ["widget-a", "fragile"],
            "predictions": {
                "widget-a": {"class_id": "7"},
                "fragile": {"class_id": 9},
                "not-selected": {"class_id": 10},
            },
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [
            {"class": "widget-a", "class_id": 7},
            {"class": "fragile", "class_id": 9},
        ],
    }


def test_parse_multi_label_annotation_assigns_stable_ids_for_missing_class_ids() -> (
    None
):
    result = parse_classification_annotation(
        classification={
            "predicted_classes": ["widget-a", "fragile", "oversized"],
            "predictions": {
                "fragile": {"class_id": 1},
            },
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [
            {"class": "widget-a", "class_id": 0},
            {"class": "fragile", "class_id": 1},
            {"class": "oversized", "class_id": 2},
        ],
    }


def test_parse_multi_label_annotation_deduplicates_classes() -> None:
    result = parse_classification_annotation(
        classification={
            "predicted_classes": ["widget-a", "widget-a"],
            "predictions": {"widget-a": {"class_id": 7}},
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [{"class": "widget-a", "class_id": 7}],
    }


def test_parse_multi_label_annotation_ignores_malformed_prediction_entries() -> None:
    result = parse_classification_annotation(
        classification={
            "predicted_classes": ["widget-a"],
            "predictions": {"widget-a": ["not", "a", "dict"]},
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [{"class": "widget-a", "class_id": 0}],
    }


def test_parse_annotation_rejects_unsupported_aliases() -> None:
    result = parse_classification_annotation(
        classification={"labels": [{"class": "widget-a", "class_id": 7}]}
    )

    assert result is None


def test_parse_annotation_rejects_empty_multi_label_annotations() -> None:
    result = parse_classification_annotation(
        classification={"predicted_classes": [], "predictions": {}}
    )

    assert result is None


def test_parse_visual_search_classification_reads_label_dicts() -> None:
    result = parse_visual_search_classification(
        candidate={
            "labels": [
                {"class": "widget-a", "class_id": "7"},
                {"class": "fragile", "class_id": 9},
            ]
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [
            {"class": "widget-a", "class_id": 7},
            {"class": "fragile", "class_id": 9},
        ],
    }


def test_parse_visual_search_classification_reads_label_strings() -> None:
    result = parse_visual_search_classification(
        candidate={"labels": ["widget-a", "fragile"]}
    )

    assert result == {
        "type": "multi_label",
        "classes": [
            {"class": "widget-a", "class_id": 0},
            {"class": "fragile", "class_id": 1},
        ],
    }


def test_parse_visual_search_classification_reads_annotation_classes() -> None:
    result = parse_visual_search_classification(
        candidate={
            "annotations": [
                {"classes": ["widget-a"]},
                {"classes": ["fragile", "widget-a"]},
            ]
        }
    )

    assert result == {
        "type": "multi_label",
        "classes": [
            {"class": "widget-a", "class_id": 0},
            {"class": "fragile", "class_id": 1},
        ],
    }


def test_parse_visual_search_classification_prefers_explicit_classification() -> None:
    result = parse_visual_search_classification(
        candidate={
            "classification": {"class": "explicit", "class_id": 3},
            "labels": [{"class": "label", "class_id": 4}],
        }
    )

    assert result == {
        "type": "single_label",
        "classes": [{"class": "explicit", "class_id": 3}],
    }
