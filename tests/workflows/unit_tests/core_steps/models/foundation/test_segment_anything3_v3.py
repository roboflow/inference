"""Unit tests for SAM3 v3 block class_mapping feature."""

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)


def _make_detections(class_names: list[str]) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float32),
        confidence=np.array([0.9] * n, dtype=np.float32),
        data={"class_name": np.array(class_names)},
    )


def _make_result(class_names: list[str]) -> list[dict]:
    return [{"predictions": _make_detections(class_names)}]


class TestApplyClassMapping:
    def test_no_mapping_returns_unchanged(self):
        result = _make_result(["cat", "dog"])
        mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {})
        assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]

    def test_full_mapping(self):
        result = _make_result(["cat", "dog"])
        mapped = SegmentAnything3BlockV3._apply_class_mapping(
            result, {"cat": "gato", "dog": "perro"}
        )
        assert list(mapped[0]["predictions"].data["class_name"]) == ["gato", "perro"]

    def test_partial_mapping(self):
        result = _make_result(["cat", "dog", "bird"])
        mapped = SegmentAnything3BlockV3._apply_class_mapping(
            result, {"cat": "gato"}
        )
        assert list(mapped[0]["predictions"].data["class_name"]) == [
            "gato", "dog", "bird"
        ]

    def test_mapping_with_no_matching_keys(self):
        result = _make_result(["cat", "dog"])
        mapped = SegmentAnything3BlockV3._apply_class_mapping(
            result, {"fish": "pez"}
        )
        assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]

    def test_multiple_images(self):
        result = [
            {"predictions": _make_detections(["cat"])},
            {"predictions": _make_detections(["dog"])},
        ]
        mapped = SegmentAnything3BlockV3._apply_class_mapping(
            result, {"cat": "gato", "dog": "perro"}
        )
        assert list(mapped[0]["predictions"].data["class_name"]) == ["gato"]
        assert list(mapped[1]["predictions"].data["class_name"]) == ["perro"]

    def test_empty_result(self):
        result = []
        mapped = SegmentAnything3BlockV3._apply_class_mapping(
            result, {"cat": "gato"}
        )
        assert mapped == []
