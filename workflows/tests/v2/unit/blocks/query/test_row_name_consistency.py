"""Every class-name consumer names a row the same way (decision 031).

One rule, shared with the painters: a row's own ``"class"`` wins only when
it is a string (``""`` included); otherwise ``class_names[class_id]`` names
the row. Each test asks the real consumers which name each row has::

    filter        DetectionsFilter on class_name: which rows equal each name
    property      detections_property(value, "class_name")
    per-class     keep_confident_rows: which rows a name's threshold drops
    prepared      prepare_drawings(...).labels         ("<name> 0.50")
    annotation    label_texts(read_rows(...), "Class")

Before this correction a non-string row value was a name for the first three
(``7`` -> ``"7"``; ``None`` -> ``"None"`` in the query, no name in per-class)
and ignored by the painters.
"""

from typing import Dict, List, Optional

import pytest
import torch
from roboflow_workflows.v2.blocks.query import QueryRuntimeError, compile_operations
from roboflow_workflows.v2.blocks.query.detections import (
    DetectionTable,
    detections_property,
    row_class_name,
)
from roboflow_workflows.v2.blocks.query.errors import OperationFailure
from roboflow_workflows.v2.blocks.transforms.per_class_confidence import (
    keep_confident_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label import label_texts
from roboflow_workflows.v2.blocks.visualization.annotations.rows import read_rows
from roboflow_workflows.v2.blocks.visualization.drawing import prepare_drawings

from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "box", 1: "disc"}
# Every name any consumer gave these rows, before or after the correction.
CANDIDATE_NAMES = ("box", "disc", "car", "", "7", "None")
MISSING = object()


def detections(
    class_ids: List[int],
    *,
    class_names: Optional[dict] = CLASS_NAMES,
    row_classes: Optional[list] = None,
) -> Detections:
    # row_classes[i] is row i's "class" value; MISSING leaves the key out.
    rows = None
    if row_classes is not None:
        rows = [{"detection_id": f"d{index}"} for index in range(len(row_classes))]
        for row, value in zip(rows, row_classes):
            if value is not MISSING:
                row["class"] = value
    count = len(class_ids)
    value = Detections(
        xyxy=torch.tensor([[row, row, row + 10, row + 10] for row in range(count)])
        .float()
        .reshape(-1, 4),
        class_id=torch.tensor(class_ids, dtype=torch.int64),
        confidence=torch.full((count,), 0.5),
        image_metadata={} if class_names is None else {"class_names": class_names},
        bboxes_metadata=rows,
    )

    return value


def class_filter() -> dict:
    statement = {
        "type": "BinaryStatement",
        "left_operand": {
            "type": "DynamicOperand",
            "operations": [
                {"type": "ExtractDetectionProperty", "property_name": "class_name"}
            ],
        },
        "comparator": {"type": "=="},
        "right_operand": {"type": "DynamicOperand", "operand_name": "name"},
    }
    operation = {
        "type": "DetectionsFilter",
        "filter_operation": {"type": "StatementGroup", "statements": [statement]},
    }

    return operation


def boxes_at(value: Detections) -> List[float]:
    # Row positions survive filtering in x_min (row i starts at x = i).
    return value.xyxy[:, 0].tolist()


def names_seen_by_filter(value: Detections) -> List[Optional[str]]:
    by_name = compile_operations([class_filter()], declared_names={"name"})
    names: List[Optional[str]] = [None] * len(value)
    for candidate in CANDIDATE_NAMES:
        for x_min in boxes_at(by_name.apply(value, {"name": candidate})):
            names[int(x_min)] = candidate

    return names


def names_seen_by_per_class(value: Detections) -> List[Optional[str]]:
    # A threshold of 1.0 for one name drops exactly the rows of that name.
    names: List[Optional[str]] = [None] * len(value)
    for candidate in CANDIDATE_NAMES:
        kept = keep_confident_rows(
            value, class_thresholds={candidate: 1.0}, default_threshold=0.0
        )
        dropped = set(range(len(value))) - {int(x) for x in boxes_at(kept)}
        for index in dropped:
            names[index] = candidate

    return names


def names_seen_by_prepared_labels(value: Detections) -> List[str]:
    (prepared,) = prepare_drawings([value])
    names = [label.rsplit(" ", 1)[0] for label in prepared.labels]

    return names


def names_seen_by_annotation_labels(value: Detections) -> List[str]:
    names = label_texts(read_rows(value), "Class", masks=None, block="test")

    return names


def names_by_consumer(value: Detections) -> Dict[str, list]:
    names = {
        "filter": names_seen_by_filter(value),
        "property": detections_property(value, "class_name"),
        "per_class": names_seen_by_per_class(value),
        "prepared": names_seen_by_prepared_labels(value),
        "annotation": names_seen_by_annotation_labels(value),
    }

    return names


@pytest.mark.parametrize(
    "row_classes, expected",
    [
        ([7, None, "car", MISSING, ""], ["box", "disc", "car", "disc", ""]),
        ([0, 1.5, True, ["car"], {"name": "car"}], ["box", "disc", "box", "disc", "box"]),
        (None, ["box", "disc", "box", "disc", "box"]),
    ],
    ids=["int-none-string-missing-empty", "other-non-strings", "no-row-metadata"],
)  # fmt: skip
def test_every_consumer_gives_each_row_the_same_name(
    row_classes: Optional[list], expected: List[str]
) -> None:
    # given
    value = detections([0, 1, 0, 1, 0], row_classes=row_classes)

    # when
    names = names_by_consumer(value)

    # then
    assert names == {consumer: expected for consumer in names}


def test_string_row_names_need_no_class_map_in_any_consumer() -> None:
    # given
    value = detections([3, 4, 3], class_names=None, row_classes=["car", "", "box"])

    # when
    names = names_by_consumer(value)

    # then
    assert names == {consumer: ["car", "", "box"] for consumer in names}


@pytest.mark.parametrize("row_class", [7, None])
def test_non_string_row_without_class_map_has_no_name_anywhere(row_class) -> None:
    # given: row 0 is named by its string; row 1 has only a non-string value
    value = detections([3, 4], class_names=None, row_classes=["car", row_class])
    note = f"bboxes_metadata['class'] is {row_class!r}, not a string"
    by_name = compile_operations([class_filter()], declared_names={"name"})

    # then: query and painters fail on row 1 and say why
    with pytest.raises(OperationFailure) as table_error:
        DetectionTable(value).class_names(require_all=False)
    assert str(table_error.value) == (
        f"image_metadata['class_names'] is missing, so class id 4 has no name; {note}"
    )
    with pytest.raises(QueryRuntimeError) as filter_error:
        by_name.apply(value, {"name": "car"})
    assert note in filter_error.value.detail
    with pytest.raises(OperationFailure, match="class id 4 has no name"):
        detections_property(value, "class_name")
    with pytest.raises(ValueError, match="row 1 with class id 4 has no name"):
        prepare_drawings([value])
    with pytest.raises(ValueError, match="class id 4 has no name"):
        names_seen_by_annotation_labels(value)
    # per-class: a row without a name uses default_threshold, as before
    assert names_seen_by_per_class(value) == ["car", None]


def test_non_string_row_with_unlisted_id_reports_the_ignored_value() -> None:
    # given
    value = detections([0, 5], row_classes=[MISSING, 7])
    table = DetectionTable(value)

    # when
    optional = table.class_names(require_all=False)
    with pytest.raises(OperationFailure) as strict_error:
        table.class_names(require_all=True)

    # then
    assert optional == ["box", None]
    assert str(strict_error.value) == (
        "class id 5 is missing from image_metadata['class_names'] (keys: [0, 1]); "
        "bboxes_metadata['class'] is 7, not a string"
    )
    with pytest.raises(ValueError, match="row 1 with class id 5 has no name"):
        prepare_drawings([value])


def test_only_non_string_row_values_without_map_count_as_no_names_attached() -> None:
    # given: V1 returns [] for class_name when no names are attached at all
    value = detections([3, 4], class_names=None, row_classes=[7, None])

    # when
    names = detections_property(value, "class_name")
    with_confidence = label_texts(
        read_rows(value), "Class and Confidence", masks=None, block="test"
    )

    # then: query and annotation both see no names attached
    assert names == []
    assert with_confidence == ["0.50", "0.50"]


def test_row_class_name_accepts_only_strings() -> None:
    # then
    assert row_class_name({"class": "car"}) == "car"
    assert row_class_name({"class": ""}) == ""
    assert row_class_name({"class": 7}) is None
    assert row_class_name({"class": None}) is None
    assert row_class_name({}) is None


class CountingRow(dict):
    """Row metadata that counts how often its own class value is looked up."""

    def __init__(self, lookups: Dict[str, int], **items) -> None:
        super().__init__(**items)
        self.lookups = lookups

    def __contains__(self, key) -> bool:
        if key == "class":
            self.lookups["class"] += 1
        return super().__contains__(key)


def test_mixed_row_values_still_resolve_once_per_table() -> None:
    # given: string, non-string and missing values over 300 rows
    count = 300
    lookups = {"class": 0}
    own = ({"class": "car"}, {"class": 7}, {})
    value = Detections(
        xyxy=torch.zeros((count, 4)),
        class_id=torch.tensor([row % 2 for row in range(count)]),
        confidence=torch.full((count,), 0.5),
        image_metadata={"class_names": CLASS_NAMES},
        bboxes_metadata=[
            CountingRow(lookups, detection_id=f"d{row}", **own[row % 3])
            for row in range(count)
        ],
    )
    by_name = compile_operations([class_filter()], declared_names={"name"})

    # when
    kept = by_name.apply(value, {"name": "box"})

    # then: one lookup per row; rows with 7 or no value and id 0 are boxes
    expected = sum(1 for row in range(count) if row % 3 != 0 and row % 2 == 0)
    assert lookups["class"] == count
    assert len(kept) == expected
