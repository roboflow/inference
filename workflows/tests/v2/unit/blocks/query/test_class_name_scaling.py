"""Class names resolve once per DetectionTable (decision 028).

A ``DetectionsFilter`` on ``class_name`` reads one name per row. Every read
reuses one resolution of the table instead of walking all rows again, so the
filter does work linear in the row count. Rows count their own ``"class"``
lookups; the work bound is that count, never a timing.

The semantics stay as before: a per-row ``class`` value wins over the
``class_names`` map, missing names fail with the same messages, and a new
carrier (after rename or filter) is read with its own metadata.
"""

from typing import Dict, List, Optional

import pytest
import torch
from roboflow_workflows.v2.blocks.query import QueryRuntimeError, compile_operations
from roboflow_workflows.v2.blocks.query.detections import DetectionTable
from roboflow_workflows.v2.blocks.query.errors import OperationFailure

from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "car", 1: "person", 2: "bus"}


class CountingRow(dict):
    """Row metadata that counts how often its per-row class name is looked up."""

    def __init__(self, lookups: Dict[str, int], **items) -> None:
        super().__init__(**items)
        self.lookups = lookups

    def __contains__(self, key) -> bool:
        if key == "class":
            self.lookups["class"] += 1
        return super().__contains__(key)


def detections(
    class_ids: List[int],
    *,
    class_names: Optional[dict] = CLASS_NAMES,
    row_names: Optional[Dict[int, str]] = None,
    lookups: Optional[Dict[str, int]] = None,
) -> Detections:
    lookups = {"class": 0} if lookups is None else lookups
    row_names = row_names or {}
    rows = []
    for row, _ in enumerate(class_ids):
        own_name = {"class": row_names[row]} if row in row_names else {}
        rows.append(CountingRow(lookups, detection_id=f"d{row}", **own_name))
    count = len(class_ids)
    image_metadata = {} if class_names is None else {"class_names": dict(class_names)}
    value = Detections(
        xyxy=torch.tensor([[row, row, row + 10, row + 10] for row in range(count)])
        .float()
        .reshape(-1, 4),
        class_id=torch.tensor(class_ids, dtype=torch.int64),
        confidence=torch.full((count,), 0.5),
        image_metadata=image_metadata,
        bboxes_metadata=rows,
    )

    return value


def class_filter(comparator: str, operand: str) -> dict:
    statement = {
        "type": "BinaryStatement",
        "left_operand": {
            "type": "DynamicOperand",
            "operations": [
                {"type": "ExtractDetectionProperty", "property_name": "class_name"}
            ],
        },
        "comparator": {"type": comparator},
        "right_operand": {"type": "DynamicOperand", "operand_name": operand},
    }
    operation = {
        "type": "DetectionsFilter",
        "filter_operation": {"type": "StatementGroup", "statements": [statement]},
    }

    return operation


def kept_ids(value: Detections) -> List[str]:
    return [row["detection_id"] for row in value.bboxes_metadata]


@pytest.mark.parametrize("count", [30, 300])
def test_class_filter_looks_up_each_row_name_once(count: int) -> None:
    # given
    lookups = {"class": 0}
    value = detections([row % 3 for row in range(count)], lookups=lookups)
    by_class = compile_operations(
        [class_filter("in (Sequence)", "classes")], declared_names={"classes"}
    )

    # when
    kept = by_class.apply(value, {"classes": ["car"]})

    # then
    assert lookups["class"] == count
    assert kept_ids(kept) == [f"d{row}" for row in range(0, count, 3)]


def test_strict_read_after_optional_read_does_not_walk_rows_again() -> None:
    # given
    lookups = {"class": 0}
    table = DetectionTable(detections([0, 1, 2, 1], lookups=lookups))

    # when
    optional = table.class_names(require_all=False)
    strict = [table.class_names(require_all=True) for _ in range(table.count)]

    # then
    assert optional == ["car", "person", "bus", "person"]
    assert strict == [optional] * table.count
    assert lookups["class"] == table.count


def test_row_name_wins_over_class_map_for_the_same_id() -> None:
    # given
    value = detections([1, 1, 0], row_names={0: "pedestrian"})
    by_class = compile_operations([class_filter("==", "name")], declared_names={"name"})

    # when
    pedestrians = by_class.apply(value, {"name": "pedestrian"})
    persons = by_class.apply(value, {"name": "person"})

    # then
    assert kept_ids(pedestrians) == ["d0"]
    assert kept_ids(persons) == ["d1"]


def test_unknown_id_fails_every_strict_read_with_the_first_missing_id() -> None:
    # given
    table = DetectionTable(detections([0, 5, 7], class_names={0: "car"}))
    expected = "class id 5 is missing from image_metadata['class_names'] (keys: [0])"

    # when
    optional = table.class_names(require_all=False)
    with pytest.raises(OperationFailure) as first:
        table.class_names(require_all=True)
    with pytest.raises(OperationFailure) as second:
        table.class_names(require_all=True)
    optional_again = table.class_names(require_all=False)

    # then
    assert optional == ["car", None, None]
    assert str(first.value) == expected
    assert str(second.value) == expected
    assert optional_again == optional


def test_unknown_id_in_a_later_row_fails_the_filter_at_its_first_row() -> None:
    # given
    value = detections([0, 1, 9])
    by_class = compile_operations([class_filter("==", "name")], declared_names={"name"})

    # when
    with pytest.raises(QueryRuntimeError) as error:
        by_class.apply(value, {"name": "car"})

    # then
    assert error.value.detail == (
        "detection 0: cannot apply to value `DetectionRow(0 of 3)` of type "
        "DetectionRow: class id 9 is missing from image_metadata['class_names'] "
        "(keys: [0, 1, 2])"
    )


@pytest.mark.parametrize("require_all", [False, True])
def test_missing_class_map_fails_every_read_naming_the_first_unnamed_row(
    require_all: bool,
) -> None:
    # given
    value = detections([3, 4, 6], class_names=None, row_names={0: "car"})
    table = DetectionTable(value)
    expected = "image_metadata['class_names'] is missing, so class id 4 has no name"

    # when
    with pytest.raises(OperationFailure) as first:
        table.class_names(require_all=require_all)
    with pytest.raises(OperationFailure) as second:
        table.class_names(require_all=require_all)

    # then
    assert str(first.value) == expected
    assert str(second.value) == expected


def test_row_names_alone_need_no_class_map() -> None:
    # given
    value = detections([3, 4], class_names=None, row_names={0: "car", 1: "dog"})

    # when
    names = DetectionTable(value).class_names(require_all=True)

    # then
    assert names == ["car", "dog"]


@pytest.mark.parametrize("class_names", [CLASS_NAMES, None])
def test_empty_table_has_no_names_in_either_mode(class_names) -> None:
    # given
    table = DetectionTable(detections([], class_names=class_names))

    # when
    optional = table.class_names(require_all=False)
    strict = table.class_names(require_all=True)

    # then
    assert optional == []
    assert strict == []


def test_filter_after_rename_reads_the_renamed_metadata() -> None:
    # given
    value = detections([0, 1, 0], row_names={2: "car"})
    rename_then_filter = compile_operations(
        [
            {
                "type": "DetectionsRename",
                "class_map": {"car": "vehicle"},
                "strict": False,
            },
            class_filter("==", "name"),
        ],
        declared_names={"name"},
    )

    # when
    vehicles = rename_then_filter.apply(value, {"name": "vehicle"})
    cars = rename_then_filter.apply(value, {"name": "car"})

    # then
    assert kept_ids(vehicles) == ["d0", "d2"]
    assert kept_ids(cars) == []


def test_second_filter_reads_the_rows_kept_by_the_first() -> None:
    # given
    value = detections([0, 1, 2, 0], row_names={3: "truck"})
    two_filters = compile_operations(
        [
            class_filter("in (Sequence)", "first"),
            class_filter("==", "second"),
        ],
        declared_names={"first", "second"},
    )

    # when
    kept = two_filters.apply(value, {"first": ["car", "truck"], "second": "truck"})

    # then
    assert kept_ids(kept) == ["d3"]
