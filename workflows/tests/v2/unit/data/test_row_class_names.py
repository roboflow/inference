"""The shared row class-name rule of ``prediction_ops`` (decisions 025, 034).

::

    name = row["class"]                 if it is a string ("" included)
           else class_names[class_id]   dict, or list indexed by id
           else None                    the consumer raises its own error
"""

from typing import Dict

import pytest
from roboflow_workflows.v2.data.prediction_ops import (
    class_names_by_id,
    row_class_name,
    row_class_names,
)

MISSING = object()


class CountingRow(dict):
    """Row metadata that counts its ``"class"`` lookups."""

    def __init__(self, lookups: Dict[str, int], **items) -> None:
        super().__init__(**items)
        self.lookups = lookups

    def __contains__(self, key) -> bool:
        if key == "class":
            self.lookups["class"] += 1
        return super().__contains__(key)


def rows_of(*values) -> list:
    return [{} if value is MISSING else {"class": value} for value in values]


@pytest.mark.parametrize(
    "value, expected",
    [("car", "car"), ("", ""), (7, None), (None, None), (True, None), (MISSING, None)],
)
def test_only_a_string_row_class_names_the_row(value, expected) -> None:
    # when
    name = row_class_name(rows_of(value)[0])

    # then
    assert name == expected


def test_a_list_map_is_read_as_indexed_by_class_id() -> None:
    # given
    metadata = {"class_names": ["box", "disc"]}

    # when
    names_by_id = class_names_by_id(metadata)

    # then
    assert names_by_id == {0: "box", 1: "disc"}


def test_a_dict_map_is_returned_as_given_and_no_map_is_none() -> None:
    # given
    names = {3: "car"}

    # when / then
    assert class_names_by_id({"class_names": names}) is names
    assert class_names_by_id({}) is None
    assert class_names_by_id(None) is None


@pytest.mark.parametrize("class_names", [{0: "box", 1: "disc"}, ["box", "disc"]])
def test_row_strings_win_and_other_values_fall_back_to_the_map(class_names) -> None:
    # given: 7, None and a missing key are no names; "" and "car" are
    rows = rows_of(7, None, "car", MISSING, "")

    # when
    names = row_class_names(
        [0, 1, 0, 1, 0],
        row_metadata=rows,
        names_by_id=class_names_by_id({"class_names": class_names}),
    )

    # then
    assert names == ["box", "disc", "car", "disc", ""]


def test_unnamed_rows_are_none_for_the_consumer_to_report() -> None:
    # given: an unknown id, a negative id (no Python wrap-around for a list)
    # and a map value of None
    rows = rows_of(MISSING, 7, MISSING)

    # when
    names = row_class_names(
        [5, -1, 2],
        row_metadata=rows,
        names_by_id=class_names_by_id({"class_names": ["box", "disc", None]}),
    )
    without_map = row_class_names([0], row_metadata=rows_of("car"), names_by_id=None)

    # then
    assert names == [None, None, None]
    assert without_map == ["car"]


def test_class_ids_above_float_precision_are_matched_exactly() -> None:
    # given: 2**53 + 1 rounds to 2**53 in float64
    big = 2**53 + 1
    names_by_id = {big: "exact", 2**53: "rounded"}

    # when
    names = row_class_names(
        [big], row_metadata=rows_of(MISSING), names_by_id=names_by_id
    )

    # then
    assert names == ["exact"]


def test_each_row_class_is_looked_up_once() -> None:
    # given: 300 mixed rows
    lookups = {"class": 0}
    values = ["car", 7, MISSING, None, ""] * 60
    rows = [
        (
            CountingRow(lookups)
            if value is MISSING
            else CountingRow(lookups, **{"class": value})
        )
        for value in values
    ]

    # when
    names = row_class_names([0] * len(rows), row_metadata=rows, names_by_id={0: "box"})

    # then
    assert lookups["class"] == len(rows)
    assert names[:5] == ["car", "box", "box", "box", ""]
