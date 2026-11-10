"""Detection UQL operations on native V2 carriers.

The street scene has four boxes (tracker ids 7, 8, 9, 10)::

    row  class   box                 conf    size  centre
    0    car     (10, 20, 50, 60)    0.875   1600  (30, 40)
    1    person  (110, 20, 130, 60)  0.5      800  (120, 40)   inside the bus
    2    car     (200, 40, 260, 80)  0.875   2400  (230, 60)
    3    bus     (100, 10, 180, 70)  0.375   4800  (140, 40)

Expected values are derived by hand from the V1 operation definitions. The
task-owned V1 probe re-runs the comparable tables through V1 tensor mode.
"""

import pytest
import torch
from roboflow_workflows.v2.blocks.query import (
    QueryRuntimeError,
    compile_condition,
    compile_operations,
)
from roboflow_workflows.v2.data.rows import PredictionRows

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

BOXES = [[10, 20, 50, 60], [110, 20, 130, 60], [200, 40, 260, 80], [100, 10, 180, 70]]
CLASS_IDS = [0, 1, 0, 2]
CONFIDENCES = [0.875, 0.5, 0.875, 0.375]
CLASS_NAMES = {0: "car", 1: "person", 2: "bus"}
TRACKER_IDS = [7, 8, 9, 10]


def street(*, rows=None, class_names=CLASS_NAMES, tracker=True) -> Detections:
    rows = list(range(4)) if rows is None else rows
    metadata = [
        {
            "detection_id": f"d{row}",
            **({"tracker_id": TRACKER_IDS[row]} if tracker else {}),
        }
        for row in rows
    ]
    image_metadata = {} if class_names is None else {"class_names": dict(class_names)}
    detections = Detections(
        xyxy=torch.tensor([BOXES[row] for row in rows], dtype=torch.float32).reshape(
            -1, 4
        ),
        class_id=torch.tensor([CLASS_IDS[row] for row in rows], dtype=torch.int64),
        confidence=torch.tensor(
            [CONFIDENCES[row] for row in rows], dtype=torch.float32
        ),
        image_metadata=image_metadata,
        bboxes_metadata=metadata,
    )
    return detections


def street_dense() -> InstanceDetections:
    boxes = street()
    masks = torch.zeros((4, 8, 8), dtype=torch.bool)
    for row in range(4):
        masks[row, row, row] = True
    return InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=masks,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )


def street_rle() -> InstanceDetections:
    boxes = street()
    return InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=InstancesRLEMasks(
            image_size=(100, 300), masks=[b"r0", b"r1", b"r2", b"r3"]
        ),
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )


def street_key_points():
    key_points = KeyPoints(
        xy=torch.tensor([[[row, row + 0.5], [row + 1, row + 1.5]] for row in range(4)]),
        class_id=torch.tensor(CLASS_IDS),
        confidence=torch.full((4, 2), 0.5),
        covariance=torch.eye(2).repeat(4, 2, 1, 1),
    )
    return key_points, street()


def street_rows() -> PredictionRows:
    return PredictionRows(
        native=street(tracker=False), columns={"tracker_id": torch.tensor(TRACKER_IDS)}
    )


def empty() -> Detections:
    return street(rows=[])


def summary(value):
    """Projection of an output that both V1 and V2 can produce."""
    if isinstance(value, PredictionRows):
        value = value.native
    if isinstance(value, tuple):
        value = value[1]
    if isinstance(value, (Detections, InstanceDetections)):
        return {
            "detection_id": [
                row["detection_id"] for row in value.bboxes_metadata or []
            ],
            "xyxy": value.xyxy.tolist(),
            "class_id": value.class_id.tolist(),
            "class_names": dict((value.image_metadata or {}).get("class_names", {})),
        }
    return value


def _property(name, *operations):
    return {"type": "DynamicOperand", "operations": [
        {"type": "ExtractDetectionProperty", "property_name": name}, *operations
    ]}  # fmt: skip


def _filter(*statements, operator="and"):
    return {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "operator": operator,
            "statements": list(statements),
        },
    }


def _binary(left, comparator, right, negate=False):
    return {
        "type": "BinaryStatement",
        "left_operand": left,
        "comparator": {"type": comparator},
        "right_operand": right,
        "negate": negate,
    }


def _parameter(name):
    return {"type": "DynamicOperand", "operand_name": name}


def _static(value):
    return {"type": "StaticOperand", "value": value}


def _kept(*rows):
    return [f"d{row}" for row in rows]


# (operations, runtime parameters, expected detection ids after the chain)
FILTER_CASES = [
    (
        [
            _filter(
                _binary(
                    _property("class_name"), "in (Sequence)", _parameter("classes")
                ),
                _binary(_property("confidence"), "(Number) >=", _static(0.5)),
            )
        ],
        {"classes": ["car", "person"]},
        _kept(0, 1, 2),
    ),
    (
        [_filter(_binary(_property("size"), "(Number) >=", _parameter("min_size")))],
        {"min_size": 2000},
        _kept(2, 3),
    ),
    (
        [_filter(_binary(_property("class_name"), "==", _static("bus"), negate=True))],
        {},
        _kept(0, 1, 2),
    ),
    (
        [
            _filter(
                _binary(
                    _property("center", {"type": "SequenceAggregate", "mode": "first"}),
                    "(Number) >=",
                    _static(100),
                )
            )
        ],
        {},
        _kept(1, 2, 3),
    ),
    (
        [
            _filter(
                _binary(
                    _property("tracker_id"), "in (Sequence)", _parameter("tracked")
                ),
                _binary(_property("class_id"), "==", _static(0)),
                operator="or",
            )
        ],
        {"tracked": [10]},
        _kept(0, 2, 3),
    ),
    ([{"type": "DetectionsSelection", "mode": "top_confidence"}], {}, _kept(0)),
    ([{"type": "DetectionsSelection", "mode": "left_most"}], {}, _kept(0)),
    ([{"type": "DetectionsSelection", "mode": "right_most"}], {}, _kept(2)),
    ([{"type": "DetectionsSelection", "mode": "first"}], {}, _kept(0)),
    ([{"type": "DetectionsSelection", "mode": "last"}], {}, _kept(3)),
    ([{"type": "SortDetections", "mode": "size"}], {}, _kept(1, 0, 2, 3)),
    ([{"type": "SortDetections", "mode": "center_x"}], {}, _kept(0, 1, 3, 2)),
    (
        [{"type": "SortDetections", "mode": "x_max", "ascending": False}],
        {},
        _kept(2, 3, 1, 0),
    ),
    (
        [{"type": "SortDetections", "mode": "y_min", "ascending": False}],
        {},
        _kept(2, 0, 1, 3),
    ),
    ([{"type": "PickDetectionsByParentClass", "parent_class": "bus"}], {}, _kept(3, 1)),
    ([{"type": "PickDetectionsByParentClass", "parent_class": "truck"}], {}, []),
    (
        [
            {"type": "SortDetections", "mode": "confidence"},
            _filter(_binary(_property("class_name"), "==", _static("car"))),
            {"type": "DetectionsSelection", "mode": "last"},
        ],
        {},
        _kept(2),
    ),
]

# (operation, expected property values)
PROPERTY_CASES = [
    ("class_name", ["car", "person", "car", "bus"]),
    ("class_id", [0, 1, 0, 2]),
    ("confidence", [0.875, 0.5, 0.875, 0.375]),
    ("x_min", [10.0, 110.0, 200.0, 100.0]),
    ("y_max", [60.0, 60.0, 80.0, 70.0]),
    ("size", [1600.0, 800.0, 2400.0, 4800.0]),
    ("center", [[30, 40], [120, 40], [230, 60], [140, 40]]),
    ("top_right", [[50, 20], [130, 20], [260, 40], [180, 10]]),
    ("bottom_left", [[10, 60], [110, 60], [200, 80], [100, 70]]),
    ("tracker_id", [7, 8, 9, 10]),
]

# (operations, expected summary fields) for box and class edits
EDIT_CASES = [
    (
        [{"type": "DetectionsOffset", "offset_x": 10, "offset_y": 4}],
        {"xyxy": [[5.0, 18.0, 55.0, 62.0], [105.0, 18.0, 135.0, 62.0],
                  [195.0, 38.0, 265.0, 82.0], [95.0, 8.0, 185.0, 72.0]]},
    ),
    (
        [{"type": "DetectionsShift", "shift_x": 5, "shift_y": -5}],
        {"xyxy": [[15.0, 15.0, 55.0, 55.0], [115.0, 15.0, 135.0, 55.0],
                  [205.0, 35.0, 265.0, 75.0], [105.0, 5.0, 185.0, 65.0]]},
    ),
    (
        [{"type": "DetectionsRename", "class_map": {"car": "vehicle"}, "strict": False}],
        {"class_id": [1024, 1, 1024, 2],
         "class_names": {1: "person", 2: "bus", 1024: "vehicle"}},
    ),
    (
        [
            {
                "type": "DetectionsRename",
                "class_map": {"car": "vehicle", "person": "human", "bus": "vehicle"},
            }
        ],
        {"class_id": [1, 0, 1, 1], "class_names": {0: "human", 1: "vehicle"}},
    ),
    (
        [
            {
                "type": "DetectionsRename",
                "class_map": {"person": "car"},
                "strict": False,
                "new_classes_id_offset": 50,
            }
        ],
        {"class_id": [0, 0, 0, 2], "class_names": {0: "car", 2: "bus"}},
    ),
]  # fmt: skip


@pytest.mark.parametrize("operations, parameters, expected", FILTER_CASES)
def test_row_selecting_operations(operations, parameters, expected) -> None:
    # given
    chain = compile_operations(operations, declared_names=set(parameters))

    # when
    result = chain.apply(street(), parameters=parameters)

    # then
    assert summary(result)["detection_id"] == expected


@pytest.mark.parametrize("name, expected", PROPERTY_CASES)
def test_detections_property_extract(name, expected) -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": name}]
    )

    # when
    result = chain.apply(street())

    # then
    assert result == expected


@pytest.mark.parametrize("operations, expected", EDIT_CASES)
def test_box_and_class_edits(operations, expected) -> None:
    # given
    chain = compile_operations(operations)

    # when
    result = summary(chain.apply(street()))

    # then
    assert {key: result[key] for key in expected} == expected
    assert result["detection_id"] == _kept(0, 1, 2, 3)


@pytest.mark.parametrize(
    "carrier", [street_dense, street_rle, street_key_points, street_rows]
)
def test_filter_keeps_every_aligned_field_of_each_carrier(carrier) -> None:
    # given
    chain = compile_operations(
        [_filter(_binary(_property("class_name"), "==", _static("car")))]
    )

    # when
    result = chain.apply(carrier())

    # then
    assert summary(result)["detection_id"] == _kept(0, 2)
    assert _aligned_extras(result) == _aligned_extras(carrier(), rows=[0, 2])


def test_sort_with_repeats_and_reordering_keeps_masks_aligned() -> None:
    # given
    chain = compile_operations(
        [{"type": "SortDetections", "mode": "x_min", "ascending": False}]
    )

    # when
    result = chain.apply(street_dense())

    # then
    assert summary(result)["detection_id"] == _kept(2, 1, 3, 0)
    assert [torch.nonzero(mask).tolist() for mask in result.mask] == [
        [[2, 2]],
        [[1, 1]],
        [[3, 3]],
        [[0, 0]],
    ]


def test_equal_confidence_keeps_input_order_in_both_directions() -> None:
    # given
    ascending = compile_operations([{"type": "SortDetections", "mode": "confidence"}])
    descending = compile_operations(
        [{"type": "SortDetections", "mode": "confidence", "ascending": False}]
    )

    # when
    up = summary(ascending.apply(street()))["detection_id"]
    down = summary(descending.apply(street()))["detection_id"]

    # then
    assert up == _kept(3, 1, 0, 2)
    assert down == _kept(0, 2, 1, 3)


def test_offset_shares_masks_and_strips_stale_host_mirror() -> None:
    # given
    detections = street_dense()
    detections.bboxes_metadata[0]["_host_xyxy"] = [10, 20, 50, 60]
    chain = compile_operations(
        [{"type": "DetectionsShift", "shift_x": 1, "shift_y": 1}]
    )

    # when
    result = chain.apply(detections)

    # then
    assert result.mask is detections.mask
    assert "_host_xyxy" not in result.bboxes_metadata[0]
    assert result.bboxes_metadata[0]["tracker_id"] == 7
    assert detections.xyxy[0].tolist() == [10.0, 20.0, 50.0, 60.0]


def test_key_point_offset_moves_boxes_only_as_v1_does() -> None:
    # given
    key_points, _ = street_key_points()
    chain = compile_operations(
        [{"type": "DetectionsOffset", "offset_x": 2, "offset_y": 2}]
    )

    # when
    moved_key_points, moved_boxes = chain.apply((key_points, street()))

    # then
    assert moved_key_points is key_points
    assert moved_boxes.xyxy[0].tolist() == [9.0, 19.0, 51.0, 61.0]


def test_key_point_selection_keeps_covariance_and_skeletons() -> None:
    # given
    chain = compile_operations([{"type": "DetectionsSelection", "mode": "right_most"}])

    # when
    key_points, boxes = chain.apply(street_key_points())

    # then
    assert key_points.xy.tolist() == [[[2.0, 2.5], [3.0, 3.5]]]
    assert key_points.covariance.shape == (1, 2, 2, 2)
    assert boxes.bboxes_metadata[0]["detection_id"] == "d2"


def test_key_points_xy_property_reads_key_points() -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "keypoints_xy"}]
    )

    # when
    result = chain.apply(street_key_points())

    # then
    assert result[1] == [[1.0, 1.5], [2.0, 2.5]]


def test_prediction_rows_column_is_the_tracker_id_source() -> None:
    # given
    chain = compile_operations(
        [
            {"type": "SortDetections", "mode": "x_min"},
            {"type": "DetectionsPropertyExtract", "property_name": "tracker_id"},
        ]
    )

    # when
    result = chain.apply(street_rows())

    # then
    assert result == [7, 10, 8, 9]


@pytest.mark.parametrize(
    "operation",
    [
        _filter(_binary(_property("confidence"), "(Number) >", _static(0))),
        {"type": "DetectionsSelection", "mode": "top_confidence"},
        {"type": "SortDetections", "mode": "size"},
        {"type": "PickDetectionsByParentClass", "parent_class": "bus"},
        {"type": "DetectionsRename", "class_map": {}, "strict": True},
        {"type": "DetectionsOffset", "offset_x": 2, "offset_y": 2},
    ],
)
def test_empty_detections_stay_empty(operation) -> None:
    # when
    result = compile_operations([operation]).apply(empty())

    # then
    assert isinstance(result, Detections)
    assert len(result) == 0


def test_empty_detections_have_empty_properties() -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "tracker_id"}]
    )

    # when
    result = chain.apply(street(rows=[], tracker=False))

    # then
    assert result == []


def test_sort_of_empty_detections_returns_the_same_object() -> None:
    # given
    detections = empty()

    # when
    result = compile_operations([{"type": "SortDetections", "mode": "size"}]).apply(
        detections
    )

    # then
    assert result is detections


def test_per_box_class_overrides_win_over_the_class_map() -> None:
    # given
    detections = street()
    detections.bboxes_metadata[1]["class"] = "pedestrian"
    names = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}]
    )
    rename = compile_operations(
        [
            {
                "type": "DetectionsRename",
                "class_map": {"pedestrian": "human"},
                "strict": False,
            }
        ]
    )

    # when
    extracted = names.apply(detections)
    renamed = rename.apply(detections)

    # then
    assert extracted == ["car", "pedestrian", "car", "bus"]
    assert renamed.bboxes_metadata[1]["class"] == "human"
    assert renamed.class_id.tolist() == [0, 1024, 0, 2]


def test_missing_class_map_gives_no_names_but_fails_inside_a_filter() -> None:
    # given
    detections = street(class_names=None)
    names = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}]
    )
    by_name = compile_operations(
        [_filter(_binary(_property("class_name"), "==", _static("car")))]
    )

    # when
    extracted = names.apply(detections)
    with pytest.raises(QueryRuntimeError) as error:
        by_name.apply(detections)

    # then
    assert extracted == []
    assert (
        error.value.path
        == "$[0].filter_operation.statements[0].left_operand.operations[0]"
    )
    assert error.value.detail.startswith("detection 0:")


def test_property_no_row_carries_fails_precisely() -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "velocity"}]
    )

    # when
    with pytest.raises(QueryRuntimeError) as error:
        chain.apply(street())

    # then
    assert "velocity" in error.value.detail


def test_single_detection_missing_metadata_property_is_none() -> None:
    # given
    chain = compile_operations(
        [_filter({"type": "UnaryStatement", "operand": _property("time_in_zone"),
                  "operator": {"type": "DoesNotExist"}})]
    )  # fmt: skip

    # when
    result = chain.apply(street())

    # then
    assert len(result) == 4


def test_rename_reads_class_map_and_strict_from_parameters() -> None:
    # given
    chain = compile_operations(
        [{"type": "DetectionsRename", "class_map": "mapping", "strict": "is_strict"}],
        declared_names={"mapping", "is_strict"},
    )
    parameters = {"mapping": {"car": "vehicle"}, "is_strict": False}

    # when
    result = chain.apply(street(), parameters=parameters)

    # then
    assert result.class_id.tolist() == [1024, 1, 1024, 2]


def test_strict_rename_names_the_uncovered_class() -> None:
    # given
    chain = compile_operations(
        [
            {
                "type": "DetectionsRename",
                "class_map": {"car": "vehicle", "person": "human"},
            }
        ]
    )

    # when
    with pytest.raises(QueryRuntimeError) as error:
        chain.apply(street())

    # then
    assert error.value.parameter == "class_map"
    assert "'bus'" in error.value.detail


def test_filter_reads_nested_parameters_and_leaves_them_unchanged() -> None:
    # given
    chain = compile_operations(
        [
            _filter(
                _binary(
                    _property("class_name"),
                    "in (Sequence)",
                    {
                        "type": "DynamicOperand",
                        "operand_name": "config",
                        "operations": [
                            {
                                "type": "LookupTable",
                                "lookup_table": {"wanted": "unused"},
                            }
                        ],
                    },
                    negate=True,
                )
            )
        ],
        declared_names={"config"},
    )
    parameters = {"config": "wanted", "other": {"nested": [1, 2]}}

    # when
    result = chain.apply(street(), parameters=parameters)

    # then
    assert len(result) == 4
    assert parameters == {"config": "wanted", "other": {"nested": [1, 2]}}


def test_filter_binds_each_row_as_a_detection_row() -> None:
    # given
    chain = compile_operations(
        [
            _filter(
                _binary(
                    {"type": "DynamicOperand", "operations": [{"type": "ToString"}]},
                    "(String) startsWith",
                    _static("DetectionRow(1 of 4)"),
                )
            )
        ]
    )

    # when
    result = chain.apply(street())

    # then
    assert summary(result)["detection_id"] == _kept(1)


def test_iterated_native_detection_tuple_is_accepted() -> None:
    # given
    detection = next(iter(street()))
    chain = compile_operations(
        [{"type": "ExtractDetectionProperty", "property_name": "center"}]
    )
    tracker = compile_operations(
        [{"type": "ExtractDetectionProperty", "property_name": "tracker_id"}]
    )

    # when
    center = chain.apply(detection)
    tracker_id = tracker.apply(detection)

    # then
    assert center == (30.0, 40.0)
    assert tracker_id == 7


@pytest.mark.parametrize(
    "value, detail",
    [
        ([1, 2], "expected Detections"),
        ((street_key_points()[0], None), "no bounding-box Detections"),
    ],
)
def test_unsupported_inputs_fail_precisely(value, detail) -> None:
    # given
    chain = compile_operations([{"type": "SortDetections", "mode": "size"}])

    # when
    with pytest.raises(QueryRuntimeError) as error:
        chain.apply(value)

    # then
    assert detail in error.value.detail
    assert error.value.operation == "SortDetections"


def test_sequence_length_and_emptiness_count_key_point_rows() -> None:
    # given
    length = compile_operations([{"type": "SequenceLength"}])
    empty_check = compile_condition(
        {"type": "StatementGroup", "statements": [
            {"type": "UnaryStatement", "operand": _parameter("_"),
             "operator": {"type": "(Sequence) is empty"}}
        ]}
    )  # fmt: skip
    key_points, boxes = street_key_points()
    no_rows = (
        KeyPoints(
            xy=key_points.xy[:0],
            class_id=key_points.class_id[:0],
            confidence=key_points.confidence[:0],
        ),
        empty(),
    )

    # when
    rows = length.apply((key_points, boxes))
    is_empty = empty_check.evaluate({"_": no_rows})

    # then
    assert rows == 4
    assert is_empty is True


def test_detections_to_dictionary_uses_the_inference_response_format() -> None:
    # given
    chain = compile_operations(
        [
            _filter(_binary(_property("class_name"), "==", _static("car"))),
            {"type": "DetectionsToDictionary"},
        ]
    )

    # when
    result = chain.apply(street())

    # then
    first = result["predictions"][0]
    assert [prediction["detection_id"] for prediction in result["predictions"]] == [
        "d0",
        "d2",
    ]
    assert (first["x"], first["y"], first["width"], first["height"]) == (
        30.0,
        40.0,
        40.0,
        40.0,
    )
    assert first["class"] == "car"


def test_detections_to_dictionary_rejects_prediction_rows_columns() -> None:
    # when
    with pytest.raises(QueryRuntimeError) as error:
        compile_operations([{"type": "DetectionsToDictionary"}]).apply(street_rows())

    # then
    assert "tracker_id" in error.value.detail


@pytest.mark.parametrize(
    "point, expected",
    [((5, 5), True), ((15, 5), False), ((10, 5), False), ((0.5, 9.5), True)],
)
def test_detection_in_zone(point, expected) -> None:
    # given
    condition = compile_condition(
        {"type": "StatementGroup", "statements": [
            _binary(_parameter("_"), "(Detection) in zone", _parameter("zone"))
        ]}
    )  # fmt: skip

    # when
    result = condition.evaluate(
        {"_": point, "zone": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    )

    # then
    assert result is expected


def test_zone_filter_on_box_centres() -> None:
    # given
    chain = compile_operations(
        [
            _filter(
                _binary(_property("center"), "(Detection) in zone", _parameter("zone"))
            )
        ],
        declared_names={"zone"},
    )

    # when
    result = chain.apply(
        street(), parameters={"zone": [[90, 0], [300, 0], [300, 100], [90, 100]]}
    )

    # then
    assert summary(result)["detection_id"] == _kept(1, 2, 3)


def _aligned_extras(value, rows=None):
    """Non-box fields that must follow every row selection."""
    if rows is not None:
        value = compile_operations(
            [_filter(_binary(_property("class_id"), "==", _static(0)))]
        ).apply(value)
    extras = {}
    native = value.native if isinstance(value, PredictionRows) else value
    if isinstance(value, PredictionRows):
        extras["columns"] = {
            name: column.tolist() for name, column in value.columns.items()
        }
    if isinstance(native, tuple):
        extras["key_points"] = native[0].xy.tolist()
        extras["covariance_rows"] = native[0].covariance.shape[0]
        native = native[1]
    if isinstance(native, InstanceDetections):
        mask = native.mask
        extras["mask"] = (
            mask.masks if isinstance(mask, InstancesRLEMasks) else mask.tolist()
        )
    extras["metadata"] = native.bboxes_metadata
    return extras
