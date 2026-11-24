"""UQL detection operations with semantic row identity and segmentation views.

Scene: the street of ``test_detection_operations`` as a modern model emits it,
with a row record (namespace ``street``) and no per-row ``detection_id``::

    row  class   box                 conf
    0    car     (10, 20, 50, 60)    0.875
    1    person  (110, 20, 130, 60)  0.5
    2    car     (200, 40, 260, 80)  0.875
    3    bus     (100, 10, 180, 70)  0.375

The segmentation view puts each row's mask on a compact 5x15 logit grid over
the 100x300 image (stride 20); one positive 2x2 cell block per row.
"""

import dataclasses

import pytest
import torch
from roboflow_workflows.v2.blocks.query import QueryRuntimeError, compile_operations
from roboflow_workflows.v2.blocks.query.detection_ops import derived_detection_id
from roboflow_workflows.v2.blocks.query.detections import detection_row
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.prediction_ops import RowSelection, take_rows
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    row_record,
    with_row_identity,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

from .test_detection_operations import street

TO_DICTIONARY = [{"type": "DetectionsToDictionary"}]
SORT_BY_CONFIDENCE = [
    {"type": "SortDetections", "mode": "confidence", "ascending": True}
]
KEEP_CARS_ABOVE_HALF = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "operator": "and",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "_",
                        "operations": [
                            {"type": "ExtractDetectionProperty", "property_name": "class_name"}
                        ],
                    },
                    "comparator": {"type": "=="},
                    "right_operand": {"type": "StaticOperand", "value": "car"},
                },
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "_",
                        "operations": [
                            {"type": "ExtractDetectionProperty", "property_name": "confidence"}
                        ],
                    },
                    "comparator": {"type": "(Number) >"},
                    "right_operand": {"type": "StaticOperand", "value": 0.5},
                },
            ],
        },
    }
]  # fmt: skip


def modern_street() -> Detections:
    detections = dataclasses.replace(street(), bboxes_metadata=None)
    return with_row_identity(detections, namespace="street")


def ids_of(prediction) -> list:
    return row_record(prediction).host_ids()


def response_ids(serialized: dict) -> list:
    return [row["detection_id"] for row in serialized["predictions"]]


def derived(*rows) -> list:
    return [derived_detection_id("street", row) for row in rows]


def street_view() -> SegmentationView:
    detections = modern_street()
    scores = torch.full((4, 5, 15), -8.0)
    for row, (x_min, y_min, _, _) in enumerate(detections.xyxy.tolist()):
        cell_y, cell_x = int(y_min) // 20, int(x_min) // 20
        scores[row, cell_y : cell_y + 2, cell_x : cell_x + 2] = 8.0
    geometry = MaskGridGeometry(
        grid_size_hw=(5, 15),
        inference_size_hw=(20, 60),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(0.2, 0.2),
        crop_size_hw=(100, 300),
        original_size_hw=(100, 300),
        static_crop_xywh=(0, 0, 300, 100),
    )
    return SegmentationView(
        detections=detections,
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )


def test_iterated_row_carries_no_inherited_row_record() -> None:
    # given
    detections = modern_street()
    iterated = list(detections)[2]

    # when
    row = detection_row(iterated)

    # then
    assert ROW_RECORD_KEY not in row.table.image_metadata
    assert row.property("class_name") == "car"
    assert ROW_RECORD_KEY in detections.image_metadata


def test_filter_sort_and_selection_gather_row_keys() -> None:
    # given
    detections = modern_street()

    # when
    sorted_rows = compile_operations(SORT_BY_CONFIDENCE).apply(detections)
    kept = compile_operations(KEEP_CARS_ABOVE_HALF).apply(sorted_rows)
    right_most = compile_operations(
        [{"type": "DetectionsSelection", "mode": "right_most"}]
    ).apply(detections)

    # then
    assert ids_of(sorted_rows) == [3, 1, 0, 2]
    assert ids_of(kept) == [0, 2]
    assert ids_of(right_most) == [2]
    assert row_record(detections).ids is None


def test_dictionary_derives_ids_from_row_keys_without_mutating_input() -> None:
    # given
    detections = modern_street()
    chain = compile_operations(SORT_BY_CONFIDENCE + TO_DICTIONARY)

    # when
    serialized = chain.apply(detections)

    # then
    assert response_ids(serialized) == derived(3, 1, 0, 2)
    assert [row["class"] for row in serialized["predictions"]] == [
        "bus",
        "person",
        "car",
        "car",
    ]
    assert [row["x"] for row in serialized["predictions"]] == [140, 120, 30, 230]
    assert [row["width"] for row in serialized["predictions"]] == [80, 20, 40, 60]
    assert detections.bboxes_metadata is None
    assert row_record(detections).ids is None


def test_derived_ids_are_stable_across_calls_and_repeats() -> None:
    # given
    repeated = take_rows(modern_street(), RowSelection.of(4, indices=[1, 1, 3]))
    chain = compile_operations(TO_DICTIONARY)

    # when
    first = response_ids(chain.apply(repeated))
    second = response_ids(chain.apply(modern_street()))

    # then
    assert first == derived(1, 1, 3)
    assert second == derived(0, 1, 2, 3)
    assert first[0] == second[1]


def test_dictionary_keeps_explicit_legacy_ids_and_fills_only_missing_ones() -> None:
    # given
    rows = [{"detection_id": "legacy-0"}, {}, {"detection_id": "legacy-2"}, None]
    detections = with_row_identity(
        dataclasses.replace(street(), bboxes_metadata=rows), namespace="street"
    )

    # when
    serialized = compile_operations(TO_DICTIONARY).apply(detections)

    # then
    assert response_ids(serialized) == [
        "legacy-0",
        *derived(1),
        "legacy-2",
        *derived(3),
    ]
    assert rows == [
        {"detection_id": "legacy-0"},
        {},
        {"detection_id": "legacy-2"},
        None,
    ]


def test_dictionary_of_legacy_rows_equals_the_v1_serializer_output() -> None:
    # given
    from roboflow_workflows.core_steps.common.serializers_tensor import (
        serialise_sv_detections,
    )

    detections = street()

    # when
    serialized = compile_operations(TO_DICTIONARY).apply(detections)

    # then
    assert serialized == serialise_sv_detections(detections)


def test_dictionary_without_ids_or_row_identity_fails_explicitly() -> None:
    # given
    detections = dataclasses.replace(street(), bboxes_metadata=None)

    # when
    with pytest.raises(QueryRuntimeError) as error:
        compile_operations(TO_DICTIONARY).apply(detections)

    # then
    assert "detection 0 has no `detection_id`" in str(error.value)
    assert "no row identity" in str(error.value)


@pytest.mark.parametrize("with_identity", [True, False])
def test_dictionary_of_empty_predictions_has_no_rows(with_identity: bool) -> None:
    # given
    empty = street(rows=[])
    if with_identity:
        empty = with_row_identity(empty, namespace="street")

    # when
    serialized = compile_operations(TO_DICTIONARY).apply(empty)

    # then
    assert serialized["predictions"] == []


def test_dictionary_keeps_per_box_class_names_and_class_map() -> None:
    # given
    rows = [{}, {"class": "pedestrian"}, {}, {}]
    detections = with_row_identity(
        dataclasses.replace(street(), bboxes_metadata=rows), namespace="street"
    )

    # when
    serialized = compile_operations(TO_DICTIONARY).apply(detections)

    # then
    assert [row["class"] for row in serialized["predictions"]] == [
        "car",
        "pedestrian",
        "car",
        "bus",
    ]
    assert [row["class_id"] for row in serialized["predictions"]] == [0, 1, 0, 2]


def test_keypoint_tuple_sorts_with_row_keys_and_serializes_boxes() -> None:
    # given
    key_points = KeyPoints(
        xy=torch.arange(16, dtype=torch.float32).reshape(4, 2, 2),
        class_id=torch.tensor([0, 1, 0, 2]),
        confidence=torch.full((4, 2), 0.5),
    )
    prediction = with_row_identity(
        (key_points, dataclasses.replace(street(), bboxes_metadata=None)),
        namespace="street",
    )

    # when
    sorted_prediction = compile_operations(SORT_BY_CONFIDENCE).apply(prediction)
    serialized = compile_operations(TO_DICTIONARY).apply(sorted_prediction)

    # then
    assert ids_of(sorted_prediction) == [3, 1, 0, 2]
    assert sorted_prediction[0].xy[:, 0, 0].tolist() == [12.0, 4.0, 0.0, 8.0]
    assert response_ids(serialized) == derived(3, 1, 0, 2)


def test_view_filter_and_sort_use_boxes_only_and_keep_compact_masks() -> None:
    # given
    view = street_view()

    # when
    kept = compile_operations(SORT_BY_CONFIDENCE + KEEP_CARS_ABOVE_HALF).apply(view)
    sizes = compile_operations(
        [{"type": "DetectionsPropertyExtract", "property_name": "size"}]
    ).apply(kept)

    # then
    assert isinstance(kept, SegmentationView)
    assert ids_of(kept.detections) == [0, 2]
    assert torch.equal(kept.scores, view.scores[[0, 2]])
    assert sizes == [1600.0, 2400.0]
    assert view.materialization_count == 0
    assert kept.materialization_count == 0


def test_view_rename_and_shift_keep_the_view_and_its_row_keys() -> None:
    # given
    view = street_view()
    chain = compile_operations(
        [
            {
                "type": "DetectionsRename",
                "class_map": {"car": "vehicle"},
                "strict": False,
            },
            {"type": "DetectionsShift", "shift_x": 5, "shift_y": 0},
        ]
    )

    # when
    edited = chain.apply(view)

    # then
    assert isinstance(edited, SegmentationView)
    assert ids_of(edited.detections) == [0, 1, 2, 3]
    assert edited.detections.xyxy[:, 0].tolist() == [15.0, 115.0, 205.0, 105.0]
    assert edited.scores is view.scores
    assert edited.materialization_count == 0


def test_view_dictionary_exports_exact_masks_once() -> None:
    # given
    view = compile_operations(KEEP_CARS_ABOVE_HALF).apply(street_view())
    from roboflow_workflows.core_steps.common.serializers_tensor import (
        serialise_sv_detections,
    )

    # when
    serialized = compile_operations(TO_DICTIONARY).apply(view)
    again = compile_operations(TO_DICTIONARY).apply(view)

    # then
    exact = view.full_res()
    expected = serialise_sv_detections(
        dataclasses.replace(
            exact, bboxes_metadata=[{"detection_id": i} for i in derived(0, 2)]
        )
    )
    assert serialized == expected
    assert again == serialized
    assert all(row["points"] for row in serialized["predictions"])
    assert view.materialization_count == 1


def test_dense_fallback_view_stays_dense_after_filtering() -> None:
    # given
    dense = street_view().full_res()
    view = SegmentationView.from_dense(
        InstanceDetections(
            xyxy=dense.xyxy,
            class_id=dense.class_id,
            confidence=dense.confidence,
            mask=dense.mask,
            image_metadata=dense.image_metadata,
            bboxes_metadata=None,
        )
    )

    # when
    kept = compile_operations(KEEP_CARS_ABOVE_HALF).apply(view)

    # then
    assert kept.is_dense_fallback
    assert ids_of(kept.detections) == [0, 2]
    assert torch.equal(kept.full_res().mask, dense.mask[[0, 2]])
