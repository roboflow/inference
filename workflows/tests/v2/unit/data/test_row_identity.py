"""Semantic row identity: one record per result, carried by native selection.

Covers decision 015: the reserved row record (implicit or tensor ids), its
checks at the kind boundary, selection of every native family, codec round
trips, the classification parent reference and the host work of minting,
selecting and reading ids.
"""

import json

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.payload_codec import decode_native, encode_native
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    select_predictions,
    take_rows,
)
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    PARENT_ROW_KEY,
    ROW_RECORD_KEY,
    RowRecord,
    RowRef,
    classification_parent,
    require_row_record,
    row_record,
    with_classification_parent,
    with_row_identity,
    without_row_record,
)
from roboflow_workflows.v2.data.rows import PredictionRows
from roboflow_workflows.v2.engine.recording import CodecRegistry
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from tests.v2.support.recording.codecs import MemoryBlobs

REGISTRY = CodecRegistry(NATIVE_RECORDING_CODECS)


def _detections(rows=3, *, device="cpu"):
    return Detections(
        xyxy=torch.arange(rows * 4, dtype=torch.float32, device=device).reshape(
            rows, 4
        ),
        class_id=torch.arange(rows, device=device),
        confidence=torch.linspace(0.9, 0.1, rows, device=device),
        image_metadata={"class_names": {0: "a"}, "parent_id": "frame"},
        bboxes_metadata=[{"label": f"row{i}"} for i in range(rows)],
    )


def _instances(*, rle):
    boxes = _detections()
    mask = (
        InstancesRLEMasks(image_size=(3, 3), masks=[b"9", b"09", b"135"])
        if rle
        else torch.arange(27).reshape(3, 3, 3) % 2 == 0
    )
    return InstanceDetections(**vars(boxes), mask=mask)


def _key_points(rows=3):
    return KeyPoints(
        xy=torch.arange(rows * 4 * 2, dtype=torch.float32).reshape(rows, 4, 2),
        class_id=torch.arange(rows),
        confidence=torch.rand(rows, 4),
        image_metadata={"parent_id": "frame"},
        key_points_metadata=[{"skeleton": i} for i in range(rows)],
    )


def _record(prediction):
    carrier = prediction[1] if isinstance(prediction, tuple) else prediction
    if isinstance(prediction, tuple) and prediction[1] is None:
        carrier = prediction[0]
    return carrier.image_metadata[ROW_RECORD_KEY]


def _ids(prediction):
    return row_record(prediction).host_ids()


def _aten_ops(function):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        result = function()
    names = [event.name for event in p.events() if event.name.startswith("aten::")]
    return result, names


# Minting --------------------------------------------------------------------------


def test_mint_runs_no_tensor_op_and_keeps_everything_else():
    source = _detections()

    identified, ops = _aten_ops(lambda: with_row_identity(source, namespace="r1"))

    assert ops == []
    assert _record(identified) == {
        "version": 1,
        "namespace": "r1",
        "rows": 3,
        "ids": None,
    }
    assert ROW_RECORD_KEY not in source.image_metadata
    assert identified.xyxy is source.xyxy
    assert identified.class_id is source.class_id
    assert identified.bboxes_metadata is source.bboxes_metadata
    assert (
        identified.image_metadata["class_names"] is source.image_metadata["class_names"]
    )
    OBJECT_DETECTION_PREDICTION_KIND.check(identified)


def test_mint_gives_each_result_its_own_namespace():
    first = with_row_identity(_detections())
    second = with_row_identity(_detections())

    assert row_record(first).namespace != row_record(second).namespace
    assert len(row_record(first).namespace) == 32


def test_mint_once_only():
    identified = with_row_identity(_detections(), namespace="r1")

    with pytest.raises(ContractError, match="already has a row identity.*'r1'"):
        with_row_identity(identified)
    with pytest.raises(ContractError, match="already has a row identity"):
        with_row_identity(select_predictions(identified, indices=[1]))
    reserved = _detections()
    reserved.image_metadata[ROW_RECORD_KEY] = None
    with pytest.raises(ContractError, match="already has a row identity"):
        with_row_identity(reserved)


def test_mint_without_image_metadata_and_on_empty_results():
    bare = Detections(
        xyxy=torch.zeros(0, 4),
        class_id=torch.zeros(0, dtype=torch.int64),
        confidence=torch.zeros(0),
    )

    identified = with_row_identity(bare, namespace="empty")

    assert identified.image_metadata == {
        ROW_RECORD_KEY: {"version": 1, "namespace": "empty", "rows": 0, "ids": None}
    }
    assert row_record(identified).host_ids() == []


@pytest.mark.parametrize("namespace", ["", 7])
def test_mint_rejects_bad_namespace(namespace):
    with pytest.raises(ContractError, match="namespace must be a non-empty string"):
        with_row_identity(_detections(), namespace=namespace)


def test_mint_rejects_unsupported_payloads():
    with pytest.raises(ContractError, match="Row identity applies to"):
        with_row_identity(torch.zeros(3))
    with pytest.raises(ContractError, match="Row identity applies to"):
        with_row_identity((_key_points(), "boxes"))


def test_keypoint_tuples_hold_one_record_on_their_authority():
    key_points, boxes = _key_points(), _detections()

    with_boxes = with_row_identity((key_points, boxes), namespace="kp")
    alone = with_row_identity((key_points, None), namespace="kp")
    bare = with_row_identity(key_points, namespace="kp")

    assert with_boxes[0] is key_points
    assert ROW_RECORD_KEY in with_boxes[1].image_metadata
    assert alone[1] is None and ROW_RECORD_KEY in alone[0].image_metadata
    assert ROW_RECORD_KEY in bare.image_metadata
    for prediction in (with_boxes, alone, (bare, None)):
        KEYPOINT_DETECTION_PREDICTION_KIND.check(prediction)


def test_keypoint_tuple_with_misaligned_rows_is_rejected():
    with pytest.raises(ContractError, match="3 instances but its Detections has 2"):
        with_row_identity((_key_points(3), _detections(2)))


# Selection ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selection",
    [
        {"mask": torch.tensor([False, True, True])},
        {"mask": np.array([False, True, True])},
        {"mask": [False, True, True]},
        {"indices": [1, 2]},
        {"indices": torch.tensor([1, 2], dtype=torch.int32)},
    ],
    ids=["tensor-mask", "numpy-mask", "list-mask", "list-indices", "int32-indices"],
)
def test_first_selection_reuses_the_positions_without_extra_ops(selection):
    plain = _detections()
    identified = with_row_identity(plain)

    _, plain_ops = _aten_ops(lambda: select_predictions(plain, **selection))
    picked, identified_ops = _aten_ops(
        lambda: select_predictions(identified, **selection)
    )

    assert identified_ops == plain_ops
    assert _ids(picked) == [1, 2]
    assert _record(picked)["ids"].dtype == torch.int64


def test_caller_index_tensor_is_not_aliased():
    identified = with_row_identity(_detections())
    indices = torch.tensor([2, 0], dtype=torch.int64)

    picked, ops = _aten_ops(lambda: select_predictions(identified, indices=indices))
    indices[0] = 99

    assert _ids(picked) == [2, 0]
    assert ops.count("aten::clone") == 1


def test_chained_selection_reorders_and_repeats_source_rows():
    identified = with_row_identity(_detections(), namespace="r1")

    sorted_rows = select_predictions(identified, indices=[2, 0, 1])
    repeated, ops = _aten_ops(lambda: select_predictions(sorted_rows, indices=[1, 1]))

    assert _ids(sorted_rows) == [2, 0, 1]
    assert _ids(repeated) == [0, 0]
    assert row_record(repeated).namespace == "r1"
    assert [row["label"] for row in repeated.bboxes_metadata] == ["row0", "row0"]
    # Three fields plus the ids: one gather each.
    assert ops.count("aten::index_select") == 4


@pytest.mark.parametrize("ids", [None, torch.tensor([5, 6, 7])])
def test_identity_selection_keeps_the_metadata_object(ids):
    identified = with_row_identity(_detections())
    _record(identified)["ids"] = ids

    picked = select_predictions(identified, indices=[0, 1, 2])

    assert picked.image_metadata is identified.image_metadata


@pytest.mark.parametrize("ids", [None, torch.tensor([5, 6, 7])])
def test_empty_selection_and_selection_from_empty(ids):
    identified = with_row_identity(_detections())
    _record(identified)["ids"] = ids

    empty = select_predictions(identified, mask=[False, False, False])
    again = select_predictions(empty, indices=[])

    assert _record(empty) == {
        "version": 1,
        "namespace": row_record(identified).namespace,
        "rows": 0,
        "ids": None,
    }
    assert _ids(empty) == [] and _ids(again) == []
    OBJECT_DETECTION_PREDICTION_KIND.check(empty)


def test_upstream_metadata_and_record_stay_untouched():
    identified = with_row_identity(_detections())
    gathered = select_predictions(identified, indices=[2, 1])
    metadata_before = dict(gathered.image_metadata)
    record_before = dict(_record(gathered))
    ids_before = _record(gathered)["ids"].clone()

    select_predictions(gathered, indices=[1, 1, 0])

    assert gathered.image_metadata == metadata_before
    assert _record(gathered).keys() == record_before.keys()
    assert _record(gathered)["rows"] == 2
    assert torch.equal(_record(gathered)["ids"], ids_before)
    assert ROW_RECORD_KEY in identified.image_metadata
    assert _record(identified)["ids"] is None


def test_selection_without_record_keeps_sharing_metadata():
    source = _detections()

    picked = select_predictions(source, indices=[1])

    assert picked.image_metadata is source.image_metadata


@pytest.mark.parametrize("rle", [False, True])
def test_instance_masks_ids_and_rows_move_together(rle):
    identified = with_row_identity(_instances(rle=rle))

    picked = select_predictions(identified, indices=[2, 0, 2])

    assert _ids(picked) == [2, 0, 2]
    assert picked.class_id.tolist() == [2, 0, 2]
    assert [row["label"] for row in picked.bboxes_metadata] == ["row2", "row0", "row2"]
    if rle:
        assert picked.mask.masks == [b"135", b"9", b"135"]
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.check(picked)
    else:
        assert torch.equal(picked.mask, _instances(rle=False).mask[[2, 0, 2]])
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(picked)


@pytest.mark.parametrize("with_boxes", [False, True])
def test_keypoint_ids_follow_the_authority(with_boxes):
    key_points = _key_points()
    identified = with_row_identity((key_points, _detections() if with_boxes else None))

    picked = select_predictions(identified, indices=[1, 1, 0])

    assert _ids(picked) == [1, 1, 0]
    assert [row["skeleton"] for row in picked[0].key_points_metadata] == [1, 1, 0]
    if with_boxes:
        assert ROW_RECORD_KEY not in picked[0].image_metadata
        assert picked[1].class_id.tolist() == [1, 1, 0]
    KEYPOINT_DETECTION_PREDICTION_KIND.check(picked)


def test_keypoints_sharing_the_detections_metadata_dict_are_rejected():
    boxes = with_row_identity(_detections())
    key_points = _key_points()
    key_points.image_metadata = boxes.image_metadata

    with pytest.raises(ContractError, match="the KeyPoints hold one too"):
        KEYPOINT_DETECTION_PREDICTION_KIND.check((key_points, boxes))
    with pytest.raises(ContractError, match="the KeyPoints hold one too"):
        select_predictions((key_points, boxes), indices=[2, 1])


def test_bare_keypoints_carry_their_record():
    identified = with_row_identity(_key_points())

    picked = select_predictions(identified, indices=[2])

    assert isinstance(picked, KeyPoints)
    assert row_record(picked).host_ids() == [2]


def test_prediction_rows_select_through_the_same_path():
    identified = with_row_identity(_detections())
    rows = PredictionRows(native=identified, columns={"score": torch.tensor([1, 2, 3])})

    picked = rows.select(indices=[2, 2])

    assert _ids(picked.native) == [2, 2]
    assert picked.columns["score"].tolist() == [3, 3]


def test_segmentation_view_select_keeps_ids_without_materializing():
    identified = with_row_identity(_detections(3))
    view = SegmentationView(
        detections=identified,
        scores=torch.randn(3, 4, 4),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )

    picked = view.select(torch.tensor([2, 2, 0]))

    assert picked.materialization_count == 0
    assert _ids(picked.detections) == [2, 2, 0]
    assert _ids(picked.full_res()) == [2, 2, 0]


def test_one_selection_covers_extra_columns_and_ids():
    identified = with_row_identity(_detections())
    selection = RowSelection.of(3, mask=torch.tensor([True, False, True]))

    picked = take_rows(identified, selection)

    assert _ids(picked) == [0, 2]
    assert selection.take(torch.tensor([10, 11, 12])).tolist() == [10, 12]


def test_meta_device_selection_reads_no_values():
    # Meta tensors have no data: any host copy or value read would raise.
    meta = torch.device("meta")
    source = with_row_identity(_detections(device=meta))
    source.bboxes_metadata = None

    picked = select_predictions(
        source, indices=torch.empty(4, dtype=torch.int64, device=meta)
    )
    again = select_predictions(
        picked, indices=torch.empty(2, dtype=torch.int64, device=meta)
    )

    assert _record(picked)["ids"].device == meta
    assert _record(again)["ids"].shape == (2,)
    OBJECT_DETECTION_PREDICTION_KIND.check(again)


# Validation ------------------------------------------------------------------------


def _with_record(**changes):
    prediction = with_row_identity(_detections(), namespace="r1")
    record = {**_record(prediction), **changes}
    prediction.image_metadata[ROW_RECORD_KEY] = record
    return prediction


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"extra": 1}, "exactly the keys"),
        ({"version": 2}, "version 2"),
        ({"version": True}, "version True"),
        ({"namespace": ""}, "namespace must be a non-empty string"),
        ({"namespace": None}, "namespace must be a non-empty string"),
        ({"rows": True}, "describes True rows"),
        ({"rows": 2}, "describes 2 rows, but the prediction has 3"),
        ({"ids": [0, 1, 2]}, "must be None or a torch.Tensor"),
        ({"ids": torch.tensor([0.0, 1.0, 2.0])}, "must be int64"),
        ({"ids": torch.tensor([0, 1, 2], dtype=torch.int32)}, "must be int64"),
        ({"ids": torch.zeros(3, 1, dtype=torch.int64)}, r"shape \(3,\)"),
        ({"ids": torch.zeros(2, dtype=torch.int64)}, r"shape \(3,\)"),
        (
            {"ids": torch.empty(3, dtype=torch.int64, device="meta")},
            "is on meta, but the prediction rows are on cpu",
        ),
    ],
)
def test_malformed_records_fail_the_kind_check(changes, match):
    prediction = _with_record(**changes)

    with pytest.raises(ContractError, match=match):
        OBJECT_DETECTION_PREDICTION_KIND.check(prediction)
    with pytest.raises(ContractError, match=match):
        row_record(prediction)


def test_record_must_be_a_dict():
    prediction = _detections()
    prediction.image_metadata[ROW_RECORD_KEY] = "r1"

    with pytest.raises(ContractError, match="exactly the keys"):
        OBJECT_DETECTION_PREDICTION_KIND.check(prediction)


def test_stale_record_on_a_hand_filtered_carrier_is_rejected():
    identified = with_row_identity(_detections())
    hand_filtered = Detections(
        xyxy=identified.xyxy[:2],
        class_id=identified.class_id[:2],
        confidence=identified.confidence[:2],
        image_metadata=identified.image_metadata,
    )

    with pytest.raises(ContractError, match="describes 3 rows.*has 2"):
        OBJECT_DETECTION_PREDICTION_KIND.check(hand_filtered)

    hand_filtered.image_metadata = without_row_record(identified.image_metadata)
    OBJECT_DETECTION_PREDICTION_KIND.check(hand_filtered)
    assert row_record(hand_filtered) is None
    assert ROW_RECORD_KEY in identified.image_metadata


def test_without_row_record_returns_metadata_without_record_unchanged():
    metadata = {"a": 1}

    assert without_row_record(metadata) is metadata
    assert without_row_record(None) is None


def test_second_keypoint_record_is_rejected():
    boxes = with_row_identity(_detections(), namespace="boxes")
    key_points = with_row_identity(_key_points(), namespace="points")

    with pytest.raises(ContractError, match="the KeyPoints hold one too"):
        KEYPOINT_DETECTION_PREDICTION_KIND.check((key_points, boxes))
    with pytest.raises(ContractError, match="the KeyPoints hold one too"):
        row_record((key_points, boxes))
    with pytest.raises(ContractError, match="already has a row identity"):
        with_row_identity((key_points, _detections()))


def test_missing_record_fails_only_correspondence_consumers():
    plain = _detections()

    OBJECT_DETECTION_PREDICTION_KIND.check(plain)
    assert row_record(plain) is None
    with pytest.raises(ContractError, match="v2/dynamic_crop needs predictions"):
        require_row_record(plain, consumer="v2/dynamic_crop")
    implicit = with_row_identity(plain)
    assert require_row_record(implicit, consumer="v2/dynamic_crop").ids is None


# Reading ids -----------------------------------------------------------------------


def test_implicit_ids_read_without_a_transfer(monkeypatch):
    record = row_record(with_row_identity(_detections(), namespace="r1"))
    columns = torch.tensor([[1, 2], [3, 4], [5, 6]])
    calls = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    assert record.host_ids() == [0, 1, 2]
    assert calls == []
    assert record.host_ids_with(columns) == ([0, 1, 2], [[1, 2], [3, 4], [5, 6]])
    assert calls == [(3, 2)]
    assert record.refs() == [RowRef("r1", 0), RowRef("r1", 1), RowRef("r1", 2)]


def test_tensor_ids_and_columns_cross_to_the_host_once(monkeypatch):
    picked = select_predictions(with_row_identity(_detections()), indices=[2, 0])
    record = row_record(picked)
    calls = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)

    ids, rectangles = record.host_ids_with(picked.xyxy.round().to(torch.int32))

    assert calls == [(2, 5)]
    assert ids == [2, 0]
    assert rectangles == [[8, 9, 10, 11], [0, 1, 2, 3]]


@pytest.mark.parametrize(
    "columns",
    [
        torch.zeros(3, 2),
        torch.zeros(3, 2, dtype=torch.bool),
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(2, 2, dtype=torch.int64),
        [[0, 0]] * 3,
    ],
)
def test_columns_must_be_integer_rows(columns):
    record = row_record(with_row_identity(_detections()))

    with pytest.raises(ContractError, match=r"integer \(3, k\) tensor"):
        record.host_ids_with(columns)


def test_columns_on_another_device_than_the_ids_are_rejected():
    record = RowRecord(namespace="r", rows=2, ids=torch.tensor([0, 1]))

    with pytest.raises(ContractError, match="columns are on meta"):
        record.host_ids_with(torch.empty(2, 4, dtype=torch.int64, device="meta"))


# RowRef and classification parents -------------------------------------------------


@pytest.mark.parametrize(
    "value, match",
    [
        ({"namespace": "r", "id": -1}, "non-negative integer"),
        ({"namespace": "r", "id": True}, "non-negative integer"),
        ({"namespace": "r", "id": 1.0}, "non-negative integer"),
        ({"namespace": "", "id": 1}, "non-empty string"),
        ({"namespace": "r"}, "exactly the keys"),
        ({"namespace": "r", "id": 1, "x": 0}, "exactly the keys"),
        (["r", 1], "exactly the keys"),
    ],
)
def test_malformed_row_references_are_rejected(value, match):
    with pytest.raises(ContractError, match=match):
        RowRef.from_dict(value)


def test_row_reference_round_trips_and_normalizes_numpy_ids():
    reference = RowRef("r", np.int64(4))

    assert type(reference.id) is int
    assert RowRef.from_dict(json.loads(json.dumps(reference.to_dict()))) == reference


def _single_label():
    return ClassificationPrediction(
        class_id=torch.tensor([2]),
        confidence=torch.rand(1, 4),
        images_metadata=[{"model": "m"}],
    )


def _multi_label():
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 3]),
        confidence=torch.rand(4),
        image_metadata={"model": "m"},
    )


@pytest.mark.parametrize("make", [_single_label, _multi_label])
def test_classification_parent_is_recorded_separately(make):
    source = make()
    parent = RowRef("r1", 2)

    tagged = with_classification_parent(source, parent)

    assert classification_parent(tagged) == parent
    assert classification_parent(source) is None
    assert with_classification_parent(tagged, parent) is tagged
    assert with_classification_parent(source, None) is source
    CLASSIFICATION_PREDICTION_KIND.check(tagged)
    with pytest.raises(ContractError, match="already refers to row"):
        with_classification_parent(tagged, RowRef("r1", 3))


def test_single_label_parent_lives_in_its_one_image_entry():
    bare = ClassificationPrediction(
        class_id=torch.tensor([1]), confidence=torch.rand(1, 3)
    )

    tagged = with_classification_parent(bare, RowRef("r1", 0))

    assert tagged.images_metadata == [{PARENT_ROW_KEY: {"namespace": "r1", "id": 0}}]
    assert bare.images_metadata is None


@pytest.mark.parametrize("make", [_single_label, _multi_label])
def test_malformed_classification_parent_fails_the_kind_check(make):
    prediction = make()
    metadata = (
        prediction.images_metadata[0]
        if isinstance(prediction, ClassificationPrediction)
        else prediction.image_metadata
    )
    metadata[PARENT_ROW_KEY] = {"namespace": "r1", "id": "2"}

    with pytest.raises(ContractError, match="non-negative integer"):
        CLASSIFICATION_PREDICTION_KIND.check(prediction)
    with pytest.raises(ContractError, match="non-negative integer"):
        classification_parent(prediction)


def test_classification_parent_rejects_bad_inputs():
    with pytest.raises(ContractError, match="parent_row must be a RowRef"):
        with_classification_parent(_single_label(), {"namespace": "r", "id": 0})
    with pytest.raises(ContractError, match="Expected inference_models"):
        classification_parent(_detections())


# Codecs ----------------------------------------------------------------------------


def _codec_cases():
    implicit = with_row_identity(_detections(), namespace="r1")
    gathered = select_predictions(implicit, indices=[2, 0, 2])
    empty = select_predictions(implicit, indices=[])
    dense = select_predictions(with_row_identity(_instances(rle=False)), indices=[1])
    rle = select_predictions(with_row_identity(_instances(rle=True)), indices=[2, 2])
    tuple_ = select_predictions(
        with_row_identity((_key_points(), _detections())), indices=[0, 2]
    )
    parent = with_classification_parent(_single_label(), RowRef("r1", 2))
    return {
        "implicit": implicit,
        "gathered": gathered,
        "empty": empty,
        "dense": dense,
        "rle": rle,
        "keypoint-tuple": tuple_,
        "classification": parent,
    }


def _assert_same_identity(decoded, expected):
    if isinstance(expected, (ClassificationPrediction,)):
        assert classification_parent(decoded) == classification_parent(expected)
        return
    actual_record, expected_record = row_record(decoded), row_record(expected)
    assert actual_record.namespace == expected_record.namespace
    assert actual_record.rows == expected_record.rows
    assert (actual_record.ids is None) == (expected_record.ids is None)
    assert actual_record.host_ids() == expected_record.host_ids()
    if actual_record.ids is not None:
        assert actual_record.ids.dtype == torch.int64


@pytest.mark.parametrize("name", list(_codec_cases()))
def test_native_json_round_trip_keeps_identity(name):
    expected = _codec_cases()[name]

    wire = json.loads(json.dumps(encode_native(expected), allow_nan=False))
    decoded = decode_native(wire)

    _assert_same_identity(decoded, expected)


@pytest.mark.parametrize("name", list(_codec_cases()))
def test_recording_round_trip_keeps_identity(name):
    expected = _codec_cases()[name]
    blobs = MemoryBlobs()

    node = REGISTRY.encoder(blobs, location="g.f").encode(expected)
    node = json.loads(json.dumps(node, allow_nan=False))
    decoded = REGISTRY.decoder(blobs, location="g.f").decode(node)

    _assert_same_identity(decoded, expected)


@pytest.mark.parametrize("with_boxes", [False, True])
@pytest.mark.parametrize("route", ["native-json", "recording"])
def test_keypoint_tuple_round_trip_then_selection_keeps_one_record(route, with_boxes):
    boxes = _detections() if with_boxes else None
    sorted_rows = select_predictions(
        with_row_identity((_key_points(), boxes), namespace="kp"), indices=[2, 0, 1]
    )
    if route == "native-json":
        decoded = decode_native(json.loads(json.dumps(encode_native(sorted_rows))))
    else:
        blobs = MemoryBlobs()
        node = REGISTRY.encoder(blobs, location="g.f").encode(sorted_rows)
        node = json.loads(json.dumps(node, allow_nan=False))
        decoded = REGISTRY.decoder(blobs, location="g.f").decode(node)

    picked = select_predictions(decoded, indices=[2, 2, 0])

    KEYPOINT_DETECTION_PREDICTION_KIND.check(decoded)
    assert _ids(decoded) == [2, 0, 1] and _ids(picked) == [1, 1, 2]
    assert row_record(picked).namespace == "kp"
    assert [row["skeleton"] for row in picked[0].key_points_metadata] == [1, 1, 2]
    if with_boxes:
        assert ROW_RECORD_KEY not in (picked[0].image_metadata or {})
        assert picked[1].class_id.tolist() == [1, 1, 2]


def test_kind_serialization_validates_the_decoded_record():
    gathered = select_predictions(with_row_identity(_detections()), indices=[1, 1])
    envelope = OBJECT_DETECTION_PREDICTION_KIND.to_serialized(gathered)

    restored = OBJECT_DETECTION_PREDICTION_KIND.to_payload(envelope)

    assert _ids(restored) == [1, 1]
    stale = json.loads(json.dumps(envelope))
    metadata = stale["value"]["carrier"]["fields"]["image_metadata"]["dict"]
    metadata[ROW_RECORD_KEY]["dict"]["rows"] = 3
    with pytest.raises(
        ContractError, match="describes 3 rows, but the prediction has 2"
    ):
        OBJECT_DETECTION_PREDICTION_KIND.to_payload(stale)
