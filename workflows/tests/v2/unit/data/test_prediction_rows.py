"""PredictionRows: native rows and numeric columns stay aligned, one authority."""

import json

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND, PredictionRows
from roboflow_workflows.v2.engine.recording import (
    CodecRegistry,
    RecordingCorruptError,
)
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from tests.v2.support.recording.codecs import MemoryBlobs

REGISTRY = CodecRegistry(NATIVE_RECORDING_CODECS)


def _frame_metadata():
    root = ImageData.from_tensor(
        torch.zeros((3, 29, 47), dtype=torch.uint8), image_id="root"
    )
    small = root.crop((7, 3, 27, 13), image_id="crop").resize((5, 10), image_id="small")

    return FrameRef.from_image(small).prediction_metadata()


def _segments(*, rle=False, labels=None):
    mask = (
        InstancesRLEMasks(image_size=(5, 10), masks=[b"a", b"b", b"c"])
        if rle
        else torch.arange(150).reshape(3, 5, 10) % 3 == 0
    )
    return InstanceDetections(
        xyxy=torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 4.0], [2.0, 0.0, 5.0, 3.0]]
        ),
        class_id=torch.tensor([7, 8, 9]),
        confidence=torch.tensor([0.1, 0.2, 0.3]),
        mask=mask,
        image_metadata=_frame_metadata(),
        bboxes_metadata=[{"label": name} for name in (labels or ("a", "b", "c"))],
    )


def _columns():
    return {
        "tracker_id": torch.tensor([10, 11, 12]),
        # Strided (3, 2) column: every second value of a (3, 4) tensor.
        "embedding": torch.arange(12, dtype=torch.float16).reshape(3, 4)[:, ::2],
    }


@pytest.mark.parametrize("rle", [False, True])
def test_two_zero_two_then_empty_selects_native_fields_and_columns_together(rle):
    rows = PredictionRows(native=_segments(rle=rle), columns=_columns())

    picked = rows.select(indices=[2, 0, 2])
    empty = picked.select(mask=torch.zeros(3, dtype=torch.bool))

    assert len(picked) == 3
    assert picked.native.class_id.tolist() == [9, 7, 9]
    assert [row["label"] for row in picked.native.bboxes_metadata] == ["c", "a", "c"]
    assert picked.columns["tracker_id"].tolist() == [12, 10, 12]
    assert torch.equal(picked.columns["embedding"], _columns()["embedding"][[2, 0, 2]])
    assert picked.columns["embedding"].dtype == torch.float16
    if rle:
        assert picked.native.mask.masks == [b"c", b"a", b"c"]
    else:
        assert torch.equal(picked.native.mask, rows.native.mask[[2, 0, 2]])
    assert picked.frame == rows.frame
    assert picked.frame.root.map_xy(2, 1) == (11.0, 5.0)

    assert len(empty) == 0
    assert empty.native.xyxy.shape == (0, 4)
    assert empty.columns["tracker_id"].shape == (0,)
    assert empty.columns["embedding"].shape == (0, 2)


def test_keypoint_rows_select_points_boxes_and_columns_together():
    points = KeyPoints(
        xy=torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
        class_id=torch.tensor([0, 1, 2]),
        confidence=torch.ones((3, 2)),
        image_metadata=_frame_metadata(),
    )
    rows = PredictionRows(
        native=(points, None), columns={"tracker_id": torch.tensor([4, 5, 6])}
    )

    picked = rows.select(indices=torch.tensor([1, 1]))

    assert picked.native[0].xy.tolist() == [[[4.0, 5.0], [6.0, 7.0]]] * 2
    assert picked.native[1] is None
    assert picked.columns["tracker_id"].tolist() == [5, 5]


def test_identity_selection_keeps_column_tensors():
    rows = PredictionRows(native=_segments(), columns=_columns())

    same = rows.select(mask=[True, True, True])

    assert same.columns["tracker_id"] is rows.columns["tracker_id"]
    assert same.native.xyxy is rows.native.xyxy


def test_columns_mapping_is_read_only_and_not_copied():
    tracker_id = torch.tensor([1, 2, 3])
    source = {"tracker_id": tracker_id}
    rows = PredictionRows(native=_segments(), columns=source)
    source["late"] = torch.zeros(3)

    assert rows.columns["tracker_id"] is tracker_id
    assert "late" not in rows.columns
    with pytest.raises(TypeError):
        rows.columns["tracker_id"] = torch.zeros(3)


@pytest.mark.parametrize(
    "columns, labels, match",
    [
        ({"label": torch.zeros(3)}, None, "also a key of the native row metadata"),
        ({"xyxy": torch.zeros(3)}, None, "shadow the native field"),
        ({"mask": torch.zeros(3)}, None, "shadow the native field"),
        ({"tracker_id": torch.zeros(2)}, None, "3 leading rows"),
        ({"tracker_id": torch.tensor(1)}, None, "3 leading rows"),
        ({"names": ["a", "b", "c"]}, None, "keep strings"),
        ({"": torch.zeros(3)}, None, "non-empty strings"),
        ({"z": torch.zeros(3, dtype=torch.complex64)}, None, "z"),
        (
            {"tracker_id": torch.zeros(3, device="meta")},
            None,
            "but the native prediction is on cpu",
        ),
    ],
)
def test_malformed_or_shadowing_columns_are_rejected(columns, labels, match):
    with pytest.raises(ContractError, match=match):
        PredictionRows(native=_segments(labels=labels), columns=columns)


def test_tracker_id_has_one_authority():
    native = _segments()
    native.bboxes_metadata = [{"tracker_id": i} for i in range(3)]

    with pytest.raises(ContractError, match="'tracker_id' is also a key"):
        PredictionRows(native=native, columns={"tracker_id": torch.arange(3)})


def test_unsupported_native_payload_is_rejected():
    with pytest.raises(ContractError, match="supports Detections"):
        PredictionRows(native=torch.zeros(3, 4))


def test_kind_round_trips_through_its_json_envelope():
    rows = PredictionRows(native=_segments(rle=True), columns=_columns())

    wire = json.loads(json.dumps(PREDICTION_ROWS_KIND.serialize(rows)))
    restored = PREDICTION_ROWS_KIND.deserialize(wire)

    PREDICTION_ROWS_KIND.check(restored)
    assert restored.native.mask == rows.native.mask
    assert restored.native.image_metadata == rows.native.image_metadata
    assert restored.frame == rows.frame
    for name, column in rows.columns.items():
        assert torch.equal(restored.columns[name], column)
        assert restored.columns[name].dtype == column.dtype
    assert PREDICTION_ROWS_KIND.deserialize(rows) is rows


def test_kind_rejects_foreign_or_malformed_values():
    from roboflow_workflows.v2.data.payload_codec import encode_native

    with pytest.raises(ContractError, match="Expected PredictionRows"):
        PREDICTION_ROWS_KIND.deserialize({"native": 1})
    with pytest.raises(ContractError, match="prediction_rows envelope"):
        PREDICTION_ROWS_KIND.deserialize(encode_native(_segments()))
    with pytest.raises(ContractError):
        PREDICTION_ROWS_KIND.check(_segments())


def test_root_output_converts_native_geometry_and_keeps_columns():
    rows = PredictionRows(native=_segments(rle=False), columns=_columns())

    own = PREDICTION_ROWS_KIND.convert_output(rows, {"coordinates_system": "own"})
    root = PREDICTION_ROWS_KIND.convert_output(rows, {})

    assert own is rows
    assert root.native.xyxy[0].tolist() == [7.0, 3.0, 11.0, 7.0]
    assert root.native.mask.shape == (3, 29, 47)
    assert root.columns["tracker_id"] is rows.columns["tracker_id"]
    assert root.frame.image_id == "root"


def test_recording_codec_round_trips_rows_and_rejects_misaligned_data():
    rows = PredictionRows(native=_segments(), columns=_columns())
    blobs = MemoryBlobs()
    node = json.loads(json.dumps(REGISTRY.encoder(blobs, location="g.f").encode(rows)))

    restored = REGISTRY.decoder(blobs, location="g.f").decode(node)

    assert isinstance(restored, PredictionRows)
    assert torch.equal(restored.native.mask, rows.native.mask)
    assert restored.columns["tracker_id"].tolist() == [10, 11, 12]
    assert restored.columns["embedding"].dtype == torch.float16

    short = PredictionRows(
        native=Detections(
            xyxy=torch.zeros((2, 4)),
            class_id=torch.zeros(2, dtype=torch.int64),
            confidence=torch.zeros(2),
        ),
    )
    short_node = REGISTRY.encoder(blobs, location="g.f").encode(short)
    node["codec"]["value"]["native"] = short_node["codec"]["value"]["native"]
    with pytest.raises(RecordingCorruptError, match="invalid prediction rows"):
        REGISTRY.decoder(blobs, location="g.f").decode(node)
