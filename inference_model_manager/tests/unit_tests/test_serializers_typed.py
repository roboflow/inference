"""Typed serializers must handle batched (list) outputs — previously only
detections did; every other family crashed with AttributeError on lists."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_model_manager.serializers_typed import (
    serialize_classification_compact,
    serialize_classification_rich,
    serialize_embeddings,
    serialize_instance_segmentation_compact,
    serialize_instance_segmentation_rich,
    serialize_keypoints_compact,
    serialize_multilabel_classification_compact,
    serialize_semantic_segmentation_compact,
    serialize_text,
)

_MODEL = SimpleNamespace(class_names=["a", "b"])


def _cls(conf=(0.9, 0.1)):
    return SimpleNamespace(confidence=list(conf), class_id=[0, 1])


def _mlc():
    return SimpleNamespace(confidence=[0.9, 0.8], class_ids=[0, 1])


def _iseg():
    return SimpleNamespace(
        xyxy=[[0, 0, 1, 1]], class_id=[0], confidence=[0.9], mask=[[1]]
    )


def _semseg():
    return SimpleNamespace(segmentation_map=[[0, 1]], confidence=[[0.9, 0.8]])


def _kp():
    return SimpleNamespace(xy=[[1, 2]], class_id=[0], confidence=[0.9])


class TestBatchedOutputs:
    def test_classification_compact_batch(self):
        out = serialize_classification_compact([_cls(), _cls()], _MODEL)
        assert len(out["batch"]) == 2
        assert out["batch"][0]["confidences"] == [0.9, 0.1]

    def test_multilabel_compact_batch(self):
        out = serialize_multilabel_classification_compact([_mlc(), _mlc()], _MODEL)
        assert len(out["batch"]) == 2
        assert out["batch"][0]["detected_classes_ids"] == [0, 1]

    def test_instance_segmentation_compact_batch(self):
        out = serialize_instance_segmentation_compact([_iseg(), _iseg()], _MODEL)
        assert len(out["batch"]) == 2

    def test_semantic_segmentation_compact_batch(self):
        out = serialize_semantic_segmentation_compact([_semseg(), _semseg()], _MODEL)
        assert len(out["batch"]) == 2

    def test_keypoints_compact_batch(self):
        out = serialize_keypoints_compact([_kp(), _kp()], _MODEL)
        assert len(out["batch"]) == 2

    def test_embeddings_batch(self):
        out = serialize_embeddings([[1.0, 2.0], [3.0, 4.0]], _MODEL)
        assert out["type"] == "roboflow-embeddings-compact-v1"

    def test_text_list_is_not_python_repr(self):
        out = serialize_text(["cap one", "cap two"], _MODEL)
        assert out == {
            "type": "roboflow-text-v1",
            "batch": [{"text": "cap one"}, {"text": "cap two"}],
        }

    def test_classification_rich_batch(self):
        out = serialize_classification_rich([_cls(), _cls((0.2, 0.8))], _MODEL)
        assert len(out["batch"]) == 2
        assert out["batch"][1]["top"][0]["class_name"] == "b"


class _RLEMasks:
    def to_coco_rle_masks(self):
        return [{"size": [4, 4], "counts": b"02"}, {"size": [4, 4], "counts": "13"}]


def _iseg_rle():
    return SimpleNamespace(
        xyxy=[[0, 0, 1, 1], [1, 1, 2, 2]],
        class_id=[0, 1],
        confidence=[0.9, 0.8],
        mask=_RLEMasks(),
    )


class TestRLEMasks:
    def test_compact_rle_masks_become_coco_dicts(self):
        out = serialize_instance_segmentation_compact(_iseg_rle(), _MODEL)
        assert out["mask"] == [
            {"format": "rle", "size": [4, 4], "counts": "02"},
            {"format": "rle", "size": [4, 4], "counts": "13"},
        ]

    def test_rich_rle_masks_are_per_detection(self):
        out = serialize_instance_segmentation_rich(_iseg_rle(), _MODEL)
        assert out["detections"][0]["mask"] == {
            "format": "rle",
            "size": [4, 4],
            "counts": "02",
        }
        assert out["detections"][1]["mask"]["counts"] == "13"

    def test_rich_dense_masks_unchanged(self):
        out = serialize_instance_segmentation_rich(_iseg(), _MODEL)
        assert out["detections"][0]["mask"] == [1]


class TestSingleOutputsUnchanged:
    def test_classification_compact_single(self):
        out = serialize_classification_compact(_cls(), _MODEL)
        assert out["confidences"] == [0.9, 0.1]
        assert "batch" not in out

    def test_text_single(self):
        assert serialize_text("hello", _MODEL)["text"] == "hello"


def test_serialize_gaze_compact():
    from types import SimpleNamespace
    from inference_model_manager.serializers_typed import serialize_gaze_compact

    out = SimpleNamespace(yaw=[0.1, 0.2], pitch=[-0.3, 0.0])
    result = serialize_gaze_compact(out, model=None)
    assert result == {
        "type": "roboflow-gaze-compact-v1",
        "yaw": [0.1, 0.2],
        "pitch": [-0.3, 0.0],
    }


def test_serialize_gaze_compact_batch():
    from types import SimpleNamespace
    from inference_model_manager.serializers_typed import serialize_gaze_compact

    outs = [
        SimpleNamespace(yaw=[0.1], pitch=[0.2]),
        SimpleNamespace(yaw=[0.3], pitch=[0.4]),
    ]
    result = serialize_gaze_compact(outs, model=None)
    assert result["type"] == "roboflow-gaze-compact-v1"
    assert result["batch"] == [
        {"yaw": [0.1], "pitch": [0.2]},
        {"yaw": [0.3], "pitch": [0.4]},
    ]


def test_serialize_structured_ocr_compact():
    from types import SimpleNamespace
    from inference_model_manager.serializers_typed import (
        serialize_structured_ocr_compact,
    )

    det = SimpleNamespace(
        xyxy=[[0, 0, 10, 10]],
        class_id=[2],
        confidence=[0.9],
        bboxes_metadata=[{"text": "word"}],
    )
    model = SimpleNamespace(class_names=["block", "line", "word"])
    out = (["word"], [det])
    result = serialize_structured_ocr_compact(out, model)
    assert result["type"] == "roboflow-structured-ocr-compact-v1"
    assert result["class_names"] == ["block", "line", "word"]
    assert result["batch"] == [
        {
            "text": "word",
            "regions": {
                "xyxy": [[0, 0, 10, 10]],
                "class_id": [2],
                "confidence": [0.9],
                "texts": ["word"],
            },
        }
    ]


def test_serialize_sam_segmentation_compact():
    from types import SimpleNamespace
    from inference_model_manager.serializers_typed import (
        serialize_sam_segmentation_compact,
    )

    preds = [
        SimpleNamespace(masks="m0", scores="s0"),
        SimpleNamespace(masks="m1", scores="s1"),
    ]
    result = serialize_sam_segmentation_compact(preds, model=None)
    assert result["type"] == "roboflow-sam-segmentation-compact-v1"
    assert result["batch"] == [
        {"masks": "m0", "scores": "s0"},
        {"masks": "m1", "scores": "s1"},
    ]


def test_serialize_sam_segmentation_compact_single():
    from types import SimpleNamespace
    from inference_model_manager.serializers_typed import (
        serialize_sam_segmentation_compact,
    )

    pred = SimpleNamespace(masks="m0", scores="s0")
    result = serialize_sam_segmentation_compact(pred, model=None)
    assert result == {
        "type": "roboflow-sam-segmentation-compact-v1",
        "masks": "m0",
        "scores": "s0",
    }


def _det_with_metadata():
    return SimpleNamespace(
        xyxy=[[0, 0, 1, 1]],
        class_id=[0],
        confidence=[0.9],
        image_metadata={"class_names": ["bolt", "screw"]},
    )


def _det_without_metadata():
    return SimpleNamespace(xyxy=[[0, 0, 1, 1]], class_id=[0], confidence=[0.9])


class TestDetectionsClassNamesMetadataFallback:
    def test_compact_falls_back_to_image_metadata_when_model_lacks_class_names(self):
        from inference_model_manager.serializers_typed import (
            serialize_detections_compact,
        )

        out = serialize_detections_compact(_det_with_metadata(), None)
        assert out["class_names"] == ["bolt", "screw"]

    def test_compact_model_class_names_win_over_metadata(self):
        from inference_model_manager.serializers_typed import (
            serialize_detections_compact,
        )

        out = serialize_detections_compact(_det_with_metadata(), _MODEL)
        assert out["class_names"] == ["a", "b"]

    def test_compact_batch_falls_back_to_first_detection_with_metadata(self):
        from inference_model_manager.serializers_typed import (
            serialize_detections_compact,
        )

        out = serialize_detections_compact(
            [_det_without_metadata(), _det_with_metadata()], None
        )
        assert out["class_names"] == ["bolt", "screw"]

    def test_compact_class_names_stay_none_without_model_or_metadata(self):
        from inference_model_manager.serializers_typed import (
            serialize_detections_compact,
        )

        out = serialize_detections_compact(
            [_det_without_metadata(), _det_without_metadata()], None
        )
        assert out["class_names"] is None

    def test_rich_falls_back_to_image_metadata(self):
        from inference_model_manager.serializers_typed import serialize_detections_rich

        out = serialize_detections_rich(_det_with_metadata(), None)
        assert out["detections"][0]["class_name"] == "bolt"
