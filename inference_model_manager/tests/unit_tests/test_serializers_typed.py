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
