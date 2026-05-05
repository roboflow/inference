"""
Tests for post_processing helpers:

  - ConfidenceFilter: 4-tier priority chain and `get_threshold()`
  - NMS helpers: per-class `conf_thresh` tensor path
"""

import pytest
import torch

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    post_process_nms_fused_model_output,
    run_nms_for_instance_segmentation,
    run_nms_for_key_points_detection,
    run_nms_for_object_detection,
)
from inference_models.weights_providers.entities import RecommendedParameters


def _od_output(box_class_conf):
    """Build (1, 4+num_classes, num_anchors) tensor from a list of
    (xywh, class_id, conf) triples. Each anchor's class scores are zero
    except the assigned class."""
    num_anchors = len(box_class_conf)
    num_classes = max(c for _, c, _ in box_class_conf) + 1
    out = torch.zeros((1, 4 + num_classes, num_anchors))
    for i, (xywh, cls, conf) in enumerate(box_class_conf):
        out[0, :4, i] = torch.tensor(xywh, dtype=torch.float32)
        out[0, 4 + cls, i] = conf
    return out


class TestRunNmsForObjectDetection:
    def test_scalar_keeps_all_above_threshold(self) -> None:
        # Three well-separated boxes, three classes, conf 0.7/0.5/0.3.
        out = _od_output(
            [
                ((10, 10, 10, 10), 0, 0.7),
                ((100, 100, 10, 10), 1, 0.5),
                ((200, 200, 10, 10), 2, 0.3),
            ]
        )
        result = run_nms_for_object_detection(out, conf_thresh=0.4)
        # 0.3 dropped; 0.7 and 0.5 kept.
        assert result[0].shape[0] == 2

    def test_per_class_tensor_drops_only_classes_above_their_threshold(
        self,
    ) -> None:
        out = _od_output(
            [
                ((10, 10, 10, 10), 0, 0.7),
                ((100, 100, 10, 10), 1, 0.5),
                ((200, 200, 10, 10), 2, 0.3),
            ]
        )
        # cat=0.6 (drops 0?), dog=0.6 (drops 0.5), fish=0.2 (keeps 0.3)
        thresholds = torch.tensor([0.6, 0.6, 0.2])
        result = run_nms_for_object_detection(out, conf_thresh=thresholds)
        kept_classes = sorted(int(c) for c in result[0][:, 5].tolist())
        assert kept_classes == [0, 2]

    def test_per_class_tensor_moved_to_output_device(self) -> None:
        out = _od_output([((10, 10, 10, 10), 0, 0.9)])
        thresholds = torch.tensor([0.5])  # CPU tensor
        result = run_nms_for_object_detection(out, conf_thresh=thresholds)
        assert result[0].shape[0] == 1


class TestPostProcessNmsFused:
    def _fused(self, rows):
        # rows: list of (x1, y1, x2, y2, conf, cls)
        return torch.tensor([rows], dtype=torch.float32)

    def test_scalar_threshold(self) -> None:
        out = self._fused(
            [
                (0, 0, 10, 10, 0.9, 0),
                (10, 10, 20, 20, 0.4, 1),
            ]
        )
        result = post_process_nms_fused_model_output(out, conf_thresh=0.5)
        assert result[0].shape[0] == 1
        assert int(result[0][0, 5]) == 0

    def test_per_class_tensor_indexes_by_class_id(self) -> None:
        out = self._fused(
            [
                (0, 0, 10, 10, 0.9, 0),  # cls 0, conf 0.9 vs thresh 0.95 → drop
                (10, 10, 20, 20, 0.4, 1),  # cls 1, conf 0.4 vs thresh 0.3 → keep
                (20, 20, 30, 30, 0.6, 2),  # cls 2, conf 0.6 vs thresh 0.5 → keep
            ]
        )
        thresholds = torch.tensor([0.95, 0.3, 0.5])
        result = post_process_nms_fused_model_output(out, conf_thresh=thresholds)
        kept = sorted(int(c) for c in result[0][:, 5].tolist())
        assert kept == [1, 2]


def _is_output(box_class_conf, num_mask_coeffs=32):
    num_anchors = len(box_class_conf)
    num_classes = max(c for _, c, _ in box_class_conf) + 1
    out = torch.zeros((1, 4 + num_classes + num_mask_coeffs, num_anchors))
    for i, (xywh, cls, conf) in enumerate(box_class_conf):
        out[0, :4, i] = torch.tensor(xywh, dtype=torch.float32)
        out[0, 4 + cls, i] = conf
    return out


class TestRunNmsForInstanceSegmentation:
    def test_per_class_tensor_drops_per_class(self) -> None:
        out = _is_output(
            [
                ((10, 10, 10, 10), 0, 0.7),
                ((100, 100, 10, 10), 1, 0.5),
                ((200, 200, 10, 10), 2, 0.3),
            ]
        )
        thresholds = torch.tensor([0.6, 0.6, 0.2])
        result = run_nms_for_instance_segmentation(out, conf_thresh=thresholds)
        kept = sorted(int(c) for c in result[0][:, 5].tolist())
        assert kept == [0, 2]


def _kp_output(box_class_conf, num_classes, kp_slots):
    num_anchors = len(box_class_conf)
    out = torch.zeros((1, 4 + num_classes + kp_slots * 3, num_anchors))
    for i, (xywh, cls, conf) in enumerate(box_class_conf):
        out[0, :4, i] = torch.tensor(xywh, dtype=torch.float32)
        out[0, 4 + cls, i] = conf
    return out


class TestRunNmsForKeyPointsDetection:
    def test_per_class_tensor_drops_per_class(self) -> None:
        out = _kp_output(
            [
                ((10, 10, 10, 10), 0, 0.7),
                ((100, 100, 10, 10), 1, 0.5),
                ((200, 200, 10, 10), 2, 0.3),
            ],
            num_classes=3,
            kp_slots=4,
        )
        thresholds = torch.tensor([0.6, 0.6, 0.2])
        result = run_nms_for_key_points_detection(
            out,
            num_classes=3,
            key_points_slots_in_prediction=4,
            conf_thresh=thresholds,
        )
        kept = sorted(int(c) for c in result[0][:, 5].tolist())
        assert kept == [0, 2]


class TestConfidenceFilter:
    @staticmethod
    def _rd(*, confidence=None, per_class=None) -> RecommendedParameters:
        return RecommendedParameters(
            confidence=confidence, per_class_confidence=per_class
        )

    def test_user_value_overrides_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(
            confidence=0.7,
            recommended_parameters=self._rd(
                confidence=0.42, per_class={"cat": 0.6, "dog": 0.3}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["cat", "dog"]) == pytest.approx(0.7)

    def test_explicit_zero_user_value_is_honored(self) -> None:
        cf = ConfidenceFilter(
            confidence=0.0,
            recommended_parameters=self._rd(confidence=0.42, per_class={"cat": 0.6}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["cat"]) == pytest.approx(0.0)

    def test_get_threshold_align_to_class_names(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        torch.testing.assert_close(
            cf.get_threshold(["cat", "dog", "fish"]),
            torch.tensor([0.6, 0.4, 0.5]),
        )

    def test_unknown_class_falls_back_to_global_optimal(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(confidence=0.5, per_class={"cat": 0.6}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["fish"]).tolist() == pytest.approx([0.5])

    def test_unknown_class_falls_back_to_default_when_no_global(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(per_class={"cat": 0.6}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["dog"]) == pytest.approx(
            INFERENCE_MODELS_DEFAULT_CONFIDENCE
        )

    def test_per_class_overrides_global_optimal(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(confidence=0.5, per_class={"cat": 0.9}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["cat"]).tolist() == pytest.approx([0.9])

    def test_global_optimal_used_when_no_per_class(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(confidence=0.42),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["any"]) == pytest.approx(0.42)

    def test_empty_per_class_treated_as_no_per_class(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(confidence=0.42, per_class={}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        assert cf.get_threshold(["any"]) == pytest.approx(0.42)

    def test_default_used_when_no_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=None,
            default_confidence=0.25,
        )
        assert cf.get_threshold(["a", "b"]) == pytest.approx(0.25)

    def test_default_used_when_recommended_parameters_is_all_none(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=RecommendedParameters(),
            default_confidence=0.25,
        )
        assert cf.get_threshold(["a"]) == pytest.approx(0.25)

    def test_get_threshold_returns_float_tensor(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=self._rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        result = cf.get_threshold(["cat", "dog", "fish"])
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (3,)

    def test_get_threshold_scalar_when_no_per_class(self) -> None:
        cf = ConfidenceFilter(
            confidence="best",
            recommended_parameters=None,
            default_confidence=0.5,
        )
        assert cf.get_threshold([]) == pytest.approx(0.5)

    def test_default_string_skips_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(
            confidence="default",
            recommended_parameters=self._rd(confidence=0.5, per_class={"cat": 0.6}),
            default_confidence=0.25,
        )
        assert cf.get_threshold(["cat", "dog"]) == pytest.approx(0.25)
