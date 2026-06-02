"""
Tests for post_processing helpers:

  - ConfidenceFilter: 4-tier priority chain and `get_threshold()`
  - NMS helpers: per-class `conf_thresh` tensor path
"""

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    align_instance_segmentation_results,
    align_instance_segmentation_results_to_rle_masks,
    align_instance_segmentation_results_to_rle_masks_batch,
    post_process_nms_fused_model_output,
    rescale_image_detections,
    rescale_key_points_detections,
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


def _decode_rles(rles, height: int, width: int) -> np.ndarray:
    if not rles:
        return np.empty((0, height, width), dtype=bool)
    decoded = mask_utils.decode(rles)
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    return decoded.transpose(2, 0, 1).astype(bool)


def _rle_alignment_inputs():
    bboxes = torch.tensor(
        [
            [1.0, 2.0, 9.0, 7.0],
            [-2.0, -1.0, 5.0, 4.0],
            [4.0, 1.0, 14.0, 12.0],
        ],
        dtype=torch.float32,
    )
    masks = torch.full((3, 8, 10), -1.0, dtype=torch.float32)
    masks[0, 2:6, 3:8] = 2.0
    masks[1, 1:4, 1:5] = 1.0
    masks[2, 4:7, 5:9] = 3.0
    return bboxes, masks


def _static_crop(offset_x: int, offset_y: int, width: int, height: int):
    return StaticCropOffset(
        offset_x=offset_x,
        offset_y=offset_y,
        crop_width=width,
        crop_height=height,
    )


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


class TestRescaleImageDetectionsClipping:

    @staticmethod
    def _meta(orig_h=400, orig_w=600) -> PreProcessingMetadata:
        return PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(height=orig_h, width=orig_w),
            size_after_pre_processing=ImageDimensions(height=orig_h, width=orig_w),
            inference_size=ImageDimensions(height=640, width=640),
            scale_width=1.0,
            scale_height=1.0,
            static_crop_offset=StaticCropOffset(
                offset_x=0,
                offset_y=0,
                crop_width=orig_w,
                crop_height=orig_h,
            ),
        )

    def test_clips_negative_x1_y1_to_zero(self) -> None:
        detections = torch.tensor(
            [[-3.0, -5.0, 200.0, 200.0, 0.9, 0.0]], dtype=torch.float32
        )
        out = rescale_image_detections(detections, self._meta(orig_h=400, orig_w=600))
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[0, 1].item() == pytest.approx(0.0)
        assert out[0, 2].item() == pytest.approx(200.0)
        assert out[0, 3].item() == pytest.approx(200.0)

    def test_clips_x2_y2_to_image_extent(self) -> None:
        detections = torch.tensor(
            [[10.0, 10.0, 700.0, 500.0, 0.9, 0.0]], dtype=torch.float32
        )
        out = rescale_image_detections(detections, self._meta(orig_h=400, orig_w=600))
        assert out[0, 0].item() == pytest.approx(10.0)
        assert out[0, 1].item() == pytest.approx(10.0)
        assert out[0, 2].item() == pytest.approx(600.0)
        assert out[0, 3].item() == pytest.approx(400.0)

    def test_in_bounds_boxes_unchanged(self) -> None:
        detections = torch.tensor(
            [[50.0, 60.0, 400.0, 350.0, 0.8, 0.0]], dtype=torch.float32
        )
        out = rescale_image_detections(detections, self._meta(orig_h=400, orig_w=600))
        assert torch.allclose(
            out[0, :4],
            torch.tensor([50.0, 60.0, 400.0, 350.0]),
            atol=1e-6,
        )

    def test_clipping_preserves_score_and_class_columns(self) -> None:
        detections = torch.tensor(
            [[-1.0, -2.0, 1000.0, 1000.0, 0.42, 7.0]], dtype=torch.float32
        )
        out = rescale_image_detections(detections, self._meta(orig_h=400, orig_w=600))
        assert out[0, 4].item() == pytest.approx(0.42)
        assert out[0, 5].item() == pytest.approx(7.0)


class TestRescaleKeyPointsDetectionsClipping:

    @staticmethod
    def _meta(orig_h=400, orig_w=600) -> PreProcessingMetadata:
        return PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(height=orig_h, width=orig_w),
            size_after_pre_processing=ImageDimensions(height=orig_h, width=orig_w),
            inference_size=ImageDimensions(height=640, width=640),
            scale_width=1.0,
            scale_height=1.0,
            static_crop_offset=StaticCropOffset(
                offset_x=0,
                offset_y=0,
                crop_width=orig_w,
                crop_height=orig_h,
            ),
        )

    def test_clips_box_coords_for_keypoint_detections(self) -> None:
        # Row layout: [x1, y1, x2, y2, conf, cls_id, kp_x, kp_y, kp_conf]
        detections = [
            torch.tensor(
                [[-5.0, 10.0, 700.0, 350.0, 0.9, 0.0, 100.0, 100.0, 0.8]],
                dtype=torch.float32,
            )
        ]
        rescale_key_points_detections(
            detections,
            [self._meta(orig_h=400, orig_w=600)],
            num_classes=1,
            key_points_slots_in_prediction=1,
        )
        out = detections[0]
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[0, 1].item() == pytest.approx(10.0)
        assert out[0, 2].item() == pytest.approx(600.0)
        assert out[0, 3].item() == pytest.approx(350.0)
        assert out[0, 4].item() == pytest.approx(0.9)
        assert out[0, 5].item() == pytest.approx(0.0)
        assert out[0, 6].item() == pytest.approx(100.0)
        assert out[0, 7].item() == pytest.approx(100.0)


class TestAlignInstanceSegmentationResultsClipping:

    @staticmethod
    def _meta(orig_h=400, orig_w=600) -> PreProcessingMetadata:
        return PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(height=orig_h, width=orig_w),
            size_after_pre_processing=ImageDimensions(height=orig_h, width=orig_w),
            inference_size=ImageDimensions(height=640, width=640),
            scale_width=1.0,
            scale_height=1.0,
            static_crop_offset=StaticCropOffset(
                offset_x=0,
                offset_y=0,
                crop_width=orig_w,
                crop_height=orig_h,
            ),
        )

    def test_clips_box_coords(self) -> None:
        bboxes = torch.tensor(
            [[10.0, 20.0, 700.0, 500.0, 0.9, 0.0]], dtype=torch.float32
        )
        masks = torch.zeros((1, 160, 160), dtype=torch.float32)
        meta = self._meta(orig_h=400, orig_w=600)
        out_bboxes, _ = align_instance_segmentation_results(
            image_bboxes=bboxes,
            masks=masks,
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=meta.original_size,
            size_after_pre_processing=meta.size_after_pre_processing,
            inference_size=meta.inference_size,
            static_crop_offset=meta.static_crop_offset,
            binarization_threshold=0.0,
        )
        # box clamped to image bounds (400×600)
        assert out_bboxes[0, 0].item() == pytest.approx(10.0)
        assert out_bboxes[0, 1].item() == pytest.approx(20.0)
        assert out_bboxes[0, 2].item() == pytest.approx(600.0)
        assert out_bboxes[0, 3].item() == pytest.approx(400.0)


class TestAlignInstanceSegmentationResultsToRleMasksBatch:

    @pytest.mark.parametrize(
        "case",
        [
            {
                "padding": (0, 0, 0, 0),
                "original_size": ImageDimensions(height=8, width=10),
                "size_after_pre_processing": ImageDimensions(height=8, width=10),
                "inference_size": ImageDimensions(height=8, width=10),
                "static_crop_offset": _static_crop(0, 0, 10, 8),
                "binarization_threshold": 0.0,
            },
            {
                "padding": (1, 1, 1, 0),
                "original_size": ImageDimensions(height=8, width=10),
                "size_after_pre_processing": ImageDimensions(height=8, width=10),
                "inference_size": ImageDimensions(height=8, width=10),
                "static_crop_offset": _static_crop(0, 0, 10, 8),
                "binarization_threshold": 0.0,
            },
            {
                "padding": (-1, 0, -1, 0),
                "original_size": ImageDimensions(height=8, width=10),
                "size_after_pre_processing": ImageDimensions(height=8, width=10),
                "inference_size": ImageDimensions(height=8, width=10),
                "static_crop_offset": _static_crop(0, 0, 10, 8),
                "binarization_threshold": 0.0,
            },
            {
                "padding": (0, 0, 0, 0),
                "original_size": ImageDimensions(height=11, width=13),
                "size_after_pre_processing": ImageDimensions(height=8, width=10),
                "inference_size": ImageDimensions(height=8, width=10),
                "static_crop_offset": _static_crop(2, 1, 10, 8),
                "binarization_threshold": 0.0,
            },
            {
                "padding": (0, 0, 0, 0),
                "original_size": ImageDimensions(height=8, width=10),
                "size_after_pre_processing": ImageDimensions(height=8, width=10),
                "inference_size": ImageDimensions(height=8, width=10),
                "static_crop_offset": _static_crop(0, 0, 10, 8),
                "binarization_threshold": 0.5,
            },
        ],
    )
    def test_batch_matches_generator_path(self, case: dict) -> None:
        bboxes, masks = _rle_alignment_inputs()
        batch_boxes = bboxes.clone()
        generator_boxes = bboxes.clone()

        actual_boxes, actual_rles = (
            align_instance_segmentation_results_to_rle_masks_batch(
                image_bboxes=batch_boxes,
                masks=masks.clone(),
                scale_width=1.0,
                scale_height=1.0,
                **case,
            )
        )
        expected_pairs = list(
            align_instance_segmentation_results_to_rle_masks(
                image_bboxes=generator_boxes,
                masks=masks.clone(),
                scale_width=1.0,
                scale_height=1.0,
                **case,
            )
        )
        expected_boxes = torch.stack([bbox for bbox, _ in expected_pairs])
        expected_rles = [rle for _, rle in expected_pairs]

        torch.testing.assert_close(actual_boxes, expected_boxes, rtol=0, atol=0)
        torch.testing.assert_close(batch_boxes, expected_boxes, rtol=0, atol=0)
        np.testing.assert_array_equal(
            _decode_rles(
                actual_rles,
                case["original_size"].height,
                case["original_size"].width,
            ),
            _decode_rles(
                expected_rles,
                case["original_size"].height,
                case["original_size"].width,
            ),
        )

    def test_empty_batch_matches_generator_path(self) -> None:
        case = {
            "padding": (0, 0, 0, 0),
            "original_size": ImageDimensions(height=8, width=10),
            "size_after_pre_processing": ImageDimensions(height=8, width=10),
            "inference_size": ImageDimensions(height=8, width=10),
            "static_crop_offset": _static_crop(0, 0, 10, 8),
            "binarization_threshold": 0.0,
        }
        bboxes = torch.empty((0, 4), dtype=torch.float32)
        masks = torch.empty((0, 8, 10), dtype=torch.float32)

        actual_boxes, actual_rles = (
            align_instance_segmentation_results_to_rle_masks_batch(
                image_bboxes=bboxes.clone(),
                masks=masks.clone(),
                scale_width=1.0,
                scale_height=1.0,
                **case,
            )
        )
        expected_pairs = list(
            align_instance_segmentation_results_to_rle_masks(
                image_bboxes=bboxes.clone(),
                masks=masks.clone(),
                scale_width=1.0,
                scale_height=1.0,
                **case,
            )
        )

        assert actual_boxes.shape == (0, 4)
        assert actual_rles == []
        assert expected_pairs == []
