"""
Tests for post_processing helpers:

  - ConfidenceFilter: 4-tier priority chain and `get_threshold()`
  - NMS helpers: per-class `conf_thresh` tensor path
  - Mask geometry: characterization of the output contract
"""

import hashlib

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
    crop_masks_to_boxes,
    post_process_nms_fused_model_output,
    rescale_image_detections,
    rescale_key_points_detections,
    run_nms_for_instance_segmentation,
    run_nms_for_key_points_detection,
    run_nms_for_object_detection,
    scale_polygons_to_image,
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


class TestAlignInstanceSegmentationResultsChunking:

    @staticmethod
    def _meta(orig_h=200, orig_w=300) -> PreProcessingMetadata:
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

    def _run(self, mask_chunk_size: int, static_crop: bool = False):
        torch.manual_seed(42)
        n = 7
        bboxes = torch.rand((n, 6), dtype=torch.float32) * 100
        masks = torch.randn((n, 160, 160), dtype=torch.float32)
        meta = self._meta()
        static_crop_offset = meta.static_crop_offset
        if static_crop:
            static_crop_offset = StaticCropOffset(
                offset_x=13,
                offset_y=7,
                crop_width=meta.original_size.width,
                crop_height=meta.original_size.height,
            )
        return align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(
                height=meta.original_size.height + (7 if static_crop else 0),
                width=meta.original_size.width + (13 if static_crop else 0),
            ),
            size_after_pre_processing=meta.size_after_pre_processing,
            inference_size=meta.inference_size,
            static_crop_offset=static_crop_offset,
            binarization_threshold=0.0,
            mask_chunk_size=mask_chunk_size,
        )

    @pytest.mark.parametrize("mask_chunk_size", [1, 2, 3, 5])
    def test_chunked_resize_matches_monolithic(self, mask_chunk_size: int) -> None:
        # given / when
        # chunk >= n runs the resize as one batch: the pre-chunking behaviour
        ref_bboxes, ref_masks = self._run(mask_chunk_size=1000)
        out_bboxes, out_masks = self._run(mask_chunk_size=mask_chunk_size)

        # then
        assert torch.equal(out_bboxes, ref_bboxes)
        assert torch.equal(out_masks, ref_masks)
        assert out_masks.dtype == torch.bool

    def test_chunked_resize_matches_monolithic_with_static_crop_canvas(self) -> None:
        # given / when
        ref_bboxes, ref_masks = self._run(mask_chunk_size=1000, static_crop=True)
        out_bboxes, out_masks = self._run(mask_chunk_size=2, static_crop=True)

        # then
        assert torch.equal(out_bboxes, ref_bboxes)
        assert torch.equal(out_masks, ref_masks)


class TestAlignInstanceSegmentationResultsGeometry:
    """Characterization of the mask output contract.

    The chunking tests above are invariance tests: they compare chunked output
    against monolithic output, so a change to the resize target moves both sides
    and both keep passing. These tests pin the geometry itself, which is what a
    resolution change would alter.
    """

    GOLDEN_SET_PIXELS = 210077
    GOLDEN_PER_INSTANCE = [30206, 30285, 30006, 29972, 29811, 30015, 29782]

    @staticmethod
    def _run(static_crop: bool = False, orig_h: int = 200, orig_w: int = 300):
        torch.manual_seed(42)
        n = 7
        bboxes = torch.rand((n, 6), dtype=torch.float32) * 100
        masks = torch.randn((n, 160, 160), dtype=torch.float32)
        static_crop_offset = StaticCropOffset(
            offset_x=13 if static_crop else 0,
            offset_y=7 if static_crop else 0,
            crop_width=orig_w,
            crop_height=orig_h,
        )
        return align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(
                height=orig_h + (7 if static_crop else 0),
                width=orig_w + (13 if static_crop else 0),
            ),
            size_after_pre_processing=ImageDimensions(height=orig_h, width=orig_w),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=static_crop_offset,
            binarization_threshold=0.0,
            mask_chunk_size=1000,
        )

    def test_masks_are_sized_to_size_after_pre_processing(self) -> None:
        # given / when
        _, masks = self._run()

        # then
        assert masks.shape[1:] == (200, 300)

    def test_masks_are_bool(self) -> None:
        # given / when
        _, masks = self._run()

        # then
        assert masks.dtype == torch.bool

    def test_non_square_image_preserves_axis_order(self) -> None:
        # given / when
        # a transposed resize would return (300, 200) and is invisible on a square image
        _, masks = self._run(orig_h=200, orig_w=300)

        # then
        assert masks.shape[1] == 200
        assert masks.shape[2] == 300

    def test_golden_output_for_fixed_seed(self) -> None:
        # given / when
        _, masks = self._run()

        # then
        assert int(masks.sum().item()) == self.GOLDEN_SET_PIXELS
        assert [int(v) for v in masks.flatten(1).sum(1)] == self.GOLDEN_PER_INSTANCE

    def test_static_crop_pastes_onto_original_size_canvas(self) -> None:
        # given / when
        _, masks = self._run(static_crop=True)

        # then
        assert masks.shape[1:] == (207, 313)

    def test_static_crop_preserves_set_pixel_count(self) -> None:
        # given / when
        # the paste relocates content onto a larger canvas; it must not lose
        # or add pixels
        _, masks = self._run(static_crop=True)

        # then
        assert int(masks.sum().item()) == self.GOLDEN_SET_PIXELS


class TestCropMasksToBoxes:
    """First coverage for crop_masks_to_boxes."""

    @staticmethod
    def _masks_and_boxes():
        masks = torch.ones((2, 40, 60), dtype=torch.float32)
        boxes = torch.tensor(
            [[4.0, 8.0, 40.0, 24.0], [0.0, 0.0, 8.0, 8.0]], dtype=torch.float32
        )
        return masks, boxes

    def test_preserves_mask_shape(self) -> None:
        # given
        masks, boxes = self._masks_and_boxes()

        # when
        result = crop_masks_to_boxes(boxes.clone(), masks.clone())

        # then
        assert result.shape == (2, 40, 60)

    def test_zeroes_everything_outside_the_scaled_box(self) -> None:
        # given
        # the default scaling of 0.25 maps box 0 to (1, 2, 10, 6) -> 9 x 4 = 36 pixels
        masks, boxes = self._masks_and_boxes()

        # when
        result = crop_masks_to_boxes(boxes.clone(), masks.clone())

        # then
        assert int((result[0] > 0).sum().item()) == 36

    def test_box_at_origin_is_cropped_from_zero(self) -> None:
        # given
        # box 1 is (0, 0, 8, 8) -> scaled (0, 0, 2, 2) -> 2 x 2 = 4 pixels
        masks, boxes = self._masks_and_boxes()

        # when
        result = crop_masks_to_boxes(boxes.clone(), masks.clone())

        # then
        assert int((result[1] > 0).sum().item()) == 4

    def test_scaling_is_a_single_scalar_for_both_axes(self) -> None:
        # given
        # documents current behaviour: one scalar is applied to x and y alike, so a
        # proto grid whose stride differs from 4, or differs per axis, is mis-cropped
        masks, boxes = self._masks_and_boxes()

        # when
        default_scaling = crop_masks_to_boxes(boxes.clone(), masks.clone())
        explicit_quarter = crop_masks_to_boxes(
            boxes.clone(), masks.clone(), scaling=0.25
        )

        # then
        assert torch.equal(default_scaling, explicit_quarter)


class TestScalePolygonsToImage:
    """Mask-space polygon coordinates must be lifted into image space.

    The legacy path does this via `post_process_polygons` / `scale_polygons`
    (inference/core/utils/postprocess.py:449, :500). The inference_models rewrite
    dropped it, which is harmless only while masks happen to be image-sized.
    """

    def test_identity_when_mask_matches_image(self) -> None:
        # given
        polygons = [np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)]

        # when
        result = scale_polygons_to_image(
            polygons=polygons,
            mask_size=ImageDimensions(height=200, width=300),
            image_size=ImageDimensions(height=200, width=300),
        )

        # then
        assert np.allclose(result[0], polygons[0])

    def test_scales_each_axis_independently(self) -> None:
        # given
        # a non-square ratio: x by 300/150 = 2.0, y by 200/50 = 4.0
        polygons = [np.array([[10.0, 10.0], [50.0, 25.0]], dtype=np.float32)]

        # when
        result = scale_polygons_to_image(
            polygons=polygons,
            mask_size=ImageDimensions(height=50, width=150),
            image_size=ImageDimensions(height=200, width=300),
        )

        # then
        assert np.allclose(result[0], np.array([[20.0, 40.0], [100.0, 100.0]]))

    def test_reduced_mask_coords_reach_image_extent(self) -> None:
        # given
        # the defect this phase fixes: a polygon touching the edge of a half-size
        # mask must land on the edge of the image, not halfway across it
        polygons = [np.array([[0.0, 0.0], [960.0, 540.0]], dtype=np.float32)]

        # when
        result = scale_polygons_to_image(
            polygons=polygons,
            mask_size=ImageDimensions(height=540, width=960),
            image_size=ImageDimensions(height=1080, width=1920),
        )

        # then
        assert result[0][:, 0].max() == pytest.approx(1920.0)
        assert result[0][:, 1].max() == pytest.approx(1080.0)

    def test_empty_polygon_list(self) -> None:
        # given / when
        result = scale_polygons_to_image(
            polygons=[],
            mask_size=ImageDimensions(height=50, width=150),
            image_size=ImageDimensions(height=200, width=300),
        )

        # then
        assert result == []


class TestMasksResolutionFactor:
    """The resolution lever: interpolate the resize target between the
    post-unpad mask grid and size_after_pre_processing."""

    PROTO = 160
    ORIG_H, ORIG_W = 200, 300

    @classmethod
    def _expected(cls, t: float):
        h = max(1, round(cls.PROTO * (1 - t) + cls.ORIG_H * t))
        w = max(1, round(cls.PROTO * (1 - t) + cls.ORIG_W * t))
        return h, w

    @classmethod
    def _run(cls, n: int = 7, static_crop: bool = False, **kwargs):
        torch.manual_seed(42)
        bboxes = torch.rand((n, 6), dtype=torch.float32) * 100
        masks = torch.randn((n, cls.PROTO, cls.PROTO), dtype=torch.float32)
        static_crop_offset = StaticCropOffset(
            offset_x=13 if static_crop else 0,
            offset_y=7 if static_crop else 0,
            crop_width=cls.ORIG_W,
            crop_height=cls.ORIG_H,
        )
        return align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(
                height=cls.ORIG_H + (7 if static_crop else 0),
                width=cls.ORIG_W + (13 if static_crop else 0),
            ),
            size_after_pre_processing=ImageDimensions(
                height=cls.ORIG_H, width=cls.ORIG_W
            ),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=static_crop_offset,
            binarization_threshold=0.0,
            mask_chunk_size=1000,
            **kwargs,
        )

    def test_factor_zero_keeps_the_post_unpad_mask_grid(self) -> None:
        # given / when
        _, masks = self._run(masks_resolution_factor=0.0)

        # then
        assert masks.shape[1:] == (self.PROTO, self.PROTO)

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    def test_factor_interpolates_the_resize_target(self, t: float) -> None:
        # given / when
        _, masks = self._run(masks_resolution_factor=t)

        # then
        assert masks.shape[1:] == self._expected(t)

    @pytest.mark.parametrize("t", [0.25, 0.5])
    def test_axis_order_survives_a_reduced_target(self, t: float) -> None:
        # given / when
        # the image is 200x300, so a transposed resize would be visible here
        _, masks = self._run(masks_resolution_factor=t)

        # then
        expected_h, expected_w = self._expected(t)
        assert masks.shape[1] == expected_h
        assert masks.shape[2] == expected_w
        assert expected_h < expected_w

    def test_output_stays_bool_at_reduced_resolution(self) -> None:
        # given / when
        _, masks = self._run(masks_resolution_factor=0.25)

        # then
        assert masks.dtype == torch.bool

    def test_no_instances_shape_follows_the_factor(self) -> None:
        # given / when
        # consumers derive the canvas from mask.shape, so the empty case must
        # agree with the populated one
        _, masks = self._run(n=0, masks_resolution_factor=0.5)

        # then
        assert masks.shape[0] == 0
        assert masks.shape[1:] == self._expected(0.5)

    def test_static_crop_canvas_scales_with_the_factor(self) -> None:
        # given / when
        _, masks = self._run(static_crop=True, masks_resolution_factor=0.5)

        # then
        # the canvas keeps its ratio to the mask grid, so the whole output is
        # reduced rather than a small mask pasted onto a full-size canvas
        target_h, target_w = self._expected(0.5)
        expected_h = round((self.ORIG_H + 7) * target_h / self.ORIG_H)
        expected_w = round((self.ORIG_W + 13) * target_w / self.ORIG_W)
        assert masks.shape[1:] == (expected_h, expected_w)


class TestRLEMasksResolutionFactor:
    """The RLE generator must honour the same factor as the dense path.

    Otherwise the resolution a caller gets depends on `mask_format`, and the
    two paths silently disagree.
    """

    PROTO = 160
    ORIG_H, ORIG_W = 200, 300

    @classmethod
    def _run(cls, n: int = 3, **kwargs):
        torch.manual_seed(42)
        bboxes = torch.rand((n, 6), dtype=torch.float32) * 100
        masks = torch.randn((n, cls.PROTO, cls.PROTO), dtype=torch.float32)
        return list(
            align_instance_segmentation_results_to_rle_masks(
                image_bboxes=bboxes.clone(),
                masks=masks.clone(),
                padding=(0, 0, 0, 0),
                scale_width=1.0,
                scale_height=1.0,
                original_size=ImageDimensions(height=cls.ORIG_H, width=cls.ORIG_W),
                size_after_pre_processing=ImageDimensions(
                    height=cls.ORIG_H, width=cls.ORIG_W
                ),
                inference_size=ImageDimensions(height=640, width=640),
                static_crop_offset=StaticCropOffset(
                    offset_x=0,
                    offset_y=0,
                    crop_width=cls.ORIG_W,
                    crop_height=cls.ORIG_H,
                ),
                binarization_threshold=0.0,
                **kwargs,
            )
        )

    def test_factor_one_declares_the_image_size(self) -> None:
        # given / when
        results = self._run(masks_resolution_factor=1.0)

        # then
        assert all(rle["size"] == [self.ORIG_H, self.ORIG_W] for _, rle in results)

    def test_reduced_factor_declares_the_reduced_size(self) -> None:
        # given / when
        results = self._run(masks_resolution_factor=0.5)

        # then
        expected_h = round(self.PROTO * 0.5 + self.ORIG_H * 0.5)
        expected_w = round(self.PROTO * 0.5 + self.ORIG_W * 0.5)
        assert all(rle["size"] == [expected_h, expected_w] for _, rle in results)

    @pytest.mark.parametrize("factor", [0.25, 0.5, 1.0])
    def test_rle_content_matches_the_dense_path(self, factor: float) -> None:
        # given / when
        # the previous version asserted decoded.shape == rle["size"], which
        # mask_utils.decode guarantees and so could never fail. Comparing the
        # decoded content against the dense path is the real invariant: the
        # two carriers must describe the same pixels at the same factor.
        rle_results = self._run(masks_resolution_factor=factor)
        torch.manual_seed(42)
        bboxes = torch.rand((3, 6), dtype=torch.float32) * 100
        masks = torch.randn((3, self.PROTO, self.PROTO), dtype=torch.float32)
        _, dense = align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(height=self.ORIG_H, width=self.ORIG_W),
            size_after_pre_processing=ImageDimensions(
                height=self.ORIG_H, width=self.ORIG_W
            ),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=self.ORIG_W, crop_height=self.ORIG_H
            ),
            binarization_threshold=0.0,
            mask_chunk_size=1000,
            masks_resolution_factor=factor,
        )

        # then
        for index, (_, rle) in enumerate(rle_results):
            decoded = mask_utils.decode(rle).astype(bool)
            assert decoded.shape == tuple(dense.shape[1:])
            assert np.array_equal(decoded, dense[index].numpy())

    def test_matches_the_dense_path_resolution(self) -> None:
        # given / when
        # mask_format must not change the resolution a caller receives
        rle_results = self._run(masks_resolution_factor=0.25)
        torch.manual_seed(42)
        bboxes = torch.rand((3, 6), dtype=torch.float32) * 100
        masks = torch.randn((3, self.PROTO, self.PROTO), dtype=torch.float32)
        _, dense = align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(height=self.ORIG_H, width=self.ORIG_W),
            size_after_pre_processing=ImageDimensions(
                height=self.ORIG_H, width=self.ORIG_W
            ),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=self.ORIG_W, crop_height=self.ORIG_H
            ),
            binarization_threshold=0.0,
            mask_chunk_size=1000,
            masks_resolution_factor=0.25,
        )

        # then
        assert rle_results[0][1]["size"] == list(dense.shape[1:])


# Captured from origin/main @ 23a01eb3a with
# `git show origin/main:<this module> > /tmp/pp_main.py`, loading it alongside
# the branch and fingerprinting its output. If origin/main ever changes this
# function these cases SHOULD fail - that is the point of a characterization
# test. Regenerate only after deciding the new behaviour is intended.
_MAIN_GOLDEN = [
    {
        "ih": 333,
        "iw": 211,
        "ox": 97,
        "oy": 61,
        "pad": (16, 18, 6, 5),
        "n": 5,
        "shape": (5, 394, 308),
        "digest": "a48aa710ef85cd88",
        "boxes": "b62c110b17ae5441",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 0,
        "oy": 0,
        "pad": (14, 9, 4, 2),
        "n": 5,
        "shape": (5, 333, 211),
        "digest": "5d04de5cff404410",
        "boxes": "5b0e73b0ea74de77",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 97,
        "oy": 61,
        "pad": (20, 23, 19, 20),
        "n": 0,
        "shape": (0, 261, 397),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 0,
        "oy": 0,
        "pad": (1, 6, 7, 19),
        "n": 0,
        "shape": (0, 200, 300),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 13,
        "oy": 61,
        "pad": (18, 6, 16, 7),
        "n": 5,
        "shape": (5, 394, 224),
        "digest": "a37811a210846e79",
        "boxes": "8cbae67b51c4aa02",
    },
    {
        "ih": 640,
        "iw": 640,
        "ox": 97,
        "oy": 0,
        "pad": (21, 2, 14, 20),
        "n": 1,
        "shape": (1, 640, 737),
        "digest": "f88d6c3e4f5c03f7",
        "boxes": "e1cacd18dd5fddf7",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 0,
        "oy": 7,
        "pad": (10, 24, 7, 16),
        "n": 1,
        "shape": (1, 340, 211),
        "digest": "d15ccb8dd99581e4",
        "boxes": "e47e86508db619f6",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 0,
        "oy": 0,
        "pad": (12, 3, 9, 12),
        "n": 0,
        "shape": (0, 200, 300),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 0,
        "oy": 0,
        "pad": (6, 1, 15, 12),
        "n": 5,
        "shape": (5, 200, 300),
        "digest": "baaed46f79ba3102",
        "boxes": "37313f9472c62672",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 97,
        "oy": 0,
        "pad": (18, 20, 6, 24),
        "n": 5,
        "shape": (5, 333, 308),
        "digest": "c760dd33356b0b6d",
        "boxes": "6b53dff165635fee",
    },
    {
        "ih": 640,
        "iw": 640,
        "ox": 13,
        "oy": 0,
        "pad": (9, 10, 0, 13),
        "n": 0,
        "shape": (0, 640, 653),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 1080,
        "iw": 1920,
        "ox": 0,
        "oy": 0,
        "pad": (0, 1, 14, 15),
        "n": 0,
        "shape": (0, 1080, 1920),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 1080,
        "iw": 1920,
        "ox": 97,
        "oy": 0,
        "pad": (23, 24, 4, 13),
        "n": 5,
        "shape": (5, 1080, 2017),
        "digest": "6982b829d1bbddc7",
        "boxes": "4c219acad8807667",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 0,
        "oy": 61,
        "pad": (13, 6, 0, 8),
        "n": 5,
        "shape": (5, 394, 211),
        "digest": "87fcc86e47722220",
        "boxes": "b21927719126483f",
    },
    {
        "ih": 640,
        "iw": 640,
        "ox": 0,
        "oy": 0,
        "pad": (5, 12, 19, 20),
        "n": 5,
        "shape": (5, 640, 640),
        "digest": "846517457eb562ae",
        "boxes": "3cedb68b2e60e5f6",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 0,
        "oy": 0,
        "pad": (6, 14, 8, 0),
        "n": 5,
        "shape": (5, 200, 300),
        "digest": "6287b10e34217d2e",
        "boxes": "bce5175b45d9adde",
    },
    {
        "ih": 640,
        "iw": 640,
        "ox": 13,
        "oy": 61,
        "pad": (2, 2, 2, 6),
        "n": 5,
        "shape": (5, 701, 653),
        "digest": "d8db8ae532bdb7ea",
        "boxes": "6798272b81011bb4",
    },
    {
        "ih": 1080,
        "iw": 1920,
        "ox": 0,
        "oy": 7,
        "pad": (11, 19, 14, 4),
        "n": 5,
        "shape": (5, 1087, 1920),
        "digest": "c9f9413897eefdd0",
        "boxes": "749ad903686c1c93",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 0,
        "oy": 61,
        "pad": (5, 20, 4, 9),
        "n": 0,
        "shape": (0, 394, 211),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 1080,
        "iw": 1920,
        "ox": 0,
        "oy": 0,
        "pad": (23, 20, 17, 6),
        "n": 5,
        "shape": (5, 1080, 1920),
        "digest": "fe15d5708b271c8f",
        "boxes": "b6b89ee145f20853",
    },
    {
        "ih": 333,
        "iw": 211,
        "ox": 97,
        "oy": 0,
        "pad": (13, 1, 3, 3),
        "n": 0,
        "shape": (0, 333, 308),
        "digest": "e3b0c44298fc1c14",
        "boxes": "e3b0c44298fc1c14",
    },
    {
        "ih": 640,
        "iw": 640,
        "ox": 0,
        "oy": 61,
        "pad": (8, 13, 19, 15),
        "n": 1,
        "shape": (1, 701, 640),
        "digest": "fbb5ace53bca0cf5",
        "boxes": "48123ae5fa70ad76",
    },
    {
        "ih": 1080,
        "iw": 1920,
        "ox": 0,
        "oy": 0,
        "pad": (7, 15, 17, 20),
        "n": 5,
        "shape": (5, 1080, 1920),
        "digest": "be0ffbd3b025fc6b",
        "boxes": "bcc7d8853de382db",
    },
    {
        "ih": 200,
        "iw": 300,
        "ox": 13,
        "oy": 0,
        "pad": (6, 23, 0, 2),
        "n": 1,
        "shape": (1, 200, 313),
        "digest": "4f01e63780b7b135",
        "boxes": "404d5eb899660b53",
    },
]


class TestBitIdenticalToMain:
    """Factor 1.0 must reproduce origin/main exactly.

    The obvious version of this test - comparing the default against an
    explicit 1.0 - compares the branch with itself and cannot fail. These
    cases carry digests of the *pre-change* implementation's output, over a
    matrix that includes non-zero letterbox padding, static crops and empty
    detections, which the rest of the suite does not exercise.
    """

    @staticmethod
    def _run(case: dict):
        torch.manual_seed(7)
        bboxes = torch.rand((case["n"], 6), dtype=torch.float32) * 100
        masks = torch.randn((case["n"], 160, 160), dtype=torch.float32)

        return align_instance_segmentation_results(
            image_bboxes=bboxes.clone(),
            masks=masks.clone(),
            padding=tuple(case["pad"]),
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(
                height=case["ih"] + case["oy"], width=case["iw"] + case["ox"]
            ),
            size_after_pre_processing=ImageDimensions(
                height=case["ih"], width=case["iw"]
            ),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=StaticCropOffset(
                offset_x=case["ox"],
                offset_y=case["oy"],
                crop_width=case["iw"],
                crop_height=case["ih"],
            ),
            binarization_threshold=0.0,
            mask_chunk_size=16,
            masks_resolution_factor=1.0,
        )

    @pytest.mark.parametrize("case", _MAIN_GOLDEN)
    def test_shape_matches_main(self, case: dict) -> None:
        # given / when
        _, masks = self._run(case)

        # then
        # asserted separately from the digest so a geometry change reports
        # readably instead of as an opaque hash mismatch
        assert tuple(masks.shape) == case["shape"]

    @pytest.mark.parametrize("case", _MAIN_GOLDEN)
    def test_content_matches_main(self, case: dict) -> None:
        # given / when
        boxes, masks = self._run(case)

        # then
        assert (
            hashlib.sha256(masks.numpy().tobytes()).hexdigest()[:16] == case["digest"]
        )
        assert hashlib.sha256(boxes.numpy().tobytes()).hexdigest()[:16] == case["boxes"]


class TestStaticCropAtReducedResolution:
    """The combination that crashed: a crop reaching the image edge, a
    letterboxed input, and a factor below 1.0.

    Rounding the canvas size and the crop offset independently can leave the
    canvas a pixel short of the pasted extent. Only crops that touch the image
    edge expose it, and only under letterbox scaling, which is why a sweep with
    zero padding and scale 1.0 finds nothing.
    """

    @staticmethod
    def _letterboxed(image_w: int, image_h: int, offset_x: int, offset_y: int):
        """Build the arguments for a crop that reaches the image edge.

        Args:
            image_w: Full image width.
            image_h: Full image height.
            offset_x: Left edge of the crop.
            offset_y: Top edge of the crop.

        Returns:
            Keyword arguments for the alignment helpers.
        """
        crop_w, crop_h = image_w - offset_x, image_h - offset_y
        scale = min(640 / crop_w, 640 / crop_h)
        new_w, new_h = round(crop_w * scale), round(crop_h * scale)
        pad_left, pad_top = (640 - new_w) // 2, (640 - new_h) // 2

        return {
            "image_bboxes": torch.tensor([[1.0, 1, 50, 50, 0.9, 0]]),
            "masks": torch.randn(1, 160, 160),
            "padding": (
                pad_left,
                pad_top,
                640 - new_w - pad_left,
                640 - new_h - pad_top,
            ),
            "scale_width": scale,
            "scale_height": scale,
            "original_size": ImageDimensions(height=image_h, width=image_w),
            "size_after_pre_processing": ImageDimensions(height=crop_h, width=crop_w),
            "inference_size": ImageDimensions(height=640, width=640),
            "static_crop_offset": StaticCropOffset(
                offset_x=offset_x,
                offset_y=offset_y,
                crop_width=crop_w,
                crop_height=crop_h,
            ),
        }

    CASES = [
        (1920, 1080, 288, 216, 0.5),
        (1920, 1080, 0, 648, 0.75),
        (245, 245, 83, 83, 1 / 3),
        (1280, 720, 417, 311, 0.1),
    ]

    @pytest.mark.parametrize("image_w,image_h,offset_x,offset_y,factor", CASES)
    def test_dense_path_places_the_mask_inside_the_canvas(
        self, image_w: int, image_h: int, offset_x: int, offset_y: int, factor: float
    ) -> None:
        # given
        torch.manual_seed(1)
        kwargs = self._letterboxed(image_w, image_h, offset_x, offset_y)

        # when
        _, masks = align_instance_segmentation_results(
            **kwargs,
            binarization_threshold=0.0,
            mask_chunk_size=16,
            masks_resolution_factor=factor,
        )

        # then
        assert masks.shape[0] == 1

    @pytest.mark.parametrize("image_w,image_h,offset_x,offset_y,factor", CASES)
    def test_rle_path_places_the_mask_inside_the_canvas(
        self, image_w: int, image_h: int, offset_x: int, offset_y: int, factor: float
    ) -> None:
        # given
        torch.manual_seed(1)
        kwargs = self._letterboxed(image_w, image_h, offset_x, offset_y)

        # when
        results = list(
            align_instance_segmentation_results_to_rle_masks(
                **kwargs, binarization_threshold=0.0, masks_resolution_factor=factor
            )
        )

        # then
        assert len(results) == 1
        decoded = mask_utils.decode(results[0][1])
        assert decoded.shape == tuple(results[0][1]["size"])


class TestEmptyAgreesWithPopulated:
    """Zero detections must report the shape the populated path would produce.

    Consumers derive the mask canvas from `mask.shape`, so a frame that
    happens to detect nothing must not describe a different canvas from the
    next frame that does. The empty path has no masks to unpad, so it has to
    compute the unpadded grid rather than read it.
    """

    @staticmethod
    def _run(n: int, padding, factor: float):
        torch.manual_seed(5)
        return align_instance_segmentation_results(
            image_bboxes=torch.rand((n, 6), dtype=torch.float32) * 50,
            masks=torch.randn((n, 160, 160), dtype=torch.float32),
            padding=padding,
            scale_width=1.0,
            scale_height=1.0,
            original_size=ImageDimensions(height=200, width=300),
            size_after_pre_processing=ImageDimensions(height=200, width=300),
            inference_size=ImageDimensions(height=640, width=640),
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=300, crop_height=200
            ),
            binarization_threshold=0.0,
            mask_chunk_size=16,
            masks_resolution_factor=factor,
        )

    @pytest.mark.parametrize("padding", [(0, 0, 0, 0), (12, 34, 12, 34), (7, 0, 41, 3)])
    @pytest.mark.parametrize("factor", [0.0, 0.25, 0.5, 1.0])
    def test_shapes_agree(self, padding, factor: float) -> None:
        # given / when
        _, empty = self._run(0, padding, factor)
        _, populated = self._run(2, padding, factor)

        # then
        assert empty.shape[1:] == populated.shape[1:]


@pytest.mark.parametrize("offset", [0, 83])
def test_empty_masks_match_populated_crop_canvas_after_rounding(offset: int) -> None:
    kwargs = dict(
        padding=(0, 0, 0, 0),
        scale_width=1.0,
        scale_height=1.0,
        original_size=ImageDimensions(height=245, width=245),
        size_after_pre_processing=ImageDimensions(height=162, width=162),
        inference_size=ImageDimensions(height=162, width=162),
        static_crop_offset=StaticCropOffset(
            offset_x=offset, offset_y=offset, crop_width=162, crop_height=162
        ),
        masks_resolution_factor=0.0,
    )
    _, populated = align_instance_segmentation_results(
        image_bboxes=torch.tensor([[0, 0, 162, 162]], dtype=torch.float32),
        masks=torch.ones((1, 81, 81)),
        **kwargs,
    )
    _, empty = align_instance_segmentation_results(
        image_bboxes=torch.empty((0, 4)),
        masks=torch.empty((0, 81, 81)),
        **kwargs,
    )

    assert empty.shape[1:] == populated.shape[1:]
    assert empty.shape[1:] == ((123, 123) if offset else (122, 122))


@pytest.mark.parametrize(
    "factor", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")]
)
def test_mask_resolution_target_rejects_invalid_factor(factor: float) -> None:
    from inference_models.models.common.roboflow.post_processing import (
        resolve_mask_target_size,
    )

    with pytest.raises(ValueError, match="masks_resolution_factor"):
        resolve_mask_target_size(
            mask_height=20,
            mask_width=30,
            size_after_pre_processing=ImageDimensions(height=200, width=300),
            masks_resolution_factor=factor,
        )
