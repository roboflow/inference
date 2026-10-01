"""
Tests for post_processing helpers:

  - ConfidenceFilter: 4-tier priority chain and `get_threshold()`
  - NMS helpers: per-class `conf_thresh` tensor path
  - Mask geometry: characterization of the output contract
"""

import numpy as np
import pytest
import torch

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    ConfidenceFilter,
    align_instance_segmentation_results,
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
