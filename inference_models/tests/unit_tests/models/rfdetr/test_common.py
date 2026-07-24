import math

import torch

from inference_models.models.rfdetr.common import (
    keypoint_precision_cholesky_to_pixel_covariance,
)


def test_covariance_identity_precision_scales_to_pixel_space() -> None:
    # given
    # log_l11 = log_l22 = 0 and l21 = 0 -> L = I -> precision = I -> covariance = I.
    precision_cholesky = torch.zeros(1, 1, 3)

    # when
    covariance = keypoint_precision_cholesky_to_pixel_covariance(
        precision_cholesky=precision_cholesky, width=20.0, height=10.0
    )

    # then
    # diag([width, height]) @ I @ diag([width, height]) = diag([width**2, height**2]).
    assert covariance.shape == (1, 1, 2, 2)
    assert torch.allclose(
        covariance[0, 0],
        torch.tensor([[400.0, 0.0], [0.0, 100.0]]),
    )


def test_covariance_matches_closed_form_inverse() -> None:
    # given
    # L = [[l11, 0], [l21, l22]] with l11 = exp(0.5), l22 = exp(-0.25), l21 = 0.3.
    log_l11, l21, log_l22 = 0.5, 0.3, -0.25
    precision_cholesky = torch.tensor([[[log_l11, l21, log_l22]]])
    width, height = 7.0, 13.0

    # when
    covariance = keypoint_precision_cholesky_to_pixel_covariance(
        precision_cholesky=precision_cholesky, width=width, height=height
    )

    # then
    l11 = math.exp(log_l11)
    l22 = math.exp(log_l22)
    det = (l11 * l22) ** 2
    cov00 = (l21**2 + l22**2) / det
    cov01 = -l11 * l21 / det
    cov11 = l11**2 / det
    expected = torch.tensor(
        [
            [width * width * cov00, width * height * cov01],
            [width * height * cov01, height * height * cov11],
        ]
    )
    assert torch.allclose(covariance[0, 0], expected, atol=1e-5)


def test_covariance_is_symmetric() -> None:
    # given
    torch.manual_seed(0)
    precision_cholesky = torch.randn(4, 5, 3)

    # when
    covariance = keypoint_precision_cholesky_to_pixel_covariance(
        precision_cholesky=precision_cholesky, width=11.0, height=17.0
    )

    # then
    assert torch.allclose(
        covariance[..., 0, 1], covariance[..., 1, 0], equal_nan=True
    )


def test_covariance_non_finite_input_yields_nan() -> None:
    # given
    precision_cholesky = torch.zeros(1, 2, 3)
    precision_cholesky[0, 1, 0] = float("inf")  # non-finite log_l11

    # when
    covariance = keypoint_precision_cholesky_to_pixel_covariance(
        precision_cholesky=precision_cholesky, width=5.0, height=5.0
    )

    # then
    # The finite keypoint is computed normally; the non-finite one is masked to NaN.
    assert torch.isfinite(covariance[0, 0]).all()
    assert torch.isnan(covariance[0, 1]).all()


def test_covariance_degenerate_precision_yields_nan() -> None:
    # given
    # Huge log magnitudes drive det(precision) to +inf -> 1 / det underflows/overflows,
    # producing non-finite pixel covariances that must be masked to NaN.
    precision_cholesky = torch.tensor([[[100.0, 0.0, 100.0]]])

    # when
    covariance = keypoint_precision_cholesky_to_pixel_covariance(
        precision_cholesky=precision_cholesky, width=5.0, height=5.0
    )

    # then
    assert torch.isnan(covariance[0, 0]).all()


class TestInstanceSegmentationMaxDetectionsCap:

    @staticmethod
    def _meta(orig_h=200, orig_w=300):
        from inference_models.entities import ImageDimensions
        from inference_models.models.common.roboflow.model_packages import (
            PreProcessingMetadata,
            StaticCropOffset,
        )

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

    @staticmethod
    def _inputs(num_queries=12, num_classes=3):
        torch.manual_seed(7)
        bboxes = torch.rand((1, num_queries, 4), dtype=torch.float32) * 0.5 + 0.25
        logits = torch.randn((1, num_queries, num_classes), dtype=torch.float32)
        masks = torch.randn((1, num_queries, 32, 32), dtype=torch.float32)
        return bboxes, logits, masks

    def _run_dense(self, max_detections):
        from inference_models.models.rfdetr.common import (
            post_process_instance_segmentation_results,
        )

        bboxes, logits, masks = self._inputs()
        return post_process_instance_segmentation_results(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=[self._meta()],
            threshold=0.0,
            num_classes=3,
            classes_re_mapping=None,
            max_detections=max_detections,
        )[0]

    def _run_rle(self, max_detections):
        from inference_models.models.rfdetr.common import (
            post_process_instance_segmentation_results_to_rle_masks,
        )

        bboxes, logits, masks = self._inputs()
        return post_process_instance_segmentation_results_to_rle_masks(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            pre_processing_meta=[self._meta()],
            threshold=0.0,
            num_classes=3,
            classes_re_mapping=None,
            max_detections=max_detections,
        )[0]

    def test_dense_cap_keeps_top_scoring_prefix(self) -> None:
        # given / when
        uncapped = self._run_dense(max_detections=None)
        capped = self._run_dense(max_detections=5)

        # then: with threshold=0 every query survives, so uncapped = num_queries
        assert uncapped.confidence.shape[0] == 12
        assert capped.confidence.shape[0] == 5
        # results are score-sorted, so the cap must select the top-5 prefix
        assert torch.equal(capped.confidence, uncapped.confidence[:5])
        assert torch.equal(capped.class_id, uncapped.class_id[:5])
        assert torch.equal(capped.xyxy, uncapped.xyxy[:5])
        assert torch.equal(capped.mask, uncapped.mask[:5])
        assert capped.mask.shape == (5, 200, 300)

    def test_dense_cap_larger_than_survivors_is_noop(self) -> None:
        # given / when
        uncapped = self._run_dense(max_detections=None)
        capped = self._run_dense(max_detections=50)

        # then
        assert torch.equal(capped.confidence, uncapped.confidence)
        assert torch.equal(capped.mask, uncapped.mask)

    def test_rle_cap_keeps_top_scoring_prefix(self) -> None:
        # given / when
        uncapped = self._run_rle(max_detections=None)
        capped = self._run_rle(max_detections=5)

        # then
        assert uncapped.confidence.shape[0] == 12
        assert capped.confidence.shape[0] == 5
        assert torch.equal(capped.confidence, uncapped.confidence[:5])
        assert torch.equal(capped.class_id, uncapped.class_id[:5])
        assert torch.equal(capped.xyxy, uncapped.xyxy[:5])
        assert capped.mask.masks == uncapped.mask.masks[:5]

    def test_dense_and_rle_agree_under_cap(self) -> None:
        # given / when
        dense = self._run_dense(max_detections=5)
        rle = self._run_rle(max_detections=5)

        # then
        assert torch.equal(dense.confidence, rle.confidence)
        assert torch.equal(dense.class_id, rle.class_id)
