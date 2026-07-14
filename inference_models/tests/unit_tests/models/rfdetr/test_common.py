import math

import torch

import inference_models.models.rfdetr.common as rfdetr_common
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
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
    assert torch.allclose(covariance[..., 0, 1], covariance[..., 1, 0], equal_nan=True)


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


def test_object_detection_post_process_preserves_topk_order_without_resort(
    monkeypatch,
) -> None:
    # given: torch.topk() returns the selected scores in descending order and the
    # post-processing masks retain their relative order.
    bboxes = torch.tensor(
        [
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2],
                [0.8, 0.8, 0.2, 0.2],
            ]
        ]
    )
    logits = torch.tensor([[[0.0, 5.0], [4.0, 0.0], [3.0, 0.0]]])
    dimensions = ImageDimensions(width=10, height=10)
    metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=dimensions,
        size_after_pre_processing=dimensions,
        inference_size=dimensions,
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(0, 0, 10, 10),
    )

    def unexpected_sort(*args, **kwargs):
        raise AssertionError("post-processing must not re-sort topk predictions")

    monkeypatch.setattr(rfdetr_common.torch, "sort", unexpected_sort)

    # when
    result = rfdetr_common.post_process_object_detection_results(
        bboxes=bboxes,
        logits=logits,
        pre_processing_meta=[metadata],
        threshold=0.5,
        num_classes=2,
        classes_re_mapping=None,
        device=torch.device("cpu"),
    )

    # then
    assert torch.equal(
        result[0].xyxy,
        torch.tensor([[4, 4, 6, 6], [1, 1, 3, 3], [7, 7, 9, 9]]),
    )
    assert torch.equal(result[0].class_id, torch.tensor([1, 0, 0]))
    assert torch.all(result[0].confidence[:-1] >= result[0].confidence[1:])
