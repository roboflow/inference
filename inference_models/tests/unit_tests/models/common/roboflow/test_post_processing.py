import torch
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow import post_processing
from inference_models.models.common.roboflow.model_packages import StaticCropOffset


def _legacy_resize_and_binarize_masks(
    masks: torch.Tensor,
    target_height: int,
    target_width: int,
    binarization_threshold: float,
) -> torch.Tensor:
    return (
        functional.resize(
            masks,
            [target_height, target_width],
            interpolation=functional.InterpolationMode.BILINEAR,
        )
        .gt_(binarization_threshold)
        .to(dtype=torch.bool)
    )


def test_align_instance_segmentation_results_matches_legacy_resize_when_chunked(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(post_processing, "MASK_ALIGNMENT_CHUNK_PIXEL_LIMIT", 9)
    masks = torch.linspace(-1.0, 1.0, steps=5 * 3 * 4).reshape(5, 3, 4)
    image_bboxes = torch.tensor(
        [
            [0.0, 0.0, 8.0, 6.0, 0.9, 1.0],
            [1.0, 1.0, 7.0, 5.0, 0.8, 2.0],
            [2.0, 1.0, 6.0, 5.0, 0.7, 3.0],
            [0.0, 2.0, 8.0, 4.0, 0.6, 4.0],
            [3.0, 0.0, 5.0, 6.0, 0.5, 5.0],
        ]
    )
    expected_masks = _legacy_resize_and_binarize_masks(
        masks=masks.clone(),
        target_height=6,
        target_width=8,
        binarization_threshold=0.2,
    )

    # when
    aligned_boxes, aligned_masks = post_processing.align_instance_segmentation_results(
        image_bboxes=image_bboxes.clone(),
        masks=masks.clone(),
        padding=(0, 0, 0, 0),
        scale_height=1.0,
        scale_width=1.0,
        original_size=ImageDimensions(height=6, width=8),
        size_after_pre_processing=ImageDimensions(height=6, width=8),
        inference_size=ImageDimensions(height=6, width=8),
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=0,
            crop_height=0,
        ),
        binarization_threshold=0.2,
    )

    # then
    assert torch.equal(aligned_masks, expected_masks)
    assert torch.allclose(aligned_boxes, image_bboxes)


def test_align_instance_segmentation_results_writes_static_crop_to_canvas_when_chunked(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(post_processing, "MASK_ALIGNMENT_CHUNK_PIXEL_LIMIT", 10)
    masks = torch.linspace(-0.5, 0.8, steps=4 * 4 * 5).reshape(4, 4, 5)
    image_bboxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 8.0, 0.9, 1.0],
            [1.0, 1.0, 9.0, 7.0, 0.8, 2.0],
            [2.0, 1.0, 8.0, 7.0, 0.7, 3.0],
            [0.0, 2.0, 10.0, 6.0, 0.6, 4.0],
        ]
    )
    expected_masks = torch.zeros((4, 13, 14), dtype=torch.bool)
    expected_masks[:, 3:11, 2:12] = _legacy_resize_and_binarize_masks(
        masks=masks.clone(),
        target_height=8,
        target_width=10,
        binarization_threshold=0.3,
    )
    expected_boxes = image_bboxes.clone()
    expected_boxes[:, :4] += torch.tensor([2.0, 3.0, 2.0, 3.0])

    # when
    aligned_boxes, aligned_masks = post_processing.align_instance_segmentation_results(
        image_bboxes=image_bboxes.clone(),
        masks=masks.clone(),
        padding=(0, 0, 0, 0),
        scale_height=1.0,
        scale_width=1.0,
        original_size=ImageDimensions(height=13, width=14),
        size_after_pre_processing=ImageDimensions(height=8, width=10),
        inference_size=ImageDimensions(height=8, width=10),
        static_crop_offset=StaticCropOffset(
            offset_x=2,
            offset_y=3,
            crop_width=10,
            crop_height=8,
        ),
        binarization_threshold=0.3,
    )

    # then
    assert torch.equal(aligned_masks, expected_masks)
    assert torch.allclose(aligned_boxes, expected_boxes)


def test_iter_aligned_instance_segmentation_results_matches_dense_alignment(
    monkeypatch,
) -> None:
    # given
    monkeypatch.setattr(post_processing, "MASK_ALIGNMENT_CHUNK_PIXEL_LIMIT", 12)
    masks = torch.linspace(-0.3, 0.9, steps=3 * 4 * 6).reshape(3, 4, 6)
    image_bboxes = torch.tensor(
        [
            [0.0, 0.0, 12.0, 8.0, 0.9, 1.0],
            [2.0, 1.0, 11.0, 7.0, 0.8, 2.0],
            [1.0, 2.0, 10.0, 6.0, 0.7, 3.0],
        ]
    )
    (
        expected_boxes,
        expected_masks,
    ) = post_processing.align_instance_segmentation_results(
        image_bboxes=image_bboxes.clone(),
        masks=masks.clone(),
        padding=(0, 0, 0, 0),
        scale_height=1.0,
        scale_width=1.0,
        original_size=ImageDimensions(height=8, width=12),
        size_after_pre_processing=ImageDimensions(height=8, width=12),
        inference_size=ImageDimensions(height=8, width=12),
        static_crop_offset=StaticCropOffset(
            offset_x=0,
            offset_y=0,
            crop_width=0,
            crop_height=0,
        ),
        binarization_threshold=0.1,
    )

    # when
    aligned_results = list(
        post_processing.iter_aligned_instance_segmentation_results(
            image_bboxes=image_bboxes.clone(),
            masks=masks.clone(),
            padding=(0, 0, 0, 0),
            scale_height=1.0,
            scale_width=1.0,
            original_size=ImageDimensions(height=8, width=12),
            size_after_pre_processing=ImageDimensions(height=8, width=12),
            inference_size=ImageDimensions(height=8, width=12),
            static_crop_offset=StaticCropOffset(
                offset_x=0,
                offset_y=0,
                crop_width=0,
                crop_height=0,
            ),
            binarization_threshold=0.1,
        )
    )

    # then
    aligned_boxes = torch.stack([box for box, _ in aligned_results])
    aligned_masks = torch.stack([mask for _, mask in aligned_results])
    assert torch.allclose(aligned_boxes, expected_boxes)
    assert torch.equal(aligned_masks, expected_masks)
