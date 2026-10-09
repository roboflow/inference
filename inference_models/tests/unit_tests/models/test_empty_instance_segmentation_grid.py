"""Empty model outputs must retain the same mask grid as populated outputs."""

from importlib import import_module

import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)


@pytest.mark.parametrize(
    "family", ["yolov5", "yolov7", "yolov8", "yolo26", "yolact", "rfdetr"]
)
@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "image_size,crop_size,offset,padding,inference_size,mask_size",
    [
        ((200, 300), (200, 300), (0, 0), (0, 0, 0, 0), 64, 16),
        ((200, 300), (100, 150), (0, 0), (0, 8, 0, 8), 64, 16),
        ((200, 300), (100, 150), (60, 40), (0, 8, 0, 8), 64, 16),
        ((245, 245), (162, 162), (83, 83), (0, 0, 0, 0), 162, 81),
        ((200, 300), (200, 300), (0, 0), (-4, -8, -4, -8), 64, 16),
        ((10, 12), (10, 12), (0, 0), (0, 0, 0, 0), 64, 16),
    ],
)
def test_empty_rle_retains_populated_grid(
    family, factor, image_size, crop_size, offset, padding, inference_size, mask_size
):
    """Compare actual family postprocessors across mask formats and counts."""
    height, width = image_size
    crop_height, crop_width = crop_size
    pad_left, pad_top, pad_right, pad_bottom = padding
    meta = PreProcessingMetadata(
        original_size=ImageDimensions(height=height, width=width),
        size_after_pre_processing=ImageDimensions(height=crop_height, width=crop_width),
        inference_size=ImageDimensions(height=inference_size, width=inference_size),
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        scale_width=(inference_size - pad_left - pad_right) / crop_width,
        scale_height=(inference_size - pad_top - pad_bottom) / crop_height,
        static_crop_offset=StaticCropOffset(
            offset_x=offset[0],
            offset_y=offset[1],
            crop_width=crop_width,
            crop_height=crop_height,
        ),
    )
    module = import_module(f"inference_models.models.{family}.common")
    outputs = {}
    for mask_format in ("dense", "rle"):
        for count in (0, 1):
            kwargs = dict(pre_processing_meta=[meta], masks_resolution_factor=factor)
            if family == "rfdetr":
                fn = (
                    module.post_process_instance_segmentation_results
                    if mask_format == "dense"
                    else module.post_process_instance_segmentation_results_to_rle_masks
                )
                kwargs.update(
                    bboxes=torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
                    logits=torch.ones((1, 1, 1)),
                    masks=torch.ones((1, 1, mask_size, mask_size)),
                    threshold=1.0 if count == 0 else 0.1,
                    num_classes=1,
                    classes_re_mapping=None,
                )
            else:
                fn = getattr(module, f"prepare_{mask_format}_masks")
                kwargs["filtered_results" if family == "yolo26" else "nms_results"] = [
                    torch.tensor([[8.0, 8.0, 48.0, 48.0, 0.9, 0.0, 1.0]])[:count]
                ]
                if family == "yolact":
                    kwargs["all_proto_data"] = torch.ones((1, mask_size, mask_size, 1))
                else:
                    kwargs["protos"] = torch.ones((1, 1, mask_size, mask_size))
                if family == "yolov8":
                    kwargs.update(
                        masks_smoothing_enabled=False, masks_binarization_threshold=0.0
                    )
            result = fn(**kwargs)[0]
            assert len(result) == count
            outputs[mask_format, count] = result

    expected_grid = outputs["dense", 1].mask_size
    for result in outputs.values():
        assert result.mask_size == expected_grid
        assert result.image_size == image_size
        assert result.mask_frame_size == image_size
    assert outputs["rle", 0].mask.mask_size == expected_grid
    assert outputs["rle", 0].mask.to_coco_rle_masks() == []
    assert outputs["rle", 0].to_supervision().mask.shape == (0, *image_size)
