"""Model results retain the image frame independently of the encoded mask grid."""

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
@pytest.mark.parametrize("mask_format", ["dense", "rle"])
@pytest.mark.parametrize("offset", [(0, 0), (60, 40)])
def test_model_result_retains_image_dimensions(family, mask_format, offset) -> None:
    module = import_module(f"inference_models.models.{family}.common")
    metadata = PreProcessingMetadata(
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        original_size=ImageDimensions(height=200, width=300),
        size_after_pre_processing=ImageDimensions(height=100, width=150),
        inference_size=ImageDimensions(height=80, width=120),
        scale_width=0.8,
        scale_height=0.8,
        static_crop_offset=StaticCropOffset(
            offset_x=offset[0], offset_y=offset[1], crop_width=150, crop_height=100
        ),
    )
    if family == "rfdetr":
        post_process = (
            module.post_process_instance_segmentation_results
            if mask_format == "dense"
            else module.post_process_instance_segmentation_results_to_rle_masks
        )
        results = post_process(
            bboxes=torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
            logits=torch.tensor([[[10.0]]]),
            masks=torch.ones((1, 1, 20, 30)),
            pre_processing_meta=[metadata],
            threshold=0.5,
            num_classes=1,
            classes_re_mapping=None,
            masks_resolution_factor=0.0,
        )
    else:
        kwargs = dict(
            nms_results=[torch.tensor([[0, 0, 120, 80, 0.9, 0, 1.0]])],
            pre_processing_meta=[metadata],
            masks_resolution_factor=0.0,
        )
        if family == "yolact":
            kwargs["all_proto_data"] = torch.ones((1, 20, 30, 1))
        else:
            kwargs["protos"] = torch.ones((1, 1, 20, 30))
            if family == "yolov8":
                kwargs.update(
                    masks_smoothing_enabled=False, masks_binarization_threshold=0.0
                )
            if family == "yolo26":
                kwargs["filtered_results"] = kwargs.pop("nms_results")
        results = getattr(module, f"prepare_{mask_format}_masks")(**kwargs)

    result = results[0]
    assert result.image_size == (200, 300)
    assert result.mask_frame_size == (200, 300)
    converted = result.to_supervision()
    assert converted.mask.shape == (1, 200, 300)
    assert converted.mask[
        0, offset[1] : offset[1] + 100, offset[0] : offset[0] + 150
    ].all()
    assert converted.mask.sum() == 100 * 150


@pytest.mark.parametrize("mask_format", ["dense", "rle"])
def test_coreml_dispatcher_forwards_resolution_factor(monkeypatch, mask_format) -> None:
    from unittest.mock import MagicMock

    from inference_models.models.rfdetr import (
        rfdetr_instance_segmentation_coreml as coreml,
    )

    model = object.__new__(coreml.RFDetrForInstanceSegmentationCoreML)
    model._class_names = ["car"]
    model._classes_re_mapping = None
    model.recommended_parameters = None
    post_process = MagicMock(return_value=[])
    target = "post_process_instance_segmentation_results"
    if mask_format == "rle":
        target += "_to_rle_masks"
    monkeypatch.setattr(coreml, target, post_process)

    model.post_process(
        model_results=(torch.empty(0), torch.empty(0), torch.empty(0)),
        pre_processing_meta=[],
        mask_format=mask_format,
        masks_resolution_factor=0.25,
    )

    assert post_process.call_args.kwargs["masks_resolution_factor"] == 0.25
