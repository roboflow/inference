from importlib import import_module

import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop

MODEL_CASES = [
    pytest.param(
        "yolov5",
        "onnx",
        "YOLOv5ForInstanceSegmentationOnnx",
        "asl_yolov5_onnx_seg_static_bs_letterbox",
        "asl_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="yolov5-onnx",
    ),
    pytest.param(
        "yolov7",
        "onnx",
        "YOLOv7ForInstanceSegmentationOnnx",
        "asl_yolov7_onnx_seg_static_bs_letterbox",
        "asl_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="yolov7-onnx",
    ),
    pytest.param(
        "yolact",
        "onnx",
        "YOLOACTForInstanceSegmentationOnnx",
        "asl_yolact_onnx_seg_static_bs_letterbox",
        "asl_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="yolact-onnx",
    ),
    pytest.param(
        "yolov8",
        "onnx",
        "YOLOv8ForInstanceSegmentationOnnx",
        "asl_yolov8n_onnx_seg_static_bs_stretch",
        "asl_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="yolov8-onnx",
    ),
    pytest.param(
        "yolo26",
        "onnx",
        "YOLO26ForInstanceSegmentationOnnx",
        "yolo26n_seg_snakes_stretch_onnx_static_package",
        "snake_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="yolo26-onnx",
    ),
    pytest.param(
        "rfdetr",
        "onnx",
        "RFDetrForInstanceSegmentationOnnx",
        "snakes_rfdetr_seg_onnx_static_bs_stretch_package",
        "snake_image_numpy",
        marks=pytest.mark.onnx_extras,
        id="rfdetr-onnx",
    ),
    pytest.param(
        "yolov8",
        "torch_script",
        "YOLOv8ForInstanceSegmentationTorchScript",
        "asl_yolov8n_torchscript_seg_static_bs_stretch",
        "asl_image_numpy",
        marks=pytest.mark.torch_models,
        id="yolov8-torch",
    ),
    pytest.param(
        "yolo26",
        "torch_script",
        "YOLO26ForInstanceSegmentationTorchScript",
        "yolo26n_seg_snakes_stretch_torch_script_package",
        "snake_image_numpy",
        marks=pytest.mark.torch_models,
        id="yolo26-torch",
    ),
    pytest.param(
        "rfdetr",
        "pytorch",
        "RFDetrForInstanceSegmentationTorch",
        "snakes_rfdetr_seg_torch_stretch_package",
        "snake_image_numpy",
        marks=pytest.mark.torch_models,
        id="rfdetr-torch",
    ),
    pytest.param(
        "yolov8",
        "trt",
        "YOLOv8ForInstanceSegmentationTRT",
        "yolov8_seg_asl_trt_package",
        "asl_image_numpy",
        marks=[pytest.mark.trt_extras, pytest.mark.gpu_only],
        id="yolov8-trt",
    ),
    pytest.param(
        "yolo26",
        "trt",
        "YOLO26ForInstanceSegmentationTRT",
        "yolo26_seg_asl_trt_package",
        "asl_image_numpy",
        marks=[pytest.mark.trt_extras, pytest.mark.gpu_only],
        id="yolo26-trt",
    ),
    pytest.param(
        "rfdetr",
        "trt",
        "RFDetrForInstanceSegmentationTRT",
        "rfdetr_seg_asl_trt_package",
        "asl_image_numpy",
        marks=[pytest.mark.trt_extras, pytest.mark.gpu_only],
        id="rfdetr-trt",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "family,backend,class_name,package_fixture,image_fixture", MODEL_CASES
)
def test_model_mask_resolution_and_crop_round_trip(
    request, monkeypatch, family, backend, class_name, package_fixture, image_fixture
) -> None:
    module = import_module(
        f"inference_models.models.{family}.{family}_instance_segmentation_{backend}"
    )
    load_kwargs = {}
    if backend == "onnx":
        load_kwargs["onnx_execution_providers"] = ["CPUExecutionProvider"]
        load_kwargs["device"] = torch.device("cpu")
    elif backend == "trt":
        load_kwargs["engine_host_code_allowed"] = True

    model = getattr(module, class_name).from_pretrained(
        model_name_or_path=request.getfixturevalue(package_fixture), **load_kwargs
    )
    image = request.getfixturevalue(image_fixture)
    image = np.pad(image, ((0, image.shape[0] % 2), (0, image.shape[1] % 2), (0, 0)))
    height, width = image.shape[:2]
    preprocessing = model._inference_config.image_pre_processing

    confidence = 0.6 if family == "yolov7" else 0.25
    accurate = None
    mask_sizes = {}
    for factor in (1.0, 0.5, 0.0):
        reference = None
        for crop in (None, "origin", "offset"):
            if crop is None:
                scene = image
                offset_x = offset_y = 0
                crop_config = None
            else:
                offset_y = 0 if crop == "origin" else height // 2
                offset_x = 0 if crop == "origin" else width // 2
                scene = np.zeros((2 * height, 2 * width, 3), dtype=image.dtype)
                scene[offset_y : offset_y + height, offset_x : offset_x + width] = image
                start = 0 if crop == "origin" else 25
                crop_config = StaticCrop(
                    enabled=True,
                    x_min=start,
                    y_min=start,
                    x_max=start + 50,
                    y_max=start + 50,
                )

            with monkeypatch.context() as patch:
                patch.setattr(preprocessing, "static_crop", crop_config)
                dense = model(
                    scene,
                    confidence=confidence,
                    mask_format="dense",
                    masks_resolution_factor=factor,
                )[0]
                rle = model(
                    scene,
                    confidence=confidence,
                    mask_format="rle",
                    masks_resolution_factor=factor,
                )[0]

            assert len(dense.xyxy) > 0, (family, backend, factor, crop)
            torch.testing.assert_close(dense.xyxy.cpu(), rle.xyxy.cpu())
            torch.testing.assert_close(dense.class_id.cpu(), rle.class_id.cpu())
            torch.testing.assert_close(dense.confidence.cpu(), rle.confidence.cpu())
            decoded = coco_rle_masks_to_torch_mask(rle.mask, device=torch.device("cpu"))
            np.testing.assert_array_equal(decoded.numpy(), dense.mask.cpu().numpy())
            assert dense.image_size == scene.shape[:2]
            assert dense.mask_frame_size == scene.shape[:2]
            if factor == 1.0:
                assert dense.mask_size == scene.shape[:2]
            else:
                assert dense.mask_size != scene.shape[:2]

            detections = dense.to_supervision()
            np.testing.assert_array_equal(detections.mask, rle.to_supervision().mask)
            assert detections.mask.shape == (len(dense.xyxy), *scene.shape[:2])
            assert detections.mask.any()
            sv.MaskAnnotator().annotate(scene.copy(), detections)
            if reference is None:
                reference = detections
                mask_sizes[factor] = dense.mask_size
                if accurate is None:
                    accurate = detections
                else:
                    np.testing.assert_allclose(detections.xyxy, accurate.xyxy, atol=1)
                    np.testing.assert_array_equal(
                        detections.class_id, accurate.class_id
                    )
                    intersection = np.logical_and(detections.mask, accurate.mask).sum(
                        axis=(1, 2)
                    )
                    union = np.logical_or(detections.mask, accurate.mask).sum(
                        axis=(1, 2)
                    )
                    assert np.all(
                        (union == 0) | (intersection / np.maximum(union, 1) > 0.8)
                    )
            else:
                expected = np.zeros_like(detections.mask)
                expected[
                    :, offset_y : offset_y + height, offset_x : offset_x + width
                ] = reference.mask
                # Resizing a canvas can round boundary pixels differently from
                # resizing the crop before placing it on that canvas.
                intersection = np.logical_and(detections.mask, expected).sum(
                    axis=(1, 2)
                )
                union = np.logical_or(detections.mask, expected).sum(axis=(1, 2))
                reduced_offset = crop == "offset" and factor < 1.0
                minimum_iou = 0.9 if reduced_offset else 0.99
                pixel_tolerance = (
                    np.ceil(
                        max(
                            scene.shape[0] / dense.mask_size[0],
                            scene.shape[1] / dense.mask_size[1],
                        )
                    )
                    if reduced_offset
                    else 1
                )
                assert np.all(
                    (union == 0) | (intersection / np.maximum(union, 1) > minimum_iou)
                ), (factor, crop)
                np.testing.assert_allclose(
                    sv.mask_to_xyxy(detections.mask),
                    sv.mask_to_xyxy(expected),
                    atol=pixel_tolerance,
                    err_msg=f"factor={factor}, crop={crop}",
                )
                np.testing.assert_allclose(
                    detections.xyxy,
                    reference.xyxy + [offset_x, offset_y, offset_x, offset_y],
                    atol=1,
                )

    np.testing.assert_allclose(
        mask_sizes[0.5],
        np.round((np.array(mask_sizes[0.0]) + np.array(mask_sizes[1.0])) / 2),
        atol=1,
    )
