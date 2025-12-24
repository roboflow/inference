import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_numpy_list(
    asl_yolact_onnx_seg_static_bs_letterbox: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_list(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_letterbox_torch_tensor(
    asl_yolact_onnx_seg_static_bs_letterbox: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_letterbox,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [66, 167, 188, 373], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [18], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.5068], atol=0.001)
    assert 14800 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_numpy(
    asl_yolact_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy, conf_thresh=0.8)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_numpy_list(
    asl_yolact_onnx_seg_static_bs_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy], conf_thresh=0.8)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch, conf_thresh=0.8)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch_list(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch], conf_thresh=0.8)

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_and_stretch_torch_tensor(
    asl_yolact_onnx_seg_static_bs_stretch: str, asl_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([asl_image_torch, asl_image_torch], dim=0), conf_thresh=0.8
    )

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[0].mask[0].cpu().numpy().sum() <= 16100
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [63, 170, 188, 377], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [21], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.9783], atol=0.001)
    assert 15000 <= predictions[1].mask[0].cpu().numpy().sum() <= 16100


####
@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_numpy(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_numpy_list(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.003)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert len(predictions) == 1
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch_list(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_seg_onnx_package_with_static_batch_size_static_crop_and_stretch_torch_tensor(
    asl_yolact_onnx_seg_static_bs_static_crop_stretch: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolact.yolact_instance_segmentation_onnx import (
        YOLOACTForInstanceSegmentationOnnx,
    )

    model = YOLOACTForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=asl_yolact_onnx_seg_static_bs_static_crop_stretch,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert len(predictions) == 2
    assert np.allclose(
        predictions[0].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[0].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[0].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[0].mask[0].cpu().numpy().sum() <= 7500
    assert np.allclose(
        predictions[1].xyxy[0].cpu().numpy(), [109, 169, 208, 311], atol=1
    )
    assert np.allclose(predictions[1].class_id[0].cpu().numpy(), [24], atol=1)
    assert np.allclose(predictions[1].confidence[0].cpu().numpy(), [0.2552], atol=0.006)
    assert 7000 <= predictions[1].mask[0].cpu().numpy().sum() <= 7500
