import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_numpy(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_numpy_custom_image_size(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([3.5349e-04, 9.9397e-01, 8.3731e-03]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_numpy(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_torch(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_torch(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([flowers_image_torch, flowers_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_torch_list(
    flowers_multi_label_resnet_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_torch, flowers_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_numpy(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_numpy_custom_image_size(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_numpy(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_torch(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_torch(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([flowers_image_torch, flowers_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_torch_list(
    flowers_multi_label_resnet_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_resnet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_torch, flowers_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([7.3851e-05, 2.3921e-01, 7.9376e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_numpy(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_numpy_custom_image_size(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0341, 0.6881, 0.2777]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_numpy(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_torch(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_torch(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([bike_image_torch, bike_image_torch], dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_torch_list(
    vehicles_multi_class_resenet_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_numpy(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_numpy_custom_image_size(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_numpy(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_torch(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_torch(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([bike_image_torch, bike_image_torch], dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_torch_list(
    vehicles_multi_class_resenet_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_resenet_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.0065, 0.2892, 0.7043], [0.0065, 0.2892, 0.7043]]).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([2, 2], dtype=torch.int64).cpu(),
    )
