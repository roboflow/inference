"""ConvNeXt classifiers trained on the Roboflow platform load with the ResNet classes.

Single-label packages tell Honda Civic from Toyota Altis and Toyota Yaris; multi-label
packages tag cat and dog.
"""

import numpy as np
import pytest
import torch

from inference_models import AutoModel


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_dynamic_bs_package_numpy(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_dynamic_bs_package_numpy_custom_image_size(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_numpy, image_size=(160, 160))

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_dynamic_bs_package_batch_numpy(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
    toyota_yaris_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([honda_civic_image_numpy, toyota_yaris_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor(
            [[1.8091e-07, 9.9999e-01, 1.0181e-05], [1.1687e-03, 3.2750e-05, 9.9880e-01]]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_dynamic_bs_package_torch(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_torch)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_dynamic_bs_package_batch_torch_list(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_torch: torch.Tensor,
    toyota_yaris_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([honda_civic_image_torch, toyota_yaris_image_torch])

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor(
            [[1.8091e-07, 9.9999e-01, 1.0181e-05], [1.1687e-03, 3.2750e-05, 9.9880e-01]]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_static_bs_package_numpy(
    convnext_single_label_onnx_static_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_static_bs_package_batch_numpy(
    convnext_single_label_onnx_static_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
    toyota_yaris_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([honda_civic_image_numpy, toyota_yaris_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor(
            [[1.8091e-07, 9.9999e-01, 1.0181e-05], [1.1687e-03, 3.2750e-05, 9.9880e-01]]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_static_bs_package_torch(
    convnext_single_label_onnx_static_bs_package: str,
    honda_civic_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_torch)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_single_label_onnx_static_bs_package_batch_torch_list(
    convnext_single_label_onnx_static_bs_package: str,
    honda_civic_image_torch: torch.Tensor,
    toyota_yaris_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = ResNetForClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_single_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([honda_civic_image_torch, toyota_yaris_image_torch])

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor(
            [[1.8091e-07, 9.9999e-01, 1.0181e-05], [1.1687e-03, 3.2750e-05, 9.9880e-01]]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_numpy(
    convnext_multi_label_onnx_dynamic_bs_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(dog_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_numpy(
    convnext_multi_label_onnx_dynamic_bs_package: str,
    dog_image_numpy: np.ndarray,
    cat_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_numpy, cat_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([3.0000e-04, 9.9910e-01, 7.6000e-03]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_torch_list(
    convnext_multi_label_onnx_dynamic_bs_package: str,
    dog_image_torch: torch.Tensor,
    cat_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_torch, cat_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([3.0000e-04, 9.9910e-01, 7.6000e-03]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_numpy(
    convnext_multi_label_onnx_static_bs_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(dog_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_numpy(
    convnext_multi_label_onnx_static_bs_package: str,
    dog_image_numpy: np.ndarray,
    cat_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_numpy, cat_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([3.0000e-04, 9.9910e-01, 7.6000e-03]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_torch_list(
    convnext_multi_label_onnx_static_bs_package: str,
    dog_image_torch: torch.Tensor,
    cat_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = ResNetForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=convnext_multi_label_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([dog_image_torch, cat_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([3.0000e-04, 9.9910e-01, 7.6000e-03]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_auto_model_loads_single_label_package_with_resnet_class(
    convnext_single_label_onnx_dynamic_bs_package: str,
    honda_civic_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )

    model = AutoModel.from_pretrained(
        model_id_or_path=convnext_single_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(honda_civic_image_numpy)

    # then
    assert isinstance(model, ResNetForClassificationOnnx)
    assert model.class_names == ["toyota_altis", "honda_civic", "toyota_yaris"]
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[1.8091e-07, 9.9999e-01, 1.0181e-05]]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_auto_model_loads_multi_label_package_with_resnet_class(
    convnext_multi_label_onnx_dynamic_bs_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForMultiLabelClassificationOnnx,
    )

    model = AutoModel.from_pretrained(
        model_id_or_path=convnext_multi_label_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(dog_image_numpy, confidence=0.5)

    # then
    assert isinstance(model, ResNetForMultiLabelClassificationOnnx)
    assert model.class_names == ["Unlabeled", "cat", "dog"]
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([1.7816e-04, 4.0206e-01, 9.9412e-01]).cpu(),
        atol=0.01,
    )
