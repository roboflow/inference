import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_numpy(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_numpy_custom_image_size(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_numpy(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_torch(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_dynamic_bs_package_batch_torch(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
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
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_numpy(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_numpy_custom_image_size(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_numpy(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_torch(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_torch(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
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
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_bs_package_batch_torch_list(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForMultiLabelClassificationOnnx,
    )

    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [flowers_image_torch, flowers_image_torch],
        confidence=0.5,
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.0066, 0.0315, 0.9680]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([2], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_numpy(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_numpy_custom_image_size(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_numpy(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_torch(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_torch(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([bike_image_torch, bike_image_torch], dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_dynamic_bs_package_batch_torch_list(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_numpy(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_numpy_custom_image_size(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100), confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_numpy(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_torch(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_torch(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([bike_image_torch, bike_image_torch], dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_class_onnx_static_bs_package_batch_torch_list(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.vit.vit_classification_onnx import (
        VITForClassificationOnnx,
    )

    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch], confidence=0.5)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int64).cpu(),
    )
