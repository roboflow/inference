import numpy as np
import torch
from inference_exp.models.vit.vit_classification_huggingface import (
    VITForClassificationHF,
    VITForMultiLabelClassificationHF,
)
from inference_exp.models.vit.vit_classification_onnx import (
    VITForClassificationOnnx,
    VITForMultiLabelClassificationOnnx,
)


def test_multi_label_hf_package_numpy(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(flowers_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_hf_package_numpy_custom_image_size(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_hf_package_batch_numpy(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_hf_package_torch(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(flowers_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_hf_package_batch_torch(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(torch.stack([flowers_image_torch, flowers_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_hf_package_batch_torch_list(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model([flowers_image_torch, flowers_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_numpy(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_numpy_custom_image_size(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_batch_numpy(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_torch(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_batch_torch(
    flowers_multi_label_vit_onnx_dynamic_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([flowers_image_torch, flowers_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_dynamic_bs_package_batch_torch_list(
    flowers_multi_label_vit_hf_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationHF.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model([flowers_image_torch, flowers_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_numpy(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_numpy_custom_image_size(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_batch_numpy(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_numpy, flowers_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_torch(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(flowers_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_batch_torch(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([flowers_image_torch, flowers_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


def test_multi_label_onnx_static_bs_package_batch_torch_list(
    flowers_multi_label_vit_onnx_static_bs_package: str,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=flowers_multi_label_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([flowers_image_torch, flowers_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor([0.0066, 0.0315, 0.9680]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_ids,
        torch.tensor([2], dtype=torch.int64),
    )


#####


def test_multi_class_hf_package_numpy(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(bike_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_hf_package_numpy_custom_image_size(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_hf_package_batch_numpy(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_hf_package_torch(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_hf_package_batch_torch(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model(torch.stack([bike_image_torch, bike_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_hf_package_batch_torch_list(
    vehicles_multi_class_vit_hf_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationHF.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_hf_package,
        device=torch.device("cpu"),
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_numpy(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_numpy_custom_image_size(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_batch_numpy(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_torch(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_batch_torch(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([bike_image_torch, bike_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_dynamic_bs_package_batch_torch_list(
    vehicles_multi_class_vit_onnx_dynamic_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_dynamic_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_numpy(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_numpy_custom_image_size(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_batch_numpy(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_torch(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_batch_torch(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model(torch.stack([bike_image_torch, bike_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )


def test_multi_class_onnx_static_bs_package_batch_torch_list(
    vehicles_multi_class_vit_onnx_static_bs_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    model = VITForClassificationOnnx.from_pretrained(
        model_name_or_path=vehicles_multi_class_vit_onnx_static_bs_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch])

    # then
    assert torch.allclose(
        predictions.confidence,
        torch.tensor([[0.9974, 0.0013, 0.0012], [0.9974, 0.0013, 0.0012]]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions.class_id,
        torch.tensor([0, 0], dtype=torch.int64),
    )
