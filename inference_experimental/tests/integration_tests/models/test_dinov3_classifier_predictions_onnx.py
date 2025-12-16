import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_numpy(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_piece_image_numpy)

    assert predictions.confidence.shape[-1] > 0
    assert predictions.class_id.shape[0] == 1


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_numpy_custom_image_size(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_piece_image_numpy, image_size=(100, 100))

    assert predictions.confidence.shape[-1] > 0
    assert predictions.class_id.shape[0] == 1


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_batch_numpy(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([chess_piece_image_numpy, chess_piece_image_numpy])

    assert predictions.class_id.shape[0] == 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_torch_tensor(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_piece_image_torch)

    assert predictions.confidence.shape[-1] > 0
    assert predictions.class_id.shape[0] == 1


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_batch_torch_tensor_stack(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(
        torch.stack([chess_piece_image_torch, chess_piece_image_torch], dim=0)
    )

    assert predictions.class_id.shape[0] == 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_classification_onnx_static_package_batch_torch_tensor_list(
    dinov3_classification_onnx_static_package: str,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForClassificationOnnx,
    )

    model = DinoV3ForClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_classification_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([chess_piece_image_torch, chess_piece_image_torch])

    assert predictions.class_id.shape[0] == 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_numpy(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_set_image_numpy)

    assert len(predictions) == 1
    assert hasattr(predictions[0], "class_ids")
    assert hasattr(predictions[0], "confidence")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_numpy_custom_image_size(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_set_image_numpy, image_size=(100, 100))

    assert len(predictions) == 1
    assert hasattr(predictions[0], "class_ids")
    assert hasattr(predictions[0], "confidence")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_batch_numpy(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_numpy: np.ndarray,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([chess_set_image_numpy, chess_set_image_numpy])

    assert len(predictions) == 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_torch_tensor(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(chess_set_image_torch)

    assert len(predictions) == 1
    assert hasattr(predictions[0], "class_ids")
    assert hasattr(predictions[0], "confidence")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_batch_torch_tensor_stack(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(
        torch.stack([chess_set_image_torch, chess_set_image_torch], dim=0)
    )

    assert len(predictions) == 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_multi_label_onnx_static_package_batch_torch_tensor_list(
    dinov3_multi_label_onnx_static_package: str,
    chess_set_image_torch: torch.Tensor,
) -> None:
    from inference_exp.models.dinov3.dinov3_classification_onnx import (
        DinoV3ForMultiLabelClassificationOnnx,
    )

    model = DinoV3ForMultiLabelClassificationOnnx.from_pretrained(
        model_name_or_path=dinov3_multi_label_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([chess_set_image_torch, chess_set_image_torch])

    assert len(predictions) == 2
