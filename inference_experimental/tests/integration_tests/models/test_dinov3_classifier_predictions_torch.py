import numpy as np
import pytest
import torch
from torchvision.transforms.functional import resize


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_numpy(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model(chess_piece_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([0.3554, 0.6446]).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_numpy_no_detection(
    dinov3_classification_torch_static_package: str,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model(flowers_image_numpy)

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([0.5260, 0.4740]).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_numpy_custom_image_size(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model(chess_piece_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([0.3554, 0.6446]).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_batch_numpy(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_numpy: np.ndarray,
    flowers_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model([chess_piece_image_numpy, flowers_image_numpy])

    # then
    assert torch.allclose(
        predictions.confidence.cpu(),
        torch.tensor([[0.3554, 0.6446], [0.5260, 0.4740]]).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_torch_tensor(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model(chess_piece_image_torch)

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor([0.3553, 0.6447])
    else:
        expected_confidence = torch.tensor([0.3556, 0.6444])
    assert torch.allclose(
        predictions.confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_batch_torch_tensor_stack(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_torch: torch.Tensor,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )
    target_h, target_w = chess_piece_image_torch.shape[-2:]
    flowers_resized = resize(flowers_image_torch, [target_h, target_w])

    # when
    predictions = model(torch.stack([chess_piece_image_torch, flowers_resized], dim=0))

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor([[0.3553, 0.6447], [0.5261, 0.4739]])
    else:
        expected_confidence = torch.tensor([[0.3556, 0.6444], [0.5260, 0.4740]])
    assert torch.allclose(
        predictions.confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_classification_torch_static_package_batch_torch_tensor_list(
    dinov3_classification_torch_static_package: str,
    chess_piece_image_torch: torch.Tensor,
    flowers_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForClassificationTorch,
    )

    model = DinoV3ForClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_classification_torch_static_package,
    )

    # when
    predictions = model([chess_piece_image_torch, flowers_image_torch])

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor([[0.3553, 0.6447], [0.5261, 0.4739]])
    else:
        expected_confidence = torch.tensor([[0.3556, 0.6444], [0.5259, 0.4741]])
    assert torch.allclose(
        predictions.confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions.class_id.cpu(),
        torch.tensor([1, 0], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_numpy(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model(chess_set_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9897,
                0.8594,
                0.9919,
                0.9998,
                0.9996,
                1.0000,
                0.9999,
                0.9979,
                0.9978,
                0.9999,
                1.0000,
                0.9955,
            ]
        ).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_numpy_custom_image_size(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model(chess_set_image_numpy, image_size=(100, 100))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9897,
                0.8594,
                0.9919,
                0.9998,
                0.9996,
                1.0000,
                0.9999,
                0.9979,
                0.9978,
                0.9999,
                1.0000,
                0.9955,
            ]
        ).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_batch_numpy(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_numpy: np.ndarray,
    chess_piece_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model([chess_set_image_numpy, chess_piece_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9897,
                0.8594,
                0.9919,
                0.9998,
                0.9996,
                1.0000,
                0.9999,
                0.9979,
                0.9978,
                0.9999,
                1.0000,
                0.9955,
            ]
        ).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                -5.9605e-08,
                -5.9605e-08,
                6.2674e-05,
                -5.9605e-08,
                1.4901e-07,
                1.1027e-06,
                4.6372e-03,
                4.4006e-04,
                9.9946e-01,
                2.2224e-04,
                -5.9605e-08,
                1.1921e-07,
            ]
        ).cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([8], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_torch_tensor(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model(chess_set_image_torch)

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor(
            [
                0.9892,
                0.8798,
                0.9916,
                0.9998,
                0.9995,
                1.0000,
                0.9998,
                0.9981,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    else:
        expected_confidence = torch.tensor(
            [
                0.9883,
                0.8467,
                0.9904,
                0.9998,
                0.9995,
                1.0000,
                0.9999,
                0.9980,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_batch_torch_tensor_stack(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_torch: torch.Tensor,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model(
        torch.stack([chess_set_image_torch, chess_piece_image_torch], dim=0)
    )

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor(
            [
                0.9892,
                0.8798,
                0.9916,
                0.9998,
                0.9995,
                1.0000,
                0.9998,
                0.9981,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    else:
        expected_confidence = torch.tensor(
            [
                0.9883,
                0.8467,
                0.9904,
                0.9998,
                0.9995,
                1.0000,
                0.9999,
                0.9980,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )
    if torch.cuda.is_available():
        expected_confidence_1 = torch.tensor(
            [
                0.0000e+00,
                0.0000e+00,
                6.5923e-05,
                0.0000e+00,
                1.1921e-07,
                1.1921e-06,
                4.1614e-03,
                4.6206e-04,
                9.9946e-01,
                2.1595e-04,
                0.0000e+00,
                1.1921e-07,
            ]
        )
    else:
        expected_confidence_1 = torch.tensor(
            [
                -5.9605e-08,
                -5.9605e-08,
                5.3078e-05,
                -5.9605e-08,
                5.9605e-08,
                1.3411e-06,
                5.6942e-03,
                3.9130e-04,
                9.9953e-01,
                2.3118e-04,
                -5.9605e-08,
                1.7881e-07,
            ]
        )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        expected_confidence_1.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([8], dtype=torch.int64).cpu(),
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_multi_label_torch_static_package_batch_torch_tensor_list(
    dinov3_multi_label_torch_static_package: str,
    chess_set_image_torch: torch.Tensor,
    chess_piece_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.dinov3.dinov3_classification_torch import (
        DinoV3ForMultiLabelClassificationTorch,
    )

    model = DinoV3ForMultiLabelClassificationTorch.from_pretrained(
        model_name_or_path=dinov3_multi_label_torch_static_package,
    )

    # when
    predictions = model([chess_set_image_torch, chess_piece_image_torch])

    # then
    if torch.cuda.is_available():
        expected_confidence = torch.tensor(
            [
                0.9892,
                0.8798,
                0.9916,
                0.9998,
                0.9995,
                1.0000,
                0.9998,
                0.9981,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    else:
        expected_confidence = torch.tensor(
            [
                0.9883,
                0.8467,
                0.9904,
                0.9998,
                0.9995,
                1.0000,
                0.9999,
                0.9980,
                0.9973,
                0.9999,
                1.0000,
                0.9949,
            ]
        )
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        expected_confidence.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[0].class_ids.cpu(),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int64).cpu(),
    )
    if torch.cuda.is_available():
        expected_confidence_1 = torch.tensor(
            [
                0.0000e+00,
                0.0000e+00,
                6.5923e-05,
                0.0000e+00,
                1.1921e-07,
                1.1921e-06,
                4.1614e-03,
                4.6206e-04,
                9.9946e-01,
                2.1595e-04,
                0.0000e+00,
                1.1921e-07,
            ]
        )
    else:
        expected_confidence_1 = torch.tensor(
            [
                -5.9605e-08,
                -5.9605e-08,
                5.3078e-05,
                -5.9605e-08,
                5.9605e-08,
                1.3411e-06,
                5.6942e-03,
                3.9130e-04,
                9.9953e-01,
                2.3118e-04,
                -5.9605e-08,
                1.7881e-07,
            ]
        )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        expected_confidence_1.cpu(),
        atol=0.0001,
    )
    assert torch.allclose(
        predictions[1].class_ids.cpu(),
        torch.tensor([8], dtype=torch.int64).cpu(),
    )
