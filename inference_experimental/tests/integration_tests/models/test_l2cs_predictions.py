import numpy as np
import torch


def test_l2cs_predictions_for_numpy_image(
    l2cs_package: str, man_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.l2cs.l2cs_onnx import L2CSNetOnnx

    model = L2CSNetOnnx.from_pretrained(l2cs_package)

    # when
    results = model(man_image_numpy)

    # then
    assert np.allclose(results.yaw.cpu().numpy(), np.array([-0.1527]), atol=0.05)
    assert np.allclose(results.pitch.cpu().numpy(), np.array([-0.1138916]), atol=0.05)


def test_l2cs_predictions_for_numpy_images_list(
    l2cs_package: str, man_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.l2cs.l2cs_onnx import L2CSNetOnnx

    model = L2CSNetOnnx.from_pretrained(l2cs_package)

    # when
    results = model([man_image_numpy, man_image_numpy])

    # then
    assert np.allclose(
        results.yaw.cpu().numpy(), np.array([-0.1527, -0.1527]), atol=0.05
    )
    assert np.allclose(
        results.pitch.cpu().numpy(), np.array([-0.1138916, -0.1138916]), atol=0.05
    )


def test_l2cs_predictions_for_torch_image(
    l2cs_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.l2cs.l2cs_onnx import L2CSNetOnnx

    model = L2CSNetOnnx.from_pretrained(l2cs_package)

    # when
    results = model(man_image_torch)

    # then
    assert np.allclose(results.yaw.cpu().numpy(), np.array([-0.1527]), atol=0.05)
    assert np.allclose(results.pitch.cpu().numpy(), np.array([-0.1138916]), atol=0.05)


def test_l2cs_predictions_for_torch_batch(
    l2cs_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.l2cs.l2cs_onnx import L2CSNetOnnx

    model = L2CSNetOnnx.from_pretrained(l2cs_package)

    # when
    results = model(torch.stack([man_image_torch, man_image_torch], dim=0))

    # then
    assert np.allclose(
        results.yaw.cpu().numpy(), np.array([-0.1527, -0.1527]), atol=0.05
    )
    assert np.allclose(
        results.pitch.cpu().numpy(), np.array([-0.1138916, -0.1138916]), atol=0.05
    )


def test_l2cs_predictions_for_torch_listh(
    l2cs_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.l2cs.l2cs_onnx import L2CSNetOnnx

    model = L2CSNetOnnx.from_pretrained(l2cs_package)

    # when
    results = model([man_image_torch, man_image_torch])

    # then
    assert np.allclose(
        results.yaw.cpu().numpy(), np.array([-0.1527, -0.1527]), atol=0.05
    )
    assert np.allclose(
        results.pitch.cpu().numpy(), np.array([-0.1138916, -0.1138916]), atol=0.05
    )
