import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_numpy(
    resnet_single_label_cls_trt_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
    )

    # when
    predictions = model(bike_image_numpy)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_numpy_batch(
    resnet_single_label_cls_trt_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy])

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3
    assert abs(predictions.confidence[1, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch(
    resnet_single_label_cls_trt_package: str,
    bike_image_torch: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch_list(
    resnet_single_label_cls_trt_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch])

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3
    assert abs(predictions.confidence[1, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch_batch(
    resnet_single_label_cls_trt_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
    )

    # when
    predictions = model(torch.stack([bike_image_torch, bike_image_torch]), dim=0)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3
    assert abs(predictions.confidence[1, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_numpy(
    resnet_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
    )

    # when
    predictions = model(dog_image_numpy)

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_numpy_batch(
    resnet_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
    )

    # when
    predictions = model([dog_image_numpy, dog_image_numpy])

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch(
    resnet_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
    )

    # when
    predictions = model(dog_image_torch)

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch_list(
    resnet_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
    )

    # when
    predictions = model([dog_image_torch, dog_image_torch])

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch_batch(
    resnet_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
    )

    # when
    predictions = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.99951171875) < 1e-3
