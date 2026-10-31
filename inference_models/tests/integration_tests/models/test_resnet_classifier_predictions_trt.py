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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.9999516010284424) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch_multiple_predictions_in_row(
    resnet_single_label_cls_trt_package: str,
    bike_image_torch: np.ndarray,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForClassificationTRT,
    )

    model = ResNetForClassificationTRT.from_pretrained(
        model_name_or_path=resnet_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(dog_image_torch)

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch_multiple_predictions_in_row(
    resnet_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
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
        engine_host_code_allowed=True,
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
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert abs(predictions[0].confidence[2].item() - 0.99951171875) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.99951171875) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_per_class_confidence_blocks_specific_class(
    resnet_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_multi_label_trt_package_numpy` above) has class 2
    confidence ~0.9995 on dog_image. Setting a 1.01 per-class threshold on
    class 2 drops it from predictions."""
    from inference_models.models.resnet.resnet_classification_trt import (
        ResNetForMultiLabelClassificationTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = ResNetForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=resnet_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.5,
        per_class_confidence={class_names[2]: 1.01},
    )
    predictions = model(dog_image_numpy, confidence="best")
    assert 2 not in predictions[0].class_ids.cpu().tolist()
