import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_numpy(
    vit_single_label_cls_trt_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForClassificationTRT,
    )

    model = VITForClassificationTRT.from_pretrained(
        model_name_or_path=vit_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(bike_image_numpy)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.7300973534584045) < 2e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_numpy_batch(
    vit_single_label_cls_trt_package: str,
    bike_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForClassificationTRT,
    )

    model = VITForClassificationTRT.from_pretrained(
        model_name_or_path=vit_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([bike_image_numpy, bike_image_numpy])

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.7300973534584045) < 2e-2
    assert abs(predictions.confidence[1, 2].item() - 0.7300973534584045) < 2e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch(
    vit_single_label_cls_trt_package: str,
    bike_image_torch: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForClassificationTRT,
    )

    model = VITForClassificationTRT.from_pretrained(
        model_name_or_path=vit_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(bike_image_torch)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.7300973534584045) < 2e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch_list(
    vit_single_label_cls_trt_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForClassificationTRT,
    )

    model = VITForClassificationTRT.from_pretrained(
        model_name_or_path=vit_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([bike_image_torch, bike_image_torch])

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.7300973534584045) < 2e-2
    assert abs(predictions.confidence[1, 2].item() - 0.7300973534584045) < 2e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_single_label_trt_package_torch_batch(
    vit_single_label_cls_trt_package: str,
    bike_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForClassificationTRT,
    )

    model = VITForClassificationTRT.from_pretrained(
        model_name_or_path=vit_single_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([bike_image_torch, bike_image_torch]), dim=0)

    # then
    assert abs(predictions.confidence[0, 2].item() - 0.7300973534584045) < 2e-2
    assert abs(predictions.confidence[1, 2].item() - 0.7300973534584045) < 2e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_numpy(
    vit_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(dog_image_numpy)

    # then
    assert abs(predictions[0].confidence[2].item() - 0.833984375) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_numpy_batch(
    vit_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([dog_image_numpy, dog_image_numpy])

    # then
    assert abs(predictions[0].confidence[2].item() - 0.833984375) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.833984375) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch(
    vit_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(dog_image_torch)

    # then
    assert abs(predictions[0].confidence[2].item() - 0.833984375) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch_list(
    vit_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([dog_image_torch, dog_image_torch])

    # then
    assert abs(predictions[0].confidence[2].item() - 0.833984375) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.833984375) < 1e-3


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_package_torch_batch(
    vit_multi_label_cls_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert abs(predictions[0].confidence[2].item() - 0.833984375) < 1e-3
    assert abs(predictions[1].confidence[2].item() - 0.833984375) < 1e-3
