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
def test_single_label_trt_package_torch_multiple_predictions_in_row(
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

    for _ in range(8):
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
def test_multi_label_trt_package_torch_multiple_predictions_in_row(
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

    for _ in range(8):
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


@pytest.mark.slow
@pytest.mark.trt_extras
def test_multi_label_trt_per_class_confidence_blocks_specific_class(
    vit_multi_label_cls_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_multi_label_trt_package_numpy` above) has class 2
    confidence ~0.834 on dog_image (so it passes 0.5 baseline threshold).
    Setting a 0.99 per-class threshold on class 2 drops it from predictions."""
    from inference_models.models.vit.vit_classification_trt import (
        VITForMultiLabelClassificationTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = VITForMultiLabelClassificationTRT.from_pretrained(
        model_name_or_path=vit_multi_label_cls_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.5,
        per_class_confidence={class_names[2]: 0.99},
    )
    predictions = model(dog_image_numpy, confidence="best")
    assert 2 not in predictions[0].class_ids.cpu().tolist()
