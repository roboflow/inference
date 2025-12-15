import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.models.owlv2.entities import ReferenceBoundingBox, ReferenceExample
from inference_exp.models.owlv2.owlv2_hf import OWLv2HF
from inference_exp.models.roboflow_instant.roboflow_instant_hf import RoboflowInstantHF


@pytest.fixture(scope="module")
def owlv2_model(owlv2_package: str) -> OWLv2HF:
    return OWLv2HF.from_pretrained(
        owlv2_package,
        device=DEFAULT_DEVICE,
        owlv2_enforce_model_compilation=True,
    )


@pytest.fixture(scope="module")
def instant_model(rf_instant_model_coin_counting_package: str) -> RoboflowInstantHF:
    return RoboflowInstantHF.from_pretrained(
        rf_instant_model_coin_counting_package,
        device=DEFAULT_DEVICE,
        owlv2_enforce_model_compilation=True,
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_owlv2_predictions_for_open_vocabulary(
    owlv2_model: OWLv2HF,
    dog_image_numpy: np.ndarray,
) -> None:
    # when
    predictions = owlv2_model(dog_image_numpy, classes=["dog", "person"])

    # then
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        np.array([[68, 257, 635, 917], [9, 354, 623, 1268]]),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_owlv2_predictions_for_reference_dataset(
    owlv2_model: OWLv2HF,
    dog_image_numpy: np.ndarray,
) -> None:
    # when
    predictions = owlv2_model.infer_with_reference_examples(
        dog_image_numpy,
        reference_examples=[
            ReferenceExample(
                image=dog_image_numpy,
                boxes=[
                    ReferenceBoundingBox(x=283, y=330, w=567, h=660, cls="dog"),
                    ReferenceBoundingBox(x=307, y=457, w=614, h=914, cls="person"),
                ],
            )
        ],
    )

    # then
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        np.array([8, 4, 709, 594]),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_instant_model_predictions(
    instant_model: RoboflowInstantHF,
    coins_counting_image_torch: torch.Tensor,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # when
    predictions = instant_model(coins_counting_image_torch)

    # then
    assert np.allclose(
        predictions[0].xyxy.cpu().numpy(),
        np.array(
            [
                [2676, 800, 2865, 970],
                [1251, 2061, 1428, 2230],
                [1707, 2575, 1892, 2764],
                [1459, 2303, 1631, 2472],
                [927, 1845, 1099, 2004],
                [1742, 2294, 1922, 2470],
                [1505, 1884, 1728, 2093],
                [1091, 2354, 1266, 2526],
            ]
        ),
        atol=5,
    )
