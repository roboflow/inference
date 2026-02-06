import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.owlv2.entities import (
    ReferenceBoundingBox,
    ReferenceExample,
)
from inference_models.models.owlv2.owlv2_hf import OWLv2HF
from inference_models.models.roboflow_instant.roboflow_instant_hf import (
    RoboflowInstantHF,
)


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
    predictions = owlv2_model(
        dog_image_numpy,
        classes=["dog", "person"],
        confidence=0.98,
        iou_threshold=0.3,
        class_agnostic_nms=False,
        max_detections=300,
    )

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
        confidence=0.99,
        iou_threshold=0.3,
        max_detections=300,
    )

    # then
    assert predictions[0].class_id.numel() == 1


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_instant_model_predictions(
    instant_model: RoboflowInstantHF,
    coins_counting_image_torch: torch.Tensor,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # when
    predictions = instant_model(
        coins_counting_image_torch,
        confidence=0.99,
        iou_threshold=0.3,
        class_agnostic_nms=False,
        max_detections=300,
    )

    # then
    assert np.allclose(
        predictions[0].class_id.cpu().numpy(),
        np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    )
