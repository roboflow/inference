import numpy as np

from inference import get_model


def test_rfdetr_base(example_image: np.ndarray) -> None:
    model = get_model("rfdetr-base")

    predictions = model.infer(example_image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"


def test_rfdetr_base_with_background_class(
    melee_image: np.ndarray, rfdetr_base_model: str
) -> None:
    model = get_model(rfdetr_base_model)

    predictions = model.infer(melee_image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"
    classes = [p.class_name for p in predictions.predictions]
    assert not any("background" in class_name for class_name in classes)
    assert set([p.class_id for p in predictions.predictions]) == set([0, 1])
