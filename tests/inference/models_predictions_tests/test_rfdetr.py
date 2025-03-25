import numpy as np

from inference import get_model


def test_rfdetr_base(example_image: np.ndarray) -> None:
    model = get_model("rfdetr-base")

    predictions = model.infer(example_image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"
