import numpy as np
import pytest

from inference.models.florence2 import Florence2


@pytest.mark.slow
def test_florence2_caption(
    example_image: np.ndarray,
) -> None:
    # model = Florence2("florence-pretrains/1")
    # since transformers 0.53.3 need newer version of florence2 weights
    model = Florence2("florence-pretrains/3")
    response = model.infer(example_image, prompt="<CAPTION>")[0].response
    assert "<CAPTION>" in response
    assert "a close up of a dog looking over a fence" in response["<CAPTION>"]
