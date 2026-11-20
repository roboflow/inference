import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference_models.models.owlv2.owlv2_hf import OWLv2HF


@pytest.mark.parametrize(
    "input_color_format, first_pixel_for_the_processor",
    [(None, [255, 0, 0]), ("rgb", [255, 0, 0]), ("bgr", [0, 0, 255])],
)
def test_text_inference_converts_declared_bgr_input_to_rgb(
    input_color_format, first_pixel_for_the_processor
) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 255
    model = OWLv2HF.__new__(OWLv2HF)
    model.pre_process = MagicMock(return_value=(None, []))
    model.forward = model.post_process = MagicMock()

    model.infer(images=image, classes=["bus"], input_color_format=input_color_format)

    (images,) = model.pre_process.call_args.args
    assert images[0, 0].tolist() == first_pixel_for_the_processor


def test_pre_process_does_not_name_the_color_format() -> None:
    # The model manager sends `input_color_format` to every method of a model
    # whose `pre_process` names it, and the reference-example methods reject it.
    assert "input_color_format" not in inspect.signature(OWLv2HF.pre_process).parameters
