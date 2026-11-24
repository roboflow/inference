import inspect
from threading import Lock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
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


@pytest.mark.parametrize("classes", ["bus", ["bus", "red bus"]])
def test_forward_validates_and_encodes_prompts_under_the_model_lock(classes):
    model = OWLv2HF.__new__(OWLv2HF)
    model._lock = Lock()
    model._device = "cpu"
    model._model = MagicMock()
    model._model.config.text_config.max_position_embeddings = 3
    prompts = ["bus"] if isinstance(classes, str) else classes
    input_ids = torch.ones((len(prompts), 3), dtype=torch.long)
    pixels = torch.zeros(1)

    def tokenize(queries):
        assert model._lock.locked()
        assert queries == prompts
        return {"input_ids": [[0] * (len(query.split()) + 2) for query in queries]}

    def encode(*, text, return_tensors):
        assert model._lock.locked()
        assert text == [prompts] and return_tensors == "pt"
        return {"input_ids": input_ids}

    model._processor = MagicMock(side_effect=encode)
    model._processor.tokenizer.side_effect = tokenize
    if isinstance(classes, str):
        model.forward(pixels, classes)
        model._model.assert_called_once_with(input_ids=input_ids, pixel_values=pixels)
    else:
        with pytest.raises(
            ModelInputError, match=r"Shorten these queries: \['red bus'\]"
        ):
            model.forward(pixels, classes)
        model._processor.assert_not_called()
        model._model.assert_not_called()
    model._processor.tokenizer.assert_called_once_with(prompts)
