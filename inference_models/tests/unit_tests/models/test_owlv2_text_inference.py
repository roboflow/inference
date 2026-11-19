import inspect
from typing import List, Optional

import numpy as np
import pytest
import torch

from inference_models.models.owlv2.owlv2_hf import OWLv2HF


def build_model_shell() -> OWLv2HF:
    """OWLv2HF with no weights loaded; the stages under test are replaced."""
    return OWLv2HF(
        model=None,
        processor=None,
        device=torch.device("cpu"),
        owlv2_class_embeddings_cache=None,
        owlv2_images_embeddings_cache=None,
        allow_url_input=True,
        allow_non_https_url=False,
        allow_url_without_fqdn=False,
        whitelisted_domains=None,
        blacklisted_domains=None,
        allow_local_storage_access_for_reference_images=False,
    )


@pytest.mark.parametrize(
    "input_color_format, first_pixel_for_the_processor",
    [(None, [255, 0, 0]), ("rgb", [255, 0, 0]), ("bgr", [0, 0, 255])],
)
def test_text_inference_converts_declared_bgr_input_to_rgb(
    monkeypatch,
    input_color_format: Optional[str],
    first_pixel_for_the_processor: List[int],
) -> None:
    # given
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 255
    model = build_model_shell()
    received = {}

    def pre_process(images, **kwargs):
        received["images"] = images
        return torch.zeros(1), []

    monkeypatch.setattr(model, "pre_process", pre_process)
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "post_process", lambda *args, **kwargs: [])

    # when
    model.infer(images=image, classes=["bus"], input_color_format=input_color_format)

    # then
    assert received["images"][0, 0].tolist() == first_pixel_for_the_processor


def test_pre_process_does_not_name_the_color_format() -> None:
    # The model manager sends `input_color_format` to every method of a model
    # whose `pre_process` names it. The reference-example methods do not
    # accept it, so naming it there would break them.
    assert "input_color_format" not in inspect.signature(OWLv2HF.pre_process).parameters
