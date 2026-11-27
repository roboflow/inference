import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import clip
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.clip.clip_onnx import ClipOnnx
from inference_models.models.clip.clip_pytorch import ClipTorch
from inference_models.models.clip.preprocessing import MEAN, STD


def weightless_clip_onnx(image_size: int) -> ClipOnnx:
    return ClipOnnx(
        visual_onnx_session=MagicMock(),
        textual_onnx_session=MagicMock(),
        image_size=image_size,
        visual_input_name="image",
        textual_input_name="text",
        device=torch.device("cpu"),
        max_batch_size=8,
    )


def weightless_clip_torch(image_size: int) -> ClipTorch:
    return ClipTorch(
        model=SimpleNamespace(visual=SimpleNamespace(input_resolution=image_size)),
        tokenizer=clip.tokenize,
        device=torch.device("cpu"),
        max_batch_size=8,
    )


@pytest.mark.parametrize("build", [weightless_clip_onnx, weightless_clip_torch])
@pytest.mark.parametrize("output_type", [None, "feature_vector"])
def test_clip_describes_raw_image_feature_vectors(build, output_type) -> None:
    # given
    model = build(224)

    # when
    info = model.get_embedding_info(output_type)

    # then
    assert info == {
        "feature_definition": "clip-image-embeds@v1",
        "output_type": "feature_vector",
        "normalization": "none",
    }


@pytest.mark.parametrize("build", [weightless_clip_onnx, weightless_clip_torch])
def test_clip_has_no_logits(build) -> None:
    # given
    model = build(224)

    # when
    with pytest.raises(ModelInputError) as error:
        model.get_embedding_info("logits")

    # then
    assert "`logits`" in str(error.value)


@pytest.mark.parametrize("build", [weightless_clip_onnx, weightless_clip_torch])
@pytest.mark.parametrize("image_size", [224, 336])
def test_clip_embedding_preprocessing_is_the_visual_tower_input(
    build, image_size
) -> None:
    # given
    model = build(image_size)

    # when
    preprocessing = model.embedding_preprocessing

    # then
    assert preprocessing == {
        "image_size": image_size,
        "resize": "shortest_edge",
        "resample": "bicubic",
        "center_crop": True,
        "rescale_factor": 1 / 255,
        "image_mean": list(MEAN),
        "image_std": list(STD),
    }
    assert json.loads(json.dumps(preprocessing)) == preprocessing
