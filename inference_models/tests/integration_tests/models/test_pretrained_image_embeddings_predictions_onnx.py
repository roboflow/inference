"""Pretrained image-embedding packages against independent reference implementations.

The packages hold the exact files of their pinned onnx-community revisions (see the
conftest fixtures). Their vectors are compared with reference vectors computed in
float32 from the original weights, by transformers for DINOv2 and SigLIP2 and by timm
for DINOv3, and stored in `fixtures/pretrained_image_embeddings_references.json`.
`development/packaging/compute_pretrained_image_embedding_references.py` regenerates
them.

For DINOv2 and DINOv3 the reference vector is the CLS token after the final layer
norm (`pooler_output` in transformers, `forward_features(x)[:, 0]` in timm), so these
tests also check that the packages return the layer-normed CLS token.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

REFERENCES = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "pretrained_image_embeddings_references.json"
    ).read_text()
)["models"]
IMAGE_NAMES = ["dog", "bike", "flowers"]
PACKAGES = [
    pytest.param("dinov2/small", "dinov2_small_embedding_package", 384, id="dinov2"),
    pytest.param("dinov3/vits16", "dinov3_vits16_embedding_package", 384, id="dinov3"),
    pytest.param(
        "siglip2/base-patch16-224",
        "siglip2_base_patch16_224_embedding_package",
        768,
        id="siglip2",
    ),
]


@pytest.fixture(scope="function")
def reference_images(
    dog_image_numpy: np.ndarray,
    bike_image_numpy: np.ndarray,
    flowers_image_numpy: np.ndarray,
) -> List[np.ndarray]:
    return [dog_image_numpy, bike_image_numpy, flowers_image_numpy]


def _load_model(package_dir: str):
    from inference_models import AutoModel
    from inference_models.models.pretrained_image_embeddings.pretrained_image_embeddings_onnx import (
        PretrainedImageEmbeddingOnnx,
    )

    model = AutoModel.from_pretrained(
        model_id_or_path=package_dir,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert isinstance(model, PretrainedImageEmbeddingOnnx)
    return model


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize("model_id, package_fixture, dimension", PACKAGES)
def test_embeddings_match_reference_implementation(
    request: pytest.FixtureRequest,
    reference_images: List[np.ndarray],
    model_id: str,
    package_fixture: str,
    dimension: int,
) -> None:
    # given
    model = _load_model(request.getfixturevalue(package_fixture))
    expected = torch.tensor(
        [REFERENCES[model_id]["embeddings"][name] for name in IMAGE_NAMES]
    )

    # when
    embeddings = model.embed_images(reference_images)

    # then
    assert tuple(embeddings.shape) == (len(IMAGE_NAMES), dimension)
    assert embeddings.dtype == torch.float32
    similarity = torch.nn.functional.cosine_similarity(
        embeddings.cpu(), expected, dim=1
    )
    assert torch.all(similarity >= 0.99), dict(zip(IMAGE_NAMES, similarity.tolist()))


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize("model_id, package_fixture, dimension", PACKAGES)
def test_batch_embeddings_match_single_image_embeddings(
    request: pytest.FixtureRequest,
    reference_images: List[np.ndarray],
    model_id: str,
    package_fixture: str,
    dimension: int,
) -> None:
    # given
    model = _load_model(request.getfixturevalue(package_fixture))

    # when
    batch = model.embed_images(reference_images).cpu()
    single = torch.cat([model.embed_images(image).cpu() for image in reference_images])

    # then
    assert tuple(single.shape) == (len(IMAGE_NAMES), dimension)
    similarity = torch.nn.functional.cosine_similarity(batch, single, dim=1)
    assert torch.all(similarity >= 0.999), dict(zip(IMAGE_NAMES, similarity.tolist()))
