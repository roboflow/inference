"""Embedding blocks on a real local CLIP or Perception Encoder package (opt-in).

Set ``V2_CLIP_PACKAGE`` and/or ``V2_PERCEPTION_ENCODER_PACKAGE`` to a local
``inference_models`` package directory. Nothing is downloaded or installed: a
test skips with its reason when its variable is unset, the directory is
missing, or the package's backend dependency is not installed (for example
the ``clip`` module of a torch CLIP package). Runs on CPU.

Image: ``workflows/tests/assets/dogs.jpg``, the existing V1 workflow test asset.
"""

import os
from pathlib import Path
from typing import Any, Dict

import cv2
import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelRequest,
    auto_model_loader,
)
from roboflow_workflows.v2.blocks.models.clip_comparison import ClipComparisonBlock
from roboflow_workflows.v2.blocks.models.embeddings import (
    ClipEmbeddingBlock,
    PerceptionEncoderEmbeddingBlock,
)
from roboflow_workflows.v2.blocks.transforms.similarity import CosineSimilarityBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow

from inference_models.errors import MissingDependencyError
from tests.v2.support.paths import WORKFLOWS_ROOT

DOGS = WORKFLOWS_ROOT / "tests" / "assets" / "dogs.jpg"
CATALOGUE = Catalogue(
    [
        ClipEmbeddingBlock,
        PerceptionEncoderEmbeddingBlock,
        ClipComparisonBlock,
        CosineSimilarityBlock,
    ],
    kinds=NATIVE_KINDS,
)
CAPTIONS = ("a photo of dogs", "a photo of a red sports car")


def _loaded_cache(variable: str) -> tuple:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is unset; point it at a local package directory")
    package = Path(value)
    if not package.is_dir():
        pytest.skip(f"{variable}={package} is not a directory")

    models = ModelCache(
        auto_model_loader(
            device=torch.device("cpu"),
            onnx_execution_providers=["CPUExecutionProvider"],
            allow_untrusted_packages=True,
        )
    )
    try:
        models.get(ModelRequest(str(package)))
    except (MissingDependencyError, ModuleNotFoundError) as error:
        pytest.skip(f"backend dependency of {package} is not installed: {error}")

    return models, str(package)


def _dogs() -> ImageData:
    pixels = cv2.imread(str(DOGS))[:, :, ::-1].copy()
    image = ImageData.from_numpy_rgb(pixels, image_id="dogs")

    return image


def _caption_workflow(block_type: str, model_id: str) -> Dict[str, Any]:
    steps = [
        {
            "type": block_type,
            "name": "image",
            "data": "$inputs.image",
            "model_id": model_id,
        }
    ]
    outputs = [
        {"type": "JsonField", "name": "embedding", "selector": "$steps.image.embedding"}
    ]
    for n, caption in enumerate(CAPTIONS):
        steps.append(
            {
                "type": block_type,
                "name": f"text_{n}",
                "data": caption,
                "model_id": model_id,
            }
        )
        steps.append(
            {
                "type": "v2/cosine_similarity@v1",
                "name": f"similarity_{n}",
                "embedding_1": "$steps.image.embedding",
                "embedding_2": f"$steps.text_{n}.embedding",
            }
        )
        outputs.append(
            {
                "type": "JsonField",
                "name": f"similarity_{n}",
                "selector": f"$steps.similarity_{n}.similarity",
            }
        )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": steps,
        "outputs": outputs,
    }

    return definition


@pytest.mark.parametrize(
    "variable, block_type",
    [
        ("V2_CLIP_PACKAGE", "v2/clip@v1"),
        ("V2_PERCEPTION_ENCODER_PACKAGE", "v2/perception_encoder@v1"),
    ],
    ids=["clip", "perception_encoder"],
)
def test_real_model_ranks_the_matching_caption_higher(
    variable: str, block_type: str
) -> None:
    models, model_id = _loaded_cache(variable)
    plan = compile_workflow(
        _caption_workflow(block_type, model_id), catalogue=CATALOGUE
    )

    (row,) = plan.create_session({"models": models}).run({"image": [_dogs()]}).rows()

    assert row["embedding"].dim() == 1 and row["embedding"].is_floating_point()
    assert row["similarity_0"] > row["similarity_1"]


def test_real_clip_comparison_picks_the_matching_caption() -> None:
    models, model_id = _loaded_cache("V2_CLIP_PACKAGE")
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {
                "type": "v2/clip_comparison@v1",
                "name": "compare",
                "images": "$inputs.image",
                "classes": list(CAPTIONS),
                "model_id": model_id,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "label",
                "selector": "$steps.compare.most_similar_class",
            }
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    (row,) = plan.create_session({"models": models}).run({"image": [_dogs()]}).rows()

    assert row["label"] == CAPTIONS[0]
