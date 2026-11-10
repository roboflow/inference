"""The local VLM blocks on real cached model packages (opt-in).

Each case runs only when its environment variable names an existing local
``inference_models`` package directory; it never downloads. Without them
every case skips, which is NOT evidence that real inference works:

    QWEN25VL_PACKAGE   roboflow_core/qwen25vl@v1
    QWEN3VL_PACKAGE    roboflow_core/qwen3vl@v1
    QWEN3_5_PACKAGE    roboflow_core/qwen3_5vl@v2
    SMOLVLM2_PACKAGE   roboflow_core/smolvlm2@v1

``VLM_DEVICE`` selects the device (default ``cpu``). The check is coarse: a
real model answers every image with non-empty text through a compiled
workflow. The step passes a per-run prompt; the answer's content is not
checked, so this does not show that the model received that prompt. The fake
model tests show the prompt wiring.
"""

import os
from pathlib import Path
from typing import Optional

import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, auto_model_loader
from roboflow_workflows.v2.blocks.models._catalogue_vlm import BLOCKS
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow

PACKAGES = {
    "roboflow_core/qwen25vl@v1": "QWEN25VL_PACKAGE",
    "roboflow_core/qwen3vl@v1": "QWEN3VL_PACKAGE",
    "roboflow_core/qwen3_5vl@v2": "QWEN3_5_PACKAGE",
    "roboflow_core/smolvlm2@v1": "SMOLVLM2_PACKAGE",
}


def _package(variable: str) -> Optional[Path]:
    value = os.environ.get(variable)
    if not value or not Path(value).is_dir():
        return None

    return Path(value)


def _image() -> ImageData:
    # Left half red, right half blue: an answer about colour is checkable by eye.
    pixels = torch.zeros((3, 224, 224), dtype=torch.uint8)
    pixels[0, :, :112] = 255
    pixels[2, :, 112:] = 255
    image = ImageData.from_tensor(pixels, image_id="red-blue")

    return image


@pytest.mark.parametrize("block_type", PACKAGES)
def test_a_real_model_answers_every_image_with_text(block_type: str) -> None:
    package = _package(PACKAGES[block_type])
    if package is None:
        pytest.skip(f"{PACKAGES[block_type]} does not name a local model package")

    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {"type": "WorkflowParameter", "name": "prompt", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": block_type,
                "name": "vlm",
                "image": "$inputs.image",
                "model_id": str(package),
                "prompt": "$inputs.prompt",
                "max_new_tokens": 32,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "answer",
                "selector": "$steps.vlm.parsed_output",
            }
        ],
    }
    loader = auto_model_loader(
        device=torch.device(os.environ.get("VLM_DEVICE", "cpu")),
        allow_untrusted_packages=True,
    )
    session = compile_workflow(definition, catalogue=Catalogue(BLOCKS)).create_session(
        {"models": ModelCache(loader)}
    )

    rows = session.run(
        {"image": [_image(), _image()], "prompt": "Which colours do you see?"}
    ).rows()

    answers = [row["answer"] for row in rows]
    assert len(answers) == 2
    assert all(isinstance(answer, str) and answer.strip() for answer in answers)
