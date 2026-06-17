"""Qwen-Image-Edit workflow block (v1).

Takes an input image and a text editing instruction and returns an edited image.
Runs locally on GPU only — no remote/hosted execution path.

Model: Qwen/Qwen-Image-Edit-2511 (HuggingFace)
Architecture key: qwen-image-edit
Task: image-editing
"""

import os
import uuid
from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Edit an image using a text instruction with **Qwen-Image-Edit-2511**, Alibaba's
diffusion-based image editing model.

Provide a source image and describe the change you want to make (e.g. *"change
the sky to a sunset"*, *"add a red hat"*, *"remove the background"*). The block
returns the edited image so it can be passed to downstream blocks or saved as an
output.

#### ⚠️ Requirements

* Requires a **local GPU** — this block cannot run on CPU or hosted inference.
* Weights are loaded from `local_weights_path` if provided, otherwise fetched
  from the Roboflow model registry using `model_id`.

#### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `prompt` | — | Required editing instruction |
| `local_weights_path` | None | Absolute path to locally downloaded weights directory |
| `num_inference_steps` | 28 | More steps → higher quality, slower |
| `guidance_scale` | 5.0 | Higher → stronger prompt adherence |
| `strength` | 0.85 | 0 = no change, 1 = ignore source image |
| `seed` | None | Set for reproducible outputs |
"""

DEFAULT_MODEL_ID = "qwen-image-edit/1"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen-Image-Edit",
            "version": "v1",
            "short_description": "Edit an image with a text instruction using Qwen-Image-Edit-2511.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Qwen",
                "image edit",
                "image editing",
                "diffusion",
                "Alibaba",
                "generative",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-paintbrush",
                "blockPriority": 6,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/qwen_image_edit@v1"]

    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Text instruction describing the desired edit (e.g. 'make the sky orange at sunset').",
        examples=["remove the background", "$inputs.edit_prompt"],
        json_schema_extra={"multiline": True},
    )

    model_id: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=DEFAULT_MODEL_ID,
        description="Roboflow model-registry id for the Qwen-Image-Edit weights. Ignored when local_weights_path is set.",
        examples=[DEFAULT_MODEL_ID],
    )

    local_weights_path: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description=(
            "Absolute path to a locally downloaded Qwen-Image-Edit weights directory. "
            "When set, skips the Roboflow registry and loads directly from disk. "
            "Useful for development before weights are registered."
        ),
        examples=["/tmp/qwen-image-edit-weights", "$inputs.weights_path"],
    )

    num_inference_steps: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=28,
        description="Number of diffusion denoising steps. More steps improve quality at the cost of speed.",
        examples=[28, 50],
    )

    guidance_scale: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=5.0,
        description="Classifier-free guidance scale. Higher values make the output adhere more strongly to the prompt.",
        examples=[5.0, 7.5],
    )

    strength: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.85,
        description="Edit strength — 0.0 leaves the source image unchanged, 1.0 ignores it entirely.",
        examples=[0.85, 0.6],
    )

    seed: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=None,
        description="Optional RNG seed for reproducible outputs. Leave unset for random results.",
        examples=[42, "$inputs.seed"],
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=True, reason=None)

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="image",
                kind=[IMAGE_KIND],
                description="The edited output image.",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [DEFAULT_MODEL_ID]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Qwen-Image-Edit requires a GPU. This block cannot run on CPU-only infrastructure.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Qwen-Image-Edit is not available on Roboflow Hosted Inference. Run a local GPU inference server.",
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]


class QwenImageEditBlockV1(WorkflowBlock):
    """Workflow block that wraps QwenImageEditHF.

    Model instances are cached by their load path so weights are only loaded
    once per process regardless of how many workflow steps use this block.
    """

    _model_cache: Dict[str, object] = {}

    def __init__(self, api_key: Optional[str]):
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        model_id: str,
        local_weights_path: Optional[str],
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: Optional[int],
    ) -> BlockResult:
        model = self._get_model(model_id=model_id, local_weights_path=local_weights_path)
        results = []
        for image in images:
            edited_pil = model.edit(
                image=image.numpy_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
            )
            edited_np = np.array(edited_pil)[:, :, ::-1]  # RGB → BGR
            parent_metadata = ImageParentMetadata(parent_id=str(uuid.uuid4()))
            results.append(
                {
                    "image": WorkflowImageData(
                        parent_metadata=parent_metadata,
                        numpy_image=edited_np,
                    )
                }
            )
        return results

    def _get_model(self, model_id: str, local_weights_path: Optional[str]):
        # Use the local path as cache key when provided so switching paths
        # forces a reload, while the same path reuses the cached instance.
        cache_key = local_weights_path if local_weights_path else model_id

        if cache_key not in QwenImageEditBlockV1._model_cache:
            if local_weights_path:
                if not os.path.isdir(local_weights_path):
                    raise ValueError(
                        f"local_weights_path '{local_weights_path}' does not exist or is not a directory."
                    )
                from inference_models.models.qwen_image_edit.qwen_image_edit_hf import QwenImageEditHF
                QwenImageEditBlockV1._model_cache[cache_key] = QwenImageEditHF.from_pretrained(
                    model_name_or_path=local_weights_path,
                    local_files_only=True,
                )
            else:
                from inference_models import AutoModel
                QwenImageEditBlockV1._model_cache[cache_key] = AutoModel.from_pretrained(
                    model_id_or_path=model_id,
                    api_key=self._api_key,
                )
        return QwenImageEditBlockV1._model_cache[cache_key]
