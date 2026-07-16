"""Qwen-Image-Edit workflow block (v1).

Takes an input image and a text editing instruction and returns an edited image.
Runs locally on GPU only — no remote/hosted execution path.

Model: Qwen/Qwen-Image-Edit (HuggingFace)
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
    BOOLEAN_KIND,
    FLOAT_KIND,
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
Edit an image using a text instruction with **Qwen-Image-Edit**, Alibaba's
diffusion-based image editing model.

Provide a source image and describe the change you want to make (e.g. *"change
the sky to a sunset"*, *"add a red hat"*, *"remove the background"*). The block
returns the edited image so it can be passed to downstream blocks or saved as an
output.

#### ⚠️ Requirements

* Requires a **local GPU** — this block cannot run on CPU or hosted inference.
* Weights are loaded from `local_weights_path` if provided, otherwise fetched
  from the Roboflow model registry using `model_id`.

#### Lightning LoRA (fast / low-VRAM)

Enable `use_lightning_lora` to fuse the lightx2v **Qwen-Image-Lightning**
step-distillation LoRA into the pipeline. The model then runs in ~4 diffusion
steps with guidance disabled — dramatically faster and feasible on consumer
GPUs. When enabled with no `local_weights_path`, the base model and LoRA are
pulled directly from HuggingFace (no Roboflow registry / API key needed). On
GPUs with limited VRAM set `INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD=sequential`.

#### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `prompt` | — | Required editing instruction |
| `local_weights_path` | None | Absolute path to locally downloaded weights directory |
| `use_lightning_lora` | False | Fuse the 4-step Qwen-Image-Lightning LoRA |
| `num_inference_steps` | auto | Auto = 4 with LoRA, 28 otherwise |
| `guidance_scale` | auto | Auto = 1.0 with LoRA, 5.0 otherwise |
| `seed` | None | Set for reproducible outputs |
"""

DEFAULT_MODEL_ID = "qwen-image-edit/1"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Qwen-Image-Edit",
            "version": "v1",
            "short_description": "Edit an image with a text instruction using Qwen-Image-Edit.",
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
                "needsGPU": True,
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

    use_lightning_lora: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        default=False,
        description=(
            "Fuse the lightx2v Qwen-Image-Lightning step-distillation LoRA into the "
            "pipeline. This lets the model run in ~4 diffusion steps (guidance "
            "disabled), making it dramatically faster and feasible on consumer GPUs. "
            "When enabled and no weights path is given, the base model and LoRA are "
            "pulled directly from HuggingFace (no Roboflow registry / API key needed)."
        ),
        examples=[True, "$inputs.use_lightning_lora"],
    )

    num_inference_steps: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=None,
        description=(
            "Number of diffusion denoising steps. More steps improve quality at the "
            "cost of speed. Leave unset to auto-select (4 with the Lightning LoRA, "
            "28 otherwise)."
        ),
        examples=[4, 28, 50],
    )

    guidance_scale: Optional[Union[Selector(kind=[FLOAT_KIND]), float]] = (
        Field(
            default=None,
            description=(
                "Classifier-free guidance scale. Higher values make the output adhere "
                "more strongly to the prompt. Leave unset to auto-select (1.0 with the "
                "Lightning LoRA, 5.0 otherwise)."
            ),
            examples=[1.0, 5.0, 7.5],
        )
    )

    seed: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=None,
        description="Optional RNG seed for reproducible outputs. Leave unset for random results.",
        examples=[42, "$inputs.seed"],
    )

    scale_megapixels: Optional[Union[Selector(kind=[FLOAT_KIND]), float]] = Field(
        default=None,
        description=(
            "Downscale inputs larger than this many megapixels before inference "
            "(never upscales). Diffusion VRAM/latency scales with pixel count, so a "
            "small cap is what keeps the model on a consumer GPU. Leave unset to "
            "auto-select (~0.35 MP with the Lightning LoRA, full size otherwise)."
        ),
        examples=[0.35, 1.0],
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
        use_lightning_lora: bool,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        seed: Optional[int],
        scale_megapixels: Optional[float],
    ) -> BlockResult:
        model = self._get_model(
            model_id=model_id,
            local_weights_path=local_weights_path,
            use_lightning_lora=use_lightning_lora,
        )
        results = []
        for image in images:
            edited_pil = model.edit(
                image=image.numpy_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                scale_megapixels=scale_megapixels,
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

    def _get_model(
        self,
        model_id: str,
        local_weights_path: Optional[str],
        use_lightning_lora: bool,
    ):
        # Use the load path as cache key when provided so switching paths forces
        # a reload, while the same path reuses the cached instance. The Lightning
        # toggle is part of the key so flipping it loads a distinct instance.
        base_key = local_weights_path if local_weights_path else model_id
        cache_key = (base_key, bool(use_lightning_lora))

        if cache_key not in QwenImageEditBlockV1._model_cache:
            # Validate the cheap precondition before importing the (heavy) backend
            # so a bad path fails fast with a clear error regardless of whether the
            # GPU model stack is importable.
            if local_weights_path and not os.path.isdir(local_weights_path):
                raise ValueError(
                    f"local_weights_path '{local_weights_path}' does not exist or is not a directory."
                )

            from inference_models.models.qwen_image_edit.qwen_image_edit_hf import (
                MODEL_ID,
                QwenImageEditHF,
            )

            if local_weights_path:
                QwenImageEditBlockV1._model_cache[cache_key] = (
                    QwenImageEditHF.from_pretrained(
                        model_name_or_path=local_weights_path,
                        local_files_only=True,
                        use_lightning_lora=use_lightning_lora,
                    )
                )
            elif use_lightning_lora:
                # Dev / offline-friendly path: pull the base model and LoRA
                # straight from HuggingFace, bypassing the Roboflow registry.
                QwenImageEditBlockV1._model_cache[cache_key] = (
                    QwenImageEditHF.from_pretrained(
                        model_name_or_path=MODEL_ID,
                        local_files_only=False,
                        use_lightning_lora=True,
                    )
                )
            else:
                from inference_models import AutoModel

                QwenImageEditBlockV1._model_cache[cache_key] = (
                    AutoModel.from_pretrained(
                        model_id_or_path=model_id,
                        api_key=self._api_key,
                        use_lightning_lora=False,
                    )
                )
        return QwenImageEditBlockV1._model_cache[cache_key]
