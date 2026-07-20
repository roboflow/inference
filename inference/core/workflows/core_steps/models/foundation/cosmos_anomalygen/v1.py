from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.draw.color import Color

from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
)
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Generate synthetic defect images with NVIDIA Cosmos AnomalyGen.

Given a clean (defect-free) image, a placement mask, and an anomaly type
(e.g. `wood+crack`), the model inpaints a realistic defect into the mask's
region. Fine-tuned per-defect checkpoints load through the same model id /
local package path as the base model. Useful for bootstrapping defect
detection datasets where real defect data is scarce.

The placement mask comes in as an instance segmentation prediction - draw it
upstream (e.g. a polygon zone converted to detections) or chain a
segmentation model.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cosmos AnomalyGen",
            "version": "v1",
            "short_description": "Inpaint realistic synthetic defects into clean images.",
            "long_description": LONG_DESCRIPTION,
            "license": "Other",
            "block_type": "model",
            "search_keywords": [
                "Cosmos",
                "AnomalyGen",
                "NVIDIA",
                "defect",
                "synthetic data",
                "inpainting",
                "anomaly",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-hammer-crash",
                "blockPriority": 5.9,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/cosmos_anomalygen@v1"]

    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The clean (defect-free) image to inpaint a defect into.",
        examples=["$inputs.image"],
    )
    segmentation_mask: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        name="Placement Mask",
        description="Segmentation prediction marking where the defect should appear.",
        examples=["$steps.model.predictions"],
    )
    anomaly_type: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The trained anomaly type to generate, as `<texture>+<defect_class>`.",
        examples=["wood+crack", "$inputs.anomaly_type"],
    )
    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="cosmos-anomalygen",
        description="The Cosmos AnomalyGen checkpoint to use (model id or local package path).",
        examples=["cosmos-anomalygen"],
    )
    guidance: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=7.0,
        description="Anomaly conditioning strength.",
        examples=[7.0],
    )
    num_steps: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=35,
        description="Number of denoising steps.",
        examples=[35],
    )
    seed: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=0,
        description="Random seed for reproducible generation.",
        examples=[0],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="image",
                kind=[IMAGE_KIND],
                description="The clean image with the synthetic defect inpainted.",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; the diffusion model needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "Cosmos AnomalyGen has no remote endpoint - the block loads "
                    "the model in-process and only supports local execution."
                ),
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
            ),
        ]

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return ["cosmos-anomalygen"]


class CosmosAnomalyGenBlockV1(WorkflowBlock):
    def __init__(
        self,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._model = None
        self._current_model_id: Optional[str] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        segmentation_mask: sv.Detections,
        anomaly_type: str,
        model_version: str,
        guidance: float,
        num_steps: int,
        seed: int,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "Cosmos AnomalyGen only supports local execution - there is no "
                "remote endpoint for this model."
            )
        model = self._resolve_model(model_id=model_version)
        numpy_image = image.numpy_image
        mask = rasterize_placement_mask(
            image=numpy_image, segmentation_mask=segmentation_mask
        )
        generated = model.generate(
            image=numpy_image,
            mask=mask,
            anomaly_type=anomaly_type,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            num_images=1,
        )
        return {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=generated[0],
            ),
        }

    def _resolve_model(self, model_id: str):
        if self._model is None or self._current_model_id != model_id:
            from inference_models import AutoModel

            extra_weights_provider_headers = get_extra_weights_provider_headers()
            self._model = AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self._api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
            )
            self._current_model_id = model_id
        return self._model


def rasterize_placement_mask(
    image: np.ndarray,
    segmentation_mask: sv.Detections,
) -> np.ndarray:
    black_image = np.zeros_like(image)
    mask_annotator = sv.MaskAnnotator(color=Color.WHITE, opacity=1.0)
    mask = mask_annotator.annotate(black_image, segmentation_mask)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
