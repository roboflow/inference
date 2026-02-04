import base64
from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import Sam3_3D_Objects_Response
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference_sdk import InferenceHTTPClient
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Generate 3D meshes and Gaussian splatting from 2D images with mask prompts.

Accepts masks as: sv.Detections (from SAM2 etc), polygon lists, binary masks, or RLE dicts.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM3D",
            "version": "v1",
            "short_description": "Generate 3D meshes and Gaussian splatting from 2D images with mask prompts.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "beta": True,
            "search_keywords": ["SAM3_3D", "3D", "mesh", "gaussian splatting"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-cube",
                "blockPriority": 9.0,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/segment_anything3_3d_objects@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    mask_input: Selector(
        kind=[LIST_OF_VALUES_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(
        description="Mask input - either instance segmentation predictions (e.g., from SAM2) or a flat list of polygon coordinates in COCO format [x1, y1, x2, y2, x3, y3, ...]",
        examples=["$steps.sam2.predictions", "$steps.detections.mask_polygon"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "mask_input"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="mesh_glb",
                kind=[STRING_KIND],
                description="Scene mesh in GLB format (base64 encoded)",
            ),
            OutputDefinition(
                name="gaussian_ply",
                kind=[STRING_KIND],
                description="Combined Gaussian splatting in PLY format (base64 encoded)",
            ),
            OutputDefinition(
                name="objects",
                kind=[LIST_OF_VALUES_KIND],
                description="List of individual objects, each with mesh_glb, gaussian_ply, and metadata (rotation, translation, scale)",
            ),
            OutputDefinition(
                name="inference_time",
                kind=[FLOAT_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class SegmentAnything3_3D_ObjectsBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        mask_input: Batch[Union[sv.Detections, List[float]]],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                mask_input=mask_input,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                mask_input=mask_input,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        mask_input: Batch[Union[sv.Detections, List[float]]],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        results = []
        model_id = "sam3-3d-objects"

        for single_image, single_mask_input in zip(images, mask_input):
            converted_mask = extract_masks_from_input(single_mask_input)

            # Convert numpy arrays to lists for JSON serialization
            if isinstance(converted_mask, list):
                serializable_mask = []
                for mask in converted_mask:
                    if isinstance(mask, np.ndarray):
                        serializable_mask.append(mask.tolist())
                    else:
                        serializable_mask.append(mask)
                converted_mask = serializable_mask

            result = client.sam3_3d_infer(
                inference_input=single_image.base64_image,
                mask_input=converted_mask,
                model_id=model_id,
            )

            # Result already comes formatted from the endpoint
            results.append({
                "mesh_glb": result.get("mesh_glb"),
                "gaussian_ply": result.get("gaussian_ply"),
                "objects": result.get("objects", []),
                "inference_time": result.get("time", 0.0),
            })

        return results

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        mask_input: Batch[Union[sv.Detections, List[float]]],
    ) -> BlockResult:
        results = []
        model_id = "sam3-3d-objects"

        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)

        for single_image, single_mask_input in zip(images, mask_input):
            converted_mask = extract_masks_from_input(single_mask_input)

            inference_request = Sam3_3D_Objects_InferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                mask_input=converted_mask,
                api_key=self._api_key,
                model_id=model_id,
            )

            response: Sam3_3D_Objects_Response = (
                self._model_manager.infer_from_request_sync(model_id, inference_request)
            )

            results.append(_format_response(response))

        return results


def extract_masks_from_input(mask_input: Any) -> Any:
    """Extract binary masks from sv.Detections, pass through other formats."""
    if isinstance(mask_input, sv.Detections):
        if len(mask_input) == 0:
            raise ValueError("sv.Detections contains no detections.")
        if mask_input.mask is not None and len(mask_input.mask) > 0:
            return list(mask_input.mask)
        raise ValueError("sv.Detections has no mask data.")
    return mask_input


def _format_response(response: Sam3_3D_Objects_Response) -> dict:
    """Format response with base64 encoded outputs."""

    def encode(data):
        return base64.b64encode(data).decode("utf-8") if data else None

    objects_list = [
        {
            "mesh_glb": encode(obj.mesh_glb),
            "gaussian_ply": encode(obj.gaussian_ply),
            "metadata": {
                "rotation": obj.metadata.rotation,
                "translation": obj.metadata.translation,
                "scale": obj.metadata.scale,
            },
        }
        for obj in response.objects
    ]

    return {
        "mesh_glb": encode(response.mesh_glb),
        "gaussian_ply": encode(response.gaussian_ply),
        "objects": objects_list,
        "inference_time": response.time,
    }
