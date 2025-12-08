from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import Sam3_3D_Objects_Response
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
)
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
Run Segment Anything 3_3D Objects model to generate 3D meshes and Gaussian splatting from 2D images.

** Dedicated inference server required (GPU strongly recommended) **

This block takes an image and mask input(s) and generates:
- Combined 3D scene mesh in GLB format
- Combined Gaussian splatting in PLY format
- Individual objects list, each with its own mesh, gaussian, and transformation metadata (rotation, translation, scale)

The mask input can be either:
- Instance segmentation predictions from another model (e.g., SAM2) - all detections will be processed
- A flat list of polygon coordinates in COCO format: [x1, y1, x2, y2, x3, y3, ...] (single mask)
- A list of flat polygon lists for multiple masks: [[x1, y1, ...], [x1, y1, ...], ...]

All detected objects/masks will be processed and returned in the objects list.
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
            raise NotImplementedError(
                "Remote execution is not supported for Segment Anything 3_3D. Run a local or dedicated inference server to use this block (GPU strongly recommended)."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        mask_input: Batch[Union[sv.Detections, List[float]]],
    ) -> BlockResult:
        results = []

        for single_image, single_mask_input in zip(images, mask_input):
            # Convert mask input to list of flat COCO format polygons
            flat_mask_input = convert_mask_input_to_flat_polygons(single_mask_input)

            # Create inference request
            model_id = "sam3-3d-objects"
            inference_request = Sam3_3D_Objects_InferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                mask_input=flat_mask_input,
                api_key=self._api_key,
                model_id=model_id,
            )

            # Load model
            self._model_manager.add_model(
                model_id=model_id,
                api_key=self._api_key,
            )

            # Run inference
            response: Sam3_3D_Objects_Response = self._model_manager.infer_from_request_sync(
                model_id, inference_request
            )

            # Convert binary data to base64 strings for workflow output
            import base64

            mesh_glb_b64 = None
            if response.mesh_glb is not None:
                mesh_glb_b64 = base64.b64encode(response.mesh_glb).decode('utf-8')

            gaussian_ply_b64 = None
            if response.gaussian_ply is not None:
                gaussian_ply_b64 = base64.b64encode(response.gaussian_ply).decode('utf-8')

            # Convert individual objects
            objects_list = []
            for obj in response.objects:
                obj_mesh_b64 = None
                if obj.mesh_glb is not None:
                    obj_mesh_b64 = base64.b64encode(obj.mesh_glb).decode('utf-8')

                obj_gaussian_b64 = None
                if obj.gaussian_ply is not None:
                    obj_gaussian_b64 = base64.b64encode(obj.gaussian_ply).decode('utf-8')

                objects_list.append({
                    "mesh_glb": obj_mesh_b64,
                    "gaussian_ply": obj_gaussian_b64,
                    "metadata": {
                        "rotation": obj.metadata.rotation,
                        "translation": obj.metadata.translation,
                        "scale": obj.metadata.scale,
                    },
                })

            result = {
                "mesh_glb": mesh_glb_b64,
                "gaussian_ply": gaussian_ply_b64,
                "objects": objects_list,
                "inference_time": response.time,
            }
            results.append(result)

        return results


def convert_mask_input_to_flat_polygons(
    mask_input: Union[sv.Detections, List]
) -> List[List[float]]:
    """
    Convert mask input to a list of flat COCO polygon formats.

    Args:
        mask_input: Either sv.Detections from instance segmentation,
                    a flat list of floats (single mask),
                    or a list of flat lists (multiple masks)

    Returns:
        List of flat polygon coordinates, each as [x1, y1, x2, y2, x3, y3, ...]

    Raises:
        ValueError: If the input format is invalid or contains no detections
    """
    # If it's a list, check if it's a single mask or multiple masks
    if isinstance(mask_input, list):
        if len(mask_input) == 0:
            raise ValueError("Empty mask input provided.")

        # Check if it's a flat list of numbers (single mask)
        if isinstance(mask_input[0], (int, float)):
            return [mask_input]

        # Check if it's a list of flat lists (multiple masks)
        if isinstance(mask_input[0], list):
            # Verify each is a flat list of numbers
            if len(mask_input[0]) > 0 and isinstance(mask_input[0][0], (int, float)):
                return mask_input

        # Otherwise return as-is (assume it's already in correct format)
        return mask_input

    # If it's sv.Detections, extract polygons from all detections
    if isinstance(mask_input, sv.Detections):
        if len(mask_input) == 0:
            raise ValueError("sv.Detections contains no detections. Cannot extract mask polygon.")

        detection_data = mask_input.data
        all_polygons = []

        # Try to get polygons from the data dictionary first
        if POLYGON_KEY_IN_SV_DETECTIONS in detection_data:
            polygons = detection_data[POLYGON_KEY_IN_SV_DETECTIONS]

            for polygon in polygons:
                flat_polygon = _convert_single_polygon_to_flat(polygon)
                if flat_polygon:
                    all_polygons.append(flat_polygon)

            if all_polygons:
                return all_polygons

        # Try to get masks from the mask attribute and convert to polygons
        if mask_input.mask is not None and len(mask_input.mask) > 0:
            for binary_mask in mask_input.mask:
                flat_polygon = _convert_binary_mask_to_flat_polygon(binary_mask)
                if flat_polygon:
                    all_polygons.append(flat_polygon)

            if all_polygons:
                return all_polygons

        raise ValueError(
            f"Could not extract polygon from sv.Detections. "
            f"No polygon key found and mask attribute is empty. "
            f"Available data keys: {list(detection_data.keys())}"
        )

    raise TypeError(
        f"mask_input must be either sv.Detections or List, got {type(mask_input)}"
    )


def _convert_single_polygon_to_flat(polygon) -> Optional[List[float]]:
    """Convert a single polygon to flat COCO format."""
    if isinstance(polygon, np.ndarray):
        # If it's a numpy array of shape (N, 2), flatten it
        if polygon.ndim == 2 and polygon.shape[1] == 2:
            return polygon.flatten().tolist()
        # If it's already flat, convert to list
        elif polygon.ndim == 1:
            return polygon.tolist()
    elif isinstance(polygon, list):
        # If it's a list of coordinate pairs, flatten it
        if len(polygon) > 0 and isinstance(polygon[0], (list, tuple, np.ndarray)):
            flat_polygon = []
            for point in polygon:
                flat_polygon.extend([float(point[0]), float(point[1])])
            return flat_polygon
        # If it's already flat
        else:
            return [float(x) for x in polygon]
    return None


def _convert_binary_mask_to_flat_polygon(binary_mask: np.ndarray) -> Optional[List[float]]:
    """Convert a binary mask to flat COCO polygon format."""
    # Ensure mask is uint8 for findContours
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    elif binary_mask.max() <= 1:
        binary_mask = binary_mask * 255

    # Find contours using OpenCV
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Get the largest contour (in case there are multiple)
    largest_contour = max(contours, key=cv2.contourArea)

    # Convert contour to flat COCO format [x1, y1, x2, y2, ...]
    # OpenCV contours are shape (N, 1, 2), we need to flatten to [x1, y1, x2, y2, ...]
    flat_polygon = []
    for point in largest_contour:
        flat_polygon.extend([float(point[0][0]), float(point[0][1])])

    return flat_polygon
