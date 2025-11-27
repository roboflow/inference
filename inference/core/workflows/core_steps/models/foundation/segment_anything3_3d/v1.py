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

This block takes an image and a mask input and generates:
- 3D mesh in GLB format
- Gaussian splatting in PLY format
- Transformation metadata (rotation, translation, scale)

The mask input can be either:
- Instance segmentation predictions from another model (e.g., SAM2)
- A flat list of polygon coordinates in COCO format: [x1, y1, x2, y2, x3, y3, ...]

When using instance segmentation predictions with multiple detections, only the first detection will be used.
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
            ),
            OutputDefinition(
                name="gaussian_ply",
                kind=[STRING_KIND],
            ),
            OutputDefinition(
                name="rotation",
                kind=[LIST_OF_VALUES_KIND],
            ),
            OutputDefinition(
                name="translation",
                kind=[LIST_OF_VALUES_KIND],
            ),
            OutputDefinition(
                name="scale",
                kind=[LIST_OF_VALUES_KIND],
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
            # Convert mask input to flat COCO format if it's sv.Detections
            flat_mask_input = convert_mask_input_to_flat_polygon(single_mask_input)

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

            result = {
                "mesh_glb": mesh_glb_b64,
                "gaussian_ply": gaussian_ply_b64,
                "rotation": response.metadata.rotation,
                "translation": response.metadata.translation,
                "scale": response.metadata.scale,
                "inference_time": response.time,
            }
            results.append(result)

        return results


def convert_mask_input_to_flat_polygon(
    mask_input: Union[sv.Detections, List[float]]
) -> List[float]:
    """
    Convert mask input to flat COCO polygon format.

    Args:
        mask_input: Either sv.Detections from instance segmentation or a flat list of floats

    Returns:
        Flat list of polygon coordinates [x1, y1, x2, y2, x3, y3, ...]

    Raises:
        ValueError: If the input format is invalid or contains no detections
    """
    # If it's already a flat list, return as-is
    if isinstance(mask_input, list):
        return mask_input

    # If it's sv.Detections, extract the polygon from the first detection
    if isinstance(mask_input, sv.Detections):
        if len(mask_input) == 0:
            raise ValueError("sv.Detections contains no detections. Cannot extract mask polygon.")

        # Warn if multiple detections are present
        if len(mask_input) > 1:
            print(f"Can only do 1 mask at a time, processing first (received {len(mask_input)} masks)")

        # Get the first detection's polygon data
        first_detection_data = mask_input.data

        # Try to get polygon from the data dictionary first
        if POLYGON_KEY_IN_SV_DETECTIONS in first_detection_data:
            polygon = first_detection_data[POLYGON_KEY_IN_SV_DETECTIONS][0]  # Get first detection's polygon

            # Convert polygon to flat COCO format
            if isinstance(polygon, np.ndarray):
                # If it's a numpy array of shape (N, 2), flatten it
                if polygon.ndim == 2 and polygon.shape[1] == 2:
                    flat_polygon = polygon.flatten().tolist()
                    return flat_polygon
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

        # Try to get mask from the mask attribute and convert to polygon
        if mask_input.mask is not None and len(mask_input.mask) > 0:
            # Get the first detection's binary mask (shape: H, W)
            binary_mask = mask_input.mask[0]

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
                raise ValueError("Could not extract contours from binary mask.")

            # Get the largest contour (in case there are multiple)
            largest_contour = max(contours, key=cv2.contourArea)

            # Convert contour to flat COCO format [x1, y1, x2, y2, ...]
            # OpenCV contours are shape (N, 1, 2), we need to flatten to [x1, y1, x2, y2, ...]
            flat_polygon = []
            for point in largest_contour:
                flat_polygon.extend([float(point[0][0]), float(point[0][1])])

            return flat_polygon

        raise ValueError(
            f"Could not extract polygon from sv.Detections. "
            f"No polygon key found and mask attribute is empty. "
            f"Available data keys: {list(first_detection_data.keys())}"
        )

    raise TypeError(
        f"mask_input must be either sv.Detections or List[float], got {type(mask_input)}"
    )
