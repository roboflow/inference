from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import DepthEstimationRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    NUMPY_ARRAY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    # Standard model configuration for UI, schema, etc.
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Depth Estimation",
            "version": "v1",
            "short_description": "Run Depth Estimation on an image.",
            "long_description": (
                """
Run depth estimation on images to generate depth maps representing 3D spatial information.

## What is Depth Estimation?

Depth estimation is a computer vision task that predicts the **distance of each pixel** from the camera, converting 2D images into 3D spatial representations. Unlike object detection (which identifies *what* and *where* objects are) or classification (which identifies *what* objects are), depth estimation tells you **how far away** each part of the scene is from the viewer.

A depth map is created where:
- Each pixel's value represents its **relative distance** from the camera
- **Lower values** (darker colors like purple/blue) indicate objects **closer** to the camera
- **Higher values** (lighter colors like yellow/green) indicate objects **further** from the camera

This enables understanding the 3D structure and spatial relationships within 2D images, which is essential for applications requiring depth perception.

## How This Block Works

This block takes one or more images as input and processes them through a depth estimation model (default: Depth-Anything-V2-Small). The model:
1. **Analyzes spatial relationships** in the image to infer depth information
2. **Generates a depth map** where each pixel contains depth/distance information
3. **Normalizes the depth values** to a 0.0-1.0 range for consistent representation
4. **Creates a visual representation** using a color-coded depth map (viridis colormap) where depth information is mapped to colors

The block outputs both the normalized depth array (for programmatic use) and a colorized visualization image (for human interpretation).

## Inputs and Outputs

**Input:**
- **images**: One or more images to estimate depth for (can be from workflow inputs or previous steps)
- **model_version**: The depth estimation model to use (default: "depth-anything-v2/small")

**Output:**
- **image**: A colorized depth map visualization image where colors represent depth (darker = closer, lighter = further)
- **normalized_depth**: A numpy array containing normalized depth values (0.0-1.0) where 0.0 represents the closest point and 1.0 represents the furthest point in the scene

## Key Configuration Options

- **model_version**: The depth estimation model to use (default: "depth-anything-v2/small") - this determines the model architecture and accuracy/speed tradeoff

## Common Use Cases

- **3D Reconstruction**: Generate depth maps from 2D images to understand 3D scene structure for reconstruction or modeling
- **Augmented Reality (AR)**: Provide depth information for realistic object placement and occlusion in AR applications
- **Autonomous Navigation**: Understand spatial relationships and distances for obstacle avoidance and path planning
- **Background Removal**: Use depth information to separate foreground objects from background more accurately
- **Photography and Cinematography**: Analyze scene composition, understand focus areas, or create depth-of-field effects
- **Robotics and Automation**: Enable robots to understand spatial relationships and distances for manipulation tasks

## Requirements

This block requires local execution (cannot run remotely). The model runs efficiently on Apple Silicon (M1-M4) devices using Metal Performance Shaders (MPS) for accelerated inference. For other platforms, it will use CPU or CUDA if available.

## Connecting to Other Blocks

The depth estimation results from this block can be connected to:
- **Visualization blocks** to overlay depth information on original images or create composite visualizations
- **Filter blocks** to filter objects or regions based on depth thresholds (e.g., only process objects within a certain distance range)
- **Transformation blocks** to modify images or detections based on depth information
- **Measurement blocks** to calculate distances, sizes, or spatial relationships using depth data
- **Conditional logic blocks** to make workflow decisions based on depth values (e.g., process only close objects)
- **Object Detection blocks** to enhance detections with depth information for 3D-aware object tracking
                """
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Depth Estimation",
                "Depth Anything",
                "Depth Anything V2",
                "Depth Anything V3",
                "Hugging Face",
                "HuggingFace",
            ],
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/depth_estimation@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    model_version: Union[
        Literal[
            "depth-anything-v2/small",
            "depth-anything-v3/small",
            "depth-anything-v3/base",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="depth-anything-v3/small",
        description="The Depth Estimation model to be used for inference.",
        examples=["depth-anything-v2/small", "$inputs.variant"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
            OutputDefinition(name="normalized_depth", kind=[NUMPY_ARRAY_KIND]),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        # Only images can be passed in as a list/batch
        return ["images"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DepthEstimationBlockV1(WorkflowBlock):
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
        model_version: str = "depth-anything-v3/small",
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for Depth Estimation. Please use a local or dedicated inference server."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str = "depth-anything-v3/small",
    ) -> BlockResult:
        # Convert each image to the format required by the model.
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]

        # Register Depth Estimation with the model manager.
        try:
            self._model_manager.add_model(model_id=model_version, api_key=self._api_key)
        except Exception as e:
            raise

        predictions = []
        for idx, image in enumerate(inference_images):
            # Run inference.
            request = DepthEstimationRequest(
                image=image,
            )

            try:
                prediction = self._model_manager.infer_from_request_sync(
                    model_id=model_version, request=request
                )
                predictions.append(prediction.response)
            except Exception as e:
                raise

        return predictions
