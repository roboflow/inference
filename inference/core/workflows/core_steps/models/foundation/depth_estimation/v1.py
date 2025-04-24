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
                🎯 This workflow block performs depth estimation on images using Apple's DepthPro model. It analyzes the spatial relationships
                and depth information in images to create a depth map where:

                📊 Each pixel's value represents its relative distance from the camera
                🔍 Lower values (darker colors) indicate closer objects
                🔭 Higher values (lighter colors) indicate further objects

                The model outputs:
                1. 🗺️ A depth map showing the relative distances of objects in the scene
                2. 📐 The camera's field of view (in degrees)
                3. 🔬 The camera's focal length

                This is particularly useful for:
                - 🏗️ Understanding 3D structure from 2D images
                - 🎨 Creating depth-aware visualizations
                - 📏 Analyzing spatial relationships in scenes
                - 🕶️ Applications in augmented reality and 3D reconstruction

                ⚡ The model runs efficiently on Apple Silicon (M1-M4) using Metal Performance Shaders (MPS) for accelerated inference.
                """
            ),
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Depth Estimation",
                "Depth Anything",
                "Depth Anything V2",
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

    model_version: str = Field(
        default="depth-anything-v2/small",
        description="The Depth Estimation model to be used for inference.",
        examples=["depth-anything-v2/small"],
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
        model_version: str = "depth-anything-v2/small",
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
        model_version: str = "depth-anything-v2/small",
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
