from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import DepthEstimationRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
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
from inference_sdk import InferenceHTTPClient


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
            return self.run_remotely(
                images=images,
                model_version=model_version,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_version: str = "depth-anything-v3/small",
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

        predictions = []
        for single_image in images:
            result = client.depth_estimation(
                inference_input=single_image.base64_image,
                model_id=model_version,
            )
            # Convert the result back to the expected format
            # Remote returns: {"normalized_depth": [...], "image": hex_string}
            normalized_depth = np.array(result.get("normalized_depth", []))

            # Return in the same format as local execution expects
            predictions.append(
                {
                    "image": single_image,
                    "normalized_depth": normalized_depth,
                }
            )

        return predictions

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
