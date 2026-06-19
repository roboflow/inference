from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    MOONDREAM2_ENABLED,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    attach_native_detection_metadata,
    native_detections_from_inference_predictions,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    STRING_KIND,
    ImageInputField,
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
from inference_sdk import InferenceHTTPClient


class BlockManifest(WorkflowBlockManifest):
    # SmolVLM needs an image and a text prompt.
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Optional text prompt to provide additional context to Moondream2.",
        examples=["my prompt", "$inputs.prompt"],
        default=None,
    )

    # Standard model configuration for UI, schema, etc.
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Moondream2",
            "version": "v1",
            "short_description": "Run Moondream2 on an image.",
            "long_description": "This workflow block runs Moondream2, a multimodal vision-language model. You can use this block to run zero-shot object detection.",
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Moondream2",
                "moondream",
                "vision language model",
                "VLM",
                "object detection",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/moondream2@v1"]

    model_version: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default="moondream2/moondream2_2b_jul24",
        description="The Moondream2 model to be used for inference.",
        examples=["moondream2/moondream2_2b_jul24", "moondream2/moondream2-2b"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        restrictions = [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; run_locally() loads a model that needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]
        if not MOONDREAM2_ENABLED:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "MOONDREAM2_ENABLED=False on Roboflow Hosted Serverless: "
                        "the Moondream2 endpoint is not registered, so "
                        "run_remotely() returns 404."
                    ),
                    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
                )
            )
        return restrictions

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return ["moondream2/moondream2_2b_jul24"]


class Moondream2BlockV1(WorkflowBlock):
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
        model_version: str,
        prompt: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode == StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_version=model_version,
                prompt=prompt,
            )
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_version=model_version,
                prompt=prompt,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
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

        prompt = prompt or ""
        # The Moondream2 detection endpoint is prompted with a single object, so
        # every returned box carries class_id == 0 / class == <prompt>.
        class_names = _build_class_names(prompt)
        predictions = []
        for image in images:
            result = client.infer_lmm(
                inference_input=image.base64_image,
                model_id=model_version,
                prompt=prompt,
                model_id_in_path=True,
            )
            # `result` is the standard inference object-detection response; its
            # `predictions` list holds detection dicts already in inference
            # center x/y/width/height format with `class` / `class_id` set. Build
            # a native Detections from those dicts (no sv.Detections round-trip).
            detections = native_detections_from_inference_predictions(
                image=image,
                predictions=result.get("predictions", []),
                prediction_type="object-detection",
                class_names=class_names,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            predictions.append({"predictions": detections})
        return predictions

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        prompt: Optional[str],
    ) -> BlockResult:
        # Use the provided prompt (or an empty string if None) for every image.
        prompt = prompt or ""
        # The model assigns class_id == 0 to the single prompted object class.
        class_names = _build_class_names(prompt)

        # Register Moondream2 with the model manager.
        self._model_manager.add_model(model_id=model_version, api_key=self._api_key)

        predictions = []
        for image in images:
            # The inference_models adapter returns native Detections (one per
            # image) straight from MoonDream2HF.detect; it tolerates the
            # `input_color_format` kwarg (forwarded to images_to_pillow) and
            # requires `classes` (the prompted object list).
            if image.is_tensor_materialised():
                model_image, image_color_format = image.tensor_image, "rgb"
            else:
                model_image, image_color_format = image.numpy_image, "bgr"
            dets = self._model_manager.run_tensor_native_inference(
                model_id=model_version,
                images=[model_image],
                input_color_format=image_color_format,
                classes=[prompt],
            )
            detections = attach_native_detection_metadata(
                dets[0],
                image=image,
                class_names=class_names,
                prediction_type="object-detection",
            )
            predictions.append({"predictions": detections})
        return predictions


def _build_class_names(prompt: str) -> Dict[int, str]:
    # Moondream2 detection is single-object prompted: class_id 0 -> prompt text.
    return {0: prompt}
