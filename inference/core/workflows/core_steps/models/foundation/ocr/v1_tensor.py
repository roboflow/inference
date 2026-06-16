from typing import Dict, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
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
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
 Retrieve the characters in an image using DocTR Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e.
ObjectDetectionBlock). An object detection model could isolate specific regions from an
image (i.e. a shipping container ID in a logistics use case) for further processing.
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis
on particular regions of an image.
"""

# Class id -> name mapping for DocTR's structured OCR detections. Sourced from the
# inference_models DocTR model's `class_names` property
# (inference_models/inference_models/models/doctr/doctr_torch.py -> ["block", "line", "word"]),
# which assigns class_id 0 to block, 1 to line, 2 to word. The serialiser needs this
# id -> name map on image_metadata to resolve every detection's class.
DOCTR_CLASS_NAMES: Dict[int, str] = {0: "block", 1: "line", 2: "word"}

PREDICTION_TYPE = "ocr"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OCR Model",
            "version": "v1",
            "short_description": "Extract text from an image using DocTR optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/ocr_model@v1", "OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="result", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return ["doctr/default"]


class OCRModelBlockV1(WorkflowBlock):

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
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        # Tensor-native local path: the DocTR core model runs through the
        # inference_models adapter's run_tensor_native_inference, which returns
        # (rendered_texts, [Detections]) - the per-image Detections already carry
        # xyxy / class_id / confidence and per-box {"text": ...} on bboxes_metadata.
        # We register the core model by id only (its InferenceRequest requires an
        # `image` we don't need just to register; the block has no version selector,
        # so the id is always "doctr/default" - the request's doctr_version_id
        # default), then attach the workflow image lineage + class_names map (and
        # guarantee detection_ids) natively - no JSON / sv.Detections round-trip.
        doctr_model_id = "doctr/default"
        self._model_manager.add_model(
            doctr_model_id,
            self._api_key,
            endpoint_type=ModelEndpointType.CORE_MODEL,
        )
        predictions = []
        for single_image in images:
            texts, detections_batch = self._model_manager.run_tensor_native_inference(
                doctr_model_id,
                images=[single_image.tensor_image],
                input_color_format="rgb",
            )
            detections = attach_native_detection_metadata(
                detections=detections_batch[0],
                image=single_image,
                class_names=DOCTR_CLASS_NAMES,
                prediction_type=PREDICTION_TYPE,
            )
            predictions.append(
                {
                    "result": texts[0],
                    "predictions": detections,
                    "parent_id": single_image.parent_metadata.parent_id,
                    "root_parent_id": single_image.workflow_root_ancestor_metadata.parent_id,
                    "prediction_type": PREDICTION_TYPE,
                }
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
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
        configuration = InferenceConfiguration(
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(configuration)
        non_empty_inference_images = [i.base64_image for i in images]
        responses = client.ocr_image(
            inference_input=non_empty_inference_images,
            generate_bounding_boxes=True,
        )
        if len(images) == 1:
            responses = [responses]
        # Remote returns JSON OCR responses; build the tensor-native Detections from
        # the standard inference detection dicts under "predictions". When
        # generate_bounding_boxes yields nothing the key may be absent/empty -
        # native_detections_from_inference_predictions handles an empty list.
        predictions = []
        for single_image, response in zip(images, responses):
            raw_predictions = response.get("predictions") or []
            detections = native_detections_from_inference_predictions(
                image=single_image,
                predictions=raw_predictions,
                prediction_type=PREDICTION_TYPE,
                class_names=DOCTR_CLASS_NAMES,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            predictions.append(
                {
                    "result": response.get("result", ""),
                    "predictions": detections,
                    "parent_id": single_image.parent_metadata.parent_id,
                    "root_parent_id": single_image.workflow_root_ancestor_metadata.parent_id,
                    "prediction_type": PREDICTION_TYPE,
                }
            )
        return predictions
