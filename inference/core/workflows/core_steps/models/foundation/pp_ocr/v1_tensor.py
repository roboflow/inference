"""Tensor-native sibling of `roboflow_core/pp_ocr@v1`.

The loader import-swaps `pp_ocr.v1.PPOCRBlockV1` for this class when
`ENABLE_TENSOR_DATA_REPRESENTATION` is enabled. The manifest is identical to the
numpy sibling (same `type`/version, params, validators); only the `predictions`
output kind changes to `TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND` and the
run-body emits a native `inference_models.Detections` instead of `sv.Detections`.

PP-OCR has no `run_tensor_native_inference` adapter, so both execution modes keep
the exact model-calling machinery of `pp_ocr.v1` (a `PPOCRInferenceRequest` served
by the local `ModelManager`, or `InferenceHTTPClient.ocr_image` remotely) and
produce the standard inference-format `OCRInferenceResponse` dicts. Those dicts are
converted to a native `Detections` HERE (no shared util is touched) so the numpy
`post_process_ocr_result` / `sv.Detections.from_inference` path is bypassed.

Byte-parity note (`class`): a PP-OCR box carries the recognised line text under
`class` with a fixed `class_id == 0`. `native_detections_from_inference_predictions`
folds `class` into a `class_id -> name` map, which for OCR would collapse every box
to a single label. So the recognised text is re-attached per box onto
`bboxes_metadata[i][CLASS_NAME_KEY]`; the tensor serializer's per-box override then
emits it as `class`, matching the numpy path's per-box `sv.Detections` class name.
The image-level `class_names` map is only a non-empty fallback required by the
serializer (never consulted, since every box carries the per-box override).
"""

from typing import List, Literal, Optional, Type

from pydantic import ConfigDict, Field, model_validator

from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    native_detections_from_inference_predictions,
)
from inference.core.workflows.core_steps.common.utils import load_core_model
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
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
from inference_models.models.base.object_detection import Detections
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import InferenceConfiguration

# Fixed, non-empty `class_id -> name` fallback map. PP-OCR emits `class_id == 0` for
# every box; the recognised text is attached per box (see module docstring), so this
# map only satisfies the tensor serializer's "CLASS_NAMES_KEY must be present" check.
CLASS_NAMES = {0: "text-region"}
PREDICTION_TYPE = "ocr"

LONG_DESCRIPTION = """
 Retrieve the characters in an image using PP-OCR (PaddleOCR) Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e.
ObjectDetectionBlock). An object detection model could isolate specific regions from an
image (i.e. a shipping container ID in a logistics use case) for further processing.
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis
on particular regions of an image.
"""


class BlockManifest(WorkflowBlockManifest):

    model_config = ConfigDict(
        json_schema_extra={
            "name": "PP-OCR",
            "version": "v1",
            "short_description": "Extract text from an image using PP-OCR (PaddleOCR) optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["ocr", "text", "paddle", "paddleocr"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/pp_ocr@v1"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    text_detection: Literal["none", "tiny", "small", "medium"] = Field(
        title="Text Detection Model Size",
        description="Text detection model size. Set to `none` to disable detection and run "
        "recognition on each input image as a single, pre-cropped text line.",
        default="small",
    )
    text_recognition: Literal["none", "tiny", "small", "medium"] = Field(
        title="Text Recognition Model Size",
        description="Text recognition model size. Set to `none` to disable recognition and "
        "return detected text boxes only, without transcribed text.",
        default="small",
    )

    @model_validator(mode="after")
    def validate_stages(self) -> "BlockManifest":
        if self.text_detection == "none" and self.text_recognition == "none":
            raise ValueError(
                "PP-OCR requires at least one of text detection or text recognition to be enabled"
            )
        return self

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
        """Return list of model_id variants that can satisfy this block.

        These are core-model ids in the standard ``<core_model>/<version>``
        form; for PP-OCR the version token encodes the two stage sizes as
        ``<text_detection>-<text_recognition>`` and is parsed back into the
        stage sizes by ``PPOCRInferenceRequest``.
        """
        return [
            "pp_ocr/small-small",
            "pp_ocr/tiny-tiny",
            "pp_ocr/medium-medium",
            "pp_ocr/small-none",
            "pp_ocr/none-small",
        ]


class PPOCRBlockV1(WorkflowBlock):
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
        text_detection: str = "small",
        text_recognition: str = "small",
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                text_detection=text_detection,
                text_recognition=text_recognition,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                text_detection=text_detection,
                text_recognition=text_recognition,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        text_detection: str = "small",
        text_recognition: str = "small",
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = PPOCRInferenceRequest(
                text_detection=text_detection,
                text_recognition=text_recognition,
                image=single_image.to_inference_format(numpy_preferred=True),
                api_key=self._api_key,
            )
            model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="pp_ocr",
            )
            result = self._model_manager.infer_from_request_sync(
                model_id, inference_request
            )
            predictions.append(
                _build_native_prediction(
                    image=single_image,
                    response=result.model_dump(by_alias=True, exclude_none=True),
                )
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        text_detection: str = "small",
        text_recognition: str = "small",
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
            model="pp_ocr",
            version=f"{text_detection}-{text_recognition}",
        )
        if len(images) == 1:
            responses = [responses]
        return [
            _build_native_prediction(image=single_image, response=response)
            for single_image, response in zip(images, responses)
        ]


def _build_native_prediction(
    image: WorkflowImageData,
    response: dict,
) -> dict:
    """Convert a single-image ``OCRInferenceResponse`` dict into the block output.

    Mirrors the output dict shape of the numpy ``post_process_ocr_result``
    (``result`` / ``predictions`` / ``parent_id`` / ``root_parent_id`` /
    ``prediction_type``) but with ``predictions`` as a native ``Detections``.
    """
    raw_predictions = response.get("predictions") or []
    detections = _native_detections_from_ocr_predictions(
        image=image,
        predictions=raw_predictions,
    )
    return {
        "result": response.get("result", ""),
        "predictions": detections,
        "parent_id": image.parent_metadata.parent_id,
        "root_parent_id": image.workflow_root_ancestor_metadata.parent_id,
        "prediction_type": PREDICTION_TYPE,
    }


def _native_detections_from_ocr_predictions(
    image: WorkflowImageData,
    predictions: List[dict],
) -> Detections:
    """Build a native ``Detections`` from PP-OCR inference-format boxes.

    Reuses the shared ``native_detections_from_inference_predictions`` (center->corner
    conversion, ``detection_id`` preservation, image lineage) with the fixed
    ``CLASS_NAMES`` fallback map, then re-attaches each box's recognised text onto
    ``bboxes_metadata[i][CLASS_NAME_KEY]`` so the tensor serializer emits it as the
    per-box ``class`` (byte-parity with the numpy path).
    """
    detections = native_detections_from_inference_predictions(
        image=image,
        predictions=predictions,
        prediction_type=PREDICTION_TYPE,
        class_names=CLASS_NAMES,
        device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
    )
    if detections.bboxes_metadata is not None:
        for entry, prediction in zip(detections.bboxes_metadata, predictions):
            entry[CLASS_NAME_KEY] = str(prediction.get(CLASS_NAME_KEY, ""))
    return detections
