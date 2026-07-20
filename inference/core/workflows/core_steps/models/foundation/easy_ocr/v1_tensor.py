from typing import Dict, List, Literal, Optional, Tuple, Type

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
from inference.core.workflows.execution_engine.constants import CLASS_NAME_KEY
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

# These are the displayed languages in the UI dropdown
LANGUAGES = Literal[
    "English",
    "Japanese",
    "Kannada",
    "Korean",
    "Latin",
    "Telugu",
    "Simplified Chinese",
]

# Dictionary of displayed_language: (model, language_code)
# This is not an extensive list of supported languages, more codes can be added
MODELS: Dict[str, Tuple[str, List[str]]] = {
    "English": ("english_g2", ["en"]),
    "Japanese": ("japanese_g2", ["en", "ja"]),
    "Kannada": ("kannada_g2", ["en", "kn"]),
    "Korean": ("korean_g2", ["en", "ko"]),
    "Latin": ("latin_g2", ["en", "la", "es", "fr", "it", "pt", "de", "pl", "nl"]),
    "Telugu": ("telugu_g2", ["en", "te"]),
    "Simplified Chinese": ("zh_sim_g2", ["en", "ch_sim"]),
}

# Fixed, non-empty `class_id -> name` fallback map. The EasyOCR inference_models model
# emits `class_id == 0` for every box; the recognised text is attached per box onto
# `bboxes_metadata[i][CLASS_NAME_KEY]` (see `_attach_local_ocr_text` /
# `_attach_remote_ocr_text`), so the tensor serializer's per-box override emits the
# recognised text as `class` - byte-parity with the numpy sibling, whose
# `sv.Detections.from_inference` carries the recognised text as the class name. This
# map only satisfies the serializer's "CLASS_NAMES_KEY must be present" check.
CLASS_NAMES: Dict[int, str] = {0: "text-region"}
PREDICTION_TYPE = "ocr"

LONG_DESCRIPTION = """
 Retrieve the characters in an image using EasyOCR Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e.
ObjectDetectionBlock). An object detection model could isolate specific regions from an
image (i.e. a shipping container ID in a logistics use case) for further processing.
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis
on particular regions of an image.

Note that EasyOCR has limitations running within containers on Apple Silicon.
"""


class BlockManifest(WorkflowBlockManifest):

    model_config = ConfigDict(
        json_schema_extra={
            "name": "EasyOCR",
            "version": "v1",
            "short_description": "Extract text from an image using EasyOCR optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/easy_ocr@v1", "EasyOCR"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    language: LANGUAGES = Field(
        title="Language",
        description="Language model to use for OCR",
        default="English",
    )
    quantize: bool = Field(
        title="Use Quantized Model",
        description="Quantized models are smaller and faster, but may be less accurate and won't work correctly on all hardware.",
        default=False,
    )

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
        return [
            "easy_ocr/english_g2",
            "easy_ocr/japanese_g2",
            "easy_ocr/kannada_g2",
            "easy_ocr/korean_g2",
            "easy_ocr/latin_g2",
            "easy_ocr/telugu_g2",
            "easy_ocr/zh_sim_g2",
        ]


class EasyOCRBlockV1(WorkflowBlock):
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
        language: LANGUAGES = "English",
        quantize: bool = False,
    ) -> BlockResult:

        if language not in MODELS:
            raise ValueError(f"Unsupported language: {language}")

        version, language_codes = MODELS.get(language, "english_g2")
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                language_codes=language_codes,
                version=version,
                quantize=quantize,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                language_codes=language_codes,
                version=version,
                quantize=quantize,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        language_codes: List[str],
        version: str = "english_g2",
        quantize: bool = False,
    ) -> BlockResult:
        model_id = f"easy_ocr/{version}"
        self._model_manager.add_model(
            model_id,
            self._api_key,
            endpoint_type=ModelEndpointType.CORE_MODEL,
        )
        results = []
        for single_image in images:
            if single_image.is_tensor_materialised():
                model_image, image_color_format = single_image.tensor_image, "rgb"
            else:
                model_image, image_color_format = single_image.numpy_image, "bgr"
            texts, dets = self._model_manager.run_tensor_native_inference(
                model_id,
                images=[model_image],
                input_color_format=image_color_format,
                confidence=0.0,  # no confidence filtering
            )
            detections = attach_native_detection_metadata(
                dets[0],
                single_image,
                class_names=CLASS_NAMES,
                prediction_type=PREDICTION_TYPE,
            )
            detections = _attach_local_ocr_text(detections)
            results.append(
                {
                    "result": texts[0],
                    "predictions": detections,
                    "parent_id": single_image.parent_metadata.parent_id,
                    "root_parent_id": single_image.workflow_root_ancestor_metadata.parent_id,
                    "prediction_type": PREDICTION_TYPE,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        language_codes: List[str],
        version: str = "english_g2",
        quantize: bool = False,
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
        predictions = client.ocr_image(
            inference_input=non_empty_inference_images,
            model="easy_ocr",
            version=version,
            quantize=quantize,
            language_codes=language_codes,
        )
        if len(images) == 1:
            predictions = [predictions]
        # Remote returns OCRInferenceResponse dicts: result text + ObjectDetectionPrediction list.
        results = []
        for single_image, prediction in zip(images, predictions):
            raw_predictions = prediction.get("predictions", []) or []
            detections = native_detections_from_inference_predictions(
                single_image,
                predictions=raw_predictions,
                prediction_type=PREDICTION_TYPE,
                class_names=CLASS_NAMES,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            detections = _attach_remote_ocr_text(detections, raw_predictions)
            results.append(
                {
                    "result": prediction.get("result", ""),
                    "predictions": detections,
                    "parent_id": single_image.parent_metadata.parent_id,
                    "root_parent_id": single_image.workflow_root_ancestor_metadata.parent_id,
                    "prediction_type": PREDICTION_TYPE,
                }
            )
        return results


def _attach_local_ocr_text(detections: Detections) -> Detections:
    """Byte-parity (`class`) for the LOCAL path.

    The EasyOCR inference_models model carries each box's recognised text on
    ``bboxes_metadata[i]["text"]`` (and ``class_id == 0`` for every box). The tensor
    serializer emits ``bboxes_metadata[i][CLASS_NAME_KEY]`` as the per-box ``class``
    when present, else it falls back to the fixed ``CLASS_NAMES`` label. So promote the
    recognised ``text`` onto ``CLASS_NAME_KEY`` to match the numpy sibling, whose
    ``sv.Detections.from_inference`` carries the recognised text as the class name.
    Mutates and returns the same object.
    """
    if detections.bboxes_metadata is not None:
        for entry in detections.bboxes_metadata:
            entry[CLASS_NAME_KEY] = str(entry.get("text", ""))
    return detections


def _attach_remote_ocr_text(
    detections: Detections, predictions: List[dict]
) -> Detections:
    """Byte-parity (`class`) for the REMOTE path.

    The inference OCR response carries each box's recognised text under ``class``, but
    ``native_detections_from_inference_predictions`` keeps only ``detection_id`` on
    ``bboxes_metadata``. Re-attach the per-box ``class`` (mirrors ``pp_ocr/v1_tensor``)
    so the serializer emits the recognised text, matching the numpy sibling. Mutates
    and returns the same object.
    """
    if detections.bboxes_metadata is not None:
        for entry, prediction in zip(detections.bboxes_metadata, predictions):
            entry[CLASS_NAME_KEY] = str(prediction.get(CLASS_NAME_KEY, ""))
    return detections
