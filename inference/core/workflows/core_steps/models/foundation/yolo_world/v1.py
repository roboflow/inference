from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

LONG_DESCRIPTION = """
Run YOLO-World, a zero-shot object detection model that can detect objects based on text descriptions without training.

## How This Block Works

This block takes one or more images as input and processes them through YOLO-World, a zero-shot object detection model. Unlike traditional object detection models that require training on specific object classes, YOLO-World can detect objects based on natural language class names you provide. The block:

1. Takes your list of class names (e.g., ["person", "car", "bicycle"]) and one or more images
2. Processes each image through the YOLO-World model to identify objects matching your specified classes
3. Returns object detection predictions with bounding boxes, class names, and confidence scores for each detected object

YOLO-World works best with short, descriptive class names for common objects. The model can handle multiple classes in a single pass, making it efficient for detecting multiple object types simultaneously. We recommend experimenting with YOLO-World to evaluate the model on your use case before using this block in production. For tips on how to effectively prompt YOLO-World, refer to the [Roboflow YOLO-World prompting guide](https://blog.roboflow.com/yolo-world-prompting-tips/).

## Common Use Cases

- **Zero-Shot Object Detection**: Detect objects in images using text descriptions without training a custom model for each class
- **Rapid Prototyping**: Quickly test object detection for new categories without dataset preparation or model training
- **Multi-Class Detection**: Detect multiple object types in a single pass by specifying multiple class names
- **Custom Object Categories**: Detect objects using natural language descriptions (e.g., "license plate", "red car", "person wearing helmet")
- **Content Filtering**: Identify specific content or objects in images for moderation or filtering purposes
- **Inventory Management**: Detect and count items in images using descriptive class names (e.g., "product on shelf", "package")

## Connecting to Other Blocks

The object detection predictions from this block can be connected to:

- **Visualization blocks** (e.g., Bounding Box Visualization) to draw detection results on images
- **Filter blocks** (e.g., Detections Filter) to filter detections based on confidence, class, or other criteria
- **Analytics blocks** (e.g., Data Aggregator, Line Counter) to count or aggregate detection results over time
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on detection results
- **Transformation blocks** (e.g., Dynamic Crop, Detections Transformation) to extract or transform regions based on detected objects
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log detection results
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "YOLO-World Model",
            "version": "v1",
            "short_description": "Run a zero-shot object detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 8,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/yolo_world_model@v1", "YoloWorldModel", "YoloWorld"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_names: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="One or more class names that you want YOLO-World to detect in the images. Provide a list of strings describing the objects (e.g., ['person', 'car', 'bicycle']). The model accepts any string as input, but works best with short, descriptive class names for common objects. For best results, use simple, clear descriptions.",
        examples=[["person", "car", "license plate"], "$inputs.class_names"],
    )
    version: Union[
        Literal[
            "v2-s",
            "v2-m",
            "v2-l",
            "v2-x",
            "s",
            "m",
            "l",
            "x",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="v2-s",
        description="The YOLO-World model variant to use. Options include 'v2-s' (small, fastest), 'v2-m' (medium), 'v2-l' (large), 'v2-x' (extra large, most accurate), and older variants ('s', 'm', 'l', 'x'). Larger models are more accurate but slower. Default is 'v2-s'.",
        examples=["v2-s", "v2-m", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.005,
        description="Confidence threshold for detections (0.0 to 1.0). Only detections with confidence scores above this threshold will be returned. Lower values return more detections (including lower confidence ones), while higher values return only high-confidence detections. Default is 0.005.",
        examples=[0.005, 0.25, 0.5, "$inputs.confidence"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class YoloWorldModelBlockV1(WorkflowBlock):

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
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                class_names=class_names,
                version=version,
                confidence=confidence,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                class_names=class_names,
                version=version,
                confidence=confidence,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = YOLOWorldInferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                yolo_world_version_id=version,
                confidence=confidence,
                text=class_names,
                api_key=self._api_key,
            )
            yolo_world_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="yolo_world",
            )
            prediction = self._model_manager.infer_from_request_sync(
                yolo_world_model_id, inference_request
            )
            predictions.append(prediction.model_dump(by_alias=True, exclude_none=True))
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
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
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(inference_configuration=configuration)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        inference_images = [i.to_inference_format() for i in images]
        image_sub_batches = list(
            make_batches(
                iterable=inference_images,
                batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )
        )
        predictions = []
        for sub_batch in image_sub_batches:
            sub_batch_predictions = client.infer_from_yolo_world(
                inference_input=[i["value"] for i in sub_batch],
                class_names=class_names,
                model_version=version,
                confidence=confidence,
            )
            predictions.extend(sub_batch_predictions)
        return self._post_process_result(images=images, predictions=predictions)

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="object-detection",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]
