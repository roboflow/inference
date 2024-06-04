from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    FloatZeroToOne,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

LONG_DESCRIPTION = """
Run YOLO-World, a zero-shot object detection model, on an image.

YOLO-World accepts one or more text classes you want to identify in an image. The model 
returns the location of objects that meet the specified class, if YOLO-World is able to 
identify objects of that class.

We recommend experimenting with YOLO-World to evaluate the model on your use case 
before using this block in production. For example on how to effectively prompt 
YOLO-World, refer to the [Roboflow YOLO-World prompting 
guide](https://blog.roboflow.com/yolo-world-prompting-tips/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "YOLO-World Model",
            "short_description": "Run a zero-shot object detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["YoloWorldModel", "YoloWorld"]
    name: str = Field(description="Unique name of step in workflows")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    class_names: Union[
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]
    ] = Field(
        description="One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects.",
        examples=[["person", "car", "license plate"], "$inputs.class_names"],
    )
    version: Union[
        Literal["v2-s", "v2-m", "v2-l", "v2-x", "s", "m", "l", "x", ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="v2-s",
        description="Variant of YoloWorld model",
        examples=["v2-s", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.005,
        description="Confidence threshold for detections",
        examples=[0.005, "$inputs.confidence"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]


class YoloWorldModelBlock(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
    ):
        self._model_manager = model_manager
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: Batch[Optional[WorkflowImageData]],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        predictions = []
        non_empty_images = [i for i in images.iter_nonempty()]
        non_empty_inference_images = [
            i.to_inference_format(numpy_preferred=True) for i in non_empty_images
        ]
        for single_image in non_empty_inference_images:
            inference_request = YOLOWorldInferenceRequest(
                image=single_image,
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
            prediction = await self._model_manager.infer_from_request(
                yolo_world_model_id, inference_request
            )
            predictions.append(prediction.model_dump(by_alias=True, exclude_none=True))
        results = self._post_process_result(
            images=non_empty_images,
            predictions=predictions,
        )
        return images.align_batch_results(
            results=results, null_element={"predictions": None}
        )

    async def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
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
        non_empty_images = [i for i in images.iter_nonempty()]
        non_empty_inference_images = [i.numpy_image for i in non_empty_images]
        image_sub_batches = list(
            make_batches(
                iterable=non_empty_inference_images,
                batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )
        )
        predictions = []
        for sub_batch in image_sub_batches:
            sub_batch_predictions = await client.infer_from_yolo_world_async(
                inference_input=[i["value"] for i in sub_batch],
                class_names=class_names,
                model_version=version,
                confidence=confidence,
            )
            predictions.extend(sub_batch_predictions)
        results = self._post_process_result(
            images=non_empty_images, predictions=predictions
        )
        return images.align_batch_results(
            results=results, null_element={"predictions": None}
        )

    def _post_process_result(
        self,
        images: List[WorkflowImageData],
        predictions: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
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
