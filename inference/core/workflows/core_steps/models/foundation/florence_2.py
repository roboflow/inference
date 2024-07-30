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
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches
from inference.models.florence2.florence2 import Florence2
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
import torch
from PIL import Image

class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Florence-2 Model",
            "short_description": "Run a multitask transformer model for a wide range of computer vision tasks.",
            "long_description": "Florence-2 is a multitask transformer model that can be used for a wide range of computer vision tasks. It is based on the Vision Transformer architecture and has been trained on a large-scale dataset of images with a wide range of labels. The model is capable of performing tasks such as image classification, object detection, and image segmentation.",
            "license": "MIT",
            "block_type": "model"
        }
    )
    type: Literal["Florence2Model", "Florence2"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    vision_task: Union[
        Literal[
            "<OPEN_VOCABULARY_DETECTION>",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        description="The computer vision task to perform.",
        default="<OPEN_VOCABULARY_DETECTION>",
        examples=["<OD>", "<CAPTION>"],
    )
    prompt: Union[
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]
    ] = Field(
        description="The accompanying prompt for the task (comma separated).",
        examples=[["red apple", "blue soda can"], "$inputs.prompt"],
    )
    version: Union[
        Literal[
            "B",
            "L"
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="B",
        description="Variant of Florence-2 Model",
        examples=["B", "$inputs.variant"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return False

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

class Florence2ModelBlock(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self.model = Florence2("florence-2-base/1")
        # CHECKPOINT = "microsoft/Florence-2-large"
        # REVISION = 'refs/pr/6'
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).eval().to(DEVICE)
        # self.processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)


    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        images: Batch[WorkflowImageData],
        vision_task: str,
        prompt: List[str],
        version: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return await self.run_locally(
                images=images,
                vision_task=vision_task,
                prompt=prompt,
                version=version,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return await self.run_locally(
                images=images,
                vision_task=vision_task,
                prompt=prompt,
                version=version,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    # def run_example(self, task_prompt, image, text_input=None):
    #     if text_input is None:
    #         prompt = task_prompt
    #     else:
    #         prompt = task_prompt + text_input
    #     print(prompt)
    #     inputs = self.processor(text=prompt, images=image, return_tensors="pt")
    #     generated_ids = self.model.generate(
    #         input_ids=inputs["input_ids"].cuda(),
    #         pixel_values=inputs["pixel_values"].cuda(),
    #         max_new_tokens=1024,
    #         early_stopping=False,
    #         do_sample=False,
    #         num_beams=3,
    #     )
    #     generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #     parsed_answer = self.processor.post_process_generation(
    #         generated_text, 
    #         task=task_prompt, 
    #         image_size=(image.width, image.height)
    #     )

    #     return parsed_answer

    async def run_locally(
        self,
        images: Batch[WorkflowImageData],
        vision_task: str,
        prompt: List[str],
        version: str,
    ) -> BlockResult:
        predictions = []
        images = [images] if not isinstance(images, list) else images
        for single_image in images:
            single_image = Image.fromarray(single_image.numpy_image)
            parsed_answer = self.model.infer(single_image, "<CAPTION>")#self.run_example(vision_task, single_image, "<and>".join(prompt))
            preds = []
            for i, bbox in enumerate(parsed_answer[vision_task]["bboxes"]):
                pred = {
                    "class": parsed_answer[vision_task]["bboxes_labels"][i],
                    "class_id": prompt.index(parsed_answer[vision_task]["bboxes_labels"][i]),
                    "confidence": 1.0,
                    "x": (bbox[0] + bbox[2]) / 2,
                    "y": (bbox[1] + bbox[3]) / 2,
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1],
                }
                preds.append(pred)
            predictions.append(preds)

        print(predictions)

        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        # predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        # predictions = attach_prediction_type_info_to_sv_detections_batch(
        #     predictions=predictions,
        #     prediction_type="object-detection",
        # )
        # predictions = attach_parents_coordinates_to_batch_of_sv_detections(
        #     images=images,
        #     predictions=predictions,
        # )
        # return [{"predictions": prediction} for prediction in predictions]
        img = Image.fromarray(images[0].numpy_image)
        return {"predictions": {"image": {"height": img.height, "width": img.width}, "predictions": predictions[0]}}