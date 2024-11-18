from typing import Literal, Union, Optional, List, Type
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from inference.core.entities.requests.inference import (
    ObjectDetectionInferenceRequest,
    ClassificationInferenceRequest,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.core.managers.base import ModelManager
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import post_process_bboxes
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    Selector,
    IMAGE_KIND,
    ImageInputField,
    ROBOFLOW_MODEL_ID_KIND,
    BOOLEAN_KIND,
    LIST_OF_VALUES_KIND,
    FloatZeroToOne,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    CLASSIFICATION_PREDICTION_KIND,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlockManifest,
    WorkflowBlock,
    BlockResult,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection And Classification",
            "version": "v1",
            "short_description": "TODO",
            "long_description": "TODO",
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/detection_and_classification@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    detector_model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        title="Detection Model",
        description="Roboflow model identifier for detection model",
        examples=["my_project/3", "$inputs.model"],
    )
    detector_class_agnostic_nms: Union[
        Optional[bool], Selector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Value to decide if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    detector_class_filter: Union[
        Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])
    ] = Field(
        default=None,
        description="List of classes to retrieve from predictions (to define subset of those "
        "which was used while model training)",
        examples=[["a", "b", "c"], "$inputs.class_filter"],
    )
    detector_confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    detector_iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="Maximum number of detections to return",
        examples=[30, "$inputs.max_detections"],
    )
    max_candidates: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    classifier_model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        title="Classifier Model",
        description="Roboflow model identifier for classification model",
        examples=["my_project/3", "$inputs.model"],
    )
    classifier_confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionAndClassificationBlockV1(WorkflowBlock):

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
        detector_model_id: str,
        detector_class_agnostic_nms: Optional[bool],
        detector_class_filter: Optional[List[str]],
        detector_confidence: float,
        detector_iou_threshold: float,
        max_detections: int,
        max_candidates: int,
        classifier_model_id: str,
        classifier_confidence: float,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "Remote execution is not supported for Detect And Classify."
            )
        self._model_manager.add_model(
            model_id=detector_model_id,
            api_key=self._api_key,
        )
        self._model_manager.add_model(
            model_id=classifier_model_id,
            api_key=self._api_key,
        )
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        request = ObjectDetectionInferenceRequest(
            api_key=self._api_key,
            model_id=detector_model_id,
            image=inference_images,
            disable_active_learning=True,
            class_agnostic_nms=detector_class_agnostic_nms,
            class_filter=detector_class_filter,
            confidence=detector_confidence,
            iou_threshold=detector_iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            source="workflow-execution",
        )
        model_instance = self._model_manager.models()[detector_model_id]
        preproc_image, preprocess_return_metadata = self._model_manager.preprocess(
            model_id=detector_model_id,
            request=request,
        )
        input_elements = len(inference_images)
        if input_elements == 1:
            prediction_results = self._model_manager.predict(
                img_in=preproc_image,
                **request.dict(),
                return_image_dims=False,
            )
            predictions = prediction_results[0]
        else:
            predictions = []
            for i in range(len(inference_images)):
                prediction_results = self._model_manager.predict(
                    img_in=np.expand_dims(preproc_image[i], axis=0),
                    **request.dict(),
                    return_image_dims=False,
                )
                p = prediction_results[0]
                predictions.append(p)
            predictions = np.concatenate(predictions, axis=0)
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=detector_confidence,
            iou_thresh=detector_iou_threshold,
            class_agnostic=detector_class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            box_format=model_instance.box_format,
        )
        infer_shape = (model_instance.img_size_h, model_instance.img_size_w)
        img_dims = preprocess_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions,
            infer_shape,
            img_dims,
            model_instance.preproc,
            resize_method=model_instance.resize_method,
            disable_preproc_static_crop=preprocess_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        crops = []
        for element_idx, batch_predictions in enumerate(predictions):
            batch_element_crops = []
            for single_detection in batch_predictions:
                x_min, y_min, x_max, y_max = np.round(single_detection[:4]).astype(
                    np.int16
                )
                confidence = single_detection[4]
                if confidence < detector_confidence:
                    continue
                class_name = model_instance.class_names[int(single_detection[6])]
                if detector_class_filter and class_name not in detector_class_filter:
                    continue
                cropped_image = images[element_idx].numpy_image[
                    y_min:y_max, x_min:x_max
                ]
                if cropped_image.size == 0:
                    continue
                batch_element_crops.append(
                    WorkflowImageData.create_crop(
                        origin_image_data=images[element_idx],
                        crop_identifier=f"{uuid4()}",
                        cropped_image=cropped_image,
                        offset_x=x_min,
                        offset_y=y_min,
                    )
                )
            crops.append(batch_element_crops)
        results_indices, flattened_crops = [], []
        for batch_idx, batch_of_crops in enumerate(crops):
            for item_id, crop in enumerate(batch_of_crops):
                flattened_crops.append(crop.to_inference_format(numpy_preferred=True))
                results_indices.append((batch_idx, item_id))
        if not flattened_crops:
            return [] * len(images)
        request = ClassificationInferenceRequest(
            api_key=self._api_key,
            model_id=classifier_model_id,
            image=flattened_crops,
            confidence=classifier_confidence,
            disable_active_learning=True,
            source="workflow-execution",
        )
        preproc_crops, preprocess_crops_return_metadata = (
            self._model_manager.preprocess(
                model_id=classifier_model_id,
                request=request,
            )
        )
        model_instance = self._model_manager.models()[classifier_model_id]
        img_dims = preprocess_crops_return_metadata["img_dims"]
        cls_results = []
        for i in range(len(flattened_crops)):
            c = self._model_manager.predict(
                img_in=np.expand_dims(preproc_crops[i], axis=0),
                **request.dict(),
                return_image_dims=False,
            )[0]
            cls_results.append(c[0])
        cls_results = np.concatenate(cls_results, axis=0)
        final_results = []
        current_batch_results = []
        previous_batch_idx = None
        for ind, (result, (batch_idx, element_idx)) in enumerate(
            zip(cls_results, results_indices)
        ):
            if previous_batch_idx is not None and batch_idx != previous_batch_idx:
                final_results.append(current_batch_results)
                current_batch_results = []
            if model_instance.multiclass:
                results = dict()
                predicted_classes = []
                for i, o in enumerate(result):
                    cls_name = model_instance.class_names[i]
                    score = float(o)
                    results[cls_name] = {"confidence": score, "class_id": i}
                    if score > classifier_confidence:
                        predicted_classes.append(cls_name)
                current_batch_results.append(
                    {
                        "predictions": {
                            "image": {
                                "width": img_dims[ind][0],
                                "height": img_dims[ind][1],
                            },
                            "predicted_classes": predicted_classes,
                            "predictions": results,
                        },
                        "crops": flattened_crops[0],
                    }
                )
            else:
                e_x = np.exp(result - np.max(result))
                preds = e_x / e_x.sum()
                results = []
                for i, cls_name in enumerate(model_instance.class_names):
                    score = float(preds[i])
                    pred = {
                        "class_id": i,
                        "class": cls_name,
                        "confidence": round(score, 4),
                    }
                    results.append(pred)
                results = sorted(results, key=lambda x: x["confidence"], reverse=True)
                current_batch_results.append(
                    {
                        "predictions": {
                            "image": {
                                "width": img_dims[ind][0],
                                "height": img_dims[ind][1],
                            },
                            "predictions": result,
                            "top": results[0]["class"],
                            "confidence": results[0]["confidence"],
                        },
                        "crops": flattened_crops[ind],
                    }
                )
        final_results.append(current_batch_results)
        return final_results
