from typing import Any, List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
)
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2poly
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import (
    AutoModel,
    Detections,
    InstanceDetections,
    InstanceSegmentationModel,
    KeyPoints,
    KeyPointsDetectionModel,
    ObjectDetectionModel,
)
from inference_models.models.base.types import PreprocessingMetadata


class InferenceModelsObjectDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        self._model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_rgb(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: List[Detections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[ObjectDetectionInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[ObjectDetectionPrediction] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    ObjectDetectionPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass


class InferenceModelsInstanceSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "instance-segmentation"

        self._model: InstanceSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_rgb(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[InstanceSegmentationInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            masks = det.mask.detach().cpu().numpy()
            polys = masks2poly(masks)
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[InstanceSegmentationPrediction] = []

            for (x1, y1, x2, y2), mask_as_poly, conf, class_id in zip(
                xyxy, polys, confs, class_ids
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    InstanceSegmentationPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        points=[
                            Point(x=point[0], y=point[1]) for point in mask_as_poly
                        ],
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                InstanceSegmentationInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass


class InferenceModelsKeyPointsDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "keypoint-detection"

        self._model: KeyPointsDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
            kwargs["key_points_threshold"] = keypoint_confidence_threshold
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_rgb(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        key_points_classes = self._model.key_points_classes
        responses: List[KeypointsDetectionInferenceResponse] = []
        for preproc_metadata, (keypoints, det) in zip(
            preprocess_return_metadata, detections_list
        ):
            if det is None:
                raise RuntimeError(
                    "Keypoints detection model does not provide instances detection - this is not supported for "
                    "models from `inference-models` package which are adapted to work with `inference`."
                )
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()
            keypoints_xy = keypoints.xy.detach().cpu().tolist()
            keypoints_class_id = keypoints.class_id.detach().cpu().tolist()
            keypoints_confidence = keypoints.confidence.detach().cpu().tolist()
            predictions: List[KeypointsPrediction] = []

            for (
                (x1, y1, x2, y2),
                conf,
                class_id,
                instance_keypoints_xy,
                instance_keypoints_class_id,
                instance_keypoints_confidence,
            ) in zip(
                xyxy,
                confs,
                class_ids,
                keypoints_xy,
                keypoints_class_id,
                keypoints_confidence,
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    KeypointsPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                        keypoints=model_keypoints_to_response(
                            instance_keypoints_xy=instance_keypoints_xy,
                            instance_keypoints_confidence=instance_keypoints_confidence,
                            instance_keypoints_class_id=instance_keypoints_class_id,
                            key_points_classes=key_points_classes,
                        ),
                    )
                )

            responses.append(
                KeypointsDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass


def model_keypoints_to_response(
    instance_keypoints_xy: List[
        List[Union[float, int]]
    ],  # (num_key_points_foc_class_of_object, 2)
    instance_keypoints_confidence: List[float],  # (instance_key_points, )
    instance_keypoints_class_id: int,
    key_points_classes: List[List[str]],
) -> List[Keypoint]:
    keypoint_classes = key_points_classes[instance_keypoints_class_id]
    results = []
    for keypoint_class_id, ((x, y), confidence, keypoint_class_name) in enumerate(
        zip(instance_keypoints_xy, instance_keypoints_confidence, keypoint_classes)
    ):
        if confidence <= 0.0:
            continue
        keypoint = Keypoint(
            x=x,
            y=y,
            confidence=confidence,
            class_id=keypoint_class_id,
            **{"class": keypoint_class_name},
        )
        results.append(keypoint)
    return results
