from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference.core.entities.requests import (
    ClassificationInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    ENFORCE_CREDITS_VERIFICATION,
    GCP_SERVERLESS,
)
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image_bgr
from inference.core.utils.postprocess import masks2poly
from inference.core.utils.visualisation import draw_detection_predictions
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import (
    AutoModel,
    ClassificationModel,
    ClassificationPrediction,
    Detections,
    InstanceDetections,
    InstanceSegmentationModel,
    KeyPoints,
    KeyPointsDetectionModel,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
    ObjectDetectionModel,
)
from inference_models.models.base.types import PreprocessingMetadata

DEFAULT_COLOR_PALETTE = [
    "#A351FB",
    "#FF4040",
    "#FFA1A0",
    "#FF7633",
    "#FFB633",
    "#D1D435",
    "#4CFB12",
    "#94CF1A",
    "#40DE8A",
    "#1B9640",
    "#00D6C1",
    "#2E9CAA",
    "#00C4FF",
    "#364797",
    "#6675FF",
    "#0019EF",
    "#863AFF",
    "#530087",
    "#CD3AFF",
    "#FF97CA",
    "#FF39C9",
]


class InferenceModelsObjectDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
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

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsInstanceSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "instance-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: InstanceSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
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

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsKeyPointsDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "keypoint-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: KeyPointsDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            extra_weights_provider_headers=extra_weights_provider_headers,
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
            load_image_bgr(
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
        keypoints_list, detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        if detections_list is None:
            raise RuntimeError(
                "Keypoints detection model does not provide instances detection - this is not supported for "
                "models from `inference-models` package which are adapted to work with `inference`."
            )
        key_points_classes = self._model.key_points_classes
        responses: List[KeypointsDetectionInferenceResponse] = []
        for preproc_metadata, keypoints, det in zip(
            preprocess_return_metadata, keypoints_list, detections_list
        ):

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

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


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


class InferenceModelsClassificationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "classification"

        extra_weights_provider_headers = get_extra_weights_provider_headers()

        self._model: Union[ClassificationModel, MultiLabelClassificationModel] = (
            AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                extra_weights_provider_headers=extra_weights_provider_headers,
                **kwargs,
            )
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        images_shapes = [i.shape[:2] for i in np_images]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs), images_shapes

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        returned_metadata: List[Tuple[int, int]],
        **kwargs,
    ) -> Union[
        List[MultiLabelClassificationInferenceResponse],
        List[ClassificationInferenceResponse],
    ]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        post_processed_predictions = self._model.post_process(
            predictions, **mapped_kwargs
        )
        if isinstance(post_processed_predictions, list):
            # multi-label classification
            return prepare_multi_label_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
                confidence_threshold=kwargs.get("confidence", 0.5),
            )
        else:
            # single-label classification
            return prepare_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
                confidence_threshold=kwargs.get("confidence", 0.5),
            )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=True)
        for response in responses:
            response.time = perf_counter() - t1
            response.inference_id = getattr(request, "id", None)

        if request.visualize_predictions:
            for response in responses:
                response.visualization = draw_predictions(
                    request, response, self.class_names
                )

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses


def prepare_multi_label_classification_response(
    post_processed_predictions: List[MultiLabelClassificationPrediction],
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[MultiLabelClassificationInferenceResponse]:
    results = []
    for prediction, image_size in zip(post_processed_predictions, image_sizes):
        image_predictions_dict = dict()
        predicted_classes = []
        for class_id, confidence in zip(
            prediction.class_ids.cpu().tolist(), prediction.confidence.cpu().tolist()
        ):
            cls_name = class_names[class_id]
            image_predictions_dict[cls_name] = {
                "confidence": confidence,
                "class_id": class_id,
            }
            if confidence > confidence_threshold:
                predicted_classes.append(cls_name)
        results.append(
            MultiLabelClassificationInferenceResponse(
                predictions=image_predictions_dict,
                predicted_classes=predicted_classes,
                image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
                # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            )
        )
    return results


def prepare_classification_response(
    post_processed_predictions: ClassificationPrediction,
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[ClassificationInferenceResponse]:
    responses = []
    for classes_confidence, image_size in zip(
        post_processed_predictions.confidence.cpu().tolist(), image_sizes
    ):
        individual_classes_predictions = []
        for i, cls_name in enumerate(class_names):
            class_score = float(classes_confidence[i])
            if class_score < confidence_threshold:
                continue
            class_prediction = {
                "class_id": i,
                "class": cls_name,
                "confidence": round(class_score, 4),
            }
            individual_classes_predictions.append(class_prediction)
        individual_classes_predictions = sorted(
            individual_classes_predictions, key=lambda x: x["confidence"], reverse=True
        )
        response = ClassificationInferenceResponse(
            image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
            # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            predictions=individual_classes_predictions,
            top=(
                individual_classes_predictions[0]["class"]
                if individual_classes_predictions
                else ""
            ),
            confidence=(
                individual_classes_predictions[0]["confidence"]
                if individual_classes_predictions
                else 0.0
            ),
        )
        responses.append(response)
    return responses


def draw_predictions(inference_request, inference_response, class_names: List[str]):
    """Draw prediction visuals on an image.

    This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

    Args:
        inference_request: The request object containing the image and parameters.
        inference_response: The response object containing the predictions and other details.
        class_names: List of class names corresponding to the model's classes.

    Returns:
        bytes: The bytes of the visualized image in JPEG format.
    """
    image = inference_request.image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_id_2_color = {
        i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }
    if isinstance(inference_response.predictions, list):
        prediction = inference_response.predictions[0]
        color = class_id_2_color.get(prediction.class_id, "#4892EA")
        draw.rectangle(
            [0, 0, image.size[1], image.size[0]],
            outline=color,
            width=inference_request.visualization_stroke_width,
        )
        text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
        text_size = font.getbbox(text)

        # set button size + 10px margins
        button_size = (text_size[2] + 20, text_size[3] + 20)
        button_img = Image.new("RGBA", button_size, color)
        # put text on button with 10px margins
        button_draw = ImageDraw.Draw(button_img)
        button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

        # put button on source image in position (0, 0)
        image.paste(button_img, (0, 0))
    else:
        if len(inference_response.predictions) > 0:
            box_color = "#4892EA"
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=box_color,
                width=inference_request.visualization_stroke_width,
            )
        row = 0
        predictions = [
            (cls_name, pred)
            for cls_name, pred in inference_response.predictions.items()
        ]
        predictions = sorted(predictions, key=lambda x: x[1].confidence, reverse=True)
        for i, (cls_name, pred) in enumerate(predictions):
            color = class_id_2_color.get(cls_name, "#4892EA")
            text = f"{cls_name} {pred.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, row))
            row += button_size[1]

    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def get_extra_weights_provider_headers() -> Optional[Dict[str, str]]:
    headers = {}
    if GCP_SERVERLESS:
        headers["x-enforce-internal-artefacts-urls"] = "true"
    if ENFORCE_CREDITS_VERIFICATION:
        headers["x-enforce-credits-verification"] = "true"
    if headers:
        return headers
    return None
