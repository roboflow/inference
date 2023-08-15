from time import perf_counter
from typing import List, Optional, Tuple, Union

import numpy as np

from inference.core.data_models import (
    InferenceResponseImage,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import FIX_BATCH_SIZE, MAX_BATCH_SIZE
from inference.core.models.mixins import ObjectDetectionMixin
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import postprocess_predictions


class ObjectDetectionBaseOnnxRoboflowInferenceModel(
    OnnxRoboflowInferenceModel, ObjectDetectionMixin
):
    """Roboflow ONNX Object detection model. This class implements an object detection specific infer method."""

    def infer(
        self, request: ObjectDetectionInferenceRequest
    ) -> Union[
        List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse
    ]:
        """Runs object detection inference on given images and returns the detections.

        Args:
            request (ObjectDetectionInferenceRequest): A request containing 1 to N inference image objects and other inference parameters (confidence, iou threshold, etc.)

        Returns:
            Union[List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse]: One to N inference response objects based on the number of inference request images in the request object, each response contains a list of predictions.

        Raises:
            ValueError: If batching is not enabled for the model and more than one image is passed in the request.
        """
        t1 = perf_counter()
        batch_size = len(request.image) if isinstance(request.image, list) else 1
        if not self.batching_enabled and batch_size > 1:
            raise ValueError(
                f"Batching is not enabled for this model, but {batch_size} images were passed in the request"
            )
        img_in, img_dims = self.preprocess(request)
        predictions = self.predict(img_in)
        predictions = predictions[:batch_size]
        request_dict = request.dict()
        predictions = self.postprocess(predictions, img_dims, **request_dict)
        responses = self.make_response(predictions, img_dims, **request_dict)
        for response in responses:
            response.time = perf_counter() - t1

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

        if self.unwrap:
            responses = responses[0]

        return responses

    def make_response(
        self,
        predictions: List[List[float]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[ObjectDetectionInferenceResponse]: A list of response objects containing object detection predictions.
        """
        responses = [
            ObjectDetectionInferenceResponse(
                predictions=[
                    ObjectDetectionPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                        }
                    )
                    for pred in batch_predictions
                    if not class_filter
                    or self.class_names[int(pred[6])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, batch_predictions in enumerate(predictions)
        ]
        return responses

    def postprocess(
        self,
        predictions: np.ndarray,
        img_dims: List[Tuple[int, int]],
        class_agnostic_nms: bool = False,
        confidence: float = 0.5,
        iou_threshold: float = 0.5,
        max_candidates: int = 3000,
        max_detections: int = 300,
        *args,
        **kwargs,
    ) -> List[List[float]]:
        """Postprocesses the object detection predictions.

        Args:
            predictions (np.ndarray): Raw predictions from the model.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression. Default is False.
            confidence (float): Confidence threshold for filtering detections. Default is 0.5.
            iou_threshold (float): IoU threshold for non-max suppression. Default is 0.5.
            max_candidates (int): Maximum number of candidate detections. Default is 3000.
            max_detections (int): Maximum number of final detections. Default is 300.

        Returns:
            List[List[float]]: The postprocessed predictions.
        """
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
        )

        infer_shape = (self.img_size_w, self.img_size_h)
        predictions = postprocess_predictions(
            predictions, infer_shape, img_dims, self.preproc, self.resize_method
        )
        return predictions

    def predict(self, img_in: np.ndarray) -> np.ndarray:
        """Runs the prediction for object detection. This method must be implemented by a subclass.

        Args:
            img_in (np.ndarray): Preprocessed image input.

        Returns:
            np.ndarray: Raw model predictions.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError

    def preprocess(
        self, request: ObjectDetectionInferenceRequest
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Preprocesses an object detection inference request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object containing images.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: Preprocessed image inputs and corresponding dimensions.
        """
        if isinstance(request.image, list):
            imgs_with_dims = [self.preproc_image(i) for i in request.image]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
            self.unwrap = False
        else:
            img_in, img_dims = self.preproc_image(request.image)
            img_dims = [img_dims]
            self.unwrap = True

        img_in /= 255.0

        if self.batching_enabled:
            batch_padding = (
                MAX_BATCH_SIZE - img_in.shape[0]
                if FIX_BATCH_SIZE or request.fix_batch_size
                else 0
            )
            width_padding = 32 - (img_in.shape[2] % 32)
            height_padding = 32 - (img_in.shape[3] % 32)
            img_in = np.pad(
                img_in,
                ((0, batch_padding), (0, 0), (0, width_padding), (0, height_padding)),
                "constant",
            )

        return img_in, img_dims
