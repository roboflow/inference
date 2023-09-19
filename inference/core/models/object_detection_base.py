from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from inference.core.data_models import (
    InferenceResponseImage,
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
        self,
        image: Any,
        class_agnostic_nms: bool = False,
        confidence: float = 0.5,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        iou_threshold: float = 0.5,
        max_candidates: int = 3000,
        max_detections: int = 300,
        return_image_dims: bool = False,
        **kwargs,
    ) -> Union[
        List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse
    ]:
        """
        Runs object detection inference on one or multiple images and returns the detections.

        Args:
            image (Any): The input image or a list of images to process.
            class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
            confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
            fix_batch_size (bool, optional): If True, fix the batch size for predictions. Useful when the model requires a fixed batch size. Defaults to False.
            max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
            max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
            return_image_dims (bool, optional): Whether to return the dimensions of the processed images along with the predictions. Defaults to False.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[List[ObjectDetectionInferenceResponse], ObjectDetectionInferenceResponse]: One or multiple object detection inference responses based on the number of processed images. Each response contains a list of predictions. If `return_image_dims` is True, it will return a tuple with predictions and image dimensions.

        Raises:
            ValueError: If batching is not enabled for the model and more than one image is passed for processing.
        """
        t1 = perf_counter()
        batch_size = len(image) if isinstance(image, list) else 1
        if not self.batching_enabled and batch_size > 1:
            raise ValueError(
                f"Batching is not enabled for this model, but {batch_size} images were passed in the request"
            )
        img_in, img_dims = self.preprocess(
            image,
            fix_batch_size=fix_batch_size,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        predictions = self.predict(img_in)
        predictions = predictions[:batch_size]
        predictions = self.postprocess(
            predictions,
            img_dims,
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        if return_image_dims:
            return predictions, img_dims
        else:
            return predictions

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
                            "class_id": int(pred[6]),
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
        disable_preproc_static_crop: bool = False,
        iou_threshold: float = 0.5,
        max_candidates: int = 3000,
        max_detections: int = 300,
        **kwargs,
    ) -> List[List[List[float]]]:
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
            predictions,
            infer_shape,
            img_dims,
            self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=disable_preproc_static_crop,
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
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Preprocesses an object detection inference request.

        Args:
            request (ObjectDetectionInferenceRequest): The request object containing images.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: Preprocessed image inputs and corresponding dimensions.
        """
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        img_in /= 255.0

        if self.batching_enabled:
            batch_padding = (
                MAX_BATCH_SIZE - img_in.shape[0]
                if FIX_BATCH_SIZE or fix_batch_size
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
