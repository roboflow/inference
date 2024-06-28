from typing import Any, List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.env import FIX_BATCH_SIZE, MAX_BATCH_SIZE
from inference.core.logger import logger
from inference.core.models.defaults import (
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_CANDIDATES,
    DEFAUlT_MAX_DETECTIONS,
)
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.validate import (
    get_num_classes_from_model_prediction_shape,
)
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import post_process_bboxes


class ObjectDetectionBaseOnnxRoboflowInferenceModel(OnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection model. This class implements an object detection specific infer method."""

    task_type = "object-detection"
    box_format = "xywh"

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        fix_batch_size: bool = False,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> Any:
        """
        Runs object detection inference on one or multiple images and returns the detections.

        Args:
            image (Any): The input image or a list of images to process.
                - can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
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
        return super().infer(
            image,
            class_agnostic_nms=class_agnostic_nms,
            confidence=confidence,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
            iou_threshold=iou_threshold,
            fix_batch_size=fix_batch_size,
            max_candidates=max_candidates,
            max_detections=max_detections,
            return_image_dims=return_image_dims,
            **kwargs,
        )

    def make_response(
        self,
        predictions: List[List[float]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        """Constructs object detection response objects based on predictions.

        Args:
            predictions (List[List[float]]): The list of predictions.
            img_dims (List[Tuple[int, int]]): Dimensions of the images.
            class_filter (Optional[List[str]]): A list of class names to filter, if provided.

        Returns:
            List[ObjectDetectionInferenceResponse]: A list of response objects containing object detection predictions.
        """

        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]

        predictions = predictions[
            : len(img_dims)
        ]  # If the batch size was fixed we have empty preds at the end
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
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        class_agnostic_nms=DEFAULT_CLASS_AGNOSTIC_NMS,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        return_image_dims: bool = False,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
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
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        predictions = predictions[0]
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            box_format=self.box_format,
        )

        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        predictions = post_process_bboxes(
            predictions,
            infer_shape,
            img_dims,
            self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        return self.make_response(predictions, img_dims, **kwargs)

    def preprocess(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
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
            batch_padding = 0
            if FIX_BATCH_SIZE or fix_batch_size:
                if MAX_BATCH_SIZE == float("inf"):
                    logger.warn(
                        "Requested fix_batch_size but MAX_BATCH_SIZE is not set. Using dynamic batching."
                    )
                    batch_padding = 0
                else:
                    batch_padding = MAX_BATCH_SIZE - img_in.shape[0]
            if batch_padding < 0:
                raise ValueError(
                    f"Requested fix_batch_size but passed in {img_in.shape[0]} images "
                    f"when the model's batch size is {MAX_BATCH_SIZE}\n"
                    f"Consider turning off fix_batch_size, changing `MAX_BATCH_SIZE` in"
                    f"your inference server config, or passing at most {MAX_BATCH_SIZE} images at a time"
                )
            width_remainder = img_in.shape[2] % 32
            height_remainder = img_in.shape[3] % 32
            if width_remainder > 0:
                width_padding = 32 - width_remainder
            else:
                width_padding = 0
            if height_remainder > 0:
                height_padding = 32 - height_remainder
            else:
                height_padding = 0
            img_in = np.pad(
                img_in,
                ((0, batch_padding), (0, 0), (0, width_padding), (0, height_padding)),
                "constant",
            )

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "disable_preproc_static_crop": disable_preproc_static_crop,
            }
        )

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray]:
        """Runs inference on the ONNX model.

        Args:
            img_in (np.ndarray): The preprocessed image(s) to run inference on.

        Returns:
            Tuple[np.ndarray]: The ONNX model predictions.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("predict must be implemented by a subclass")

    def validate_model_classes(self) -> None:
        output_shape = self.get_model_output_shape()
        num_classes = get_num_classes_from_model_prediction_shape(
            output_shape[2], masks=0
        )
        try:
            assert num_classes == self.num_classes
        except AssertionError:
            raise ValueError(
                f"Number of classes in model ({num_classes}) does not match the number of classes in the environment ({self.num_classes})"
            )
