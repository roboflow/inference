from time import perf_counter
from typing import Any, List, Tuple, Union

import numpy as np

from inference.core.data_models import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.exceptions import InvalidMaskDecodeArgument
from inference.core.models.mixins import InstanceSegmentationMixin
from inference.core.models.roboflow import OnnxRoboflowInferenceModel
from inference.core.nms import w_np_non_max_suppression
from inference.core.utils.postprocess import (
    mask2poly,
    postprocess_predictions,
    process_mask_accurate,
    process_mask_fast,
    process_mask_tradeoff,
    scale_polys,
)


class InstanceSegmentationBaseOnnxRoboflowInferenceModel(
    OnnxRoboflowInferenceModel, InstanceSegmentationMixin
):
    """Roboflow ONNX Instance Segmentation model.

    This class implements an instance segmentation specific inference method
    for ONNX models provided by Roboflow.
    """

    def infer(
        self,
        image: Any,
        class_agnostic_nms: bool = False,
        confidence: float = 0.5,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        iou_threshold: float = 0.5,
        mask_decode_mode: str = "accurate",
        max_candidates: int = 3000,
        max_detections: int = 300,
        return_image_dims: bool = False,
        tradeoff_factor: float = 0.5,
        **kwargs,
    ) -> Union[
        List[List[List[float]]], Tuple[List[List[List[float]]], List[Tuple[int, int]]]
    ]:
        """
        Process an image or list of images for instance segmentation.

        Args:
            image (Any): An image or a list of images for processing.
            class_agnostic_nms (bool, optional): Whether to use class-agnostic non-maximum suppression. Defaults to False.
            confidence (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            iou_threshold (float, optional): IoU threshold for non-maximum suppression. Defaults to 0.5.
            mask_decode_mode (str, optional): Decoding mode for masks. Choices are "accurate", "tradeoff", and "fast". Defaults to "accurate".
            max_candidates (int, optional): Maximum number of candidate detections. Defaults to 3000.
            max_detections (int, optional): Maximum number of detections after non-maximum suppression. Defaults to 300.
            return_image_dims (bool, optional): Whether to return the dimensions of the processed images. Defaults to False.
            tradeoff_factor (float, optional): Tradeoff factor used when `mask_decode_mode` is set to "tradeoff". Must be in [0.0, 1.0]. Defaults to 0.5.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.
            **kwargs: Additional parameters to customize the inference process.

        Returns:
            Union[List[List[List[float]]], Tuple[List[List[List[float]]], List[Tuple[int, int]]]]: The list of predictions, with each prediction being a list of lists. Optionally, also returns the dimensions of the processed images.

        Raises:
            InvalidMaskDecodeArgument: If an invalid `mask_decode_mode` is provided or if the `tradeoff_factor` is outside the allowed range.

        Notes:
            - Processes input images and normalizes them.
            - Makes predictions using the ONNX runtime.
            - Applies non-maximum suppression to the predictions.
            - Decodes the masks according to the specified mode.
        """
        t1 = perf_counter()

        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        img_in /= 255.0
        predictions, protos = self.infer_onnx(img_in)
        predictions = w_np_non_max_suppression(
            predictions,
            conf_thresh=confidence,
            iou_thresh=iou_threshold,
            class_agnostic=class_agnostic_nms,
            max_detections=max_detections,
            max_candidate_detections=max_candidates,
            num_masks=32,
        )
        infer_shape = (self.img_size_w, self.img_size_h)
        predictions = np.array(predictions)
        masks = []
        if predictions.shape[1] > 0:
            for i, (pred, proto, img_dim) in enumerate(
                zip(predictions, protos, img_dims)
            ):
                if mask_decode_mode == "accurate":
                    batch_masks = process_mask_accurate(
                        proto, pred[:, 7:], pred[:, :4], img_in.shape[2:]
                    )
                    output_mask_shape = img_in.shape[2:]
                elif mask_decode_mode == "tradeoff":
                    if not 0 <= tradeoff_factor <= 1:
                        raise InvalidMaskDecodeArgument(
                            f"Invalid tradeoff_factor: {tradeoff_factor}. Must be in [0.0, 1.0]"
                        )
                    batch_masks = process_mask_tradeoff(
                        proto,
                        pred[:, 7:],
                        pred[:, :4],
                        img_in.shape[2:],
                        tradeoff_factor,
                    )
                    output_mask_shape = batch_masks.shape[1:]
                elif mask_decode_mode == "fast":
                    batch_masks = process_mask_fast(
                        proto, pred[:, 7:], pred[:, :4], img_in.shape[2:]
                    )
                    output_mask_shape = batch_masks.shape[1:]
                else:
                    raise InvalidMaskDecodeArgument(
                        f"Invalid mask_decode_mode: {mask_decode_mode}. Must be one of ['accurate', 'fast', 'tradeoff']"
                    )
                polys = mask2poly(batch_masks)
                pred[:, :4] = postprocess_predictions(
                    [pred[:, :4]],
                    infer_shape,
                    [img_dim],
                    self.preproc,
                    resize_method=self.resize_method,
                    disable_preproc_static_crop=disable_preproc_static_crop,
                )[0]
                polys = scale_polys(
                    output_mask_shape,
                    polys,
                    img_dim,
                    self.preproc,
                    resize_method=self.resize_method,
                )
                masks.append(polys)
        else:
            masks.append([])
        if return_image_dims:
            return predictions, masks, img_dims
        else:
            return predictions, masks

    def make_response(
        self,
        predictions: List[List[List[float]]],
        masks: List[List[List[float]]],
        img_dims: List[Tuple[int, int]],
        class_filter: List[str] = [],
    ) -> Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
    ]:
        """
        Create instance segmentation inference response objects for the provided predictions and masks.

        Args:
            predictions (List[List[List[float]]]): List of prediction data, one for each image.
            masks (List[List[List[float]]]): List of masks corresponding to the predictions.
            img_dims (List[Tuple[int, int]]): List of image dimensions corresponding to the processed images.
            class_filter (List[str], optional): List of class names to filter predictions by. Defaults to an empty list (no filtering).

        Returns:
            Union[InstanceSegmentationInferenceResponse, List[InstanceSegmentationInferenceResponse]]: A single instance segmentation response or a list of instance segmentation responses based on the number of processed images.

        Notes:
            - For each image, constructs an `InstanceSegmentationInferenceResponse` object.
            - Each response contains a list of `InstanceSegmentationPrediction` objects.
        """
        responses = [
            InstanceSegmentationInferenceResponse(
                predictions=[
                    InstanceSegmentationPrediction(
                        # Passing args as a dictionary here since one of the args is 'class' (a protected term in Python)
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "points": [Point(x=point[0], y=point[1]) for point in mask],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                        }
                    )
                    for pred, mask in zip(batch_predictions, batch_masks)
                    if not class_filter
                    or self.class_names[int(pred[6])] in class_filter
                ],
                image=InferenceResponseImage(
                    width=img_dims[ind][1], height=img_dims[ind][0]
                ),
            )
            for ind, (batch_predictions, batch_masks) in enumerate(
                zip(predictions, masks)
            )
        ]
        return responses

    def infer_onnx(self, img_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs inference on the ONNX model.

        Args:
            img_in (np.ndarray): The preprocessed image(s) to run inference on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The ONNX model predictions and the ONNX model protos.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("infer_onnx must be implemented by a subclass")
