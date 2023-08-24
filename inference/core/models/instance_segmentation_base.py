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
        iou_threshold: float = 0.5,
        mask_decode_mode: str = "accurate",
        max_candidates: int = 3000,
        max_detections: int = 300,
        return_image_dims: bool = False,
        tradeoff_factor: float = 0.5,
        *args,
        **kwargs,
    ) -> Union[
        List[List[List[float]]], Tuple[List[List[List[float]]], List[Tuple[int, int]]]
    ]:
        """Takes an instance segmentation inference request, preprocesses all images, runs inference on all images, and returns the postprocessed detections in the form of inference response objects.

        Args:
            request (InstanceSegmentationInferenceRequest): A request containing 1 to N inference image objects and other inference parameters (confidence, iou threshold, etc.).

        Returns:
            Union[List[InstanceSegmentationInferenceResponse], InstanceSegmentationInferenceResponse]: One to N inference response objects based on the number of inference request images in the inference request object. Each inference response object contains a list of predictions.
        """
        t1 = perf_counter()

        if isinstance(image, list):
            imgs_with_dims = [self.preproc_image(i) for i in image]
            imgs, img_dims = zip(*imgs_with_dims)
            img_in = np.concatenate(imgs, axis=0)
            unwrap = False
        else:
            img_in, img_dims = self.preproc_image(image)
            img_dims = [img_dims]
            unwrap = True

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
                    self.resize_method,
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
