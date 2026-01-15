from typing import List, Optional, Tuple, Union

import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.exceptions import InvalidMaskDecodeArgument
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.onnx import run_session_via_iobinding
from inference.core.utils.postprocess import (
    masks2poly,
    post_process_bboxes,
    post_process_polygons,
    process_mask_accurate,
    process_mask_fast,
    process_mask_tradeoff,
)
from inference.models.yolov11.yolov11_instance_segmentation import (
    YOLOv11InstanceSegmentation,
)

DEFAULT_CONFIDENCE = 0.4
DEFAULT_MASK_DECODE_MODE = "accurate"
DEFAULT_TRADEOFF_FACTOR = 0.0


class YOLO26InstanceSegmentation(YOLOv11InstanceSegmentation):
    """YOLO26 Instance Segmentation model with end-to-end ONNX output.

    YOLO26 exports with NMS already applied, outputting:
    - predictions: (batch, num_detections, 38) where 38 = 6 + 32 mask coefficients
      Format: [x1, y1, x2, y2, confidence, class_index, mask_coeff_0, ..., mask_coeff_31]
    - protos: (batch, 32, H, W) mask prototypes
    """

    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and mask prototypes.
        """
        with self._session_lock:
            predictions, protos = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )


        return predictions, protos

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, np.ndarray],
        preprocess_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        mask_decode_mode: str = DEFAULT_MASK_DECODE_MODE,
        tradeoff_factor: float = DEFAULT_TRADEOFF_FACTOR,
        **kwargs,
    ) -> Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
    ]:
        """Postprocesses the instance segmentation predictions.

        YOLO26 predictions come with NMS already applied, so we just need to:
        1. Filter by confidence
        2. Decode masks
        3. Format response
        """
        predictions, protos = predictions
        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preprocess_return_metadata["img_dims"]
        img_in_shape = preprocess_return_metadata["im_shape"]

        # Filter by confidence and process each batch
        masks = []
        filtered_predictions = []

        for batch_idx, batch_preds in enumerate(predictions):
            # Filter by confidence (conf is at index 4)
            keep = batch_preds[:, 4] > confidence
            batch_preds = batch_preds[keep]
            filtered_predictions.append(batch_preds)

            if batch_preds.size == 0:
                masks.append([])
                continue

            # Get mask coefficients (starting at index 6)
            mask_coeffs = batch_preds[:, 6:]
            boxes = batch_preds[:, :4]
            proto = protos[batch_idx]
            img_dim = img_dims[batch_idx]

            # Decode masks based on mode
            if mask_decode_mode == "accurate":
                batch_masks = process_mask_accurate(
                    proto, mask_coeffs, boxes, img_in_shape[2:]
                )
                output_mask_shape = img_in_shape[2:]
            elif mask_decode_mode == "tradeoff":
                if not 0 <= tradeoff_factor <= 1:
                    raise InvalidMaskDecodeArgument(
                        f"Invalid tradeoff_factor: {tradeoff_factor}. Must be in [0.0, 1.0]"
                    )
                batch_masks = process_mask_tradeoff(
                    proto, mask_coeffs, boxes, img_in_shape[2:], tradeoff_factor
                )
                output_mask_shape = batch_masks.shape[1:]
            elif mask_decode_mode == "fast":
                batch_masks = process_mask_fast(
                    proto, mask_coeffs, boxes, img_in_shape[2:]
                )
                output_mask_shape = batch_masks.shape[1:]
            else:
                raise InvalidMaskDecodeArgument(
                    f"Invalid mask_decode_mode: {mask_decode_mode}. Must be one of ['accurate', 'fast', 'tradeoff']"
                )

            # Convert masks to polygons
            polys = masks2poly(batch_masks)

            # Post-process bounding boxes
            batch_preds[:, :4] = post_process_bboxes(
                [boxes],
                infer_shape,
                [img_dim],
                self.preproc,
                resize_method=self.resize_method,
                disable_preproc_static_crop=preprocess_return_metadata[
                    "disable_preproc_static_crop"
                ],
            )[0]

            # Post-process polygons
            polys = post_process_polygons(
                img_dim,
                polys,
                output_mask_shape,
                self.preproc,
                resize_method=self.resize_method,
            )
            masks.append(polys)
            filtered_predictions[batch_idx] = batch_preds

        return self.make_response(filtered_predictions, masks, img_dims, **kwargs)

    def make_response(
        self,
        predictions: List[np.ndarray],
        masks: List[List[List[Tuple[float, float]]]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        """Constructs instance segmentation response objects.

        YOLO26 prediction format: [x1, y1, x2, y2, conf, class_idx, mask_coeffs...]
        """
        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]

        responses = []
        for ind, (batch_predictions, batch_masks) in enumerate(zip(predictions, masks)):
            preds_list = []
            for pred, mask in zip(batch_predictions, batch_masks):
                class_idx = int(pred[5])  # class index is at position 5
                if class_filter and self.class_names[class_idx] not in class_filter:
                    continue
                preds_list.append(
                    InstanceSegmentationPrediction(
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "points": [Point(x=point[0], y=point[1]) for point in mask],
                            "confidence": pred[4],
                            "class": self.class_names[class_idx],
                            "class_id": class_idx,
                        }
                    )
                )
            responses.append(
                InstanceSegmentationInferenceResponse(
                    predictions=preds_list,
                    image=InferenceResponseImage(
                        width=img_dims[ind][1], height=img_dims[ind][0]
                    ),
                )
            )
        return responses

    def validate_model_classes(self) -> None:
        pass
