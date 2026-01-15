from typing import List, Tuple, Optional
import numpy as np

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.keypoints import model_keypoints_to_response
from inference.core.utils.postprocess import post_process_bboxes, post_process_keypoints
from inference.core.utils.onnx import run_session_via_iobinding
from inference.models.yolov11.yolov11_keypoints_detection import (
    YOLOv11KeypointsDetection,
)

DEFAULT_CONFIDENCE = 0.4


class YOLO26KeypointsDetection(YOLOv11KeypointsDetection):
    """YOLO26 Keypoints Detection model with end-to-end ONNX output.
    
    YOLO26 exports with NMS already applied, outputting:
    - predictions: (batch, num_detections, 57) for COCO pose (17 keypoints * 3 + 6)
      Format: [x1, y1, x2, y2, confidence, class_index, kp0_x, kp0_y, kp0_conf, ...]
    """
    
    def predict(self, img_in: np.ndarray, **kwargs) -> Tuple[np.ndarray, ...]:
        """Performs inference on the given image using the ONNX session.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: Predictions with boxes, confidence, class, and keypoints.
        """
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )[0]
        
        # YOLO26 end-to-end format: (batch, num_det, 6 + num_keypoints*3)
        # [x1, y1, x2, y2, conf, class_idx, keypoints...]
        boxes = predictions[:, :, :4]
        confs = predictions[:, :, 4:5]
        class_indices = predictions[:, :, 5:6]
        keypoints = predictions[:, :, 6:]
        
        # Reformat to match expected format: [boxes, conf, class_idx, keypoints]
        predictions = np.concatenate([boxes, confs, class_indices, keypoints], axis=2)
        
        return (predictions,)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray],
        preproc_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Postprocesses the keypoints detection predictions.
        
        YOLO26 predictions come with NMS already applied, so we just need to:
        1. Filter by confidence
        2. Scale coordinates to original image size
        3. Format response
        """
        predictions = predictions[0]
        infer_shape = (self.img_size_h, self.img_size_w)
        img_dims = preproc_return_metadata["img_dims"]
        
        # Number of keypoint values (x, y, conf per keypoint)
        num_keypoint_values = predictions.shape[2] - 6  # subtract boxes(4) + conf(1) + class(1)
        
        # Filter by confidence and process each batch
        filtered_predictions = []
        for batch_preds in predictions:
            # Filter by confidence (conf is at index 4)
            keep = batch_preds[:, 4] > confidence
            batch_preds = batch_preds[keep]
            filtered_predictions.append(batch_preds)
        
        # Post-process bounding boxes
        filtered_predictions = post_process_bboxes(
            predictions=filtered_predictions,
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        
        # Post-process keypoints
        filtered_predictions = post_process_keypoints(
            predictions=filtered_predictions,
            keypoints_start_index=6,  # keypoints start at index 6
            infer_shape=infer_shape,
            img_dims=img_dims,
            preproc=self.preproc,
            resize_method=self.resize_method,
            disable_preproc_static_crop=preproc_return_metadata[
                "disable_preproc_static_crop"
            ],
        )
        
        return self.make_response(filtered_predictions, img_dims, **kwargs)

    def make_response(
        self,
        predictions: List[np.ndarray],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Constructs keypoints detection response objects.
        
        YOLO26 prediction format: [x1, y1, x2, y2, conf, class_idx, keypoints...]
        """
        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]
        
        keypoint_confidence_threshold = 0.0
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
            
        responses = []
        for ind, batch_predictions in enumerate(predictions):
            preds_list = []
            for pred in batch_predictions:
                class_idx = int(pred[5])  # class index is at position 5
                if class_filter and self.class_names[class_idx] not in class_filter:
                    continue
                    
                # Keypoints start at index 6
                keypoints_data = pred[6:]
                
                preds_list.append(
                    KeypointsPrediction(
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[class_idx],
                            "class_id": class_idx,
                            "keypoints": model_keypoints_to_response(
                                keypoints_metadata=self.keypoints_metadata,
                                keypoints=keypoints_data,
                                predicted_object_class_id=class_idx,
                                keypoint_confidence_threshold=keypoint_confidence_threshold,
                            ),
                        }
                    )
                )
            responses.append(
                KeypointsDetectionInferenceResponse(
                    predictions=preds_list,
                    image=InferenceResponseImage(
                        width=img_dims[ind][1], height=img_dims[ind][0]
                    ),
                )
            )
        return responses

    def validate_model_classes(self) -> None:
        pass
