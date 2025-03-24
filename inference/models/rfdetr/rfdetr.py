import os
from typing import Any, List, Tuple

import numpy as np

from inference.core.logger import logger
from inference.core.models.defaults import (
    DEFAULT_CLASS_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU_THRESH,
    DEFAULT_MAX_CANDIDATES,
    DEFAUlT_MAX_DETECTIONS,
)
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding


class RFDETRObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection with the RFDETR model.

    This class is responsible for performing object detection using the RFDETR model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    preprocess_means = [0.485, 0.456, 0.406]
    preprocess_stds = [0.229, 0.224, 0.225]

    COCO_CLASSES = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }


    @property
    def weights_file(self) -> str:
        """Gets the weights file for the RFDETR model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return os.path.join(self.cache_dir, "weights.onnx")

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
        self.resize_method = "Stretch to"
        return super().preprocess(
            image,
            disable_preproc_auto_orient,
            disable_preproc_contrast,
            disable_preproc_grayscale,
            disable_preproc_static_crop,
            fix_batch_size,
            **kwargs,
        )

    def predict(self, img_in: ImageMetaType, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session with the RFDETR model.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class IDs.
        """
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )

        bboxes = predictions[0]
        logits = predictions[1]

        return (bboxes, logits)

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
            preproc_return_metadata (PreprocessReturnMetadata): Metadata from preprocessing.
            class_agnostic_nms (bool): Whether to apply class-agnostic non-max suppression.
            confidence (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for non-max suppression.
            max_candidates (int): Maximum number of candidate detections.
            max_detections (int): Maximum number of final detections.

        Returns:
            List[ObjectDetectionInferenceResponse]: The post-processed predictions.
        """
        bboxes = predictions[0]
        logits = predictions[1]

        logits = 1 / (1 + np.exp(-logits))

        img_dims = preproc_return_metadata["img_dims"]

        processed_predictions = []

        for batch_idx in range(bboxes.shape[0]):
            orig_h, orig_w = img_dims[batch_idx]

            boxes = bboxes[batch_idx]
            image_logits = logits[batch_idx]

            scores = np.max(image_logits, axis=1)
            labels = np.argmax(image_logits, axis=1)

            confidence_mask = scores > confidence
            filtered_boxes = boxes[confidence_mask]
            filtered_scores = scores[confidence_mask]
            filtered_labels = labels[confidence_mask]

            converted_boxes = []
            for box in filtered_boxes:
                cx, cy, w, h = box
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h])
                converted_box = np.array([x1, y1, x2, y2]) * scale_fct

                converted_boxes.append(converted_box)

            converted_boxes = np.array(converted_boxes)

            if len(converted_boxes) > 0:
                batch_predictions = np.column_stack(
                    (
                        converted_boxes,
                        filtered_scores.reshape(-1, 1),
                        np.zeros((len(filtered_scores), 1)),
                        filtered_labels.reshape(-1, 1),
                    )
                )
            else:
                batch_predictions = np.zeros((0, 7))

            processed_predictions.append(batch_predictions)

        return self.make_response(processed_predictions, img_dims, **kwargs)

    def validate_model_classes(self) -> None:
        pass
