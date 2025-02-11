from typing import Tuple

import cv2
import numpy as np
import onnxruntime


class LocalCODETRObjectDetection:
    """Local ONNX Object detection model for CO-DETR without Roboflow dependencies."""

    def __init__(self, onnx_path: str):
        """Initialize the model with a local ONNX file.

        Args:
            onnx_path (str): Path to the ONNX model file
        """
        # Initialize ONNX session
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.onnx_session.get_inputs()[0].name

        # COCO classes (you can modify this based on your model)
        with open("coco_classes.txt", "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the input image."""
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and pad image
        target_size = (800, 800)  # Adjust based on your model's requirements
        h, w = img.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Create padding
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Normalize and transpose
        img_input = img_padded.transpose(2, 0, 1).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension
        img_input = img_input / 255.0  # Normalize

        return img_input

    def postprocess(
        self, predictions: np.ndarray, confidence_threshold: float = 0.3
    ) -> list:
        """Postprocess the raw predictions."""
        # Extract boxes and scores
        boxes = predictions[0, :, :4]
        scores = predictions[0, :, 4:]

        # Get confidence scores and class ids
        confidences = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Filter by confidence
        mask = confidences > confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        results = []
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            results.append(
                {
                    "bbox": box,
                    "confidence": float(conf),
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id],
                }
            )

        return results

    def infer(self, img: np.ndarray) -> list:
        """Run inference on an image.

        Args:
            img (np.ndarray): Input image in BGR format

        Returns:
            list: List of detection results
        """
        # Preprocess
        img_input = self.preprocess(img)

        # Run inference
        predictions = self.onnx_session.run(None, {self.input_name: img_input})[0]

        # Postprocess
        results = self.postprocess(predictions)

        return results
