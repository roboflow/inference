import os
from typing import Any, List, Tuple, Union

import cv2
import numpy as np

from inference.core.entities.requests.inference import (
    InferenceRequest,
    InferenceRequestImage,
)
from inference.core.env import (
    DISABLE_PREPROC_AUTO_ORIENT,
    FIX_BATCH_SIZE,
    MAX_BATCH_SIZE,
    USE_PYTORCH_FOR_PREPROCESSING,
)
from inference.core.logger import logger
from inference.core.models.defaults import (
    DEFAULT_CONFIDENCE,
    DEFAUlT_MAX_DETECTIONS,
)
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding
from inference.core.utils.preprocess import letterbox_image

if USE_PYTORCH_FOR_PREPROCESSING:
    import torch


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

    def preproc_image(
        self,
        image: Union[Any, InferenceRequestImage],
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses an inference request image by loading it, then applying any pre-processing specified by the Roboflow platform, then scaling it to the inference input dimensions.

        Args:
            image (Union[Any, InferenceRequestImage]): An object containing information necessary to load the image for inference.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing a numpy array of the preprocessed image pixel data and a tuple of the images original size.
        """
        np_image, is_bgr = load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient
            or "auto-orient" not in self.preproc.keys()
            or DISABLE_PREPROC_AUTO_ORIENT,
        )
        preprocessed_image, img_dims = self.preprocess_image(
            np_image,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        preprocessed_image = preprocessed_image.astype(np.float32)
        preprocessed_image /= 255.0

        for i in range(3):
            preprocessed_image[:, :, i] = (
                preprocessed_image[:, :, i] - means[i]
            ) / stds[i]

        if USE_PYTORCH_FOR_PREPROCESSING:
            preprocessed_image = torch.from_numpy(
                np.ascontiguousarray(preprocessed_image)
            )
            if torch.cuda.is_available():
                preprocessed_image = preprocessed_image.cuda()
            preprocessed_image = (
                preprocessed_image.permute(2, 0, 1).unsqueeze(0).contiguous().float()
            )

        if self.resize_method == "Stretch to":
            if isinstance(preprocessed_image, np.ndarray):
                preprocessed_image = preprocessed_image.astype(np.float32)
                resized = cv2.resize(
                    preprocessed_image,
                    (self.img_size_w, self.img_size_h),
                )
            elif USE_PYTORCH_FOR_PREPROCESSING:
                resized = torch.nn.functional.interpolate(
                    preprocessed_image,
                    size=(self.img_size_h, self.img_size_w),
                    mode="bilinear",
                )
            else:
                raise ValueError(
                    f"Received an image of unknown type, {type(preprocessed_image)}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        elif self.resize_method == "Fit (black edges) in":
            resized = letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h)
            )
        elif self.resize_method == "Fit (white edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(255, 255, 255),
            )
        elif self.resize_method == "Fit (grey edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(114, 114, 114),
            )

        if is_bgr:
            if isinstance(resized, np.ndarray):
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                resized = resized[:, [2, 1, 0], :, :]

        if isinstance(resized, np.ndarray):
            img_in = np.transpose(resized, (2, 0, 1))
            img_in = img_in.astype(np.float32)
            img_in = np.expand_dims(img_in, axis=0)
        elif USE_PYTORCH_FOR_PREPROCESSING:
            img_in = resized.float()
        else:
            raise ValueError(
                f"Received an image of unknown type, {type(resized)}; "
                "This is most likely a bug. Contact Roboflow team through github issues "
                "(https://github.com/roboflow/inference/issues) providing full context of the problem"
            )
        return img_in, img_dims

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
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        img_in = img_in.astype(np.float32)

        if self.batching_enabled:
            batch_padding = 0
            if FIX_BATCH_SIZE or fix_batch_size:
                if MAX_BATCH_SIZE == float("inf"):
                    logger.warning(
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

            if isinstance(img_in, np.ndarray):
                img_in = np.pad(
                    img_in,
                    (
                        (0, batch_padding),
                        (0, 0),
                        (0, width_padding),
                        (0, height_padding),
                    ),
                    "constant",
                )
            elif USE_PYTORCH_FOR_PREPROCESSING:
                img_in = torch.nn.functional.pad(
                    img_in,
                    (
                        0,
                        height_padding,
                        0,
                        width_padding,
                        0,
                        0,
                        0,
                        batch_padding,
                    ),
                    mode="constant",
                    value=0,
                )
            else:
                raise ValueError(
                    f"Received an image of unknown type, {type(img_in)}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "disable_preproc_static_crop": disable_preproc_static_crop,
            }
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

    def sigmoid_stable(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        bboxes, logits = predictions
        bboxes = bboxes.astype(np.float32)
        logits = logits.astype(np.float32)

        batch_size, num_queries, num_classes = logits.shape
        logits_sigmoid = self.sigmoid_stable(logits)

        img_dims = preproc_return_metadata["img_dims"]

        processed_predictions = []

        for batch_idx in range(batch_size):
            orig_h, orig_w = img_dims[batch_idx]

            logits_flat = logits_sigmoid[batch_idx].reshape(-1)

            sorted_indices = np.argsort(-logits_flat)[:max_detections]
            topk_scores = logits_flat[sorted_indices]

            conf_mask = topk_scores > confidence
            sorted_indices = sorted_indices[conf_mask]
            topk_scores = topk_scores[conf_mask]

            topk_boxes = sorted_indices // num_classes
            topk_labels = sorted_indices % num_classes

            selected_boxes = bboxes[batch_idx, topk_boxes]

            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            boxes_xyxy = np.concatenate([xy_min, xy_max], axis=1)

            scale_fct = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
            boxes_xyxy *= scale_fct

            batch_predictions = np.column_stack(
                (
                    boxes_xyxy,
                    topk_scores,
                    np.zeros((len(topk_scores), 1), dtype=np.float32),
                    topk_labels,
                )
            )

            processed_predictions.append(batch_predictions)

        return self.make_response(processed_predictions, img_dims, **kwargs)

    def validate_model_classes(self) -> None:
        pass
