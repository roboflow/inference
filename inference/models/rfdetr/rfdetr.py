import os
import time
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime
from PIL import Image

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.responses.inference import InferenceResponseImage
from inference.core.env import (
    DISABLE_PREPROC_AUTO_ORIENT,
    FIX_BATCH_SIZE,
    MAX_BATCH_SIZE,
    ONNXRUNTIME_EXECUTION_PROVIDERS,
    REQUIRED_ONNX_PROVIDERS,
    RFDETR_ONNX_MAX_RESOLUTION,
    TENSORRT_CACHE_PATH,
    USE_PYTORCH_FOR_PREPROCESSING,
)
from inference.core.exceptions import (
    CannotInitialiseModelError,
    ModelArtefactError,
    OnnxProviderNotAvailable,
)
from inference.core.logger import logger
from inference.core.models.defaults import DEFAULT_CONFIDENCE, DEFAUlT_MAX_DETECTIONS
from inference.core.models.instance_segmentation_base import (
    InstanceSegmentationBaseOnnxRoboflowInferenceModel,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Point,
)
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.onnx import has_trt
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import (
    ImageMetaType,
    get_onnxruntime_execution_providers,
    run_session_via_iobinding,
)
from inference.core.utils.postprocess import mask2poly
from inference.core.utils.preprocess import letterbox_image

if USE_PYTORCH_FOR_PREPROCESSING:
    import torch

    CUDA_IS_AVAILABLE = torch.cuda.is_available()

ROBOFLOW_BACKGROUND_CLASS = "background_class83422"


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

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the RFDETR model.

        Returns:
            str: Path to the ONNX weights file.
        """
        return "weights.onnx"

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
        if isinstance(image, Image.Image) and USE_PYTORCH_FOR_PREPROCESSING:
            if CUDA_IS_AVAILABLE:
                np_image = torch.from_numpy(np.asarray(image, copy=False)).cuda()
            else:
                np_image = torch.from_numpy(np.asarray(image, copy=False))
            is_bgr = False
        else:
            np_image, is_bgr = load_image(
                image,
                disable_preproc_auto_orient=disable_preproc_auto_orient
                or "auto-orient" not in self.preproc.keys()
                or DISABLE_PREPROC_AUTO_ORIENT,
            )
        if USE_PYTORCH_FOR_PREPROCESSING:
            if not isinstance(np_image, torch.Tensor):
                np_image = torch.from_numpy(np_image)
            if torch.cuda.is_available():
                np_image = np_image.cuda()

        preprocessed_image, img_dims = self.preprocess_image(
            np_image,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        if USE_PYTORCH_FOR_PREPROCESSING:
            preprocessed_image = (
                preprocessed_image.permute(2, 0, 1).unsqueeze(0).contiguous()
            )
            preprocessed_image = preprocessed_image.float()

            preprocessed_image /= 255.0

            means = torch.tensor(
                self.preprocess_means, device=preprocessed_image.device
            ).view(3, 1, 1)
            stds = torch.tensor(
                self.preprocess_stds, device=preprocessed_image.device
            ).view(3, 1, 1)
            preprocessed_image = (preprocessed_image - means) / stds
        else:
            preprocessed_image = preprocessed_image.astype(np.float32)
            preprocessed_image /= 255.0

            preprocessed_image[:, :, 0] = (
                preprocessed_image[:, :, 0] - self.preprocess_means[0]
            ) / self.preprocess_stds[0]
            preprocessed_image[:, :, 1] = (
                preprocessed_image[:, :, 1] - self.preprocess_means[1]
            ) / self.preprocess_stds[1]
            preprocessed_image[:, :, 2] = (
                preprocessed_image[:, :, 2] - self.preprocess_means[2]
            ) / self.preprocess_stds[2]

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
        if not USE_PYTORCH_FOR_PREPROCESSING:
            img_in = img_in.astype(np.float32)
        else:
            img_in = img_in.float()

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
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )
        bboxes = predictions[0]
        logits = predictions[1]

        return (bboxes, logits)

    def sigmoid_stable(self, x):
        # More efficient, branchless, numerically stable sigmoid computation
        z = np.exp(-np.abs(x))
        return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

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

            # Use argpartition for better performance when max_detections is smaller than logits_flat
            partition_indices = np.argpartition(-logits_flat, max_detections)[
                :max_detections
            ]
            sorted_indices = partition_indices[
                np.argsort(-logits_flat[partition_indices])
            ]
            topk_scores = logits_flat[sorted_indices]

            conf_mask = topk_scores > confidence
            sorted_indices = sorted_indices[conf_mask]
            topk_scores = topk_scores[conf_mask]

            topk_boxes = sorted_indices // num_classes
            topk_labels = sorted_indices % num_classes

            if self.is_one_indexed:
                class_filter_mask = topk_labels != self.background_class_index

                topk_labels[topk_labels > self.background_class_index] -= 1
                topk_scores = topk_scores[class_filter_mask]
                topk_labels = topk_labels[class_filter_mask]
                topk_boxes = topk_boxes[class_filter_mask]

            selected_boxes = bboxes[batch_idx, topk_boxes]

            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            boxes_xyxy = np.concatenate([xy_min, xy_max], axis=1)

            if self.resize_method == "Stretch to":
                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
                boxes_xyxy *= scale_fct
            else:
                input_h, input_w = self.img_size_h, self.img_size_w

                scale = min(input_w / orig_w, input_h / orig_h)
                scaled_w = int(orig_w * scale)
                scaled_h = int(orig_h * scale)

                pad_x = (input_w - scaled_w) / 2
                pad_y = (input_h - scaled_h) / 2

                boxes_input = boxes_xyxy * np.array(
                    [input_w, input_h, input_w, input_h], dtype=np.float32
                )

                boxes_input[:, 0] -= pad_x
                boxes_input[:, 1] -= pad_y
                boxes_input[:, 2] -= pad_x
                boxes_input[:, 3] -= pad_y

                boxes_xyxy = boxes_input / scale

            np.clip(
                boxes_xyxy,
                [0, 0, 0, 0],
                [orig_w, orig_h, orig_w, orig_h],
                out=boxes_xyxy,
            )

            batch_predictions = np.column_stack(
                (
                    boxes_xyxy,
                    topk_scores,
                    np.zeros((len(topk_scores), 1), dtype=np.float32),
                    topk_labels,
                )
            )
            batch_predictions = batch_predictions[
                batch_predictions[:, 6] < len(self.class_names)
            ]

            processed_predictions.append(batch_predictions)

        res = self.make_response(processed_predictions, img_dims, **kwargs)
        return res

    def initialize_model(self, **kwargs) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        logger.debug("Getting model artefacts")
        self.get_model_artifacts(**kwargs)

        input_resolution = self.environment.get("RESOLUTION")
        if input_resolution is None:
            input_resolution = self.preproc.get("resize", {}).get("width")
        if isinstance(input_resolution, (list, tuple)):
            input_resolution = input_resolution[0]
        try:
            input_resolution = int(input_resolution)
        except (TypeError, ValueError):
            input_resolution = None
        if (
            input_resolution is not None
            and input_resolution >= RFDETR_ONNX_MAX_RESOLUTION
        ):
            logger.error(
                "NOT loading '%s' model, input resolution is '%s', ONNX max resolution limit set to '%s' (limit can be increased via RFDETR_ONNX_MAX_RESOLUTION env variable)",
                self.endpoint,
                input_resolution,
                RFDETR_ONNX_MAX_RESOLUTION,
            )
            raise CannotInitialiseModelError(f"Resolution too high for RFDETR")

        logger.debug("Creating inference session")
        if self.load_weights or not self.has_model_metadata:
            t1_session = perf_counter()
            providers = get_onnxruntime_execution_providers(
                ONNXRUNTIME_EXECUTION_PROVIDERS
            )

            if not self.load_weights:
                providers = [
                    "CPUExecutionProvider"
                ]  # "OpenVINOExecutionProvider" dropped until further investigation is done

            try:
                session_options = onnxruntime.SessionOptions()
                session_options.log_severity_level = 3
                # TensorRT does better graph optimization for its EP than onnx
                if has_trt(providers):
                    session_options.graph_optimization_level = (
                        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                expanded_execution_providers = []
                for ep in self.onnxruntime_execution_providers:
                    if ep == "TensorrtExecutionProvider":
                        ep = (
                            "TensorrtExecutionProvider",
                            {
                                "trt_max_workspace_size": str(1 << 30),
                                "trt_engine_cache_enable": True,
                                "trt_engine_cache_path": os.path.join(
                                    TENSORRT_CACHE_PATH, self.endpoint
                                ),
                                "trt_fp16_enable": True,
                                "trt_dump_subgraphs": False,
                                "trt_force_sequential_engine_build": False,
                                "trt_dla_enable": False,
                            },
                        )
                    expanded_execution_providers.append(ep)

                if "OpenVINOExecutionProvider" in expanded_execution_providers:
                    expanded_execution_providers.remove("OpenVINOExecutionProvider")

                self.onnx_session = onnxruntime.InferenceSession(
                    self.cache_file(self.weights_file),
                    providers=expanded_execution_providers,
                    sess_options=session_options,
                )
            except Exception as e:
                self.clear_cache()
                raise ModelArtefactError(
                    f"Unable to load ONNX session. Cause: {e}"
                ) from e
            logger.debug(f"Session created in {perf_counter() - t1_session} seconds")

            inputs = self.onnx_session.get_inputs()[0]
            input_shape = inputs.shape
            self.batch_size = input_shape[0]
            self.img_size_h = input_shape[2]
            self.img_size_w = input_shape[3]
            self.input_name = inputs.name
            if isinstance(self.img_size_h, str) or isinstance(self.img_size_w, str):
                if "resize" in self.preproc:
                    self.img_size_h = int(self.preproc["resize"]["height"])
                    self.img_size_w = int(self.preproc["resize"]["width"])
                else:
                    self.img_size_h = 640
                    self.img_size_w = 640

            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

            model_metadata = {
                "batch_size": self.batch_size,
                "img_size_h": self.img_size_h,
                "img_size_w": self.img_size_w,
            }
            logger.debug(f"Writing model metadata to memcache")
            self.write_model_metadata_to_memcache(model_metadata)
            if not self.load_weights:  # had to load weights to get metadata
                del self.onnx_session
        else:
            if not self.has_model_metadata:
                raise ValueError(
                    "This should be unreachable, should get weights if we don't have model metadata"
                )
            logger.debug(f"Loading model metadata from memcache")
            metadata = self.model_metadata_from_memcache()
            self.batch_size = metadata["batch_size"]
            self.img_size_h = metadata["img_size_h"]
            self.img_size_w = metadata["img_size_w"]
            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

        if ROBOFLOW_BACKGROUND_CLASS in self.class_names:
            self.is_one_indexed = True
            self.background_class_index = self.class_names.index(
                ROBOFLOW_BACKGROUND_CLASS
            )
            self.class_names = (
                self.class_names[: self.background_class_index]
                + self.class_names[self.background_class_index + 1 :]
            )
        else:
            self.is_one_indexed = False
        logger.debug("Model initialisation finished.")

    def validate_model_classes(self) -> None:
        pass


class RFDETRInstanceSegmentation(
    RFDETRObjectDetection, InstanceSegmentationBaseOnnxRoboflowInferenceModel
):
    task_type = "instance-segmentation"

    def initialize_model(self, **kwargs) -> None:
        super().initialize_model(**kwargs)
        mask_shape = self.onnx_session.get_outputs()[2].shape
        self.mask_shape = mask_shape[2:]

    def predict(self, img_in: ImageMetaType, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session with the RFDETR model.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class IDs.
        """
        with self._session_lock:
            predictions = run_session_via_iobinding(
                self.onnx_session, self.input_name, img_in
            )
        bboxes = predictions[0]
        logits = predictions[1]
        masks = predictions[2]

        return (bboxes, logits, masks)

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        bboxes, logits, masks = predictions
        bboxes = bboxes.astype(np.float32)
        logits = logits.astype(np.float32)

        batch_size, num_queries, num_classes = logits.shape
        logits_sigmoid = self.sigmoid_stable(logits)

        img_dims = preproc_return_metadata["img_dims"]

        processed_predictions = []
        processed_masks = []

        for batch_idx in range(batch_size):
            orig_h, orig_w = img_dims[batch_idx]

            logits_flat = logits_sigmoid[batch_idx].reshape(-1)

            # Use argpartition for better performance when max_detections is smaller than logits_flat
            partition_indices = np.argpartition(-logits_flat, max_detections)[
                :max_detections
            ]
            sorted_indices = partition_indices[
                np.argsort(-logits_flat[partition_indices])
            ]
            topk_scores = logits_flat[sorted_indices]

            conf_mask = topk_scores > confidence
            sorted_indices = sorted_indices[conf_mask]
            topk_scores = topk_scores[conf_mask]

            topk_boxes = sorted_indices // num_classes
            topk_labels = sorted_indices % num_classes

            if self.is_one_indexed:
                class_filter_mask = topk_labels != self.background_class_index

                topk_labels[topk_labels > self.background_class_index] -= 1
                topk_scores = topk_scores[class_filter_mask]
                topk_labels = topk_labels[class_filter_mask]
                topk_boxes = topk_boxes[class_filter_mask]

            selected_boxes = bboxes[batch_idx, topk_boxes]
            selected_masks = masks[batch_idx, topk_boxes]

            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            xy_min = cxcy - 0.5 * wh
            xy_max = cxcy + 0.5 * wh
            boxes_xyxy = np.concatenate([xy_min, xy_max], axis=1)

            if self.resize_method == "Stretch to":
                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
                boxes_xyxy *= scale_fct
            else:
                input_h, input_w = self.img_size_h, self.img_size_w

                scale = min(input_w / orig_w, input_h / orig_h)
                scaled_w = int(orig_w * scale)
                scaled_h = int(orig_h * scale)

                pad_x = (input_w - scaled_w) / 2
                pad_y = (input_h - scaled_h) / 2

                boxes_input = boxes_xyxy * np.array(
                    [input_w, input_h, input_w, input_h], dtype=np.float32
                )

                boxes_input[:, 0] -= pad_x
                boxes_input[:, 1] -= pad_y
                boxes_input[:, 2] -= pad_x
                boxes_input[:, 3] -= pad_y

                boxes_xyxy = boxes_input / scale

            np.clip(
                boxes_xyxy,
                [0, 0, 0, 0],
                [orig_w, orig_h, orig_w, orig_h],
                out=boxes_xyxy,
            )

            batch_predictions = np.column_stack(
                (
                    boxes_xyxy,
                    topk_scores,
                    np.zeros((len(topk_scores), 1), dtype=np.float32),
                    topk_labels,
                )
            )
            valid_pred_mask = batch_predictions[:, 6] < len(self.class_names)

            outputs_predictions = []
            outputs_polygons = []
            class_filter_local = kwargs.get("class_filter")
            for i, pred in enumerate(batch_predictions):
                if not valid_pred_mask[i]:
                    continue
                # Early class filtering to avoid unnecessary mask processing
                if class_filter_local:
                    try:
                        pred_class_name = self.class_names[int(pred[6])]
                    except Exception:
                        continue
                    if pred_class_name not in class_filter_local:
                        continue
                mask = selected_masks[i]
                # Per-mask optional upscaling for better polygon quality without retaining all high-res masks
                mask_decode_mode = kwargs.get("mask_decode_mode", "accurate")
                if mask_decode_mode == "accurate":
                    target_res = (orig_w, orig_h)
                    if mask.shape[1] != target_res[0] or mask.shape[0] != target_res[1]:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            target_res,
                            interpolation=cv2.INTER_LINEAR,
                        )
                elif mask_decode_mode == "tradeoff":
                    tradeoff_factor = kwargs.get("tradeoff_factor", 0.0)
                    mask_res = (mask.shape[1], mask.shape[0])  # (w, h)
                    full_res = (orig_w, orig_h)  # (w, h)
                    target_res = (
                        int(
                            mask_res[0] * (1 - tradeoff_factor)
                            + full_res[0] * tradeoff_factor
                        ),
                        int(
                            mask_res[1] * (1 - tradeoff_factor)
                            + full_res[1] * tradeoff_factor
                        ),
                    )
                    if mask.shape[1] != target_res[0] or mask.shape[0] != target_res[1]:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            target_res,
                            interpolation=cv2.INTER_LINEAR,
                        )
                # Ensure binary for polygonization
                mask_bin = (mask > 0).astype(np.uint8)
                points = mask2poly(mask_bin)
                # Scale polygon points back to original image coordinates if needed
                new_points = []
                prediction_h, prediction_w = mask_bin.shape[0], mask_bin.shape[1]
                for point in points:
                    if self.resize_method == "Stretch to":
                        new_x = point[0] * (orig_w / prediction_w)
                        new_y = point[1] * (orig_h / prediction_h)
                    else:
                        scale = max(orig_w / prediction_w, orig_h / prediction_h)
                        pad_x = (orig_w - prediction_w * scale) / 2
                        pad_y = (orig_h - prediction_h * scale) / 2
                        new_x = point[0] * scale + pad_x
                        new_y = point[1] * scale + pad_y
                    new_points.append(np.array([new_x, new_y]))
                outputs_polygons.append(new_points)
                outputs_predictions.append(list(pred))

            processed_predictions.append(outputs_predictions)
            processed_masks.append(outputs_polygons)

        res = self.make_response(
            processed_predictions, processed_masks, img_dims, **kwargs
        )
        return res

    def make_response(
        self,
        predictions: List[List[List[float]]],
        masks: List[List[List[np.ndarray]]],
        img_dims: List[Tuple[int, int]],
        class_filter: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        """Constructs instance segmentation response objects from preprocessed predictions and polygons."""
        # Align to actual number of real images; predictions/masks may include padded slots
        if isinstance(img_dims, dict) and "img_dims" in img_dims:
            img_dims = img_dims["img_dims"]
        effective_len = min(len(img_dims), len(predictions), len(masks))

        responses = []
        for ind in range(effective_len):
            batch_predictions = predictions[ind]
            batch_masks = masks[ind]
            preds_out = []
            for pred, mask in zip(batch_predictions, batch_masks):
                if class_filter and self.class_names[int(pred[6])] not in class_filter:
                    continue
                preds_out.append(
                    InstanceSegmentationPrediction(
                        **{
                            "x": (pred[0] + pred[2]) / 2,
                            "y": (pred[1] + pred[3]) / 2,
                            "width": pred[2] - pred[0],
                            "height": pred[3] - pred[1],
                            "confidence": pred[4],
                            "class": self.class_names[int(pred[6])],
                            "class_id": int(pred[6]),
                            "points": [Point(x=point[0], y=point[1]) for point in mask],
                        }
                    )
                )
            responses.append(
                InstanceSegmentationInferenceResponse(
                    predictions=preds_out,
                    image=InferenceResponseImage(
                        width=img_dims[ind][1], height=img_dims[ind][0]
                    ),
                )
            )
        return responses
