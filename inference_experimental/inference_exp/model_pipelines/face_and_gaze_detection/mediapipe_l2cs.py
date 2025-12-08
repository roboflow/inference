from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, KeyPoints
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import ModelPipelineInitializationError, ModelRuntimeError
from inference_exp.models.l2cs.l2cs_onnx import (
    DEFAULT_GAZE_MAX_BATCH_SIZE,
    L2CSGazeDetection,
    L2CSNetOnnx,
)
from inference_exp.models.mediapipe_face_detection.face_detection import (
    MediaPipeFaceDetector,
)


class FaceAndGazeDetectionMPAndL2CS:

    @classmethod
    def with_models(
        cls, models: List[Any], **kwargs
    ) -> "FaceAndGazeDetectionMPAndL2CS":
        if len(models) != 2:
            raise ModelPipelineInitializationError(
                message="Model pipeline `face-and-gaze-detection` requires two models tu run - face detector "
                "and gaze detector. If you run `inference` locally, verify the parameter of pipeline loader "
                "to make sure that two models parameters' are provided. If you use Roboflow hosted solution, "
                "contact us to get help.",
                help_url="https://todo",
            )
        face_detector, gaze_detector = models
        if not isinstance(face_detector, MediaPipeFaceDetector):
            raise ModelPipelineInitializationError(
                message="Model pipeline `face-and-gaze-detection` requires first model to be `MediaPipeFaceDetector` - "
                "if you run `inference` locally, make sure that you initialized the pipeline pointing model of "
                "matching type.",
                help_url="https://todo",
            )
        if not isinstance(gaze_detector, L2CSNetOnnx):
            raise ModelPipelineInitializationError(
                message="Model pipeline `face-and-gaze-detection` requires second model to be `L2CSNet` - "
                "if you run `inference` locally, make sure that you initialized the pipeline pointing model of "
                "matching type.",
                help_url="https://todo",
            )
        return FaceAndGazeDetectionMPAndL2CS.from_pretrained(
            face_detector=face_detector, gaze_detector=gaze_detector, **kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        face_detector: Union[str, MediaPipeFaceDetector],
        gaze_detector: Union[str, L2CSNetOnnx],
        onnx_execution_providers: Optional[List[Union[str, tuple]]] = None,
        default_onnx_trt_options: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = DEFAULT_GAZE_MAX_BATCH_SIZE,
        **kwargs,
    ) -> "FaceAndGazeDetectionMPAndL2CS":
        if isinstance(face_detector, str):
            face_detector = MediaPipeFaceDetector.from_pretrained(
                model_name_or_path=face_detector
            )
        if isinstance(gaze_detector, str):
            gaze_detector = L2CSNetOnnx.from_pretrained(
                model_name_or_path=gaze_detector,
                onnx_execution_providers=onnx_execution_providers,
                default_onnx_trt_options=default_onnx_trt_options,
                device=device,
                max_batch_size=max_batch_size,
            )
        return cls(
            face_detector=face_detector,
            gaze_detector=gaze_detector,
        )

    def __init__(
        self,
        face_detector: MediaPipeFaceDetector,
        gaze_detector: L2CSNetOnnx,
    ):
        self._face_detector = face_detector
        self._gaze_detector = gaze_detector

    @property
    def class_names(self) -> List[str]:
        return self._face_detector.class_names

    @property
    def key_points_classes(self) -> List[List[str]]:
        return self._face_detector.key_points_classes

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        conf_threshold: float = 0.25,
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections], List[L2CSGazeDetection]]:
        key_points, detections = self._face_detector(
            images,
            input_color_format=input_color_format,
            conf_thresh=conf_threshold,
            **kwargs,
        )
        crops, crops_images_bounds = crop_images_to_detections(
            images=images,
            detections=detections,
            device=self._gaze_detector.device,
        )
        gaze_detections = self._gaze_detector(crops, input_color_format="rgb", **kwargs)
        gaze_detections_dispatched = []
        for start, end in crops_images_bounds:
            gaze_detections_dispatched.append(
                L2CSGazeDetection(
                    yaw=gaze_detections.yaw[start:end],
                    pitch=gaze_detections.pitch[start:end],
                )
            )
        return key_points, detections, gaze_detections_dispatched

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        conf_threshold: float = 0.25,
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections], List[L2CSGazeDetection]]:
        return self.infer(
            images=images,
            input_color_format=input_color_format,
            conf_threshold=conf_threshold,
            **kwargs,
        )


def crop_images_to_detections(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    detections: List[Detections],
    device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    if isinstance(images, np.ndarray):
        input_color_format = input_color_format or "bgr"
        if input_color_format != "rgb":
            images = np.ascontiguousarray(images[:, :, ::-1])
        prepared_images = [torch.from_numpy(images).permute(2, 0, 1).to(device)]
    elif isinstance(images, torch.Tensor):
        input_color_format = input_color_format or "rgb"
        images = images.to(device)
        if len(images.shape) == 3:
            images = images.unsqueeze(dim=0)
        if input_color_format != "rgb":
            images = images[:, [2, 1, 0], :, :]
        prepared_images = [i for i in images]
    elif isinstance(images, list) and len(images) == 0:
        raise ModelRuntimeError(
            message="Detected empty input to the model",
            help_url="https://todo",
        )
    elif isinstance(images, list) and isinstance(images[0], np.ndarray):
        prepared_images = []
        input_color_format = input_color_format or "bgr"
        for image in images:
            if input_color_format != "rgb":
                image = np.ascontiguousarray(image[:, :, ::-1])
            prepared_images.append(torch.from_numpy(image).permute(2, 0, 1).to(device))
    elif isinstance(images, list) and isinstance(images[0], torch.Tensor):
        prepared_images = []
        input_color_format = input_color_format or "rgb"
        for image in images:
            if input_color_format != "rgb":
                image = image[[2, 1, 0], :, :]
            prepared_images.append(image.to(device))
    else:
        raise ModelRuntimeError(
            message=f"Detected unknown input batch element: {type(images)}",
            help_url="https://todo",
        )
    crops = []
    crops_images_bounds = []
    for image, image_detections in zip(prepared_images, detections):
        start_bound = len(crops)
        for xyxy in image_detections.xyxy:
            x_min, y_min, x_max, y_max = xyxy.tolist()
            crop = image[:, y_min:y_max, x_min:x_max]
            if crop.numel() == 0:
                continue
            crops.append(crop)
        end_bound = len(crops)
        crops_images_bounds.append((start_bound, end_bound))
    return crops, crops_images_bounds
