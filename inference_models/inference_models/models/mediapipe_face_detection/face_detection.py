from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import Detections, KeyPoints, KeyPointsDetectionModel
from inference_models.entities import ColorFormat, ImageDimensions
from inference_models.errors import MissingDependencyError, ModelRuntimeError
from inference_models.models.common.model_packages import get_model_package_contents

try:
    import mediapipe as mp
    from mediapipe.tasks.python.components.containers import Detection
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import face detection model from MediaPipe - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-models` library directly in your Python "
        f"program, make sure the following extras of the package are installed: `mediapipe`."
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


class MediaPipeFaceDetector(
    KeyPointsDetectionModel[List[mp.Image], ImageDimensions, List[Detection]]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "MediaPipeFaceDetector":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["mediapipe_face_detector.tflite"],
        )
        face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=model_package_content[
                        "mediapipe_face_detector.tflite"
                    ]
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
            )
        )
        return cls(face_detector=face_detector)

    def __init__(self, face_detector: mp.tasks.vision.FaceDetector):
        self._face_detector = face_detector
        self._thread_lock = Lock()

    @property
    def class_names(self) -> List[str]:
        return ["face"]

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [["right-eye", "left-eye", "nose", "mouth", "right-ear", "left-ear"]]

    @property
    def skeletons(self) -> List[List[Tuple[int, int]]]:
        return [[(5, 1), (1, 2), (4, 0), (0, 2), (2, 3)]]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> Tuple[List[mp.Image], List[ImageDimensions]]:
        if isinstance(images, np.ndarray):
            input_color_format = input_color_format or "bgr"
            if input_color_format != "rgb":
                images = np.ascontiguousarray(images[:, :, ::-1])
            preprocessed_images = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=images.astype(np.uint8)
            )
            dimensions = ImageDimensions(height=images.shape[0], width=images.shape[1])
            return [preprocessed_images], [dimensions]
        if isinstance(images, torch.Tensor):
            input_color_format = input_color_format or "rgb"
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=0)
            if input_color_format != "rgb":
                images = images[:, [2, 1, 0], :, :]
            images = images.permute(0, 2, 3, 1)
            preprocessed_images, dimensions = [], []
            for image in images:
                np_image = np.ascontiguousarray(image.cpu().numpy())
                preprocessed_images.append(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=np_image.astype(np.uint8)
                    )
                )
                dimensions.append(
                    ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
                )
            return preprocessed_images, dimensions
        if not isinstance(images, list):
            raise ModelRuntimeError(
                message="Pre-processing supports only np.array or torch.Tensor or list of above.",
                help_url="https://todo",
            )
        if not len(images):
            raise ModelRuntimeError(
                message="Detected empty input to the model", help_url="https://todo"
            )
        if isinstance(images[0], np.ndarray):
            input_color_format = input_color_format or "bgr"
            preprocessed_images, dimensions = [], []
            for image in images:
                if input_color_format != "rgb":
                    image = np.ascontiguousarray(image[:, :, ::-1])
                preprocessed_images.append(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=image.astype(np.uint8)
                    )
                )
                dimensions.append(
                    ImageDimensions(height=image.shape[0], width=image.shape[1])
                )
            return preprocessed_images, dimensions
        if isinstance(images[0], torch.Tensor):
            input_color_format = input_color_format or "rgb"
            preprocessed_images, dimensions = [], []
            for image in images:
                if input_color_format != "rgb":
                    image = image[[2, 1, 0], :, :]
                np_image = image.cpu().permute(1, 2, 0).numpy()
                preprocessed_images.append(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=np_image.astype(np.uint8)
                    )
                )
                dimensions.append(
                    ImageDimensions(height=np_image.shape[0], width=np_image.shape[1])
                )
            return preprocessed_images, dimensions
        raise ModelRuntimeError(
            message=f"Detected unknown input batch element: {type(images[0])}",
            help_url="https://todo",
        )

    def forward(
        self, pre_processed_images: List[mp.Image], **kwargs
    ) -> List[List[Detection]]:
        results = []
        with self._thread_lock:
            for input_image in pre_processed_images:
                image_faces = self._face_detector.detect(image=input_image).detections
                results.append(image_faces)
        return results

    def post_process(
        self,
        model_results: List[List[Detection]],
        pre_processing_meta: List[ImageDimensions],
        conf_thresh: float = 0.25,
        **kwargs,
    ) -> Tuple[List[KeyPoints], List[Detections]]:
        final_key_points, final_detections = [], []
        for image_results, image_dimensions in zip(model_results, pre_processing_meta):
            detections_xyxy, detections_class_id, detections_confidence = [], [], []
            key_points_xy, key_points_class_id, key_points_confidence = [], [], []
            for detection in image_results:
                if detection.categories[0].score < conf_thresh:
                    continue
                xyxy = (
                    detection.bounding_box.origin_x,
                    detection.bounding_box.origin_y,
                    detection.bounding_box.origin_x + detection.bounding_box.width,
                    detection.bounding_box.origin_y + detection.bounding_box.height,
                )
                detections_xyxy.append(xyxy)
                detections_class_id.append(0)
                detections_confidence.append(detection.categories[0].score)
                detection_key_points = []
                for keypoint in detection.keypoints:
                    detection_key_points.append(
                        (
                            keypoint.x * image_dimensions.width,
                            keypoint.y * image_dimensions.height,
                        )
                    )
                key_points_xy.append(detection_key_points)
                key_points_class_id.append(0)
                key_points_confidence.append([1.0] * len(detection_key_points))
            detections = Detections(
                xyxy=torch.tensor(detections_xyxy).round().int(),
                class_id=torch.tensor(detections_class_id).int(),
                confidence=torch.tensor(detections_confidence),
            )
            key_points = KeyPoints(
                xy=torch.tensor(key_points_xy).round().int(),
                class_id=torch.tensor(key_points_class_id).int(),
                confidence=torch.tensor(key_points_confidence),
            )
            final_key_points.append(key_points)
            final_detections.append(detections)
        return final_key_points, final_detections
