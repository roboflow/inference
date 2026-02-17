from time import perf_counter
from typing import List, Optional, Tuple

from mediapipe.tasks.python.components.containers.bounding_box import BoundingBox
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.detections import Detection
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.entities.responses.gaze import (
    GazeDetectionInferenceResponse,
    GazeDetectionPrediction,
)
from inference.core.entities.responses.inference import FaceDetectionPrediction, Point
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    GAZE_MAX_BATCH_SIZE,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr
from inference_models import AutoModelPipeline
from inference_models.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
    FaceAndGazeDetectionMPAndL2CS,
)


class InferenceModelsGazeAdapter(Model):
    """Roboflow ONNX Gaze model.

    This class is responsible for handling the ONNX Gaze model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        gaze_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for gaze detection inference.
    """

    def __init__(self, *args, api_key: str = None, **kwargs):
        """Initializes the Gaze with the given arguments and keyword arguments."""
        super().__init__()
        self.task_type = "gaze-detection"
        self.api_key = api_key if api_key else API_KEY

        extra_weights_provider_headers = get_extra_weights_provider_headers()
        self._pipeline: FaceAndGazeDetectionMPAndL2CS = (
            AutoModelPipeline.from_pretrained(
                "face-and-gaze-detection",
                api_key=self.api_key,
                extra_weights_provider_headers=extra_weights_provider_headers,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            )
        )

    def infer_from_request(
        self, request: GazeDetectionInferenceRequest
    ) -> List[GazeDetectionInferenceResponse]:
        """Detect faces and gazes in image(s).

        Args:
            request (GazeDetectionInferenceRequest): The request object containing the image.

        Returns:
            List[GazeDetectionInferenceResponse]: The list of response objects containing the faces and corresponding gazes.
        """
        timer_start = perf_counter()
        if isinstance(request.image, list):
            if len(request.image) > GAZE_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be inferred with gaze detection at one time is {GAZE_MAX_BATCH_SIZE}"
                )
            imgs = request.image
        else:
            imgs = [request.image]
        np_imgs = [load_image_bgr(img) for img in imgs]
        avg_image_loading_time = (perf_counter() - timer_start) / len(np_imgs)
        if not request.do_run_face_detection:
            predictions_time_start = perf_counter()
            gaze_detections = self._pipeline._gaze_detector.infer(images=np_imgs)
            avg_image_prediction_time = (perf_counter() - predictions_time_start) / len(
                np_imgs
            )
            predictions = []
            for i, image in enumerate(np_imgs):
                image_yaw = gaze_detections.yaw[i].item()
                image_pitch = gaze_detections.pitch[i].item()
                faces = [
                    Detection(
                        bounding_box=BoundingBox(
                            origin_x=0,
                            origin_y=0,
                            width=image.shape[1],
                            height=image.shape[0],
                        ),
                        categories=[Category(score=1.0, category_name="face")],
                        keypoints=[],
                    )
                ]
                gazes = [(image_yaw, image_pitch)]
                image_predictions = self._make_response(
                    faces=faces,
                    gazes=gazes,
                    imgW=image.shape[1],
                    imgH=image.shape[0],
                    time_total=avg_image_prediction_time + avg_image_loading_time,
                    time_face_det=0,
                    time_gaze_det=avg_image_prediction_time,
                )
                predictions.append(image_predictions)
            return predictions

        predictions_time_start = perf_counter()
        landmarks, faces, gazes = self._pipeline(images=np_imgs)
        # prepare response
        avg_image_prediction_time = (perf_counter() - predictions_time_start) / len(
            np_imgs
        )
        response = []
        for i in range(len(np_imgs)):
            imgH, imgW, _ = np_imgs[i].shape
            faces_per_img = faces[i]
            landmarks_per_img = landmarks[i]
            gazes_per_img = gazes[i]
            processed_faces_for_image = []
            processed_gazes_for_image = []
            for detection_id in range(faces_per_img.xyxy.shape[0]):
                min_x, min_y, max_x, max_y = faces_per_img.xyxy[detection_id].tolist()
                width = max_x - min_x
                height = max_y - min_y
                score = faces_per_img.confidence[detection_id].item()
                detection_keypoints = landmarks_per_img.xy[detection_id].tolist()
                processed_keypoints = []
                for x, y in detection_keypoints:
                    processed_keypoints.append(
                        NormalizedKeypoint(x=x / imgW, y=y / imgH)
                    )
                face_detection_mp = Detection(
                    bounding_box=BoundingBox(
                        origin_x=min_x,
                        origin_y=min_y,
                        width=width,
                        height=height,
                    ),
                    categories=[Category(score=score, category_name="face")],
                    keypoints=processed_keypoints,
                )
                processed_faces_for_image.append(face_detection_mp)
                if gazes_per_img is None:
                    processed_gazes_for_image.append(None)
                else:
                    processed_gazes_for_image.append(
                        (
                            gazes_per_img.yaw[detection_id].item(),
                            gazes_per_img.pitch[detection_id].item(),
                        )
                    )
            response.append(
                self._make_response(
                    processed_faces_for_image,
                    processed_gazes_for_image,
                    imgW,
                    imgH,
                    avg_image_prediction_time + avg_image_loading_time,
                )
            )
        return response

    def _make_response(
        self,
        faces: List[Detection],
        gazes: List[Optional[Tuple[float, float]]],
        imgW: int,
        imgH: int,
        time_total: float,
        time_face_det: float = None,
        time_gaze_det: float = None,
    ) -> GazeDetectionInferenceResponse:
        """Prepare response object from detected faces and corresponding gazes.

        Args:
            faces (List[Detection]): The detected faces.
            gazes (List[tuple(float, float)]): The detected gazes (yaw, pitch).
            imgW (int): The width (px) of original image.
            imgH (int): The height (px) of original image.
            time_total (float): The processing time.
            time_face_det (float): The processing time.
            time_gaze_det (float): The processing time.

        Returns:
            GazeDetectionInferenceResponse: The response object including the detected faces and gazes info.
        """
        predictions = []
        for face, gaze in zip(faces, gazes):
            landmarks = []
            for keypoint in face.keypoints:
                x = min(max(int(keypoint.x * imgW), 0), imgW - 1)
                y = min(max(int(keypoint.y * imgH), 0), imgH - 1)
                landmarks.append(Point(x=x, y=y))

            bbox = face.bounding_box
            x_center = bbox.origin_x + bbox.width / 2
            y_center = bbox.origin_y + bbox.height / 2
            score = face.categories[0].score

            prediction = GazeDetectionPrediction(
                face=FaceDetectionPrediction(
                    **dict(
                        x=x_center,
                        y=y_center,
                        width=bbox.width,
                        height=bbox.height,
                        confidence=score,
                        class_name="face",
                        landmarks=landmarks,
                    )
                ),
                yaw=gaze[0] if gaze is not None else None,
                pitch=gaze[1] if gaze is not None else None,
            )
            predictions.append(prediction)

        return GazeDetectionInferenceResponse(
            predictions=predictions,
            time=time_total,
            time_face_det=time_face_det,
            time_gaze_det=time_gaze_det,
        )
