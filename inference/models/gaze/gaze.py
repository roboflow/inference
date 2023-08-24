from time import perf_counter
from typing import List, Optional, Tuple, Union

import numpy as np
import math
import onnxruntime
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision

import mediapipe as mp
from mediapipe.tasks.python.components.containers.detections import Detection
from mediapipe.tasks.python.components.containers.bounding_box import BoundingBox
from mediapipe.tasks.python.components.containers.category import Category
from inference.models.gaze.l2cs import L2CS

from inference.core.data_models import (
    GazeDetectionInferenceRequest,
    GazeDetectionInferenceResponse,
    Point,
    FaceDetectionPrediction,
    GazeDetectionPrediction
)
from inference.core.env import (
    GAZE_MAX_BATCH_SIZE,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
)
from inference.core.exceptions import OnnxProviderNotAvailable
from inference.core.models.roboflow import OnnxRoboflowCoreModel


class L2C2Wrapper(L2CS):
    """Roboflow L2CS Gaze detection model.

    This class is responsible for converting L2CS model to ONNX model.
    It is ONLY intended for internal usage.

    Workflow:
        After training a L2CS model, create an instance of this wrapper class.
        Load the trained weights file, and save it as ONNX model.
    """
    def __init__(self):
        self.device = torch.device('cpu')
        self.num_bins = 90
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], self.num_bins)
        self._gaze_softmax = nn.Softmax(dim=1)
        self._gaze_idx_tensor = torch.FloatTensor([i for i in range(90)]).to(self.device)

    def load_L2CS_model(self, file_path="/tmp/cache/gaze/L2CS/L2CSNet_gaze360_resnet50_90bins.pkl"):
        super().load_state_dict(torch.load(file_path, map_location=self.device))
        super().to(self.device)

    def saveas_ONNX_model(self, file_path="/tmp/cache/gaze/L2CS/L2CSNet_gaze360_resnet50_90bins.onnx"):
        dummy_input = torch.randn(1, 3, 448, 448)
        dynamic_axes = {'input': {0: 'batch_size'}, 'output_yaw': {0: 'batch_size'}, 'output_pitch': {0: 'batch_size'}}
        torch.onnx.export(self, dummy_input, file_path, input_names=["input"],
                          output_names=['output_yaw', 'output_pitch'],
                          dynamic_axes=dynamic_axes, verbose=False)

    def forward(self, x):
        idx_tensor = torch.stack([self._gaze_idx_tensor for i in range(x.shape[0])], dim=0)
        gaze_yaw, gaze_pitch = super().forward(x)

        yaw_predicted = self._gaze_softmax(gaze_yaw)
        yaw_radian = (torch.sum(yaw_predicted * idx_tensor, dim=1) * 4 - 180) * np.pi / 180

        pitch_predicted = self._gaze_softmax(gaze_pitch)
        pitch_radian = (torch.sum(pitch_predicted * idx_tensor, dim=1) * 4 - 180) * np.pi / 180

        return yaw_radian, pitch_radian


class GazeOnnxRoboflowCoreModel(OnnxRoboflowCoreModel):
    """Roboflow ONNX Gaze model.

    This class is responsible for handling the ONNX Gaze model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        gaze_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for gaze detection inference.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the GazeOnnxRoboflowCoreModel with the given arguments and keyword arguments."""

        t1 = perf_counter()
        super().__init__(*args, **kwargs)
        # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
        self.log("Creating inference sessions")

        # TODO: convert face detector (TensorflowLite) to ONNX model

        self.gaze_onnx_session = onnxruntime.InferenceSession(
            self.cache_file("L2CSNet_gaze360_resnet50_90bins.onnx"),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": TENSORRT_CACHE_PATH,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        if REQUIRED_ONNX_PROVIDERS:
            available_providers = onnxruntime.get_available_providers()
            for provider in REQUIRED_ONNX_PROVIDERS:
                if provider not in available_providers:
                    raise OnnxProviderNotAvailable(
                        f"Required ONNX Execution Provider {provider} is not availble. Check that you are using the correct docker image on a supported device."
                    )

        # init face detector
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.cache_file("mediapipe_face_detector.tflite")),
                running_mode=mp.tasks.vision.RunningMode.IMAGE
            )
        )

        # additional settings for gaze detection
        self._gaze_transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.log(f"GAZE model loaded in {perf_counter() - t1:.2f} seconds")

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: The list of file names.
        """
        return ["mediapipe_face_detector.tflite", "L2CSNet_gaze360_resnet50_90bins.onnx"]

    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int,
                                         image_height: int) -> Tuple[int, int]:
        """Converts normalized value pair to pixel coordinates.

        Args:
            normalized_x (int): The normalized x-axis value.
            normalized_y (int): The normalized y-axis value.
            image_width (int): The width (px) of image.
            image_height (int): The height (px) of image.

        Returns:
            Tuple[int, int]: The response object including the detected faces and gazes info.
        """
        x_px = max(min(math.floor(normalized_x * image_width), image_width - 1), 0)
        y_px = max(min(math.floor(normalized_y * image_height), image_height - 1), 0)
        return x_px, y_px

    def _make_response(self, faces: List[Detection], gazes: List[Tuple[float, float]], imgW: int, imgH: int,
                       time_load_img: float, time_face_det: float,
                       time_gaze_det: float) -> GazeDetectionInferenceResponse:
        """Prepare response object from detected faces and corresponding gazes.

        Args:
            faces (List[Detection]): The detected faces.
            gazes (List[tuple(float, float)]): The detected gazes (yaw, pitch).
            imgW (int): The width (px) of original image.
            imgH (int): The height (px) of original image.
            time_load_img (float): The processing time for loading image.
            time_face_det (float): The processing time for face detection.
            time_gaze_det (float): The processing time for gaze detection.

        Returns:
            GazeDetectionInferenceResponse: The response object including the detected faces and gazes info.
        """
        predictions = []
        for i in range(len(faces)):
            face = faces[i]
            gaze = gazes[i]

            landmarks = []
            for keypoint in face.keypoints:
                x, y = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y, imgW, imgH)
                landmarks.append(Point(x=x, y=y))

            bbox = face.bounding_box
            x_center = bbox.origin_x + bbox.width / 2
            y_center = bbox.origin_y + bbox.height / 2
            score = face.categories[0].score

            prediction = GazeDetectionPrediction(
                face=FaceDetectionPrediction(x=x_center, y=y_center, width=bbox.width, height=bbox.height,
                                             confidence=score, class_name="face", landmarks=landmarks),
                yaw=gaze[0], pitch=gaze[1])
            predictions.append(prediction)

        response = GazeDetectionInferenceResponse(predictions=predictions,
                                                  time_total=(time_load_img + time_face_det + time_gaze_det),
                                                  time_load_img=time_load_img, time_face_det=time_face_det,
                                                  time_gaze_det=time_gaze_det)
        return response

    def _detect_gaze(self, pil_imgs: List[Image.Image]) -> List[Tuple[float, float]]:
        """Detect faces and gazes in an image.

        Args:
            pil_imgs (List[Image.Image]): The PIL image list, each image is a cropped facial image.

        Returns:
            List[Tuple[float, float]]: Yaw (radian) and Pitch (radian).
        """
        ret = []
        for i in range(0, len(pil_imgs), GAZE_MAX_BATCH_SIZE):
            img_batch = []
            for j in range(i, min(len(pil_imgs), i + GAZE_MAX_BATCH_SIZE)):
                img = self._gaze_transformations(pil_imgs[j])
                img = np.expand_dims(img, axis=0).astype(np.float32)
                img_batch.append(img)

            img_batch = np.concatenate(img_batch, axis=0)
            onnx_input_image = {self.gaze_onnx_session.get_inputs()[0].name: img_batch}
            yaw, pitch = self.gaze_onnx_session.run(None, onnx_input_image)

            for j in range(len(img_batch)):
                ret.append((yaw[j], pitch[j]))

        return ret

    def _crop_face_img(self, pil_img: Image.Image, face: Detection) -> Image.Image:
        """Extract facial area in an image.

        Args:
            pil_img (Image.Image): The PIL image.
            face (mediapipe.tasks.python.components.containers.detections.Detection): The detected face.

        Returns:
            Image.Image: Cropped face image.
        """
        # extract face area
        bbox = face.bounding_box
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        x_max = bbox.origin_x + bbox.width
        y_max = bbox.origin_y + bbox.height
        face_img = pil_img.crop((x_min, y_min, x_max, y_max))
        face_img = face_img.resize((224, 224))
        return face_img

    def detect(self, request: GazeDetectionInferenceRequest) -> List[GazeDetectionInferenceResponse]:
        """Detect faces and gazes in image(s).

        Args:
            request (GazeDetectionInferenceRequest): The request object containing the image.

        Returns:
            List[GazeDetectionInferenceResponse]: The list of response objects containing the faces and corresponding gazes.
        """

        if isinstance(request.image, list):
            if len(request.image) > GAZE_MAX_BATCH_SIZE:
                raise ValueError(
                    f"The maximum number of images that can be embedded at once is {GAZE_MAX_BATCH_SIZE}"
                )
            imgs = request.image
        else:
            imgs = [request.image]

        # load pil images
        t1 = perf_counter()
        num_img = len(imgs)
        pil_imgs = [self.load_image(img.type, img.value) for img in imgs]
        time_load_img = (perf_counter() - t1) / num_img

        # face detection
        # TODO: face detection for batch
        t1 = perf_counter()
        faces = []
        for pil_img in pil_imgs:
            if not request.is_cropped_face_image:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
                faces_per_img = self.face_detector.detect(mp_img).detections
            else:
                faces_per_img = [
                    Detection(bounding_box=BoundingBox(origin_x=0, origin_y=0, width=pil_img[0], height=pil_img[1]),
                              categories=[Category(score=1.0, category_name="face")], keypoints=[])]
            faces.append(faces_per_img)
        time_face_det = (perf_counter() - t1) / num_img

        # gaze detection
        t1 = perf_counter()
        face_imgs = []
        for i, pil_img in enumerate(pil_imgs):
            if not request.is_cropped_face_image:
                face_imgs.extend([self._crop_face_img(pil_img, face) for face in faces[i]])
            else:
                face_imgs.append(pil_img.resize((224, 224)))
        gazes = self._detect_gaze(face_imgs)
        time_gaze_det = (perf_counter() - t1) / num_img

        # prepare response
        response = []
        idx_gaze = 0
        for i in range(len(pil_imgs)):
            imgW, imgH = pil_imgs[i].size
            faces_per_img = faces[i]
            gazes_per_img = gazes[idx_gaze:idx_gaze + len(faces_per_img)]
            response.append(self._make_response(faces_per_img, gazes_per_img, imgW, imgH, time_load_img, time_face_det, time_gaze_det))

        return response

    def infer(self, request: GazeDetectionInferenceRequest) -> List[GazeDetectionInferenceResponse]:
        """Routes the request to the appropriate inference function.

        Args:
            request (GazeDetectionInferenceRequest): The request object containing the inference details.

        Returns:
            List[GazeDetectionInferenceResponse]: The list of response objects containing the faces and corresponding gazes.
        """
        if isinstance(request, GazeDetectionInferenceRequest):
            infer_func = self.detect
        else:
            raise ValueError(
                f"Request type {type(request)} is not a valid GazeDetectionInferenceRequest"
            )
        return infer_func(request)
