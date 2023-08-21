from time import perf_counter
from typing import List, Optional, Tuple, Union

import numpy as np
import math
import onnxruntime
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
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
    GazeDetectionPrediction,
    InferenceRequestImage,
)
from inference.core.env import (
    GAZE_MAX_BATCH_SIZE,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
)
from inference.core.exceptions import OnnxProviderNotAvailable
from inference.core.models.roboflow import OnnxRoboflowCoreModel


class GazeOnnxRoboflowCoreModel(OnnxRoboflowCoreModel):
    """Roboflow ONNX Gaze model.

    This class is responsible for handling the ONNX Gaze model, including
    loading the model, preprocessing the input, and performing inference.

    Attributes:
        face_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for face detection inference.
        gaze_onnx_session (onnxruntime.InferenceSession): ONNX Runtime session for gaze detection inference.
        resolution (int): The resolution of the input image.
        gaze_preprocess (function): Function to preprocess the image.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the GazeOnnxRoboflowCoreModel with the given arguments and keyword arguments."""

        t1 = perf_counter()
        super().__init__(*args, **kwargs)
        # # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
        # self.log("Creating inference sessions")
        # self.face_onnx_session = onnxruntime.InferenceSession(
        #     self.cache_file("detector.tflite"),
        #     providers=[
        #         (
        #             "TensorrtExecutionProvider",
        #             {
        #                 "trt_engine_cache_enable": True,
        #                 "trt_engine_cache_path": TENSORRT_CACHE_PATH,
        #             },
        #         ),
        #         "CUDAExecutionProvider",
        #         "CPUExecutionProvider",
        #     ],
        # )
        #
        # self.gaze_onnx_session = onnxruntime.InferenceSession(
        #     self.cache_file("L2CSNet_gaze360.pkl"),
        #     providers=[
        #         (
        #             "TensorrtExecutionProvider",
        #             {
        #                 "trt_engine_cache_enable": True,
        #                 "trt_engine_cache_path": TENSORRT_CACHE_PATH,
        #             },
        #         ),
        #         "CUDAExecutionProvider",
        #         "CPUExecutionProvider",
        #     ],
        # )
        #
        # if REQUIRED_ONNX_PROVIDERS:
        #     available_providers = onnxruntime.get_available_providers()
        #     for provider in REQUIRED_ONNX_PROVIDERS:
        #         if provider not in available_providers:
        #             raise OnnxProviderNotAvailable(
        #                 f"Required ONNX Execution Provider {provider} is not availble. Check that you are using the correct docker image on a supported device."
        #             )

        # init face detector
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.cache_file("mediapipe_face_detector.tflite")),
                running_mode=mp.tasks.vision.RunningMode.IMAGE
            )
        )

        # init gaze detector
        self.is_gpu_available = torch.cuda.is_available()
        self.gpu = torch.device('cuda:0' if self.is_gpu_available else 'cpu')
        self.gaze_detector = L2CS(block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_bins=90)
        self.gaze_detector.load_state_dict(
            torch.load(self.cache_file("L2CSNet_gaze360_resnet50_90bins.pkl"), map_location=self.gpu))
        if self.is_gpu_available:
            self.gaze_detector.cuda(self.gpu)
        self.gaze_detector.eval()

        # additional settings for gaze detection
        self._gaze_transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self._gaze_softmax = nn.Softmax(dim=1)
        self._gaze_idx_tensor = torch.FloatTensor([i for i in range(90)])
        if self.is_gpu_available:
            self._gaze_idx_tensor = self._gaze_idx_tensor.cuda(self.gpu)

        # self.resolution = self.face_onnx_session.get_inputs()[0].shape[2]
        # self.clip_preprocess = _transform(self.resolution)
        self.log(f"GAZE model loaded in {perf_counter() - t1:.2f} seconds")

    def get_infer_bucket_file_list(self) -> List[str]:
        """Gets the list of files required for inference.

        Returns:
            List[str]: The list of file names.
        """
        return ["mediapipe_face_detector.tflite", "L2CSNet_gaze360_resnet50_90bins.pkl"]

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
            gazes (List[tuple(float, float)]): The detected gazes.
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
                pitch=gaze[0], yaw=gaze[1])
            predictions.append(prediction)

        response = GazeDetectionInferenceResponse(predictions=predictions,
                                                  time_total=(time_load_img + time_face_det + time_gaze_det),
                                                  time_load_img=time_load_img, time_face_det=time_face_det,
                                                  time_gaze_det=time_gaze_det)
        return response

    def _detect_gaze(self, pil_img: Image.Image, face: Detection) -> Tuple[float, float]:
        """Detect faces and gazes in an image.

        Args:
            pil_img (Image.Image): The PIL image.
            face (mediapipe.tasks.python.components.containers.detections.Detection): The detected face.

        Returns:
            Tuple[float, float]: Pitch (degree) and Yaw (degree).
        """
        # extract face area
        bbox = face.bounding_box
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        x_max = bbox.origin_x + bbox.width
        y_max = bbox.origin_y + bbox.height

        with torch.no_grad():
            # crop image
            img = pil_img.crop((x_min, y_min, x_max, y_max))
            # img = img.resize((224, 224))
            img = self._gaze_transformations(img)
            img = Variable(img)
            if self.is_gpu_available:
                img = img.cuda(self.gpu)
            img = img.unsqueeze(0)

            # gaze prediction
            gaze_pitch, gaze_yaw = self.gaze_detector(img)
            pitch_predicted = self._gaze_softmax(gaze_pitch)
            yaw_predicted = self._gaze_softmax(gaze_yaw)
            pitch_predicted = torch.sum(pitch_predicted.data[0] * self._gaze_idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * self._gaze_idx_tensor) * 4 - 180
            pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
            yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted

    def _detect_one_img(self, is_cropped_face_image: bool,
                        image: InferenceRequestImage) -> GazeDetectionInferenceResponse:
        """Detect faces and gazes in an image.

        Args:
            is_cropped_face_image (bool): If true, skip face detection and use the whole image as face image; if false, run face detection first and crop facial area, then do gaze detection.
            image (InferenceRequestImage): The object containing information necessary to load the image for inference.

        Returns:
            GazeDetectionInferenceResponse: The response object containing the faces and corresponding gazes.
        """

        # load pil image
        t1 = perf_counter()
        pil_img = self.load_image(image.type, image.value)
        time_load_img = perf_counter() - t1

        # face detection
        t1 = perf_counter()
        if not is_cropped_face_image:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
            faces = self.face_detector.detect(mp_img).detections
        else:
            faces = [Detection(bounding_box=BoundingBox(origin_x=0, origin_y=0, width=pil_img[0], height=pil_img[1]),
                               categories=[Category(score=1.0, category_name="face")], keypoints=[])]
        time_face_det = perf_counter() - t1

        # gaze detection
        t1 = perf_counter()
        gazes = [self._detect_gaze(pil_img, face) for face in faces]
        time_gaze_det = perf_counter() - t1

        imgW, imgH = pil_img.size
        return self._make_response(faces, gazes, imgW, imgH, time_load_img, time_face_det, time_gaze_det)

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

        response = [self._detect_one_img(request.is_cropped_face_image, img) for img in imgs]
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
