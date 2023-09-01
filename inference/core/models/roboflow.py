import json
import os
import shutil
import traceback
import urllib
from io import BytesIO
from time import perf_counter, sleep
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime
import requests
from PIL import Image

from inference.core.data_models import (
    InferenceRequest,
    InferenceRequestImage,
    InferenceResponse,
)
from inference.core.devices.utils import get_device_id
from inference.core.env import (
    API_BASE_URL,
    API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    CORE_MODEL_BUCKET,
    INFER_BUCKET,
    LICENSE_SERVER,
    MODEL_CACHE_DIR,
    ONNXRUNTIME_EXECUTION_PROVIDERS,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
)
from inference.core.exceptions import (
    MissingApiKeyError,
    OnnxProviderNotAvailable,
    TensorrtRoboflowAPIError,
)
from inference.core.models.base import Model
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import get_onnxruntime_execution_providers
from inference.core.utils.preprocess import prepare

if AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY_ID:
    import boto3

    s3 = boto3.client("s3")

NUM_S3_RETRY = 3
SLEEP_SECONDS_BETWEEN_RETRIES = 3


def get_api_data(api_url):
    """Fetch API data from a given URL.

    Args:
        api_url (str): The URL to fetch data from.

    Raises:
        TensorrtRoboflowAPIError: If an error occurs while fetching the data.

    Returns:
        dict: JSON response from the API.
    """
    try:
        r = requests.get(api_url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if "error" in r.json():
            raise TensorrtRoboflowAPIError(
                f"An error occurred when calling the Roboflow API to acquire the model artifacts. The endpoint to debug is {api_url}. The error was: {r.json()['error']}."
            )
        raise TensorrtRoboflowAPIError(
            f"An error occurred when calling the Roboflow API to acquire the model artifacts. The endpoint to debug is {api_url}. The error was: {e}."
        )
    api_data = r.json()
    return api_data


class RoboflowInferenceModel(Model):
    """Base Roboflow inference model."""

    def __init__(
        self,
        model_id: str,
        cache_dir_root=MODEL_CACHE_DIR,
        api_key=None,
    ):
        """
        Initialize the RoboflowInferenceModel object.

        Args:
            model_id (str): The unique identifier for the model.
            cache_dir_root (str, optional): The root directory for the cache. Defaults to MODEL_CACHE_DIR.
            api_key (str, optional): API key for authentication. Defaults to None.
        """
        super().__init__()
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        if not self.api_key and not (AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID):
            raise MissingApiKeyError(
                "No API Key Found, must provide an API Key in each request or as an environment variable on server startup"
            )

        self.dataset_id, self.version_id = model_id.split("/")
        self.endpoint = model_id
        self.device_id = get_device_id()

        self.cache_dir = os.path.join(cache_dir_root, self.endpoint)
        os.makedirs(self.cache_dir, exist_ok=True)

    def cache_file(self, f: str) -> str:
        """Get the cache file path for a given file.

        Args:
            f (str): Filename.

        Returns:
            str: Full path to the cached file.
        """
        return os.path.join(self.cache_dir, f)

    def clear_cache(self) -> None:
        """Clear the cache directory."""

        shutil.rmtree(self.cache_dir)

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> str:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        image = load_image(inference_request.image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        for box in inference_response.predictions:
            color = tuple(
                int(self.colors.get(box.class_name, "#4892EA")[i : i + 2], 16)
                for i in (1, 3, 5)
            )
            x1 = int(box.x - box.width / 2)
            x2 = int(box.x + box.width / 2)
            y1 = int(box.y - box.height / 2)
            y2 = int(box.y + box.height / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=inference_request.visualization_stroke_width,
            )
            if hasattr(box, "points"):
                points = np.array([(int(p.x), int(p.y)) for p in box.points], np.int32)
                if len(points) > 2:
                    cv2.polylines(
                        image,
                        [points],
                        isClosed=True,
                        color=color,
                        thickness=inference_request.visualization_stroke_width,
                    )
            if inference_request.visualization_labels:
                text = f"{box.class_name} {box.confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                button_size = (text_width + 20, text_height + 20)
                button_img = np.full(
                    (button_size[1], button_size[0], 3), color[::-1], dtype=np.uint8
                )
                cv2.putText(
                    button_img,
                    text,
                    (10, 10 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                end_x = min(x1 + button_size[0], image.shape[1])
                end_y = min(y1 + button_size[1], image.shape[0])
                image[y1:end_y, x1:end_x] = button_img[: end_y - y1, : end_x - x1]

        image = Image.fromarray(image)
        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    @property
    def get_class_names(self):
        return self.class_names

    def get_device_id(self) -> str:
        """
        Get the device identifier on which the model is deployed.

        Returns:
            str: Device identifier.
        """
        return self.device_id

    def get_infer_bucket_file_list(self) -> List[str]:
        """Get a list of inference bucket files.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            List[str]: A list of inference bucket files.
        """
        raise NotImplementedError(
            self.__class__.__name__ + ".get_infer_bucket_file_list"
        )

    def get_model_artifacts(self) -> None:
        """Fetch or load the model artifacts.

        Downloads the model artifacts from S3 or the Roboflow API if they are not already cached.

        Raises:
            Exception: If it fails to download the model artifacts from S3.
            TensorrtRoboflowAPIError: If an error occurs while fetching data from the Roboflow API.
        """
        infer_bucket_files = self.get_infer_bucket_file_list()
        infer_bucket_files.append(self.weights_file)
        if all([os.path.exists(self.cache_file(f)) for f in infer_bucket_files]):
            self.log("Model artifacts already downloaded, loading model from cache")
            if "environment.json" in infer_bucket_files:
                with self.open_cache("environment.json", "r") as f:
                    self.environment = json.load(f)

            if "class_names.txt" in infer_bucket_files:
                self.class_names = []
                with self.open_cache("class_names.txt", "r") as f:
                    for l in f.readlines():
                        self.class_names.append(l.strip("\n"))
            elif "CLASS_MAP" in self.environment:
                self.class_names = []
                for i in range(len(self.environment["CLASS_MAP"].keys())):
                    self.class_names.append(self.environment["CLASS_MAP"][str(i)])
            if "COLORS" in self.environment.keys():
                self.colors = json.loads(self.environment["COLORS"])
            else:
                # then no colors have been saved to S3
                colors_order = [
                    "#4892EA",
                    "#00EEC3",
                    "#FE4EF0",
                    "#F4004E",
                    "#FA7200",
                    "#EEEE17",
                    "#90FF00",
                    "#78C1D2",
                    "#8C29FF",
                ]

                self.colors = {}
                i = 1
                for c in self.class_names:
                    self.colors[c] = colors_order[i % len(colors_order)]
                    i += 1
        else:
            # If AWS keys are available, then we can download model artifacts directly
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                self.log("Downloading model artifacts from S3")
                for f in infer_bucket_files:
                    success = False
                    attempts = 0
                    while not success and attempts < NUM_S3_RETRY:
                        try:
                            s3.download_file(
                                INFER_BUCKET,
                                f"{self.endpoint}/{f}",
                                self.cache_file(f),
                            )
                            success = True
                        except Exception as e:
                            attempts += 1
                            print(
                                f"Failed to download model artifacts after {attempts} attempts | Infer Bucket = {INFER_BUCKET} | Object Path = {self.endpoint}/{f} | Weights File = {self.weights_file}",
                                traceback.format_exc(),
                            )
                            sleep(SLEEP_SECONDS_BETWEEN_RETRIES)
                    if not success:
                        raise Exception(f"Failed to download model artifacts.")

                if "environment.json" in infer_bucket_files:
                    with self.open_cache("environment.json", "r") as f:
                        self.environment = json.load(f)

                if "class_names.txt" in infer_bucket_files:
                    self.class_names = []
                    with self.open_cache("class_names.txt", "r") as f:
                        for l in f.readlines():
                            self.class_names.append(l.strip("\n"))
                elif "CLASS_MAP" in self.environment:
                    self.class_names = []
                    for i in range(len(self.environment["CLASS_MAP"].keys())):
                        self.class_names.append(self.environment["CLASS_MAP"][str(i)])

                if "COLORS" in self.environment.keys():
                    self.colors = json.loads(self.environment["COLORS"])
                else:
                    # then no colors have been saved to S3
                    colors_order = [
                        "#4892EA",
                        "#00EEC3",
                        "#FE4EF0",
                        "#F4004E",
                        "#FA7200",
                        "#EEEE17",
                        "#90FF00",
                        "#78C1D2",
                        "#8C29FF",
                    ]

                    self.colors = {}
                    i = 1
                    for c in self.class_names:
                        self.colors[c] = colors_order[i % len(colors_order)]
                        i += 1

            else:
                self.log("Downloading model artifacts from Roboflow API")
                # AWS Keys are not available so we use the API Key to hit the Roboflow API which returns a signed link for downloading model artifacts
                self.api_url = f"{API_BASE_URL}/ort/{self.endpoint}?api_key={self.api_key}&device={self.device_id}&nocache=true&dynamic=true"
                if LICENSE_SERVER:
                    self.api_url = (
                        f"http://{LICENSE_SERVER}/proxy?url="
                        + urllib.parse.quote(self.api_url, safe="~()*!'")
                    )
                api_data = get_api_data(self.api_url)
                if "ort" not in api_data.keys():
                    raise TensorrtRoboflowAPIError(
                        f"An error occurred when calling the Roboflow API to acquire the model artifacts. The endpoint to debug is {self.api_url}. The error was: Key 'tensorrt' not in api_data."
                    )

                api_data = api_data["ort"]

                if "classes" in api_data:
                    self.class_names = api_data["classes"]
                else:
                    self.class_names = None

                if "colors" in api_data:
                    self.colors = api_data["colors"]

                if LICENSE_SERVER:
                    license_server_base_url = f"http://{LICENSE_SERVER}/proxy?url="
                    weights_url = license_server_base_url + urllib.parse.quote(
                        api_data["model"], safe="~()*!'"
                    )

                    def get_env_url(api_data):
                        return license_server_base_url + urllib.parse.quote(
                            api_data["environment"], safe="~()*!'"
                        )

                else:
                    weights_url = api_data["model"]

                    def get_env_url(api_data):
                        return api_data["environment"]

                t1 = perf_counter()
                r = requests.get(weights_url)
                with self.open_cache(self.weights_file, "wb") as f:
                    f.write(r.content)
                if perf_counter() - t1 > 120:
                    self.log(
                        "Weights download took longer than 120 seconds, refreshing API request"
                    )
                    api_data = get_api_data(self.api_url)
                env_url = get_env_url(api_data)
                self.environment = requests.get(env_url).json()
                with open(self.cache_file("environment.json"), "w") as f:
                    json.dump(self.environment, f)
                if not self.class_names and "CLASS_MAP" in self.environment:
                    self.class_names = []
                    for i in range(len(self.environment["CLASS_MAP"].keys())):
                        self.class_names.append(self.environment["CLASS_MAP"][str(i)])

        if not os.path.exists(self.cache_file("class_names.txt")):
            with self.open_cache("class_names.txt", "w") as f:
                for c in self.class_names:
                    f.write(f"{c}\n")
        self.num_classes = len(self.class_names)
        self.preproc = json.loads(self.environment["PREPROCESSING"])

        if self.preproc.get("resize"):
            self.resize_method = self.preproc["resize"].get("format", "Stretch to")
            if self.resize_method not in [
                "Stretch to",
                "Fit (black edges) in",
                "Fit (white edges) in",
            ]:
                self.resize_method = "Stretch to"
        else:
            self.resize_method = "Stretch to"
        self.log(f"Resize method is '{self.resize_method}'")

    def initialize_model(self) -> None:
        """Initialize the model.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError(self.__class__.__name__ + ".initialize_model")

    @staticmethod
    def letterbox_image(
        image: Image.Image, size: Tuple[int, int], c: int = 0
    ) -> Image.Image:
        """Resize image to fit within the size specified while maintaining aspect ratio by padding the image

        Args:
            image (Image.Image): PIL image to be resized
            size (Tuple[int, int]): Desired width and height in pixels of the output image
            c (int, optional): Color to use for padding. Defaults to 0. Not using this as it seems to be less performant. This should be investigated.

        Returns:
            Image.Image: PIL image of dimensions `size` containing the original image plus any required padding
        """
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", size, (114, 114, 114))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def load_preprocessed_image(self, image: InferenceRequestImage) -> Image.Image:
        """Loads and preprocesses an image for inference.

        Args:
            image (InferenceRequestImage): The image information to load and preprocess.

        Returns:
            Image.Image: The loaded and preprocessed PIL image.
        """
        pil_image = load_image(image)
        return self.preprocess_image(pil_image)

    def open_cache(self, f: str, mode: str, encoding: str = None):
        """Opens a cache file with the given filename, mode, and encoding.

        Args:
            f (str): Filename to open from cache.
            mode (str): Mode in which to open the file (e.g., 'r' for read, 'w' for write).
            encoding (str, optional): Encoding to use when opening the file. Defaults to None.

        Returns:
            file object: The opened file object.
        """
        return open(self.cache_file(f), mode, encoding=encoding)

    def preproc_image(
        self, image: Union[Any, InferenceRequestImage]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preproccess an inference request image by loading it, then applying any preprocs specified by the Roboflow platform, then scaling it to the inference input dimensions

        Args:
            image (InferenceRequestImage): An object containing information necessary to load the image for inference

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing an numpy array of the preprocessed image pixel data and a tuple of the images original size
        """
        pil_image = load_image(image)
        preprocessed_image, img_dims = self.preprocess_image(pil_image)

        if self.resize_method == "Stretch to":
            resized = preprocessed_image.resize(
                (self.img_size_w, self.img_size_h), Image.BICUBIC
            )
        elif self.resize_method == "Fit (black edges) in":
            resized = self.letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h)
            )
        elif self.resize_method == "Fit (white edges) in":
            resized = self.letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h), c=255
            )

        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img_in = np.expand_dims(img_in, axis=0)

        return img_in, img_dims

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocesses the given image using specified preprocessing steps.

        Args:
            image (Image.Image): The PIL image to preprocess.

        Returns:
            Image.Image: The preprocessed PIL image.
        """
        return prepare(image, self.preproc)

    @property
    def weights_file(self) -> str:
        """Abstract property representing the file containing the model weights.

        Raises:
            NotImplementedError: This property must be implemented in subclasses.

        Returns:
            str: The file path to the weights file.
        """
        raise NotImplementedError(self.__class__.__name__ + ".weights_file")


class RoboflowCoreModel(RoboflowInferenceModel):
    """Base Roboflow inference model (Inherits from CvModel since all Roboflow models are CV models currently)."""

    def __init__(
        self,
        model_id: str,
        api_key=None,
    ):
        """Initializes the RoboflowCoreModel instance.

        Args:
            model_id (str): The identifier for the specific model.
            api_key ([type], optional): The API key for authentication. Defaults to None.
        """
        super().__init__(model_id, api_key=api_key)
        self.download_weights()

    def download_weights(self) -> None:
        """Downloads the model weights from the configured source.

        This method includes handling for AWS access keys and error handling.
        """
        if any(
            [
                not os.path.exists(self.cache_file(f))
                for f in self.get_infer_bucket_file_list()
            ]
        ):
            self.log("Downloading model artifacts")
            if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                infer_bucket_files = self.get_infer_bucket_file_list()
                for f in infer_bucket_files:
                    success = False
                    attempts = 0
                    while not success and attempts < NUM_S3_RETRY:
                        try:
                            local_file_path = self.cache_file(f)
                            s3.download_file(
                                CORE_MODEL_BUCKET,
                                f"{self.endpoint}/{f}",
                                local_file_path,
                            )

                            # Check file size
                            if (
                                os.path.getsize(local_file_path) <= 4096
                                and attempts < 2
                            ):
                                attempts += 1
                                continue

                            success = True
                        except Exception as e:
                            attempts += 1
                            print(
                                f"Failed to download model artifacts after {attempts} attempts | Infer Bucket = {INFER_BUCKET} | Object Path = {self.endpoint}/{f}",
                                traceback.format_exc(),
                            )
                            sleep(SLEEP_SECONDS_BETWEEN_RETRIES)
                    if not success:
                        raise Exception(f"Failed to download model artifacts.")
            else:
                # AWS Keys are not available so we use the API Key to hit the Roboflow API which returns a signed link for downloading model artifacts
                self.api_url = f"{API_BASE_URL}/core_model/{self.endpoint}?api_key={self.api_key}&device={self.device_id}&nocache=true"
                if LICENSE_SERVER:
                    self.api_url = (
                        f"http://{LICENSE_SERVER}/proxy?url="
                        + urllib.parse.quote(self.api_url, safe="~()*!'")
                    )
                api_data = get_api_data(self.api_url)
                if "weights" not in api_data.keys():
                    raise TensorrtRoboflowAPIError(
                        f"An error occurred when calling the Roboflow API to acquire the model artifacts. The endpoint to debug is {self.api_url}. The error was: Key 'tensorrt' not in api_data."
                    )

                weights_url_keys = api_data["weights"].keys()

                for weights_url_key in weights_url_keys:
                    if LICENSE_SERVER:
                        license_server_base_url = f"http://{LICENSE_SERVER}/proxy?url="
                        weights_url = license_server_base_url + urllib.parse.quote(
                            api_data["weights"][weights_url_key], safe="~()*!'"
                        )
                    else:
                        weights_url = api_data["weights"][weights_url_key]
                    t1 = perf_counter()
                    attempts = 0
                    success = False
                    while attempts < 3 and not success:
                        r = requests.get(weights_url)
                        filename = weights_url.split("?")[0].split("/")[-1]
                        file_path = self.open_cache(f"{filename}", "wb")
                        with file_path as f:
                            f.write(r.content)

                        # Check file size
                        if os.path.getsize(file_path.name) <= 4096:
                            if attempts < 2:
                                attempts += 1
                                continue
                            else:
                                raise Exception(
                                    f"Failed to download model artifacts from API after 2 attempts."
                                )
                        else:
                            success = True

                    if perf_counter() - t1 > 120:
                        self.log(
                            "Weights download took longer than 120 seconds, refreshing API request"
                        )
                        api_data = get_api_data(self.api_url)
        else:
            self.log("Model artifacts already downloaded, loading from cache")

    def get_device_id(self) -> str:
        """Returns the device ID associated with this model.

        Returns:
            str: The device ID.
        """
        return self.device_id

    def get_infer_bucket_file_list(self) -> List[str]:
        """Abstract method to get the list of files to be downloaded from the inference bucket.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            List[str]: A list of filenames.
        """
        raise NotImplementedError(
            "get_infer_bucket_file_list not implemented for OnnxRoboflowCoreModel"
        )

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Abstract method to preprocess an image.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            Image.Image: The preprocessed PIL image.
        """
        raise NotImplementedError(self.__class__.__name__ + ".preprocess_image")


class OnnxRoboflowInferenceModel(RoboflowInferenceModel):
    """Roboflow Inference Model that operates using an ONNX model file."""

    def __init__(
        self,
        model_id: str,
        onnxruntime_execution_providers: List[
            str
        ] = get_onnxruntime_execution_providers(ONNXRUNTIME_EXECUTION_PROVIDERS),
        *args,
        **kwargs,
    ):
        """Initializes the OnnxRoboflowInferenceModel instance.

        Args:
            model_id (str): The identifier for the specific ONNX model.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(model_id, *args, **kwargs)
        self.onnxruntime_execution_providers = onnxruntime_execution_providers
        for ep in self.onnxruntime_execution_providers:
            if ep == "TensorrtExecutionProvider":
                ep = (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": os.path.join(
                            TENSORRT_CACHE_PATH, self.endpoint
                        ),
                        "trt_fp16_enable": True,
                    },
                )
        self.initialize_model()

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt"]

    def initialize_model(self) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        self.get_model_artifacts()
        self.log("Creating inference session")
        t1_session = perf_counter()
        # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
        self.onnx_session = onnxruntime.InferenceSession(
            self.cache_file(self.weights_file),
            providers=self.onnxruntime_execution_providers,
        )
        self.log(f"Session created in {perf_counter() - t1_session} seconds")

        if REQUIRED_ONNX_PROVIDERS:
            available_providers = onnxruntime.get_available_providers()
            for provider in REQUIRED_ONNX_PROVIDERS:
                if provider not in available_providers:
                    raise OnnxProviderNotAvailable(
                        f"Required ONNX Execution Provider {provider} is not availble. Check that you are using the correct docker image on a supported device."
                    )

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
            self.log(f"Model {self.endpoint} is loaded with dynamic batching enabled")
        else:
            self.batching_enabled = False
            self.log(f"Model {self.endpoint} is loaded with dynamic batching disabled")

    @property
    def weights_file(self) -> str:
        """Returns the file containing the ONNX model weights.

        Returns:
            str: The file path to the weights file.
        """
        return "weights.onnx"


class OnnxRoboflowCoreModel(RoboflowCoreModel):
    """Roboflow Inference Model that operates using an ONNX model file."""

    pass
