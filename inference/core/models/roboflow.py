import itertools
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime
from filelock import FileLock
from PIL import Image

from inference.core.env import (
    API_KEY,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    CORE_MODEL_BUCKET,
    DISABLE_PREPROC_AUTO_ORIENT,
    INFER_BUCKET,
    LAMBDA,
    MAX_BATCH_SIZE,
    MODEL_CACHE_DIR,
    MODEL_VALIDATION_DISABLED,
    ONNXRUNTIME_EXECUTION_PROVIDERS,
    REQUIRED_ONNX_PROVIDERS,
    TENSORRT_CACHE_PATH,
    USE_PYTORCH_FOR_PREPROCESSING,
)
from inference.core.logger import logger

if USE_PYTORCH_FOR_PREPROCESSING:
    import torch

from inference.core.cache import cache
from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    clear_cache,
    get_cache_dir,
    get_cache_file_path,
    initialise_cache,
    load_json_from_cache,
    load_text_file_from_cache,
    save_bytes_in_cache,
    save_json_in_cache,
    save_text_lines_in_cache,
)
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.requests.inference import (
    InferenceRequest,
    InferenceRequestImage,
)
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.exceptions import ModelArtefactError, OnnxProviderNotAvailable
from inference.core.models.base import Model
from inference.core.models.utils.batching import create_batches
from inference.core.models.utils.onnx import has_trt
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_instant_model_data,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import get_onnxruntime_execution_providers
from inference.core.utils.preprocess import letterbox_image, prepare
from inference.core.utils.roboflow import get_model_id_chunks
from inference.core.utils.visualisation import draw_detection_predictions
from inference.models.aliases import resolve_roboflow_model_alias

NUM_S3_RETRY = 5
SLEEP_SECONDS_BETWEEN_RETRIES = 3
MODEL_METADATA_CACHE_EXPIRATION_TIMEOUT = 3600  # 1 hour

S3_CLIENT = None
if AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY_ID:
    try:
        import boto3
        from botocore.config import Config

        from inference.core.utils.s3 import download_s3_files_to_directory

        config = Config(retries={"max_attempts": NUM_S3_RETRY, "mode": "standard"})
        S3_CLIENT = boto3.client("s3", config=config)
    except:
        logger.debug("Error loading boto3")
        pass

DEFAULT_COLOR_PALETTE = [
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


class RoboflowInferenceModel(Model):
    """Base Roboflow inference model."""

    def __init__(
        self,
        model_id: str,
        cache_dir_root=MODEL_CACHE_DIR,
        api_key=None,
        load_weights=True,
    ):
        """
        Initialize the RoboflowInferenceModel object.

        Args:
            model_id (str): The unique identifier for the model.
            cache_dir_root (str, optional): The root directory for the cache. Defaults to MODEL_CACHE_DIR.
            api_key (str, optional): API key for authentication. Defaults to None.
        """
        super().__init__()
        self.load_weights = load_weights
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)
        self.dataset_id, self.version_id = get_model_id_chunks(model_id=model_id)
        self.endpoint = model_id
        self.device_id = GLOBAL_DEVICE_ID
        self.cache_dir = os.path.join(cache_dir_root, self.endpoint)
        self.keypoints_metadata: Optional[dict] = None
        initialise_cache(model_id=self.endpoint)

    def cache_file(self, f: str) -> str:
        """Get the cache file path for a given file.

        Args:
            f (str): Filename.

        Returns:
            str: Full path to the cached file.
        """
        return get_cache_file_path(file=f, model_id=self.endpoint)

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clear the cache directory.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        clear_cache(model_id=self.endpoint, delete_from_disk=delete_from_disk)

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=self.colors,
        )

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

    @property
    def cache_key(self):
        return f"metadata:{self.endpoint}"

    @staticmethod
    def model_metadata_from_memcache_endpoint(endpoint):
        model_metadata = cache.get(f"metadata:{endpoint}")
        return model_metadata

    def model_metadata_from_memcache(self):
        model_metadata = cache.get(self.cache_key)
        return model_metadata

    def write_model_metadata_to_memcache(self, metadata):
        cache.set(
            self.cache_key, metadata, expire=MODEL_METADATA_CACHE_EXPIRATION_TIMEOUT
        )

    @property
    def has_model_metadata(self):
        return self.model_metadata_from_memcache() is not None

    def get_model_artifacts(self) -> None:
        """Fetch or load the model artifacts.

        Downloads the model artifacts from S3 or the Roboflow API if they are not already cached.
        """
        self.cache_model_artefacts()
        self.load_model_artifacts_from_cache()

    def cache_model_artefacts(self) -> None:
        infer_bucket_files = self.get_all_required_infer_bucket_file()
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            return None
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None
        self.download_model_artifacts_from_roboflow_api()

    def get_all_required_infer_bucket_file(self) -> List[str]:
        infer_bucket_files = self.get_infer_bucket_file_list()
        infer_bucket_files.append(self.weights_file)
        logger.debug(f"List of files required to load model: {infer_bucket_files}")
        return [f for f in infer_bucket_files if f is not None]

    def download_model_artefacts_from_s3(self) -> None:
        try:
            logger.debug("Downloading model artifacts from S3")
            infer_bucket_files = self.get_all_required_infer_bucket_file()
            cache_directory = get_cache_dir()
            s3_keys = [f"{self.endpoint}/{file}" for file in infer_bucket_files]
            download_s3_files_to_directory(
                bucket=self.model_artifact_bucket,
                keys=s3_keys,
                target_dir=cache_directory,
                s3_client=S3_CLIENT,
            )
        except Exception as error:
            raise ModelArtefactError(
                f"Could not obtain model artefacts from S3 with keys {s3_keys}. Cause: {error}"
            ) from error

    @property
    def model_artifact_bucket(self):
        return INFER_BUCKET

    def download_model_artifacts_from_roboflow_api(self) -> None:
        logger.debug("Downloading model artifacts from Roboflow API")

        # Use the same lock file pattern as in clear_cache
        lock_dir = MODEL_CACHE_DIR + "/_file_locks"  # Dedicated lock directory
        os.makedirs(lock_dir, exist_ok=True)  # Ensure lock directory exists.
        lock_file = os.path.join(lock_dir, f"{os.path.basename(self.cache_dir)}.lock")
        try:
            lock = FileLock(lock_file, timeout=120)  # 120 second timeout for downloads
            with lock:
                if self.version_id is not None:
                    api_data = get_roboflow_model_data(
                        api_key=self.api_key,
                        model_id=self.endpoint,
                        endpoint_type=ModelEndpointType.ORT,
                        device_id=self.device_id,
                    )
                    if "ort" not in api_data.keys():
                        raise ModelArtefactError(
                            "Could not find `ort` key in roboflow API model description response."
                        )
                    api_data = api_data["ort"]
                    if "classes" in api_data:
                        save_text_lines_in_cache(
                            content=api_data["classes"],
                            file="class_names.txt",
                            model_id=self.endpoint,
                        )
                    if "model" not in api_data:
                        raise ModelArtefactError(
                            "Could not find `model` key in roboflow API model description response."
                        )
                    if "environment" not in api_data:
                        raise ModelArtefactError(
                            "Could not find `environment` key in roboflow API model description response."
                        )
                    environment = get_from_url(api_data["environment"])
                    model_weights_response = get_from_url(
                        api_data["model"], json_response=False
                    )
                else:
                    api_data = get_roboflow_instant_model_data(
                        api_key=self.api_key,
                        model_id=self.endpoint,
                    )
                    if (
                        "modelFiles" not in api_data
                        or "ort" not in api_data["modelFiles"]
                        or "model" not in api_data["modelFiles"]["ort"]
                    ):
                        raise ModelArtefactError(
                            "Could not find `modelFiles` key or `modelFiles`.`ort` or `modelFiles`.`ort`.`model` key in roboflow API model description response."
                        )
                    if "environment" not in api_data:
                        raise ModelArtefactError(
                            "Could not find `environment` key in roboflow API model description response."
                        )
                    model_weights_response = get_from_url(
                        api_data["modelFiles"]["ort"]["model"], json_response=False
                    )
                    environment = api_data["environment"]
                    if "classes" in api_data:
                        save_text_lines_in_cache(
                            content=api_data["classes"],
                            file="class_names.txt",
                            model_id=self.endpoint,
                        )

                save_bytes_in_cache(
                    content=model_weights_response.content,
                    file=self.weights_file,
                    model_id=self.endpoint,
                )
                if "colors" in api_data:
                    environment["COLORS"] = api_data["colors"]
                save_json_in_cache(
                    content=environment,
                    file="environment.json",
                    model_id=self.endpoint,
                )
                if "keypoints_metadata" in api_data:
                    # TODO: make sure backend provides that
                    save_json_in_cache(
                        content=api_data["keypoints_metadata"],
                        file="keypoints_metadata.json",
                        model_id=self.endpoint,
                    )
        except Exception as e:
            logger.error(f"Error downloading model artifacts: {e}")
            raise
        finally:
            try:
                if os.path.exists(lock_file):
                    os.unlink(lock_file)  # Clean up lock file
            except OSError:
                pass  # Best effort cleanup

    def load_model_artifacts_from_cache(self) -> None:
        logger.debug("Model artifacts already downloaded, loading model from cache")
        infer_bucket_files = self.get_all_required_infer_bucket_file()
        if "environment.json" in infer_bucket_files:
            self.environment = load_json_from_cache(
                file="environment.json",
                model_id=self.endpoint,
                object_pairs_hook=OrderedDict,
            )
        if "class_names.txt" in infer_bucket_files:
            self.class_names = load_text_file_from_cache(
                file="class_names.txt",
                model_id=self.endpoint,
                split_lines=True,
                strip_white_chars=True,
            )
        else:
            self.class_names = get_class_names_from_environment_file(
                environment=self.environment
            )
        self.colors = get_color_mapping_from_environment(
            environment=self.environment,
            class_names=self.class_names,
        )
        if "keypoints_metadata.json" in infer_bucket_files:
            self.keypoints_metadata = parse_keypoints_metadata(
                load_json_from_cache(
                    file="keypoints_metadata.json",
                    model_id=self.endpoint,
                    object_pairs_hook=OrderedDict,
                )
            )
        self.num_classes = len(self.class_names)
        if "PREPROCESSING" not in self.environment:
            raise ModelArtefactError(
                "Could not find `PREPROCESSING` key in environment file."
            )
        if issubclass(type(self.environment["PREPROCESSING"]), dict):
            self.preproc = self.environment["PREPROCESSING"]
        else:
            self.preproc = json.loads(self.environment["PREPROCESSING"])
        if self.preproc.get("resize"):
            self.resize_method = self.preproc["resize"].get("format", "Stretch to")
            if self.resize_method not in [
                "Stretch to",
                "Fit (black edges) in",
                "Fit (white edges) in",
                "Fit (grey edges) in",
            ]:
                self.resize_method = "Stretch to"
        else:
            self.resize_method = "Stretch to"
        logger.debug(f"Resize method is '{self.resize_method}'")
        self.multiclass = self.environment.get("MULTICLASS", False)

    def initialize_model(self) -> None:
        """Initialize the model.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError(self.__class__.__name__ + ".initialize_model")

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

    def preprocess_image(
        self,
        image: np.ndarray,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses the given image using specified preprocessing steps.

        Args:
            image (Image.Image): The PIL image to preprocess.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Image.Image: The preprocessed PIL image.
        """
        return prepare(
            image,
            self.preproc,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

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
        infer_bucket_files = self.get_infer_bucket_file_list()
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            logger.debug("Model artifacts already downloaded, loading from cache")
            return None
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None
        self.download_model_from_roboflow_api()

    def download_model_from_roboflow_api(self) -> None:
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.CORE_MODEL,
            device_id=self.device_id,
        )
        if "weights" not in api_data:
            raise ModelArtefactError(
                f"`weights` key not available in Roboflow API response while downloading model weights."
            )
        for weights_url_key in api_data["weights"]:
            weights_url = api_data["weights"][weights_url_key]
            t1 = perf_counter()
            model_weights_response = get_from_url(weights_url, json_response=False)
            filename = weights_url.split("?")[0].split("/")[-1]
            save_bytes_in_cache(
                content=model_weights_response.content,
                file=filename,
                model_id=self.endpoint,
            )
            if perf_counter() - t1 > 120:
                logger.debug(
                    "Weights download took longer than 120 seconds, refreshing API request"
                )
                api_data = get_roboflow_model_data(
                    api_key=self.api_key,
                    model_id=self.endpoint,
                    endpoint_type=ModelEndpointType.CORE_MODEL,
                    device_id=self.device_id,
                )

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

    @property
    def weights_file(self) -> str:
        """Abstract property representing the file containing the model weights. For core models, all model artifacts are handled through get_infer_bucket_file_list method."""
        return None

    @property
    def model_artifact_bucket(self):
        return CORE_MODEL_BUCKET


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
        if self.load_weights or not self.has_model_metadata:
            self.onnxruntime_execution_providers = onnxruntime_execution_providers
            expanded_execution_providers = []
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
                expanded_execution_providers.append(ep)
            self.onnxruntime_execution_providers = expanded_execution_providers

        self.initialize_model()
        self.image_loader_threadpool = ThreadPoolExecutor(max_workers=None)
        try:
            self.validate_model()
        except ModelArtefactError as e:
            logger.error(f"Unable to validate model artifacts, clearing cache: {e}")
            self.clear_cache()
            raise ModelArtefactError from e

    def infer(self, image: Any, **kwargs) -> Any:
        """Runs inference on given data.
        - image:
            can be a BGR numpy array, filepath, InferenceRequestImage, PIL Image, byte-string, etc.
        """
        input_elements = len(image) if isinstance(image, list) else 1
        max_batch_size = MAX_BATCH_SIZE if self.batching_enabled else self.batch_size
        if (input_elements == 1) or (max_batch_size == float("inf")):
            return super().infer(image, **kwargs)
        logger.debug(
            f"Inference will be executed in batches, as there is {input_elements} input elements and "
            f"maximum batch size for a model is set to: {max_batch_size}"
        )
        inference_results = []
        for batch_input in create_batches(sequence=image, batch_size=max_batch_size):
            batch_inference_results = super().infer(batch_input, **kwargs)
            inference_results.append(batch_inference_results)
        return self.merge_inference_results(inference_results=inference_results)

    def merge_inference_results(self, inference_results: List[Any]) -> Any:
        return list(itertools.chain(*inference_results))

    def validate_model(self) -> None:
        if MODEL_VALIDATION_DISABLED:
            logger.debug("Model validation disabled.")
            return None
        logger.debug("Starting model validation")
        if not self.load_weights:
            return
        try:
            assert self.onnx_session is not None
        except AssertionError as e:
            raise ModelArtefactError(
                "ONNX session not initialized. Check that the model weights are available."
            ) from e
        try:
            self.run_test_inference()
        except Exception as e:
            raise ModelArtefactError(f"Unable to run test inference. Cause: {e}") from e
        try:
            self.validate_model_classes()
        except Exception as e:
            raise ModelArtefactError(
                f"Unable to validate model classes. Cause: {e}"
            ) from e
        logger.debug("Model validation finished")

    def run_test_inference(self) -> None:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Running test inference. Image size: {test_image.shape}")
        result = self.infer(test_image, usage_inference_test_run=True)
        logger.debug(f"Test inference finished.")
        return result

    def get_model_output_shape(self) -> Tuple[int, int, int]:
        test_image = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
        logger.debug(f"Getting model output shape. Image size: {test_image.shape}")
        test_image, _ = self.preprocess(test_image)
        output = self.predict(test_image)[0]
        logger.debug(f"Model output shape test finished.")
        return output.shape

    def validate_model_classes(self) -> None:
        pass

    def get_infer_bucket_file_list(self) -> list:
        """Returns the list of files to be downloaded from the inference bucket for ONNX model.

        Returns:
            list: A list of filenames specific to ONNX models.
        """
        return ["environment.json", "class_names.txt"]

    def initialize_model(self) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        logger.debug("Getting model artefacts")
        self.get_model_artifacts()
        logger.debug("Creating inference session")
        if self.load_weights or not self.has_model_metadata:
            t1_session = perf_counter()
            # Create an ONNX Runtime Session with a list of execution providers in priority order. ORT attempts to load providers until one is successful. This keeps the code across devices identical.
            providers = self.onnxruntime_execution_providers

            if not self.load_weights:
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            try:
                session_options = onnxruntime.SessionOptions()
                session_options.log_severity_level = 3
                # TensorRT does better graph optimization for its EP than onnx
                if has_trt(providers):
                    session_options.graph_optimization_level = (
                        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                self.onnx_session = onnxruntime.InferenceSession(
                    self.cache_file(self.weights_file),
                    providers=providers,
                    sess_options=session_options,
                )
            except Exception as e:
                self.clear_cache()
                raise ModelArtefactError(
                    f"Unable to load ONNX session. Cause: {e}"
                ) from e
            logger.debug(f"Session created in {perf_counter() - t1_session} seconds")

            if REQUIRED_ONNX_PROVIDERS:
                available_providers = onnxruntime.get_available_providers()
                for provider in REQUIRED_ONNX_PROVIDERS:
                    if provider not in available_providers:
                        raise OnnxProviderNotAvailable(
                            f"Required ONNX Execution Provider {provider} is not availble. "
                            "Check that you are using the correct docker image on a supported device. "
                            "Export list of available providers as ONNXRUNTIME_EXECUTION_PROVIDERS environmental variable, "
                            "consult documentation for more details."
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

        logger.debug("Model initialisation finished.")

    def load_image(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        if isinstance(image, list):
            preproc_image = partial(
                self.preproc_image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            imgs_with_dims = self.image_loader_threadpool.map(preproc_image, image)
            imgs, img_dims = zip(*imgs_with_dims)
            if isinstance(imgs[0], np.ndarray):
                img_in = np.concatenate(imgs, axis=0)
            elif USE_PYTORCH_FOR_PREPROCESSING:
                img_in = torch.cat(imgs, dim=0)
            else:
                raise ValueError(
                    f"Received a list of images of unknown type, {type(imgs[0])}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )
        else:
            img_in, img_dims = self.preproc_image(
                image,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                disable_preproc_contrast=disable_preproc_contrast,
                disable_preproc_grayscale=disable_preproc_grayscale,
                disable_preproc_static_crop=disable_preproc_static_crop,
            )
            img_dims = [img_dims]
        return img_in, img_dims

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


def get_class_names_from_environment_file(environment: Optional[dict]) -> List[str]:
    if environment is None:
        raise ModelArtefactError(
            f"Missing environment while attempting to get model class names."
        )
    if class_mapping_not_available_in_environment(environment=environment):
        raise ModelArtefactError(
            f"Missing `CLASS_MAP` in environment or `CLASS_MAP` is not dict."
        )
    class_names = []
    for i in range(len(environment["CLASS_MAP"].keys())):
        class_names.append(environment["CLASS_MAP"][str(i)])
    return class_names


def class_mapping_not_available_in_environment(environment: dict) -> bool:
    return "CLASS_MAP" not in environment or not issubclass(
        type(environment["CLASS_MAP"]), dict
    )


def get_color_mapping_from_environment(
    environment: Optional[dict], class_names: List[str]
) -> Dict[str, str]:
    if color_mapping_available_in_environment(environment=environment):
        return environment["COLORS"]
    return {
        class_name: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }


def color_mapping_available_in_environment(environment: Optional[dict]) -> bool:
    return (
        environment is not None
        and "COLORS" in environment
        and issubclass(type(environment["COLORS"]), dict)
    )


def is_model_artefacts_bucket_available() -> bool:
    # TODO: download from GCS directly if GCP_SERVERLESS is true
    return (
        AWS_ACCESS_KEY_ID is not None
        and AWS_SECRET_ACCESS_KEY is not None
        and LAMBDA
        and S3_CLIENT is not None
    )


def parse_keypoints_metadata(metadata: list) -> dict:
    return {
        e["object_class_id"]: {int(key): value for key, value in e["keypoints"].items()}
        for e in metadata
    }
