from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np
import requests
from requests import HTTPError

from inference_sdk.http.entities import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
    HTTPClientMode,
    ImagesReference,
    InferenceConfiguration,
    ModelDescription,
    RegisteredModels,
    ServerInfo,
)
from inference_sdk.http.errors import (
    HTTPCallErrorError,
    HTTPClientError,
    InvalidModelIdentifier,
    InvalidParameterError,
    ModelNotInitializedError,
    ModelNotSelectedError,
    ModelTaskTypeNotSupportedError,
    WrongClientModeError,
)
from inference_sdk.http.utils.iterables import unwrap_single_element_list
from inference_sdk.http.utils.loaders import (
    load_static_inference_input,
    load_stream_inference_input,
)
from inference_sdk.http.utils.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    response_contains_jpeg_image,
    transform_base64_visualisation,
    transform_visualisation_bytes,
)
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    inject_images_into_payload,
)

SUCCESSFUL_STATUS_CODE = 200
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
}
NEW_INFERENCE_ENDPOINTS = {
    INSTANCE_SEGMENTATION_TASK: "/infer/instance_segmentation",
    OBJECT_DETECTION_TASK: "/infer/object_detection",
    CLASSIFICATION_TASK: "/infer/classification",
    KEYPOINTS_DETECTION_TASK: "/infer/keypoints_detection",
}
CLIP_ARGUMENT_TYPES = {"image", "text"}


def wrap_errors(function: callable) -> callable:
    def decorate(*args, **kwargs) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError as error:
            if "application/json" in error.response.headers.get("Content-Type", ""):
                api_message = error.response.json().get("message")
            else:
                api_message = error.response.text
            raise HTTPCallErrorError(
                description=str(error),
                status_code=error.response.status_code,
                api_message=api_message,
            ) from error
        except ConnectionError as error:
            raise HTTPClientError(
                f"Error with server connection: {str(error)}"
            ) from error

    return decorate


class InferenceHTTPClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
    ):
        self.__api_url = api_url
        self.__api_key = api_key
        self.__inference_configuration = InferenceConfiguration.init_default()
        self.__client_mode = _determine_client_mode(api_url=api_url)
        self.__selected_model: Optional[str] = None

    @property
    def inference_configuration(self) -> InferenceConfiguration:
        return self.__inference_configuration

    @property
    def client_mode(self) -> HTTPClientMode:
        return self.__client_mode

    @property
    def selected_model(self) -> Optional[str]:
        return self.__selected_model

    @contextmanager
    def use_configuration(
        self, inference_configuration: InferenceConfiguration
    ) -> Generator["InferenceHTTPClient", None, None]:
        previous_configuration = self.__inference_configuration
        self.__inference_configuration = inference_configuration
        try:
            yield self
        finally:
            self.__inference_configuration = previous_configuration

    def configure(
        self, inference_configuration: InferenceConfiguration
    ) -> "InferenceHTTPClient":
        self.__inference_configuration = inference_configuration
        return self

    def select_api_v0(self) -> "InferenceHTTPClient":
        self.__client_mode = HTTPClientMode.V0
        return self

    def select_api_v1(self) -> "InferenceHTTPClient":
        self.__client_mode = HTTPClientMode.V1
        return self

    @contextmanager
    def use_api_v0(self) -> Generator["InferenceHTTPClient", None, None]:
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V0
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    @contextmanager
    def use_api_v1(self) -> Generator["InferenceHTTPClient", None, None]:
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V1
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    def select_model(self, model_id: str) -> "InferenceHTTPClient":
        self.__selected_model = model_id
        return self

    @contextmanager
    def use_model(self, model_id: str) -> Generator["InferenceHTTPClient", None, None]:
        previous_model = self.__selected_model
        self.__selected_model = model_id
        try:
            yield self
        finally:
            self.__selected_model = previous_model

    @wrap_errors
    def get_server_info(self) -> ServerInfo:
        response = requests.get(f"{self.__api_url}/info")
        response.raise_for_status()
        response_payload = response.json()
        return ServerInfo.from_dict(response_payload)

    def infer_on_stream(
        self,
        input_uri: str,
        model_id: Optional[str] = None,
    ) -> Generator[Tuple[Union[str, int], np.ndarray, dict], None, None]:
        for reference, frame in load_stream_inference_input(
            input_uri=input_uri,
            image_extensions=self.__inference_configuration.image_extensions_for_directory_scan,
        ):
            prediction = self.infer(
                inference_input=frame,
                model_id=model_id,
            )
            yield reference, frame, prediction

    @wrap_errors
    def infer(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        if self.__client_mode is HTTPClientMode.V0:
            return self.infer_from_api_v0(
                inference_input=inference_input,
                model_id=model_id,
            )
        return self.infer_from_api_v1(
            inference_input=inference_input,
            model_id=model_id,
        )

    def infer_from_api_v0(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model identifier: {model_id} in use."
            )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=None,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        params = {
            "api_key": self.__api_key,
        }
        params.update(self.__inference_configuration.to_legacy_call_parameters())
        results = []
        for element in encoded_inference_inputs:
            image, scaling_factor = element
            response = requests.post(
                f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
                headers=DEFAULT_HEADERS,
                params=params,
                data=image,
            )
            api_key_safe_raise_for_status(response=response)
            if response_contains_jpeg_image(response=response):
                visualisation = transform_visualisation_bytes(
                    visualisation=response.content,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                parsed_response = {"visualization": visualisation}
            else:
                parsed_response = response.json()
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=scaling_factor,
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def infer_from_api_v1(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        self.__ensure_v1_client_mode()
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_description = self.get_model_description(model_id=model_id_to_be_used)
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=model_description,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        if model_description.task_type not in NEW_INFERENCE_ENDPOINTS:
            raise ModelTaskTypeNotSupportedError(
                f"Model task {model_description.task_type} is not supported by API v1 client."
            )
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        payload = {
            "api_key": self.__api_key,
            "model_id": model_id_to_be_used,
        }
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        results = []
        for element in encoded_inference_inputs:
            image, scaling_factor = element
            payload["image"] = {"type": "base64", "value": image}
            response = requests.post(
                f"{self.__api_url}{endpoint}",
                json=payload,
                headers=DEFAULT_HEADERS,
            )
            api_key_safe_raise_for_status(response=response)
            parsed_response = response.json()
            if parsed_response.get("visualization") is not None:
                parsed_response["visualization"] = transform_base64_visualisation(
                    visualisation=parsed_response["visualization"],
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=scaling_factor,
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def get_model_description(
        self, model_id: str, allow_loading: bool = True
    ) -> ModelDescription:
        self.__ensure_v1_client_mode()
        registered_models = self.list_loaded_models()
        matching_models = [
            e for e in registered_models.models if e.model_id == model_id
        ]
        if len(matching_models) > 0:
            return matching_models[0]
        if allow_loading is True:
            self.load_model(model_id=model_id)
            return self.get_model_description(model_id=model_id, allow_loading=False)
        raise ModelNotInitializedError(
            f"Model {model_id} is not initialised and cannot retrieve its description."
        )

    @wrap_errors
    def list_loaded_models(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.get(f"{self.__api_url}/model/registry")
        response.raise_for_status()
        response_payload = response.json()
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def load_model(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.post(
            f"{self.__api_url}/model/add",
            json={
                "model_id": model_id,
                "api_key": self.__api_key,
            },
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        response_payload = response.json()
        if set_as_default:
            self.__selected_model = model_id
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_model(self, model_id: str) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.post(
            f"{self.__api_url}/model/remove",
            json={
                "model_id": model_id,
            },
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        response_payload = response.json()
        if model_id == self.__selected_model:
            self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_all_models(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.post(f"{self.__api_url}/model/clear")
        response.raise_for_status()
        response_payload = response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def prompt_cogvlm(
        self,
        visual_prompt: ImagesReference,
        text_prompt: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> dict:
        self.__ensure_v1_client_mode()  # Lambda does not support CogVLM, so we require v1 mode of client
        encoded_image = load_static_inference_input(
            inference_input=visual_prompt,
        )
        payload = {
            "api_key": self.__api_key,
            "model_id": "cogvlm",
            "prompt": text_prompt,
        }
        payload = inject_images_into_payload(
            payload=payload,
            encoded_images=encoded_image,
        )
        if chat_history is not None:
            payload["history"] = chat_history
        response = requests.post(
            f"{self.__api_url}/llm/cogvlm",
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @wrap_errors
    def ocr_image(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        results = []
        for element in encoded_inference_inputs:
            image, _ = element
            payload["image"] = {"type": "base64", "value": image}
            response = requests.post(
                self.__wrap_url_with_api_key(f"{self.__api_url}/doctr/ocr"),
                json=payload,
                headers=DEFAULT_HEADERS,
            )
            api_key_safe_raise_for_status(response=response)
            results.append(response.json())
        return unwrap_single_element_list(sequence=results)

    @wrap_errors
    def detect_gazes(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        self.__ensure_v1_client_mode()  # Lambda does not support Gaze, so we require v1 mode of client
        return self._post_images(
            inference_input=inference_input, endpoint="/gaze/gaze_detection"
        )

    @wrap_errors
    def get_clip_image_embeddings(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        return self._post_images(
            inference_input=inference_input, endpoint="/clip/embed_image"
        )

    @wrap_errors
    def get_clip_text_embeddings(
        self, text: Union[str, List[str]]
    ) -> Union[dict, List[dict]]:
        payload = self.__initialise_payload()
        payload["text"] = text
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/embed_text"),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    @wrap_errors
    def clip_compare(
        self,
        subject: Union[str, ImagesReference],
        prompt: Union[str, List[str], ImagesReference, List[ImagesReference]],
        subject_type: str = "image",
        prompt_type: str = "text",
    ) -> Union[dict, List[dict]]:
        """
        Both `subject_type` and `prompt_type` must be either "image" or "text"
        """
        if (
            subject_type not in CLIP_ARGUMENT_TYPES
            or prompt_type not in CLIP_ARGUMENT_TYPES
        ):
            raise InvalidParameterError(
                f"Could not accept `subject_type` and `prompt_type` with values different than {CLIP_ARGUMENT_TYPES}"
            )
        payload = self.__initialise_payload()
        payload["subject_type"] = subject_type
        payload["prompt_type"] = prompt_type
        if subject_type == "image":
            encoded_image = load_static_inference_input(
                inference_input=subject,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_image, key="subject"
            )
        else:
            payload["subject"] = subject
        if prompt_type == "image":
            encoded_inference_inputs = load_static_inference_input(
                inference_input=prompt,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_inference_inputs, key="prompt"
            )
        else:
            payload["prompt"] = prompt
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/compare"),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    def _post_images(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        endpoint: str,
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if model_id is not None:
            payload["model_id"] = model_id
        payload = inject_images_into_payload(
            payload=payload,
            encoded_images=encoded_inference_inputs,
        )
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}{endpoint}"),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    def __initialise_payload(self) -> dict:
        if self.__client_mode is not HTTPClientMode.V0:
            return {"api_key": self.__api_key}
        return {}

    def __wrap_url_with_api_key(self, url: str) -> str:
        if self.__client_mode is not HTTPClientMode.V0:
            return url
        return f"{url}?api_key={self.__api_key}"

    def __ensure_v1_client_mode(self) -> None:
        if self.__client_mode is not HTTPClientMode.V1:
            raise WrongClientModeError("Use client mode `v1` to run this operation.")


def _determine_client_downsizing_parameters(
    client_downsizing_disabled: bool,
    model_description: Optional[ModelDescription],
    default_max_input_size: int,
) -> Tuple[Optional[int], Optional[int]]:
    if client_downsizing_disabled:
        return None, None
    if (
        model_description is None
        or model_description.input_height is None
        or model_description.input_width is None
    ):
        return default_max_input_size, default_max_input_size
    return model_description.input_height, model_description.input_width


def _determine_client_mode(api_url: str) -> HTTPClientMode:
    if "roboflow.com" in api_url:
        return HTTPClientMode.V0
    return HTTPClientMode.V1


def _ensure_model_is_selected(model_id: Optional[str]) -> None:
    if model_id is None:
        raise ModelNotSelectedError("No model was selected to be used.")
