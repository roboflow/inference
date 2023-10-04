from typing import Any, Optional, Union, List, Tuple, Generator, TypeVar

import numpy as np
import requests
from requests import HTTPError

from clients.http.entities import (
    ServerInfo,
    RegisteredModels,
    InferenceConfiguration,
    HTTPClientMode,
    ImagesReference, ModelDescription, CLASSIFICATION_TASK, OBJECT_DETECTION_TASK, INSTANCE_SEGMENTATION_TASK,
)
from clients.http.errors import (
    HTTPClientError,
    HTTPCallErrorError,
    InvalidModelIdentifier, ModelNotInitializedError, ModelTaskTypeNotSupportedError,
)
from clients.http.utils.loaders import (
    load_static_inference_input,
    load_stream_inference_input,
)
from clients.http.utils.post_processing import (
    response_contains_jpeg_image,
    transform_visualisation_bytes,
    transform_base64_visualisation,
)

SUCCESSFUL_STATUS_CODE = 200
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
}
NEW_INFERENCE_ENDPOINTS = {
    INSTANCE_SEGMENTATION_TASK: "/infer/instance_segmentation",
    OBJECT_DETECTION_TASK: "/infer/object_detection",
    CLASSIFICATION_TASK: "/infer/classification",
}

T = TypeVar("T")


def wrap_errors(function: callable) -> callable:
    def decorate(*args, **kwargs) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError as error:
            if "application/json" in error.response.headers.get("Content-Type"):
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
        client_mode: HTTPClientMode = HTTPClientMode.V1,
    ):
        self.__api_url = api_url
        self.__api_key = api_key
        self.__inference_configuration = InferenceConfiguration.init_default()
        self.__client_mode = client_mode
        self.__selected_model: Optional[str] = None

    def configure(
        self, inference_configuration: InferenceConfiguration
    ) -> "InferenceHTTPClient":
        self.__inference_configuration = inference_configuration
        return self

    def use_legacy_client(self) -> "InferenceHTTPClient":
        self.__client_mode = HTTPClientMode.V0
        return self

    def use_new_client(self) -> "InferenceHTTPClient":
        self.__client_mode = HTTPClientMode.V1
        return self

    def use_model(self, model_id: str) -> "InferenceHTTPClient":
        self.__selected_model = model_id
        return self

    @wrap_errors
    def get_server_info(self) -> ServerInfo:
        # API KEY NOT NEEDED!
        response = requests.get(f"{self.__api_url}/info")
        response.raise_for_status()
        response_payload = response.json()
        return ServerInfo.from_dict(response_payload)

    def infer_on_stream(
        self,
        input_uri: str,
        model_id: Optional[str] = None,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        for frame in load_stream_inference_input(
            input_uri=input_uri,
            image_extensions=self.__inference_configuration.image_extensions_for_directory_scan,
        ):
            prediction = self.infer(
                inference_input=frame,
                model_id=model_id,
            )
            yield frame, prediction

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
        model_description = self.get_model_description(model_id=model_id_to_be_used)
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model identifier: {model_id} in use."
            )
        params = {
            "api_key": self.__api_key,
        }
        params.update(self.__inference_configuration.to_legacy_call_parameters())
        results = []
        for element in encoded_inference_inputs:
            response = requests.post(
                f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
                headers=DEFAULT_HEADERS,
                params=params,
                data=element,
            )
            response.raise_for_status()
            if response_contains_jpeg_image(response=response):
                visualisation = transform_visualisation_bytes(
                    visualisation=response.content,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                results.append({"visualization": visualisation})
            else:
                results.append(response.json())
        return unwrap_single_element_list(sequence=results)

    def infer_from_api_v1(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        model_id_to_be_used = model_id or self.__selected_model
        model_description = self.get_model_description(model_id=model_id_to_be_used)
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = {
            "api_key": self.__api_key,
            "model_id": model_id_to_be_used,
        }
        if model_description.task_type not in NEW_INFERENCE_ENDPOINTS:
            raise ModelTaskTypeNotSupportedError(
                f"Model task {model_description.task_type} is not supported by API v1 client."
            )
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        results = []
        for element in encoded_inference_inputs:
            payload["image"] = {"type": "base64", "value": element}
            response = requests.post(
                f"{self.__api_url}{endpoint}",
                json=payload,
                headers=DEFAULT_HEADERS,
            )
            response.raise_for_status()
            parsed_response = response.json()
            if parsed_response.get("visualization") is not None:
                parsed_response["visualization"] = transform_base64_visualisation(
                    visualisation=parsed_response["visualization"],
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def get_model_description(self, model_id: str, allow_loading: bool = True) -> ModelDescription:
        registered_models = self.list_loaded_models()
        matching_models = [e for e in registered_models.models if e.model_id == model_id]
        if len(matching_models) > 0:
            return matching_models[0]
        if allow_loading is True:
            self.load_model(model_id=model_id)
            return self.get_model_description(model_id=model_id, allow_loading=False)
        raise ModelNotInitializedError(f"Model {model_id} is not initialised and cannot retrieve its description.")

    @wrap_errors
    def list_loaded_models(self) -> RegisteredModels:
        # This may be problematic due to route structure in API - given LAMBDA is used - it will not return 404,
        # but attempts to load models that does not exist...
        response = requests.get(f"{self.__api_url}/model/registry")
        response.raise_for_status()
        response_payload = response.json()
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def load_model(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
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
        response = requests.post(f"{self.__api_url}/model/clear")
        response.raise_for_status()
        response_payload = response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)


def unwrap_single_element_list(sequence: List[T]) -> Union[T, List[T]]:
    if len(sequence) == 1:
        return sequence[0]
    return sequence
