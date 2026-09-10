import base64
import os
import warnings
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlencode

import aiohttp
import numpy as np
import requests
from aiohttp import ClientConnectionError, ClientResponseError
from requests import HTTPError, Response

from inference_sdk.config import (
    EXECUTION_ID_HEADER,
    InferenceSDKGuidanceWarning,
    execution_id,
)
from inference_sdk.http.entities import (
    ACTION_RECOGNITION_TASK,
    ALL_ROBOFLOW_API_URLS,
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
    ApiKeyTransport,
    HTTPClientMode,
    ImagesReference,
    InferenceConfiguration,
    ModelDescription,
    RegisteredModels,
    ServerInfo,
    VideoReference,
)
from inference_sdk.http.errors import (
    APIKeyNotProvided,
    FeatureDeprecatedError,
    HTTPCallErrorError,
    HTTPClientError,
    InvalidInputFormatError,
    InvalidModelIdentifier,
    InvalidParameterError,
    ModelNotInitializedError,
    ModelNotSelectedError,
    ModelTaskTypeNotSupportedError,
    RetryError,
    WrongClientModeError,
)
from inference_sdk.http.utils.aliases import (
    resolve_ocr_path,
    resolve_roboflow_model_alias,
)
from inference_sdk.http.utils.depth_maps import (
    decode_depth_estimation_result,
    warn_depth_map_json_format_deprecated,
)
from inference_sdk.http.utils.executors import (
    UNKNOWN_MODEL_ID,
    RequestMethod,
    collect_remote_processing_metadata_from_headers,
    collect_remote_processing_metadata_from_response,
    execute_requests_packages,
    execute_requests_packages_async,
    send_post_request,
)
from inference_sdk.http.utils.iterables import unwrap_single_element_list
from inference_sdk.http.utils.loaders import (
    load_nested_batches_of_inference_input,
    load_static_inference_input,
    load_static_inference_input_async,
    load_stream_inference_input,
    uri_is_http_link,
)
from inference_sdk.http.utils.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    combine_clip_embeddings,
    decode_workflow_outputs,
    filter_model_descriptions,
    response_contains_jpeg_image,
    transform_base64_visualisation,
    transform_visualisation_bytes,
)
from inference_sdk.http.utils.profilling import save_workflows_profiler_trace
from inference_sdk.http.utils.request_building import (
    ImagePlacement,
    RequestData,
    prepare_requests_data,
)
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    deduct_api_key_from_string,
    inject_images_into_payload,
    inject_nested_batches_of_images_into_payload,
)
from inference_sdk.utils.decorators import deprecated, experimental

SUCCESSFUL_STATUS_CODE = 200
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
}

_DEFAULT_API_KEY_TRANSPORT_WARNED = False


def _warn_about_default_api_key_transport_once() -> None:
    global _DEFAULT_API_KEY_TRANSPORT_WARNED
    if _DEFAULT_API_KEY_TRANSPORT_WARNED:
        return
    _DEFAULT_API_KEY_TRANSPORT_WARNED = True
    warnings.warn(
        "This client sends the Roboflow API key through the legacy channel "
        "(query parameter / request body). Sending it as the "
        "`Authorization: Bearer <api_key>` header is recommended and works "
        "with inference servers from release 1.5.0 onward. Opt in with "
        "InferenceConfiguration(api_key_transport='header'); use 'both' for "
        "a transition period safe with every server version. Set "
        "api_key_transport='legacy' "
        "explicitly to keep the current behaviour and silence this warning.",
        InferenceSDKGuidanceWarning,
    )


# Routes taking an image
NEW_INFERENCE_ENDPOINTS = {
    INSTANCE_SEGMENTATION_TASK: "/infer/instance_segmentation",
    OBJECT_DETECTION_TASK: "/infer/object_detection",
    CLASSIFICATION_TASK: "/infer/classification",
    KEYPOINTS_DETECTION_TASK: "/infer/keypoints_detection",
}
# Routes taking a video clip
VIDEO_INFERENCE_ENDPOINTS = {
    ACTION_RECOGNITION_TASK: "/infer/action_recognition",
}
CLIP_ARGUMENT_TYPES = {"image", "text"}

BufferFillingStrategy = Literal[
    "WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST", "ADAPTIVE_DROP_LATEST"
]
BufferConsumptionStrategy = Literal["LAZY", "EAGER"]

if TYPE_CHECKING:
    from inference_sdk.webrtc.client import WebRTCClient


def _collect_processing_time_from_response(
    response: requests.Response,
    model_id: str = UNKNOWN_MODEL_ID,
) -> None:
    collect_remote_processing_metadata_from_response(
        response=response,
        model_id=model_id,
    )


def wrap_errors(function: callable) -> callable:
    def decorate(*args, **kwargs) -> Any:
        try:
            return function(*args, **kwargs)
        except RetryError as error:
            if error.status_code is not None:
                raise HTTPCallErrorError(
                    description=f"Original request failed and retry did not succeed. Status code shows the "
                    f"response of the last request executed.",
                    status_code=error.status_code,
                    api_message=None,
                ) from error
            raise HTTPClientError(
                f"Original request failed and retry did not succeed. Details: {error}"
            ) from error
        except HTTPError as error:
            if "application/json" in error.response.headers.get("Content-Type", ""):
                error_data = error.response.json()
                api_message = (
                    error_data.get("message") or error_data.get("detail") or "N/A"
                )
                if "inner_error_message" in error_data:
                    more_details = error_data["inner_error_message"]
                    api_message = f"{api_message}. More details: {more_details}"
            else:
                api_message = error.response.text
            raise HTTPCallErrorError(
                description=str(error),
                status_code=error.response.status_code,
                api_message=api_message,
            ) from error
        except (ConnectionError, requests.exceptions.ConnectionError) as error:
            raise HTTPClientError(
                f"Error with server connection: {deduct_api_key_from_string(str(error))}"
            ) from error

    return decorate


def wrap_errors_async(function: callable) -> callable:
    async def decorate(*args, **kwargs) -> Any:
        try:
            return await function(*args, **kwargs)
        except ClientResponseError as error:
            raise HTTPCallErrorError(
                description=deduct_api_key_from_string(value=str(error)),
                status_code=error.status,
                api_message=deduct_api_key_from_string(error.message),
            ) from error
        except ClientConnectionError as error:
            raise HTTPClientError(
                f"Error with server connection: {deduct_api_key_from_string(str(error))}"
            ) from error

    return decorate


class InferenceHTTPClient:
    """HTTP client for making inference requests to Roboflow's API.

    This client handles authentication, request formatting, and error handling for
    interacting with Roboflow's inference endpoints. It supports both synchronous
    and asynchronous requests.

    Attributes:
        inference_configuration (InferenceConfiguration): Configuration settings for
            inference requests.
        client_mode (HTTPClientMode): The API version mode being used (V0 or V1).
        selected_model (Optional[str]): Currently selected model identifier, if any.

    Example:
        ```python
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="http://localhost:9001", # use local inference server
            # api_key="<YOUR API KEY>" # optional to access your private data and models
        )

        result = client.run_workflow(
            workspace_name="roboflow-docs",
            workflow_id="model-comparison",
            images={
                "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
            },
            parameters={
                "model1": "yolov8n-640",
                "model2": "yolov11n-640"
            }
        )
        ```
    """

    @classmethod
    def init(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
    ) -> "InferenceHTTPClient":
        """Initialize a new InferenceHTTPClient instance.

        Args:
            api_url (str): The base URL for the inference API.
            api_key (Optional[str], optional): API key for authentication. Defaults to None.

        Returns:
            InferenceHTTPClient: A new instance of the InferenceHTTPClient.
        """
        return cls(api_url=api_url, api_key=api_key)

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
    ):
        """Initialize a new InferenceHTTPClient instance.

        The channel used to send the API key (query/body vs
        `Authorization: Bearer` header) is controlled by the
        `api_key_transport` field of `InferenceConfiguration` - see
        `configure()` / `use_configuration()`.

        Args:
            api_url (str): The base URL for the inference API.
            api_key (Optional[str], optional): API key for authentication. Defaults to None.
        """
        self.__api_url = api_url
        self.__api_key = api_key
        self.__inference_configuration = InferenceConfiguration.init_default()
        self.__client_mode = _determine_client_mode(api_url=api_url)
        self.__selected_model: Optional[str] = None
        self.__webrtc_client: Optional["WebRTCClient"] = None
        self.__webrtc_client_transport: Optional[ApiKeyTransport] = None
        self.__webrtc_transport_stickiness_warned = False

    @property
    def inference_configuration(self) -> InferenceConfiguration:
        """Get the current inference configuration.

        Returns:
            InferenceConfiguration: The current inference configuration settings.
        """
        return self.__inference_configuration

    @property
    def client_mode(self) -> HTTPClientMode:
        """Get the current client mode.

        Returns:
            HTTPClientMode: The current API version mode (V0 or V1).
        """
        return self.__client_mode

    @property
    def selected_model(self) -> Optional[str]:
        """Get the currently selected model identifier.

        Returns:
            Optional[str]: The identifier of the currently selected model, if any.
        """
        return self.__selected_model

    @property
    def webrtc(self) -> "WebRTCClient":
        """Lazy accessor for the WebRTC client namespace.

        Returns:
            WebRTCClient: Namespaced WebRTC API bound to this HTTP client.
        """
        from inference_sdk.webrtc.client import WebRTCClient

        if self.__webrtc_client is None:
            # The transport is captured ONCE here - later configuration
            # changes do not re-sync it (see __warn_if_webrtc_transport_is_stale).
            self.__webrtc_client_transport = self.__resolved_api_key_transport()
            self.__webrtc_client = WebRTCClient(
                self.__api_url,
                self.__api_key,
                api_key_transport=self.__webrtc_client_transport.value,
            )
        return self.__webrtc_client

    @contextmanager
    def use_configuration(
        self, inference_configuration: InferenceConfiguration
    ) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use a different inference configuration.

        Args:
            inference_configuration (InferenceConfiguration): The temporary configuration to use.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance with temporary configuration.
        """
        previous_configuration = self.__inference_configuration
        self.__inference_configuration = inference_configuration
        self.__warn_if_webrtc_transport_is_stale()
        try:
            yield self
        finally:
            self.__inference_configuration = previous_configuration
            self.__warn_if_webrtc_transport_is_stale()

    def configure(
        self, inference_configuration: InferenceConfiguration
    ) -> "InferenceHTTPClient":
        """Configure the client with new inference settings.

        Args:
            inference_configuration (InferenceConfiguration): The new configuration to apply.

        Returns:
            InferenceHTTPClient: The client instance with updated configuration.
        """
        self.__inference_configuration = inference_configuration
        self.__warn_if_webrtc_transport_is_stale()
        return self

    def select_api_v0(self) -> "InferenceHTTPClient":
        """Select API version 0 for client operations.

        Returns:
            InferenceHTTPClient: The client instance with API v0 selected.
        """
        self.__client_mode = HTTPClientMode.V0
        return self

    def select_api_v1(self) -> "InferenceHTTPClient":
        """Select API version 1 for client operations.

        Returns:
            InferenceHTTPClient: The client instance with API v1 selected.
        """
        self.__client_mode = HTTPClientMode.V1
        return self

    @contextmanager
    def use_api_v0(self) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use API version 0 for client operations.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using API v0.
        """
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V0
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    @contextmanager
    def use_api_v1(self) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use API version 1 for client operations.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using API v1.
        """
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V1
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    def select_model(self, model_id: str) -> "InferenceHTTPClient":
        """Select a model for inference operations.

        Args:
            model_id (str): The identifier of the model to select.

        Returns:
            InferenceHTTPClient: The client instance with the selected model.
        """
        self.__selected_model = model_id
        return self

    @contextmanager
    def use_model(self, model_id: str) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use a specific model for inference operations.

        Args:
            model_id (str): The identifier of the model to use.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using the specified model.
        """
        previous_model = self.__selected_model
        self.__selected_model = model_id
        try:
            yield self
        finally:
            self.__selected_model = previous_model

    @wrap_errors
    def get_server_info(self) -> ServerInfo:
        """Get information about the inference server.

        Returns:
            ServerInfo: Information about the server configuration and status.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        response = requests.get(f"{self.__api_url}/info")
        response.raise_for_status()
        response_payload = response.json()
        return ServerInfo.from_dict(response_payload)

    def infer_on_stream(
        self,
        input_uri: str,
        model_id: Optional[str] = None,
    ) -> Generator[Tuple[Union[str, int], np.ndarray, dict], None, None]:
        """Run inference on a video stream or sequence of images.

        Args:
            input_uri (str): URI of the input stream or directory.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Yields:
            Generator[Tuple[Union[str, int], np.ndarray, dict], None, None]: Tuples of (frame reference, frame data, prediction).
        """
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
        """Run inference on one or more images.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return self.infer_from_api_v0(
                inference_input=inference_input,
                model_id=model_id,
            )
        return self.infer_from_api_v1(
            inference_input=inference_input,
            model_id=model_id,
        )

    @wrap_errors_async
    async def infer_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference asynchronously on one or more images.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return await self.infer_from_api_v0_async(
                inference_input=inference_input,
                model_id=model_id,
            )
        return await self.infer_from_api_v1_async(
            inference_input=inference_input,
            model_id=model_id,
        )

    def infer_from_api_v0(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using API v0.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            ModelNotSelectedError: If no model is selected.
            APIKeyNotProvided: If API key is required but not provided.
            InvalidModelIdentifier: If the model identifier format is invalid.
        """
        requests_data = self._prepare_infer_from_api_v0_request_data(
            inference_input=inference_input,
            model_id=model_id,
        )
        responses = self._execute_infer_from_api_request(
            requests_data=requests_data,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            if response_contains_jpeg_image(response=response):
                visualisation = transform_visualisation_bytes(
                    visualisation=response.content,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                parsed_response = {"visualization": visualisation}
            else:
                parsed_response = response.json()
                if parsed_response.get("visualization") is not None:
                    parsed_response["visualization"] = transform_base64_visualisation(
                        visualisation=parsed_response["visualization"],
                        expected_format=self.__inference_configuration.output_visualisation_format,
                    )
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=request_data.image_scaling_factors[0],
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def _execute_infer_from_api_request(
        self,
        requests_data: List[RequestData],
    ) -> List[Response]:
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return responses

    def _prepare_infer_from_api_v0_request_data(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> List[RequestData]:
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        _ensure_api_key_provided(api_key=self.__api_key)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model id: {model_id}. Expected format: project_id/model_version_id."
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
        params = self.__legacy_api_key_payload()
        params.update(self.__inference_configuration.to_legacy_call_parameters())

        execution_id_value = execution_id.get()
        headers = DEFAULT_HEADERS
        if execution_id_value:
            headers = headers.copy()
            headers[EXECUTION_ID_HEADER] = execution_id_value

        requests_data = prepare_requests_data(
            url=f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(headers),
            parameters=params,
            payload=None,
            max_batch_size=1,
            image_placement=ImagePlacement.DATA,
        )
        return requests_data

    async def infer_from_api_v0_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using API v0 asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            ModelNotSelectedError: If no model is selected.
            APIKeyNotProvided: If API key is required but not provided.
            InvalidModelIdentifier: If the model identifier format is invalid.
        """
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        _ensure_api_key_provided(api_key=self.__api_key)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model id: {model_id}. Expected format: project_id/model_version_id."
            )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=None,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        params = self.__legacy_api_key_payload()
        params.update(self.__inference_configuration.to_legacy_call_parameters())

        execution_id_value = execution_id.get()
        headers = DEFAULT_HEADERS
        if execution_id_value:
            headers = headers.copy()
            headers[EXECUTION_ID_HEADER] = execution_id_value

        requests_data = prepare_requests_data(
            url=f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(headers),
            parameters=params,
            payload=None,
            max_batch_size=1,
            image_placement=ImagePlacement.DATA,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            if not issubclass(type(response), dict):
                visualisation = transform_visualisation_bytes(
                    visualisation=response,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                parsed_response = {"visualization": visualisation}
            else:
                parsed_response = response
                if parsed_response.get("visualization") is not None:
                    parsed_response["visualization"] = transform_base64_visualisation(
                        visualisation=parsed_response["visualization"],
                        expected_format=self.__inference_configuration.output_visualisation_format,
                    )
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=request_data.image_scaling_factors[0],
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def infer_from_api_v1(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        requests_data = self._prepare_infer_from_api_v1_request_data(
            inference_input=inference_input,
            model_id=model_id,
        )
        responses = self._execute_infer_from_api_request(
            requests_data=requests_data,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            parsed_response = response.json()
            if not issubclass(type(parsed_response), list):
                parsed_response = [parsed_response]
            for parsed_response_element, scaling_factor in zip(
                parsed_response, request_data.image_scaling_factors
            ):
                if parsed_response_element.get("visualization") is not None:
                    parsed_response_element["visualization"] = (
                        transform_base64_visualisation(
                            visualisation=parsed_response_element["visualization"],
                            expected_format=self.__inference_configuration.output_visualisation_format,
                        )
                    )
                parsed_response_element = adjust_prediction_to_client_scaling_factor(
                    prediction=parsed_response_element,
                    scaling_factor=scaling_factor,
                )
                results.append(parsed_response_element)
        return unwrap_single_element_list(sequence=results)

    def _prepare_infer_from_api_v1_request_data(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> List[RequestData]:
        self.__ensure_v1_client_mode()
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_description = self.get_model_description(model_id=model_id_to_be_used)
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=model_description,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        _ensure_task_takes_an_image(task_type=model_description.task_type)
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        payload = {
            **self.__legacy_api_key_payload(),
            "model_id": model_id_to_be_used,
        }
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        query_params = self.__inference_configuration.to_api_v1_query_parameters()
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}{endpoint}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            parameters=query_params,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        return requests_data

    async def infer_from_api_v1_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        self.__ensure_v1_client_mode()
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_description = await self.get_model_description_async(
            model_id=model_id_to_be_used
        )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=model_description,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        _ensure_task_takes_an_image(
            task_type=model_description.task_type, asynchronous=True
        )
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        payload = {
            **self.__legacy_api_key_payload(),
            "model_id": model_id_to_be_used,
        }
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        query_params = self.__inference_configuration.to_api_v1_query_parameters()
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}{endpoint}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            parameters=query_params,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, parsed_response in zip(requests_data, responses):
            if not issubclass(type(parsed_response), list):
                parsed_response = [parsed_response]
            for parsed_response_element, scaling_factor in zip(
                parsed_response, request_data.image_scaling_factors
            ):
                if parsed_response_element.get("visualization") is not None:
                    parsed_response_element["visualization"] = (
                        transform_base64_visualisation(
                            visualisation=parsed_response_element["visualization"],
                            expected_format=self.__inference_configuration.output_visualisation_format,
                        )
                    )
                parsed_response_element = adjust_prediction_to_client_scaling_factor(
                    prediction=parsed_response_element,
                    scaling_factor=scaling_factor,
                )
                results.append(parsed_response_element)
        return unwrap_single_element_list(sequence=results)

    def get_model_description(
        self, model_id: str, allow_loading: bool = True
    ) -> ModelDescription:
        """Get the description of a model.

        Args:
            model_id (str): The identifier of the model.
            allow_loading (bool, optional): Whether to load the model if not already loaded. Defaults to True.

        Returns:
            ModelDescription: Description of the model.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            ModelNotInitializedError: If the model is not initialized and cannot be loaded.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        registered_models = self.list_loaded_models()
        matching_model = filter_model_descriptions(
            descriptions=registered_models.models,
            model_id=de_aliased_model_id,
        )
        if matching_model is None and allow_loading is True:
            registered_models = self.load_model(model_id=de_aliased_model_id)
            matching_model = filter_model_descriptions(
                descriptions=registered_models.models,
                model_id=de_aliased_model_id,
            )
        if matching_model is not None:
            return matching_model
        raise ModelNotInitializedError(
            f"Model {model_id} (de-aliased: {de_aliased_model_id}) is not initialised and cannot "
            f"retrieve its description."
        )

    async def get_model_description_async(
        self, model_id: str, allow_loading: bool = True
    ) -> ModelDescription:
        """Get the description of a model asynchronously.

        Args:
            model_id (str): The identifier of the model.
            allow_loading (bool, optional): Whether to load the model if not already loaded. Defaults to True.

        Returns:
            ModelDescription: Description of the model.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            ModelNotInitializedError: If the model is not initialized and cannot be loaded.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        registered_models = await self.list_loaded_models_async()
        matching_model = filter_model_descriptions(
            descriptions=registered_models.models,
            model_id=de_aliased_model_id,
        )
        if matching_model is None and allow_loading is True:
            registered_models = await self.load_model_async(
                model_id=de_aliased_model_id
            )
            matching_model = filter_model_descriptions(
                descriptions=registered_models.models,
                model_id=de_aliased_model_id,
            )
        if matching_model is not None:
            return matching_model
        raise ModelNotInitializedError(
            f"Model {model_id} (de-aliased: {de_aliased_model_id}) is not initialised and cannot "
            f"retrieve its description."
        )

    @wrap_errors
    def list_loaded_models(self) -> RegisteredModels:
        """List all models currently loaded on the server.

        Returns:
            RegisteredModels: Information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        url = f"{self.__api_url}/model/registry"
        if self.__resolved_api_key_transport() is not ApiKeyTransport.HEADER:
            url = f"{url}?api_key={self.__api_key}"
        response = requests.get(url, headers=self.__headers_with_auth(None))
        response.raise_for_status()
        response_payload = response.json()
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def list_loaded_models_async(self) -> RegisteredModels:
        """List all models currently loaded on the server asynchronously.

        Returns:
            RegisteredModels: Information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        url = f"{self.__api_url}/model/registry"
        if self.__resolved_api_key_transport() is not ApiKeyTransport.HEADER:
            url = f"{url}?api_key={self.__api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=self.__headers_with_auth(None)
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
                return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def load_model(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
        """Load a model onto the server.

        Args:
            model_id (str): The identifier of the model to load.
            set_as_default (bool, optional): Whether to set this model as the default. Defaults to False.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        response = requests.post(
            f"{self.__api_url}/model/add",
            json={
                "model_id": de_aliased_model_id,
                **self.__legacy_api_key_payload(),
            },
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
        )
        response.raise_for_status()
        response_payload = response.json()
        if set_as_default:
            self.__selected_model = de_aliased_model_id
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def load_model_async(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
        """Load a model onto the server asynchronously.

        Args:
            model_id (str): The identifier of the model to load.
            set_as_default (bool, optional): Whether to set this model as the default. Defaults to False.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        payload = {
            "model_id": de_aliased_model_id,
            **self.__legacy_api_key_payload(),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.__api_url}/model/add",
                json=payload,
                headers=self.__headers_with_auth(DEFAULT_HEADERS),
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        if set_as_default:
            self.__selected_model = de_aliased_model_id
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_model(self, model_id: str) -> RegisteredModels:
        """Unload a model from the server.

        Args:
            model_id (str): The identifier of the model to unload.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        response = requests.post(
            f"{self.__api_url}/model/remove",
            json={
                "model_id": de_aliased_model_id,
            },
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
        )
        response.raise_for_status()
        response_payload = response.json()
        if (
            de_aliased_model_id == self.__selected_model
            or model_id == self.__selected_model
        ):
            self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def unload_model_async(self, model_id: str) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.__api_url}/model/remove",
                json={
                    "model_id": de_aliased_model_id,
                },
                headers=self.__headers_with_auth(DEFAULT_HEADERS),
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        if (
            de_aliased_model_id == self.__selected_model
            or model_id == self.__selected_model
        ):
            self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_all_models(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.post(
            f"{self.__api_url}/model/clear",
            headers=self.__headers_with_auth(None),
        )
        response.raise_for_status()
        response_payload = response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def unload_all_models_async(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.__api_url}/model/clear",
                headers=self.__headers_with_auth(None),
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def ocr_image(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model: str = "doctr",
        version: Optional[str] = None,
        quantize: Optional[bool] = None,
        generate_bounding_boxes: Optional[bool] = None,
        language_codes: Optional[List[str]] = None,
    ) -> Union[dict, List[dict]]:
        """Run OCR on input image(s).

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for OCR.
            model (str, optional): OCR model to use ('doctr', 'trocr', 'easy_ocr' or 'pp_ocr'). Defaults to "doctr".
            version (Optional[str], optional): Model version to use. Defaults to None.
                For trocr, supported versions are: 'trocr-small-printed', 'trocr-base-printed', 'trocr-large-printed'.
                For pp_ocr, the version selects the detection and recognition stages as
                '{detection}-{recognition}', where each stage is one of 'none', 'tiny', 'small' or 'medium'
                (default 'small-small'). Passing a single token (e.g. 'small') applies it to both stages.
                Setting a stage to 'none' skips it: 'small-none' runs detection only (boxes without text),
                'none-small' runs recognition only (each full input image is read as a single text line).
                'none-none' is invalid.
            quantize: (Optional[bool]): flag of EasyOCR to decide which version of model to load
            generate_bounding_boxes: (Optional[bool]): flag of some models (like DocTR) to decide if output variant
                with sv.Detections(...) compatible bounding boxes should be returned (due to historical reasons, some
                old implementations were flattening detected OCR structure into text and were only returning that as
                results).
            language_codes: (Optional[List[str]]): Parameter of EasyOCR that dictates the code of languages that
                model should recognise (leave blank for default for given OCR model version).
        Returns:
            Union[dict, List[dict]]: OCR results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if version:
            key = f"{model.lower()}_version_id"
            payload[key] = version
        if quantize is not None:
            payload["quantize"] = quantize
        if generate_bounding_boxes is not None:
            payload["generate_bounding_boxes"] = generate_bounding_boxes
        if language_codes is not None:
            payload["language_codes"] = language_codes
        model_path = resolve_ocr_path(model_name=model)
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{model_path}")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = [r.json() for r in responses]
        return unwrap_single_element_list(sequence=results)

    @wrap_errors_async
    async def ocr_image_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model: str = "doctr",
        version: Optional[str] = None,
        quantize: Optional[bool] = None,
        generate_bounding_boxes: Optional[bool] = None,
        language_codes: Optional[List[str]] = None,
    ) -> Union[dict, List[dict]]:
        """Run OCR on input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for OCR.
            model (str, optional): OCR model to use ('doctr', 'trocr', 'easy_ocr' or 'pp_ocr'). Defaults to "doctr".
            version (Optional[str], optional): Model version to use. Defaults to None.
                For trocr, supported versions are: 'trocr-small-printed', 'trocr-base-printed', 'trocr-large-printed'.
                For pp_ocr, the version selects the detection and recognition stages as
                '{detection}-{recognition}', where each stage is one of 'none', 'tiny', 'small' or 'medium'
                (default 'small-small'). Passing a single token (e.g. 'small') applies it to both stages.
                Setting a stage to 'none' skips it: 'small-none' runs detection only (boxes without text),
                'none-small' runs recognition only (each full input image is read as a single text line).
                'none-none' is invalid.
            quantize: (Optional[bool]): flag of EasyOCR to decide which version of model to load
            generate_bounding_boxes: (Optional[bool]): flag of some models (like DocTR) to decide if output variant
                with sv.Detections(...) compatible bounding boxes should be returned (due to historical reasons, some
                old implementations were flattening detected OCR structure into text and were only returning that as
                results).
            language_codes: (Optional[List[str]]): Parameter of EasyOCR that dictates the code of languages that
                model should recognise (leave blank for default for given OCR model version).
        Returns:
            Union[dict, List[dict]]: OCR results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if version:
            key = f"{model.lower()}_version_id"
            payload[key] = version
        if quantize is not None:
            payload["quantize"] = quantize
        if generate_bounding_boxes is not None:
            payload["generate_bounding_boxes"] = generate_bounding_boxes
        if language_codes is not None:
            payload["language_codes"] = language_codes
        model_path = resolve_ocr_path(model_name=model)
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{model_path}")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return unwrap_single_element_list(sequence=responses)

    def detect_gazes(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        """Deprecated. Always raises FeatureDeprecatedError.

        Gaze detection has been removed from inference along with the
        MediaPipe dependency. This helper short-circuits client-side and
        never issues a network call.

        Raises:
            FeatureDeprecatedError: Always.
        """
        raise FeatureDeprecatedError(
            feature="InferenceHTTPClient.detect_gazes",
            reason="MediaPipe dependency removed from inference.",
            removal_release="end of Q2 2026",
        )

    async def detect_gazes_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        """Deprecated. Always raises FeatureDeprecatedError.

        Gaze detection has been removed from inference along with the
        MediaPipe dependency. This helper short-circuits client-side and
        never issues a network call.

        Raises:
            FeatureDeprecatedError: Always.
        """
        raise FeatureDeprecatedError(
            feature="InferenceHTTPClient.detect_gazes_async",
            reason="MediaPipe dependency removed from inference.",
            removal_release="end of Q2 2026",
        )

    @wrap_errors
    def get_clip_image_embeddings(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input image(s).

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {}
        if clip_version is not None:
            extra_payload["clip_version_id"] = clip_version
        result = self._post_images(
            inference_input=inference_input,
            endpoint="/clip/embed_image",
            extra_payload=extra_payload,
        )
        result = combine_clip_embeddings(embeddings=result)
        return unwrap_single_element_list(result)

    @wrap_errors_async
    async def get_clip_image_embeddings_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {}
        if clip_version is not None:
            extra_payload["clip_version_id"] = clip_version
        result = await self._post_images_async(
            inference_input=inference_input,
            endpoint="/clip/embed_image",
            extra_payload=extra_payload,
        )
        result = combine_clip_embeddings(embeddings=result)
        return unwrap_single_element_list(result)

    @wrap_errors
    def get_clip_text_embeddings(
        self,
        text: Union[str, List[str]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input text(s).

        Args:
            text (Union[str, List[str]]): Input text(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input text(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = self.__initialise_payload()
        payload["text"] = text
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        headers = DEFAULT_HEADERS.copy()
        execution_id_value = execution_id.get()
        if execution_id_value is not None:
            headers[EXECUTION_ID_HEADER] = execution_id_value
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/embed_text"),
            json=payload,
            headers=self.__headers_with_auth(headers),
        )
        _collect_processing_time_from_response(
            response, model_id=clip_version or "clip"
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    @wrap_errors_async
    async def get_clip_text_embeddings_async(
        self,
        text: Union[str, List[str]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input text(s) asynchronously.

        Args:
            text (Union[str, List[str]]): Input text(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input text(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = self.__initialise_payload()
        payload["text"] = text
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.__wrap_url_with_api_key(f"{self.__api_url}/clip/embed_text"),
                json=payload,
                headers=self.__headers_with_auth(DEFAULT_HEADERS),
                # Billing parameters travel on the URL via __wrap_url_with_api_key; kept explicit because aioresponses tests pin this kwarg.
                params=None,
            ) as response:
                response.raise_for_status()
                collect_remote_processing_metadata_from_headers(
                    headers=response.headers,
                    fallback_model_id=clip_version or "clip",
                )
                response_payload = await response.json()
        return unwrap_single_element_list(sequence=response_payload)

    @wrap_errors
    def clip_compare(
        self,
        subject: Union[str, ImagesReference],
        prompt: Union[str, List[str], ImagesReference, List[ImagesReference]],
        subject_type: str = "image",
        prompt_type: str = "text",
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Compare a subject against prompts using CLIP embeddings.

        Args:
            subject (Union[str, ImagesReference]): The subject to compare (image or text).
            prompt (Union[str, List[str], ImagesReference, List[ImagesReference]]): The prompt(s) to compare against.
            subject_type (str, optional): Type of subject ('image' or 'text'). Defaults to "image".
            prompt_type (str, optional): Type of prompt(s) ('image' or 'text'). Defaults to "text".
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Comparison results between subject and prompt(s).

        Raises:
            InvalidParameterError: If subject_type or prompt_type is invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
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
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
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

        headers = DEFAULT_HEADERS.copy()
        execution_id_value = execution_id.get()
        if execution_id_value is not None:
            headers[EXECUTION_ID_HEADER] = execution_id_value
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/compare"),
            json=payload,
            headers=self.__headers_with_auth(headers),
        )
        _collect_processing_time_from_response(
            response, model_id=clip_version or "clip"
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @wrap_errors_async
    async def clip_compare_async(
        self,
        subject: Union[str, ImagesReference],
        prompt: Union[str, List[str], ImagesReference, List[ImagesReference]],
        subject_type: str = "image",
        prompt_type: str = "text",
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Compare a subject against prompts using CLIP embeddings asynchronously.

        Args:
            subject (Union[str, ImagesReference]): The subject to compare (image or text).
            prompt (Union[str, List[str], ImagesReference, List[ImagesReference]]): The prompt(s) to compare against.
            subject_type (str, optional): Type of subject ('image' or 'text'). Defaults to "image".
            prompt_type (str, optional): Type of prompt(s) ('image' or 'text'). Defaults to "text".
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Comparison results between subject and prompt(s).

        Raises:
            InvalidParameterError: If subject_type or prompt_type is invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
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
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        if subject_type == "image":
            encoded_image = await load_static_inference_input_async(
                inference_input=subject,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_image, key="subject"
            )
        else:
            payload["subject"] = subject
        if prompt_type == "image":
            encoded_inference_inputs = await load_static_inference_input_async(
                inference_input=prompt,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_inference_inputs, key="prompt"
            )
        else:
            payload["prompt"] = prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.__wrap_url_with_api_key(f"{self.__api_url}/clip/compare"),
                json=payload,
                headers=self.__headers_with_auth(DEFAULT_HEADERS),
                # Billing parameters travel on the URL via __wrap_url_with_api_key; kept explicit because aioresponses tests pin this kwarg.
                params=None,
            ) as response:
                response.raise_for_status()
                collect_remote_processing_metadata_from_headers(
                    headers=response.headers,
                    fallback_model_id=clip_version or "clip",
                )
                return await response.json()

    @wrap_errors
    def get_perception_encoder_image_embeddings(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        perception_encoder_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get Perception Encoder embeddings for input image(s)."""
        extra_payload = {}
        if perception_encoder_version is not None:
            extra_payload["perception_encoder_version_id"] = perception_encoder_version
        result = self._post_images(
            inference_input=inference_input,
            endpoint="/perception_encoder/embed_image",
            extra_payload=extra_payload,
        )
        return unwrap_single_element_list(result)

    @wrap_errors
    def get_perception_encoder_text_embeddings(
        self,
        text: Union[str, List[str]],
        perception_encoder_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get Perception Encoder embeddings for input text(s)."""
        payload = self.__initialise_payload()
        payload["text"] = text
        if perception_encoder_version is not None:
            payload["perception_encoder_version_id"] = perception_encoder_version

        headers = DEFAULT_HEADERS.copy()
        execution_id_value = execution_id.get()
        if execution_id_value is not None:
            headers[EXECUTION_ID_HEADER] = execution_id_value
        response = requests.post(
            self.__wrap_url_with_api_key(
                f"{self.__api_url}/perception_encoder/embed_text"
            ),
            json=payload,
            headers=self.__headers_with_auth(headers),
        )
        _collect_processing_time_from_response(
            response,
            model_id=perception_encoder_version or "perception_encoder",
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    @wrap_errors
    def infer_lmm(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: str,
        prompt: Optional[str] = None,
        model_id_in_path: bool = False,
        max_new_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using a Large Multimodal Model (LMM).

        This method supports various vision-language models including Florence-2,
        Moondream2, SmolVLM, Qwen2.5-VL, Qwen3-VL, and PaliGemma.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for inference. Can be file paths, URLs, base64 strings, numpy arrays, or PIL images.
            model_id (str): The identifier of the LMM model to use. Examples include:
                - "florence-2-base", "florence-2-large" for Florence-2
                - "moondream2/moondream2_2b_jul24" for Moondream2
                - "smolvlm2/smolvlm-2.2b-instruct" for SmolVLM
                - "qwen25-vl-7b" for Qwen2.5-VL
                - "qwen3vl-2b-instruct" for Qwen3-VL
            prompt (Optional[str], optional): Text prompt to guide the model. Defaults to None.
            model_id_in_path (bool, optional): If True, includes model_id in the URL path
                (e.g., /infer/lmm/florence-2-base) which enables path-based routing.
                If False (default), model_id is only sent in the request body.
            max_new_tokens (Optional[int], optional): Maximum number of tokens to generate.
                If not provided, the server-side model default is used.
            enable_thinking (Optional[bool], optional): Enables reasoning mode for models
                that support it. If not provided, the server-side model default is used.

        Returns:
            Union[dict, List[dict]]: Inference results containing the model response.
                The structure depends on the specific model used.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {"model_id": model_id}
        if prompt is not None:
            extra_payload["prompt"] = prompt
        if max_new_tokens is not None:
            extra_payload["max_new_tokens"] = max_new_tokens
        if enable_thinking is not None:
            extra_payload["enable_thinking"] = enable_thinking

        if model_id_in_path:
            endpoint = f"/infer/lmm/{model_id}"
        else:
            endpoint = "/infer/lmm"

        result = self._post_images(
            inference_input=inference_input,
            endpoint=endpoint,
            extra_payload=extra_payload,
        )
        return result

    @wrap_errors_async
    async def infer_lmm_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: str,
        prompt: Optional[str] = None,
        model_id_in_path: bool = False,
        max_new_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using a Large Multimodal Model (LMM) asynchronously.

        This method supports various vision-language models including Florence-2,
        Moondream2, SmolVLM, Qwen2.5-VL, Qwen3-VL, and PaliGemma.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for inference. Can be file paths, URLs, base64 strings, numpy arrays, or PIL images.
            model_id (str): The identifier of the LMM model to use.
            prompt (Optional[str], optional): Text prompt to guide the model. Defaults to None.
            model_id_in_path (bool, optional): If True, includes model_id in the URL path
                (e.g., /infer/lmm/florence-2-base) which enables path-based routing.
                If False (default), model_id is only sent in the request body.
            max_new_tokens (Optional[int], optional): Maximum number of tokens to generate.
                If not provided, the server-side model default is used.
            enable_thinking (Optional[bool], optional): Enables reasoning mode for models
                that support it. If not provided, the server-side model default is used.

        Returns:
            Union[dict, List[dict]]: Inference results containing the model response.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {"model_id": model_id}
        if prompt is not None:
            extra_payload["prompt"] = prompt
        if max_new_tokens is not None:
            extra_payload["max_new_tokens"] = max_new_tokens
        if enable_thinking is not None:
            extra_payload["enable_thinking"] = enable_thinking

        if model_id_in_path:
            endpoint = f"/infer/lmm/{model_id}"
        else:
            endpoint = "/infer/lmm"

        result = await self._post_images_async(
            inference_input=inference_input,
            endpoint=endpoint,
            extra_payload=extra_payload,
        )
        return result

    @wrap_errors
    def depth_estimation(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: str = "depth-anything-v3/small",
        model_id_in_path: bool = False,
        depth_map_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run depth estimation on input image(s).

        This method estimates depth maps from images using models like Depth Anything.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for depth estimation. Can be file paths, URLs, base64 strings, numpy arrays,
                or PIL images.
            model_id (str, optional): The depth estimation model to use. Defaults to
                "depth-anything-v3/small". Supported models include:
                - "depth-anything-v2/small"
                - "depth-anything-v3/small"
                - "depth-anything-v3/base"
            model_id_in_path (bool, optional): If True, includes model_id in the URL path
                (e.g., /infer/depth-estimation/depth-anything-v3/small), which enables
                path-based routing. If False (default), model_id is only sent in the
                request body.
            depth_map_format (str, optional): Requested serialization for
                `normalized_depth` on the wire: "json" (default, legacy nested
                float list), "png16" (compact base64 16-bit PNG, typically >10x
                smaller payload, decoded client-side to a numpy array) or "png8"
                (smaller still, 256 depth levels). The "json" default is
                deprecated: in one of the first `inference` releases of 2027 the
                default becomes "png16" in a breaking way (`normalized_depth`
                turns into a numpy.ndarray), and an
                InferenceSDKDeprecationWarning is emitted when "json" is used
                (shown once per process under default warning filters).
                Servers that predate this field ignore it and return the
                legacy list.

        Returns:
            Union[dict, List[dict]]: Depth estimation results containing:
                - normalized_depth: Per-image normalized ordinal depth as a list,
                  where 1 is nearest and 0 is farthest. Values are not physical
                  distances or directly comparable across images or model families
                  without calibration. Nested float list for "json" (default);
                  numpy array for "png16"/"png8" (PNG payloads are decoded
                  automatically; legacy servers returning float lists pass
                  through unchanged regardless of the requested format)
                - image: Hex-encoded visualization of the depth map

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if depth_map_format == "json":
            warn_depth_map_json_format_deprecated()
        extra_payload = {
            "model_id": model_id,
            "depth_map_format": depth_map_format,
        }
        if model_id_in_path:
            endpoint = f"/infer/depth-estimation/{model_id}"
        else:
            endpoint = "/infer/depth-estimation"
        result = self._post_images(
            inference_input=inference_input,
            endpoint=endpoint,
            extra_payload=extra_payload,
        )
        return decode_depth_estimation_result(result)

    @wrap_errors_async
    async def depth_estimation_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: str = "depth-anything-v3/small",
        model_id_in_path: bool = False,
        depth_map_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run depth estimation on input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for depth estimation.
            model_id (str, optional): The depth estimation model to use. Defaults to
                "depth-anything-v3/small".
            model_id_in_path (bool, optional): If True, includes model_id in the URL path
                for path-based routing. If False (default), model_id is only sent in the
                request body.
            depth_map_format (str, optional): Requested serialization for
                `normalized_depth` on the wire: "json" (default, legacy nested
                float list), "png16" (compact base64 16-bit PNG decoded
                client-side to a numpy array) or "png8" (smaller still, 256
                depth levels). The "json" default is deprecated: in one of the
                first `inference` releases of 2027 the default becomes "png16"
                in a breaking way (`normalized_depth` turns into a
                numpy.ndarray), and an InferenceSDKDeprecationWarning is
                emitted when "json" is used (shown once per process under
                default warning filters). Servers that predate this field
                ignore it and return the legacy list.

        Returns:
            Union[dict, List[dict]]: Depth estimation results containing per-image
                normalized ordinal depth, where 1 is nearest and 0 is farthest. `normalized_depth`
                is a nested float list for "json" (default) and a numpy array for
                "png16"/"png8".

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if depth_map_format == "json":
            warn_depth_map_json_format_deprecated()
        extra_payload = {
            "model_id": model_id,
            "depth_map_format": depth_map_format,
        }
        if model_id_in_path:
            endpoint = f"/infer/depth-estimation/{model_id}"
        else:
            endpoint = "/infer/depth-estimation"
        result = await self._post_images_async(
            inference_input=inference_input,
            endpoint=endpoint,
            extra_payload=extra_payload,
        )
        return decode_depth_estimation_result(result)

    @wrap_errors
    def sam2_segment_image(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: Optional[List[dict]] = None,
        sam2_version_id: str = "hiera_tiny",
        multimask_output: bool = True,
        mask_input_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run Segment Anything 2 (SAM2) segmentation on input image(s).

        This method performs instance segmentation using SAM2, which can segment
        objects based on point or box prompts.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for segmentation. Can be file paths, URLs, base64 strings, numpy arrays,
                or PIL images.
            prompts (Optional[List[dict]], optional): List of prompt dictionaries. Each prompt
                can contain:
                - "box": {"x": float, "y": float, "width": float, "height": float}
                - "points": [{"x": float, "y": float, "positive": bool}, ...]
                Defaults to None (automatic segmentation).
            sam2_version_id (str, optional): Version of SAM2 model to use. Options are
                "hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus".
                Defaults to "hiera_tiny".
            multimask_output (bool, optional): Whether to output multiple masks per prompt.
                Defaults to True.
            mask_input_format (str, optional): Format for mask output. Defaults to "json".

        Returns:
            Union[dict, List[dict]]: Segmentation results containing predictions with masks,
                confidence scores, and bounding boxes.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {
            "sam2_version_id": sam2_version_id,
            "multimask_output": multimask_output,
            "format": mask_input_format,
        }
        if prompts is not None:
            extra_payload["prompts"] = {"prompts": prompts}
        result = self._post_images(
            inference_input=inference_input,
            endpoint="/sam2/segment_image",
            extra_payload=extra_payload,
        )
        return result

    @wrap_errors_async
    async def sam2_segment_image_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: Optional[List[dict]] = None,
        sam2_version_id: str = "hiera_tiny",
        multimask_output: bool = True,
        mask_input_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run Segment Anything 2 (SAM2) segmentation on input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s)
                for segmentation.
            prompts (Optional[List[dict]], optional): List of prompt dictionaries.
                Defaults to None.
            sam2_version_id (str, optional): Version of SAM2 model. Defaults to "hiera_tiny".
            multimask_output (bool, optional): Whether to output multiple masks. Defaults to True.
            mask_input_format (str, optional): Format for mask output. Defaults to "json".

        Returns:
            Union[dict, List[dict]]: Segmentation results.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {
            "sam2_version_id": sam2_version_id,
            "multimask_output": multimask_output,
            "format": mask_input_format,
        }
        if prompts is not None:
            extra_payload["prompts"] = {"prompts": prompts}
        result = await self._post_images_async(
            inference_input=inference_input,
            endpoint="/sam2/segment_image",
            extra_payload=extra_payload,
        )
        return result

    @wrap_errors
    def sam3_3d_infer(
        self,
        inference_input: ImagesReference,
        mask_input: Any,
        model_id: str = "sam3-3d-objects",
        *,
        output_meshes: bool = True,
        output_scene: bool = True,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_distillations: bool = False,
    ) -> dict:
        """Generate 3D meshes and Gaussian splatting from a 2D image with mask prompts.

        This method uses SAM3 3D to generate 3D representations from 2D images
        with mask prompts.

        Args:
            inference_input (ImagesReference): Input image for 3D generation.
                Can be a file path, URL, base64 string, numpy array, or PIL image.
            mask_input (Any): Mask input in any supported format:
                - Polygon coordinates: [x1, y1, x2, y2, ...]
                - Binary mask (as numpy array or base64)
                - RLE dictionary
                - List of any of the above for multiple masks
            model_id (str, optional): The SAM3 3D model to use. Defaults to "sam3-3d-objects".
            output_meshes (bool, optional): SAM3 3D always outputs object gaussians, and can
                optionally output object meshes if output_meshes is True. Defaults to True.
            output_scene (bool, optional): Output the combined scene reconstruction in
                addition to individual object reconstructions. Defaults to True.
            with_mesh_postprocess (bool, optional): Enable mesh postprocessing. Defaults to True.
            with_texture_baking (bool, optional): Enable texture baking for meshes. Defaults to True.
            use_distillations (bool, optional): Use the distilled versions of the model components.

        Returns:
            dict: Response containing base64-encoded 3D outputs:
                - mesh_glb: Scene mesh in GLB format (base64 encoded) if output_meshes=True, otherwise None.
                - gaussian_ply: Combined Gaussian splatting in PLY format (base64 encoded)
                - objects: List of individual objects, each containing:
                    - mesh_glb: Object mesh (base64) if output_scene=True and output_meshes=True, otherwise None.
                    - gaussian_ply: Object Gaussian (base64) if output_scene=True, otherwise None.
                    - metadata: {"rotation": [...], "translation": [...], "scale": [...]}
                - time: Inference time in seconds

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["model_id"] = model_id
        payload["mask_input"] = mask_input
        payload["output_meshes"] = output_meshes
        payload["output_scene"] = output_scene
        payload["with_mesh_postprocess"] = with_mesh_postprocess
        payload["with_texture_baking"] = with_texture_baking
        payload["use_distillations"] = use_distillations

        url = self.__wrap_url_with_api_key(f"{self.__api_url}/sam3_3d/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return responses[0].json()

    @wrap_errors_async
    async def sam3_3d_infer_async(
        self,
        inference_input: ImagesReference,
        mask_input: Any,
        model_id: str = "sam3-3d-objects",
        *,
        output_meshes: bool = True,
        output_scene: bool = True,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_distillations: bool = False,
    ) -> dict:
        """Generate 3D meshes and Gaussian splatting from a 2D image asynchronously.

        Args:
            inference_input (ImagesReference): Input image for 3D generation.
            mask_input (Any): Mask input in any supported format.
            model_id (str, optional): The SAM3 3D model to use. Defaults to "sam3-3d-objects".
            output_meshes (bool, optional): SAM3 3D always outputs object gaussians, and can
                optionally output object meshes if output_meshes is True. Defaults to True.
            output_scene (bool, optional): Output the combined scene reconstruction in
                addition to individual object reconstructions. Defaults to True.
            with_mesh_postprocess (bool, optional): Enable mesh postprocessing. Defaults to True.
            with_texture_baking (bool, optional): Enable texture baking for meshes. Defaults to True.
            use_distillations (bool, optional): Use the distilled versions of the model components.

        Returns:
            dict: Response containing base64-encoded 3D outputs:
                - mesh_glb: Scene mesh in GLB format (base64 encoded) if output_meshes=True, otherwise None.
                - gaussian_ply: Combined Gaussian splatting in PLY format (base64 encoded)
                - objects: List of individual objects, each containing:
                    - mesh_glb: Object mesh (base64) if output_scene=True and output_meshes=True, otherwise None.
                    - gaussian_ply: Object Gaussian (base64) if output_scene=True, otherwise None.
                    - metadata: {"rotation": [...], "translation": [...], "scale": [...]}
                - time: Inference time in seconds

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["model_id"] = model_id
        payload["mask_input"] = mask_input
        payload["output_meshes"] = output_meshes
        payload["output_scene"] = output_scene
        payload["with_mesh_postprocess"] = with_mesh_postprocess
        payload["with_texture_baking"] = with_texture_baking
        payload["use_distillations"] = use_distillations

        url = self.__wrap_url_with_api_key(f"{self.__api_url}/sam3_3d/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return responses[0]

    @wrap_errors
    def sam3_concept_segment(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: List[dict],
        model_id: str = "sam3/sam3_final",
        output_prob_thresh: float = 0.5,
        nms_iou_threshold: Optional[float] = None,
        format: str = "polygon",
    ) -> Union[dict, List[dict]]:
        """Run SAM3 promptable concept segmentation (PCS) on input image(s).

        Performs zero-shot instance segmentation using text or visual prompts.

        Args:
            inference_input: Input image(s) for segmentation.
            prompts: List of prompt dicts, each with keys like "type", "text",
                "output_prob_thresh", "boxes", "box_labels".
            model_id: SAM3 model to use. Defaults to "sam3/sam3_final".
            output_prob_thresh: Global confidence threshold. Defaults to 0.5.
            nms_iou_threshold: IoU threshold for cross-prompt NMS. None disables NMS.
            format: Output mask format, "polygon" or "rle". Defaults to "polygon".

        Returns:
            Segmentation results with prompt_results containing predictions.
        """
        extra_payload = {
            "model_id": model_id,
            "prompts": prompts,
            "output_prob_thresh": output_prob_thresh,
            "format": format,
        }
        if nms_iou_threshold is not None:
            extra_payload["nms_iou_threshold"] = nms_iou_threshold
        return self._post_images(
            inference_input=inference_input,
            endpoint="/sam3/concept_segment",
            extra_payload=extra_payload,
        )

    @wrap_errors_async
    async def sam3_concept_segment_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: List[dict],
        model_id: str = "sam3/sam3_final",
        output_prob_thresh: float = 0.5,
        nms_iou_threshold: Optional[float] = None,
        format: str = "polygon",
    ) -> Union[dict, List[dict]]:
        """Run SAM3 promptable concept segmentation (PCS) asynchronously.

        Args:
            inference_input: Input image(s) for segmentation.
            prompts: List of prompt dicts.
            model_id: SAM3 model to use. Defaults to "sam3/sam3_final".
            output_prob_thresh: Global confidence threshold. Defaults to 0.5.
            nms_iou_threshold: IoU threshold for cross-prompt NMS. None disables NMS.
            format: Output mask format, "polygon" or "rle". Defaults to "polygon".

        Returns:
            Segmentation results with prompt_results containing predictions.
        """
        extra_payload = {
            "model_id": model_id,
            "prompts": prompts,
            "output_prob_thresh": output_prob_thresh,
            "format": format,
        }
        if nms_iou_threshold is not None:
            extra_payload["nms_iou_threshold"] = nms_iou_threshold
        return await self._post_images_async(
            inference_input=inference_input,
            endpoint="/sam3/concept_segment",
            extra_payload=extra_payload,
        )

    @wrap_errors
    def sam3_visual_segment(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: Optional[List[dict]] = None,
        multimask_output: bool = True,
        mask_input_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run SAM3 promptable visual segmentation (PVS) on input image(s).

        Performs instance segmentation using point or box prompts.

        Args:
            inference_input: Input image(s) for segmentation.
            prompts: List of prompt dicts with "box" and/or "points" keys.
                Defaults to None (automatic segmentation).
            multimask_output: Whether to output multiple masks per prompt.
                Defaults to True.
            mask_input_format: Format for mask output. Defaults to "json".

        Returns:
            Segmentation results containing predictions with masks.
        """
        extra_payload = {
            "multimask_output": multimask_output,
            "format": mask_input_format,
        }
        if prompts is not None:
            extra_payload["prompts"] = {"prompts": prompts}
        return self._post_images(
            inference_input=inference_input,
            endpoint="/sam3/visual_segment",
            extra_payload=extra_payload,
        )

    @wrap_errors_async
    async def sam3_visual_segment_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        prompts: Optional[List[dict]] = None,
        multimask_output: bool = True,
        mask_input_format: str = "json",
    ) -> Union[dict, List[dict]]:
        """Run SAM3 promptable visual segmentation (PVS) asynchronously.

        Args:
            inference_input: Input image(s) for segmentation.
            prompts: List of prompt dicts. Defaults to None.
            multimask_output: Whether to output multiple masks. Defaults to True.
            mask_input_format: Format for mask output. Defaults to "json".

        Returns:
            Segmentation results containing predictions with masks.
        """
        extra_payload = {
            "multimask_output": multimask_output,
            "format": mask_input_format,
        }
        if prompts is not None:
            extra_payload["prompts"] = {"prompts": prompts}
        return await self._post_images_async(
            inference_input=inference_input,
            endpoint="/sam3/visual_segment",
            extra_payload=extra_payload,
        )

    @wrap_errors
    def sam3_embed_image(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        image_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Generate SAM3 image embeddings.

        Args:
            inference_input: Input image(s) to embed.
            image_id: Optional cache ID for embeddings. Defaults to None.

        Returns:
            Embedding results with image_id and processing time.
        """
        extra_payload = {}
        if image_id is not None:
            extra_payload["image_id"] = image_id
        return self._post_images(
            inference_input=inference_input,
            endpoint="/sam3/embed_image",
            extra_payload=extra_payload if extra_payload else None,
        )

    @wrap_errors_async
    async def sam3_embed_image_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        image_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Generate SAM3 image embeddings asynchronously.

        Args:
            inference_input: Input image(s) to embed.
            image_id: Optional cache ID for embeddings. Defaults to None.

        Returns:
            Embedding results with image_id and processing time.
        """
        extra_payload = {}
        if image_id is not None:
            extra_payload["image_id"] = image_id
        return await self._post_images_async(
            inference_input=inference_input,
            endpoint="/sam3/embed_image",
            extra_payload=extra_payload if extra_payload else None,
        )

    @deprecated(
        reason="Please use run_workflow(...) method. This method will be removed end of Q2 2024"
    )
    @wrap_errors
    def infer_from_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_name: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        use_cache: bool = True,
        enable_profiling: bool = False,
        workflow_version_id: Optional[str] = None,
        disable_sinks: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference using a workflow specification.

        Triggers inference from workflow specification at the inference HTTP
        side. Either (`workspace_name` and `workflow_name`) or `workflow_specification` must be
        provided. In the first case - definition of workflow will be fetched
        from Roboflow API, in the latter - `workflow_specification` will be
        used. `images` and `parameters` will be merged into workflow inputs,
        the distinction is made to make sure the SDK can easily serialise
        images and prepare a proper payload. Supported images are numpy arrays,
        PIL.Image and base64 images, links to images and local paths.
        `excluded_fields` will be added to request to filter out results
        of workflow execution at the server side.

        Args:
            workspace_name (Optional[str], optional): Name of the workspace containing the workflow. Defaults to None.
            workflow_name (Optional[str], optional): Name of the workflow. Defaults to None.
            specification (Optional[dict], optional): Direct workflow specification. Defaults to None.
            images (Optional[Dict[str, Any]], optional): Images to process. Defaults to None.
            parameters (Optional[Dict[str, Any]], optional): Additional parameters for the workflow. Defaults to None.
            excluded_fields (Optional[List[str]], optional): Fields to exclude from results. Defaults to None.
            use_cache (bool, optional): Whether to use cached results. Defaults to True.
            enable_profiling (bool, optional): Whether to enable profiling. Defaults to False.
            disable_sinks (bool, optional): Whether to disable sink writes and outbound
                notifications/uploads. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Results of the workflow execution.

        Raises:
            InvalidParameterError: If neither workflow identifiers nor specification is provided.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        return self._run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_name,
            specification=specification,
            images=images,
            parameters=parameters,
            excluded_fields=excluded_fields,
            legacy_endpoints=True,
            use_cache=use_cache,
            enable_profiling=enable_profiling,
            workflow_version_id=workflow_version_id,
            disable_sinks=disable_sinks,
        )

    @wrap_errors
    def run_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        use_cache: bool = True,
        enable_profiling: bool = False,
        workflow_version_id: Optional[str] = None,
        disable_sinks: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference using a workflow specification.

        Triggers inference from workflow specification at the inference HTTP
        side. Either (`workspace_name` and `workflow_id`) or `workflow_specification` must be
        provided. In the first case - definition of workflow will be fetched
        from Roboflow API, in the latter - `workflow_specification` will be
        used. `images` and `parameters` will be merged into workflow inputs,
        the distinction is made to make sure the SDK can easily serialise
        images and prepare a proper payload. Supported images are numpy arrays,
        PIL.Image and base64 images, links to images and local paths.
        `excluded_fields` will be added to request to filter out results
        of workflow execution at the server side.

        **Important!**
        Method is not compatible with inference server <=0.9.18. Please migrate to newer version of
        the server before end of Q2 2024. Until that is done - use old method: infer_from_workflow(...).

        Note:
            Method is not compatible with inference server <=0.9.18. Please migrate to newer version of
            the server before end of Q2 2024. Until that is done - use old method: infer_from_workflow(...).

        Args:
            workspace_name (Optional[str], optional): Name of the workspace containing the workflow. Defaults to None.
            workflow_id (Optional[str], optional): ID of the workflow. Defaults to None.
            specification (Optional[dict], optional): Direct workflow specification. Defaults to None.
            images (Optional[Dict[str, Any]], optional): Images to process. Defaults to None.
            parameters (Optional[Dict[str, Any]], optional): Additional parameters for the workflow. Defaults to None.
            excluded_fields (Optional[List[str]], optional): Fields to exclude from results. Defaults to None.
            use_cache (bool, optional): Whether to use cached results. Defaults to True.
            enable_profiling (bool, optional): Whether to enable profiling. Defaults to False.
            disable_sinks (bool, optional): Whether to disable sink writes and outbound
                notifications/uploads. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Results of the workflow execution.

        Raises:
            InvalidParameterError: If neither workflow identifiers nor specification is provided.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        return self._run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            specification=specification,
            images=images,
            parameters=parameters,
            excluded_fields=excluded_fields,
            legacy_endpoints=False,
            use_cache=use_cache,
            enable_profiling=enable_profiling,
            workflow_version_id=workflow_version_id,
            disable_sinks=disable_sinks,
        )

    def _run_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        legacy_endpoints: bool = False,
        use_cache: bool = True,
        enable_profiling: bool = False,
        workflow_version_id: Optional[str] = None,
        disable_sinks: bool = False,
    ) -> List[Dict[str, Any]]:
        response = self._execute_workflow_request(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            specification=specification,
            images=images,
            parameters=parameters,
            excluded_fields=excluded_fields,
            legacy_endpoints=legacy_endpoints,
            use_cache=use_cache,
            enable_profiling=enable_profiling,
            workflow_version_id=workflow_version_id,
            disable_sinks=disable_sinks,
        )
        response_data = response.json()
        workflow_outputs = response_data["outputs"]
        profiler_trace = response_data.get("profiler_trace", [])
        if enable_profiling:
            save_workflows_profiler_trace(
                directory=self.__inference_configuration.profiling_directory,
                profiler_trace=profiler_trace,
            )
        return decode_workflow_outputs(
            workflow_outputs=workflow_outputs,
            expected_format=self.__inference_configuration.output_visualisation_format,
        )

    def _execute_workflow_request(
        self,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        legacy_endpoints: bool = False,
        use_cache: bool = True,
        enable_profiling: bool = False,
        workflow_version_id: Optional[str] = None,
        disable_sinks: bool = False,
    ) -> Response:
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not (named_workflow_specified != (specification is not None)):
            raise InvalidParameterError(
                "Parameters (`workspace_name`, `workflow_id` / `workflow_name`) can be used mutually exclusive with "
                "`specification`, but at least one must be set."
            )
        if images is None:
            images = {}
        if parameters is None:
            parameters = {}
        payload = {
            **self.__legacy_api_key_payload(),
            "use_cache": use_cache,
            "enable_profiling": enable_profiling,
        }
        if disable_sinks:
            payload["disable_sinks"] = True
        inputs = {}
        for image_name, image in images.items():
            loaded_image = load_nested_batches_of_inference_input(
                inference_input=image,
            )
            inject_nested_batches_of_images_into_payload(
                payload=inputs,
                encoded_images=loaded_image,
                key=image_name,
            )
        inputs.update(parameters)
        payload["inputs"] = inputs
        if excluded_fields is not None:
            payload["excluded_fields"] = excluded_fields
        if specification is not None:
            payload["specification"] = specification
        if specification is not None:
            if legacy_endpoints:
                url = f"{self.__api_url}/infer/workflows"
            else:
                url = f"{self.__api_url}/workflows/run"
        else:
            if workflow_version_id is not None:
                payload["workflow_version_id"] = workflow_version_id
            if legacy_endpoints:
                url = f"{self.__api_url}/infer/workflows/{workspace_name}/{workflow_id}"
            else:
                url = f"{self.__api_url}/{workspace_name}/workflows/{workflow_id}"
        response = send_post_request(
            url=url,
            payload=payload,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            enable_retries=self.__inference_configuration.workflow_run_retries_enabled,
        )
        return response

    @wrap_errors
    def infer_on_video(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        """Run a video model over one clip, sent whole.

        Args:
            video_reference (VideoReference): URL or local path of the clip.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            dict: `timeline` over the clip, with `source_fps`, `frame_count` and `windows_classified`.

        Raises:
            InvalidInputFormatError: If the reference is neither a URL nor an existing path.
            ModelTaskTypeNotSupportedError: If the model takes images (API v1 only).
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return self.infer_on_video_from_api_v0(
                video_reference=video_reference,
                model_id=model_id,
            )
        return self.infer_on_video_from_api_v1(
            video_reference=video_reference,
            model_id=model_id,
        )

    @wrap_errors_async
    async def infer_on_video_async(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        """Run a video model over one clip asynchronously. See ``infer_on_video``.

        Args:
            video_reference (VideoReference): URL or local path of the clip.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            dict: `timeline` over the clip, with `source_fps`, `frame_count` and `windows_classified`.

        Raises:
            InvalidInputFormatError: If the reference is neither a URL nor an existing path.
            ModelTaskTypeNotSupportedError: If the model takes images (API v1 only).
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return await self.infer_on_video_from_api_v0_async(
                video_reference=video_reference,
                model_id=model_id,
            )
        return await self.infer_on_video_from_api_v1_async(
            video_reference=video_reference,
            model_id=model_id,
        )

    def infer_on_video_from_api_v0(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        url, params, data, headers = self.__build_v0_video_request(
            video_reference=video_reference,
            model_id=model_id,
        )
        response = requests.post(url, params=params, data=data, headers=headers)
        api_key_safe_raise_for_status(response=response)
        return response.json()

    async def infer_on_video_from_api_v0_async(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        url, params, data, headers = self.__build_v0_video_request(
            video_reference=video_reference,
            model_id=model_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, params=params, data=data, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    def infer_on_video_from_api_v1(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        model_id = self.__resolve_video_model_id(model_id=model_id)
        task_type = self.get_model_description(model_id=model_id).task_type
        url, payload = self.__build_v1_video_request(
            video_reference=video_reference,
            model_id=model_id,
            task_type=task_type,
        )
        response = requests.post(
            url, json=payload, headers=self.__headers_with_auth(DEFAULT_HEADERS)
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    async def infer_on_video_from_api_v1_async(
        self,
        video_reference: VideoReference,
        model_id: Optional[str] = None,
    ) -> dict:
        model_id = self.__resolve_video_model_id(model_id=model_id)
        description = await self.get_model_description_async(model_id=model_id)
        url, payload = self.__build_v1_video_request(
            video_reference=video_reference,
            model_id=model_id,
            task_type=description.task_type,
            asynchronous=True,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=self.__headers_with_auth(DEFAULT_HEADERS)
            ) as response:
                response.raise_for_status()
                return await response.json()

    def __resolve_video_model_id(self, model_id: Optional[str]) -> str:
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        return resolve_roboflow_model_alias(model_id=model_id_to_be_used)

    def __build_v0_video_request(
        self,
        video_reference: VideoReference,
        model_id: Optional[str],
    ) -> Tuple[str, dict, Optional[str], dict]:
        video_type, video = _resolve_video_payload(video_reference=video_reference)
        model_id = self.__resolve_video_model_id(model_id=model_id)
        model_id_chunks = model_id.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model id: {model_id}. Expected format: project_id/model_version_id."
            )
        params = self.__legacy_api_key_payload()
        class_filter = self.__inference_configuration.class_filter
        if class_filter:
            params["class_filter"] = ",".join(class_filter)
        url = f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}"
        headers = dict(self.__headers_with_auth(DEFAULT_HEADERS) or {})
        if video_type == "url":
            params["image"] = video
            return url, params, None, headers
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return url, params, video, headers

    def __build_v1_video_request(
        self,
        video_reference: VideoReference,
        model_id: str,
        task_type: str,
        asynchronous: bool = False,
    ) -> Tuple[str, dict]:
        if task_type not in VIDEO_INFERENCE_ENDPOINTS:
            image_door = "infer_async()" if asynchronous else "infer()"
            raise ModelTaskTypeNotSupportedError(
                f"Model task {task_type} takes images, not a clip. Use {image_door} "
                f"for one image or infer_on_stream() to classify a video frame by frame."
            )
        video_type, video = _resolve_video_payload(video_reference=video_reference)
        payload = self.__initialise_payload()
        payload["model_id"] = model_id
        payload["video"] = {"type": video_type, "value": video}
        class_filter = self.__inference_configuration.class_filter
        if class_filter is not None:
            payload["class_filter"] = class_filter
        url = self.__wrap_url_with_api_key(
            f"{self.__api_url}{VIDEO_INFERENCE_ENDPOINTS[task_type]}"
        )
        return url, payload

    @wrap_errors
    def infer_from_yolo_world(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        class_names: List[str],
        model_version: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> List[dict]:
        """Run inference using YOLO-World model.

        Args:
            inference_input: Input image(s) to run inference on. Can be a single image
                reference or a list of image references.
            class_names: List of class names to detect in the image(s).
            model_version: Optional version of YOLO-World model to use. If not specified,
                uses the default version.
            confidence: Optional confidence threshold for detections. If not specified,
                uses the model's default threshold.

        Returns:
            List of dictionaries containing detection results for each input image.
            Each dictionary contains bounding boxes, class labels, and confidence scores
            for detected objects.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["text"] = class_names
        if model_version is not None:
            payload["yolo_world_version_id"] = model_version
        if confidence is not None:
            payload["confidence"] = confidence
        url = self.__wrap_url_with_api_key(f"{self.__api_url}/yolo_world/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return [r.json() for r in responses]

    @wrap_errors_async
    async def infer_from_yolo_world_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        class_names: List[str],
        model_version: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> List[dict]:
        """Run inference using YOLO-World model asynchronously.

        Args:
            inference_input: Input image(s) to run inference on. Can be a single image
                reference or a list of image references.
            class_names: List of class names to detect in the image(s).
            model_version: Optional version of YOLO-World model to use. If not specified,
                uses the default version.
            confidence: Optional confidence threshold for detections. If not specified,
                uses the model's default threshold.

        Returns:
            List of dictionaries containing detection results for each input image.
            Each dictionary contains bounding boxes, class labels, and confidence scores
            for detected objects.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["text"] = class_names
        if model_version is not None:
            payload["yolo_world_version_id"] = model_version
        if confidence is not None:
            payload["confidence"] = confidence
        url = self.__wrap_url_with_api_key(f"{self.__api_url}/yolo_world/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        return await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def start_inference_pipeline_with_workflow(
        self,
        video_reference: Union[str, int, List[Union[str, int]]],
        workflow_specification: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        image_input_name: str = "image",
        workflows_parameters: Optional[Dict[str, Any]] = None,
        workflows_thread_pool_workers: int = 4,
        cancel_thread_pool_tasks_on_exit: bool = True,
        video_metadata_input_name: str = "video_metadata",
        max_fps: Optional[Union[float, int]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = "DROP_OLDEST",
        source_buffer_consumption_strategy: Optional[
            BufferConsumptionStrategy
        ] = "EAGER",
        video_source_properties: Optional[Dict[str, float]] = None,
        batch_collection_timeout: Optional[float] = None,
        results_buffer_size: int = 64,
    ) -> dict:
        """Starts an inference pipeline using a workflow specification.

        Args:
            video_reference: Path to video file, camera index, or list of video sources.
                Can be a string path, integer camera index, or list of either.
            workflow_specification: Optional workflow specification dictionary. Mutually
                exclusive with workspace_name/workflow_id.
            workspace_name: Optional name of workspace containing workflow. Must be used
                with workflow_id.
            workflow_id: Optional ID of workflow to use. Must be used with workspace_name.
            image_input_name: Name of the image input node in workflow. Defaults to "image".
            workflows_parameters: Optional parameters to pass to workflow.
            workflows_thread_pool_workers: Number of worker threads for workflow execution.
                Defaults to 4.
            cancel_thread_pool_tasks_on_exit: Whether to cancel pending tasks when exiting.
                Defaults to True.
            video_metadata_input_name: Name of video metadata input in workflow.
                Defaults to "video_metadata".
            max_fps: Optional maximum FPS to process video at.
            source_buffer_filling_strategy: Strategy for filling source buffer when full.
                One of: "WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST",
                "ADAPTIVE_DROP_LATEST". Defaults to "DROP_OLDEST".
            source_buffer_consumption_strategy: Strategy for consuming from source buffer.
                One of: "LAZY", "EAGER". Defaults to "EAGER".
            video_source_properties: Optional dictionary of video source properties.
            batch_collection_timeout: Optional timeout for batch collection in seconds.
            results_buffer_size: Size of results buffer. Defaults to 64.

        Returns:
            dict: Response containing pipeline initialization details.

        Raises:
            InvalidParameterError: If workflow specification parameters are invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not (named_workflow_specified != (workflow_specification is not None)):
            raise InvalidParameterError(
                "Parameters (`workspace_name`, `workflow_id`) can be used mutually exclusive with "
                "`workflow_specification`, but at least one must be set."
            )
        payload = {
            **self.__legacy_api_key_payload(),
            "video_configuration": {
                "type": "VideoConfiguration",
                "video_reference": video_reference,
                "max_fps": max_fps,
                "source_buffer_filling_strategy": source_buffer_filling_strategy,
                "source_buffer_consumption_strategy": source_buffer_consumption_strategy,
                "video_source_properties": video_source_properties,
                "batch_collection_timeout": batch_collection_timeout,
            },
            "processing_configuration": {
                "type": "WorkflowConfiguration",
                "workflow_specification": workflow_specification,
                "workspace_name": workspace_name,
                "workflow_id": workflow_id,
                "image_input_name": image_input_name,
                "workflows_parameters": workflows_parameters,
                "workflows_thread_pool_workers": workflows_thread_pool_workers,
                "cancel_thread_pool_tasks_on_exit": cancel_thread_pool_tasks_on_exit,
                "video_metadata_input_name": video_metadata_input_name,
            },
            "sink_configuration": {
                "type": "MemorySinkConfiguration",
                "results_buffer_size": results_buffer_size,
            },
        }
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/initialise",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        response.raise_for_status()
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def list_inference_pipelines(self) -> List[dict]:
        """Lists all active inference pipelines on the server.

        This method retrieves information about all currently running inference pipelines
        on the server, including their IDs and status.

        Returns:
            List[dict]: A list of dictionaries containing information about each active
                inference pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = self.__legacy_api_key_payload()
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/list",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def get_inference_pipeline_status(self, pipeline_id: str) -> dict:
        """Gets the current status of a specific inference pipeline.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to check.

        Returns:
            dict: A dictionary containing the current status and details of the pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = self.__legacy_api_key_payload()
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/status",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def pause_inference_pipeline(self, pipeline_id: str) -> dict:
        """Pauses a running inference pipeline.

        Sends a request to pause the specified inference pipeline. The pipeline must be
        currently running for this operation to succeed.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to pause.

        Returns:
            dict: A dictionary containing the response from the server about the pause operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = self.__legacy_api_key_payload()
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/pause",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def resume_inference_pipeline(self, pipeline_id: str) -> dict:
        """Resumes a paused inference pipeline.

        Sends a request to resume the specified inference pipeline. The pipeline must be
        currently paused for this operation to succeed.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to resume.

        Returns:
            dict: A dictionary containing the response from the server about the resume operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = self.__legacy_api_key_payload()
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/resume",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def terminate_inference_pipeline(self, pipeline_id: str) -> dict:
        """Terminates a running inference pipeline.

        Sends a request to terminate the specified inference pipeline. This will stop all
        processing and free up associated resources.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to terminate.

        Returns:
            dict: A dictionary containing the response from the server about the termination operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = self.__legacy_api_key_payload()
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/terminate",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def consume_inference_pipeline_result(
        self,
        pipeline_id: str,
        excluded_fields: Optional[List[str]] = None,
    ) -> dict:
        """Consumes and returns the next available result from an inference pipeline.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to consume results from.
            excluded_fields: Optional list of field names to exclude from the result. If None,
                no fields will be excluded.

        Returns:
            dict: A dictionary containing the next available result from the pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            InvalidParameterError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        if excluded_fields is None:
            excluded_fields = []
        payload = {
            **self.__legacy_api_key_payload(),
            "excluded_fields": excluded_fields,
        }
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/consume",
            json=payload,
            headers=self.__headers_with_auth(None),
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    def _ensure_pipeline_id_not_empty(self, pipeline_id: str) -> None:
        if not pipeline_id:
            raise InvalidParameterError("Empty `pipeline_id` parameter detected")

    def _post_images(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        endpoint: str,
        model_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if model_id is not None:
            payload["model_id"] = model_id
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{endpoint}")
        if extra_payload is not None:
            payload.update(extra_payload)
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = [r.json() for r in responses]
        return unwrap_single_element_list(sequence=results)

    async def _post_images_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        endpoint: str,
        model_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if model_id is not None:
            payload["model_id"] = model_id
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{endpoint}")
        if extra_payload is not None:
            payload.update(extra_payload)
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=self.__headers_with_auth(DEFAULT_HEADERS),
            # Billing parameters travel on the URL query string instead - see
            # __wrap_url_with_api_key - so passing them here too would
            # double-append them onto the final request.
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return unwrap_single_element_list(sequence=responses)

    def __warn_if_webrtc_transport_is_stale(self) -> None:
        # The webrtc namespace captures the api-key transport once, at first
        # `client.webrtc` access, and deliberately keeps it (a streaming
        # session should not change auth mid-flight). This warns - once per
        # client - when a configuration change diverges from that captured
        # value, so the change is not silently ignored for streaming.
        if self.__webrtc_client is None or self.__webrtc_transport_stickiness_warned:
            return
        configured = (
            self.__inference_configuration.api_key_transport or ApiKeyTransport.LEGACY
        )
        if configured is self.__webrtc_client_transport:
            return
        self.__webrtc_transport_stickiness_warned = True
        warnings.warn(
            f"api_key_transport changed to '{configured.value}', but this "
            f"client's WebRTC namespace was already initialised with "
            f"'{self.__webrtc_client_transport.value}' and keeps it - the "
            "transport is captured once at first `client.webrtc` access. "
            "Build a new InferenceHTTPClient to stream with a different "
            "transport.",
            InferenceSDKGuidanceWarning,
        )

    def __resolved_api_key_transport(self) -> ApiKeyTransport:
        # None means the user made no choice - resolve to the legacy channel
        # (today's default) and recommend moving to the header transport once
        # per process. An explicit "legacy" stays silent.
        transport = self.__inference_configuration.api_key_transport
        if transport is None:
            _warn_about_default_api_key_transport_once()
            return ApiKeyTransport.LEGACY
        return transport

    def __auth_headers(self) -> Dict[str, str]:
        # Non-empty only in "both" / "header" transport modes - the header is
        # the same form the SDK already uses toward the platform API
        # (inference_sdk/webrtc/model_workflows.py). The key travels in a
        # header - never in the URL - so request exceptions (whose text embeds
        # the URL) cannot leak it.
        if (
            self.__resolved_api_key_transport() is ApiKeyTransport.LEGACY
            or self.__api_key is None
        ):
            return {}
        return {"Authorization": f"Bearer {self.__api_key}"}

    def __headers_with_auth(
        self, headers: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, str]]:
        # Returns the input untouched in legacy mode so shared dicts
        # (DEFAULT_HEADERS) are never mutated and wire behaviour stays
        # byte-identical for the default transport.
        auth_headers = self.__auth_headers()
        if not auth_headers:
            return headers
        if headers is None:
            return auth_headers
        return {**headers, **auth_headers}

    def __legacy_api_key_payload(self) -> dict:
        # The `api_key` entry for query-params / JSON-body dicts. Suppressed
        # only in "header" mode - "both" keeps the legacy channels intact.
        if self.__resolved_api_key_transport() is ApiKeyTransport.HEADER:
            return {}
        return {"api_key": self.__api_key}

    def __initialise_payload(self) -> dict:
        if (
            self.__client_mode is not HTTPClientMode.V0
            and self.__resolved_api_key_transport() is not ApiKeyTransport.HEADER
        ):
            return {"api_key": self.__api_key}
        return {}

    def __wrap_url_with_api_key(self, url: str) -> str:
        # The one URL seam every hand-built request method routes through, so
        # it also appends the current billing query parameters (explicit
        # configuration, or the outbound forwarding-authority context read at
        # send time) - the standard v0/v1 `infer()` methods serialize those
        # through `InferenceConfiguration` instead, and never call this.
        query_params: Dict[str, Any] = {}
        if (
            self.__client_mode is HTTPClientMode.V0
            and self.__resolved_api_key_transport() is not ApiKeyTransport.HEADER
        ):
            query_params["api_key"] = self.__api_key
        billing_query_parameters = (
            self.__inference_configuration.to_billing_query_parameters()
        )
        if billing_query_parameters:
            query_params.update(billing_query_parameters)
        if not query_params:
            return url
        return f"{url}?{urlencode(query_params)}"

    def __ensure_v1_client_mode(self) -> None:
        if self.__client_mode is not HTTPClientMode.V1:
            raise WrongClientModeError("Use client mode `v1` to run this operation.")


def _resolve_video_payload(video_reference: VideoReference) -> Tuple[str, str]:
    """Turn what the caller handed over into a transport and a value.

    A URL is forwarded for the server to fetch. A local path is read here on
    purpose, rather than by falling through an image loader that happens to
    skip decoding. Anything else is an error, not a guess. Encoded bytes are
    not accepted, because an image reference does not accept them either.
    """
    if not isinstance(video_reference, str):
        raise InvalidInputFormatError(
            f"Unknown type of video reference ({type(video_reference).__name__}). "
            "Pass a URL or a local path."
        )
    if uri_is_http_link(uri=video_reference):
        return "url", video_reference
    if os.path.isfile(video_reference):
        with open(video_reference, "rb") as clip:
            return "base64", base64.b64encode(clip.read()).decode("utf-8")
    raise InvalidInputFormatError(
        f"Video reference is neither a URL nor an existing file: {video_reference!r}. "
        "Pass a URL or a local path."
    )


def _ensure_task_takes_an_image(task_type: str, asynchronous: bool = False) -> None:
    """Refuse a video model at the image door, and say where the clip goes."""
    if task_type in NEW_INFERENCE_ENDPOINTS:
        return
    if task_type in VIDEO_INFERENCE_ENDPOINTS:
        clip_door = "infer_on_video_async()" if asynchronous else "infer_on_video()"
        # infer_on_stream has no async twin, so it keeps its name either way.
        raise ModelTaskTypeNotSupportedError(
            f"Model task {task_type} takes a clip, not an image. Use {clip_door} "
            f"to send a clip whole, or infer_on_stream() to classify a video "
            f"frame by frame with an image model."
        )
    raise ModelTaskTypeNotSupportedError(
        f"Model task {task_type} is not supported by API v1 client."
    )


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
    if any(api_url.startswith(roboflow_url) for roboflow_url in ALL_ROBOFLOW_API_URLS):
        return HTTPClientMode.V0
    return HTTPClientMode.V1


def _ensure_model_is_selected(model_id: Optional[str]) -> None:
    if model_id is None:
        raise ModelNotSelectedError("No model was selected to be used.")


def _ensure_api_key_provided(api_key: Optional[str]) -> None:
    if api_key is None:
        raise APIKeyNotProvided("API key must be provided in this case")
