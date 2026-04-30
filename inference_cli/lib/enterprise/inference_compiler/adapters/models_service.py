from typing import Any, Dict, List, Literal, Optional

import backoff
import requests
from pydantic import BaseModel, Field
from requests import Timeout

from inference_cli.lib.enterprise.inference_compiler.constants import (
    ROBOFLOW_API_HOST,
    ROBOFLOW_API_KEY,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    RetryError,
    RuntimeConfigurationError,
)
from inference_cli.lib.enterprise.inference_compiler.utils.http import (
    handle_response_errors,
)


class SignedURLDetails(BaseModel):
    type: Literal["signed-url-details-v1"]
    upload_url: str = Field(alias="uploadUrl")
    method: str = Field(alias="method")
    extension_headers: dict = Field(alias="extensionHeaders")
    max_file_size: int = Field(alias="maxFileSize")

    class Config:
        populate_by_name = True


class ExternalFileUploadSpecs(BaseModel):
    type: Literal["external-file-upload-specs-v1"]
    file_handle: str = Field(alias="fileHandle")
    signed_url_details: SignedURLDetails = Field(alias="signedUrlDetails")

    class Config:
        populate_by_name = True


class ModelPackageRegistrationResponse(BaseModel):
    model_id: str = Field(alias="modelId")
    model_package_id: str = Field(alias="modelPackageId")
    file_upload_specs: List[ExternalFileUploadSpecs] = Field(alias="filesUploadSpecs")

    class Config:
        populate_by_name = True


class FileConfirmation(BaseModel):
    file_handle: str = Field(alias="fileHandle")
    md5_hash: Optional[str] = Field(alias="md5Hash", default=None)

    class Config:
        populate_by_name = True


class ExternalPublicTRTTimingCompilationEntryV1(BaseModel):
    type: Literal["external-public-trt-timing-cache-entry-v1"]
    cache_key: str = Field(alias="cacheKey")
    compilation_features: Dict[str, Any] = Field(alias="compilationFeatures")
    file_handle: str = Field(alias="fileHandle")
    download_url: str = Field(alias="downloadUrl")
    md5_hash: Optional[str] = Field(alias="md5Hash", default=None)

    class Config:
        populate_by_name = True


class ExternalPrivateTRTTimingCompilationEntryV1(BaseModel):
    type: Literal["external-private-trt-timing-cache-entry-v1"]
    cache_key: str = Field(alias="cacheKey")
    compilation_features: Dict[str, Any] = Field(alias="compilationFeatures")
    file_handle: str = Field(alias="fileHandle")
    download_url: str = Field(alias="downloadUrl")
    md5_hash: Optional[str] = Field(alias="md5Hash", default=None)

    class Config:
        populate_by_name = True


class PrivateTRTTimingCacheEntryRegistrationResults(BaseModel):
    cache_key: str = Field(alias="cacheKey")
    upload_specs: ExternalFileUploadSpecs = Field(alias="uploadSpecs")

    class Config:
        populate_by_name = True


class ExternalPrivateTRTTimingCacheListEntryV1(BaseModel):
    type: Literal["external-private-trt-timing-cache-list-entry-v1"]
    cache_key: str = Field(alias="cacheKey")
    compilation_features: Dict[str, Any] = Field(alias="compilationFeatures")
    sealed: bool

    class Config:
        populate_by_name = True


class PrivateTRTTimingCacheEntriesList(BaseModel):
    cache_entries: List[ExternalPrivateTRTTimingCacheListEntryV1] = Field(
        alias="cacheEntries"
    )
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)

    class Config:
        populate_by_name = True


class ModelsServiceClient:
    @classmethod
    def init(
        cls,
        api_key: Optional[str] = None,
    ) -> "ModelsServiceClient":
        if api_key is None:
            api_key = ROBOFLOW_API_KEY
        if api_key is None:
            raise RuntimeConfigurationError(
                "Could not initialize Models Service client without a Roboflow API key. "
                "Set the key explicitly or use the environment variable `ROBOFLOW_API_KEY`. If you need help finding "
                "your Roboflow API key, "
                "visit: https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key"
            )
        return cls(
            api_host=ROBOFLOW_API_HOST,
            api_key=api_key,
        )

    def __init__(
        self,
        api_host: str,
        api_key: str,
    ):
        self._api_host = api_host
        self._api_key = api_key

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def register_model_package(
        self,
        model_id: str,
        package_manifest: dict,
        file_handles: List[str],
        model_features: Optional[dict] = None,
    ):
        try:
            payload = {
                "modelId": model_id,
                "packageManifest": package_manifest,
                "fileHandles": file_handles,
            }
            if model_features:
                payload["modelFeatures"] = model_features
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/register",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return ModelPackageRegistrationResponse.model_validate(response.json())

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def confirm_model_package_artefacts(
        self,
        model_id: str,
        model_package_id: str,
        confirmations: List[FileConfirmation],
        seal_model_package: Optional[bool] = None,
    ) -> None:
        try:
            payload: Dict[str, Any] = {
                "modelId": model_id,
                "modelPackageId": model_package_id,
                "confirmations": [c.model_dump(by_alias=True) for c in confirmations],
            }
            if seal_model_package:
                payload["sealModelPackage"] = seal_model_package
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/artefacts/confirm",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def add_model_package_artefacts(
        self,
        model_id: str,
        model_package_id: str,
        file_handles: List[str],
    ) -> ModelPackageRegistrationResponse:
        try:
            payload = {
                "modelId": model_id,
                "modelPackageId": model_package_id,
                "fileHandles": file_handles,
            }
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/artefacts/add",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return ModelPackageRegistrationResponse.model_validate(response.json())  # type: ignore

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def remove_model_package_artefacts(
        self,
        model_id: str,
        model_package_id: str,
        file_handles: List[str],
    ) -> None:
        try:
            payload = {
                "modelId": model_id,
                "modelPackageId": model_package_id,
                "fileHandles": file_handles,
            }
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/artefacts/remove",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def seal_model_package(self, model_id: str, package_id: str) -> None:
        payload: Dict[str, Any] = {"modelId": model_id, "modelPackageId": package_id}
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/seal",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def un_seal_model_package(self, model_id: str, model_package_id: str) -> None:
        payload: Dict[str, Any] = {
            "modelId": model_id,
            "modelPackageId": model_package_id,
        }
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/un-seal",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def delete_model_package(self, model_id: str, model_package_id: str) -> None:
        payload: Dict[str, Any] = {
            "modelId": model_id,
            "modelPackageId": model_package_id,
        }
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/delete",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def un_delete_model_package(self, model_id: str, model_package_id: str) -> None:
        payload: Dict[str, Any] = {
            "modelId": model_id,
            "modelPackageId": model_package_id,
        }
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/model-packages/un-delete",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def get_public_trt_timing_cache(
        self, compilation_features: Dict[str, Any]
    ) -> ExternalPublicTRTTimingCompilationEntryV1:
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/trt-compilation/timing-cache/public/get",
                json={"compilationFeatures": compilation_features},
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return ExternalPublicTRTTimingCompilationEntryV1.model_validate(  # type: ignore
            response.json()["cacheEntry"]
        )

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def get_private_trt_timing_cache(
        self, compilation_features: Dict[str, Any]
    ) -> ExternalPrivateTRTTimingCompilationEntryV1:
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/trt-compilation/timing-cache/private/get",
                json={"compilationFeatures": compilation_features},
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return ExternalPrivateTRTTimingCompilationEntryV1.model_validate(  # type: ignore
            response.json()["cacheEntry"]
        )

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def register_private_trt_timing_cache(
        self,
        compilation_features: Dict[str, Any],
    ) -> PrivateTRTTimingCacheEntryRegistrationResults:
        try:
            response = requests.post(
                f"{self._api_host}/models/v1/external/trt-compilation/timing-cache/private/register",
                json={"compilationFeatures": compilation_features},
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return PrivateTRTTimingCacheEntryRegistrationResults.model_validate(  # type: ignore
            response.json()
        )

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def confirm_private_trt_timing_cache_upload(
        self, cache_key: str, confirmation: FileConfirmation
    ) -> None:
        try:
            payload: Dict[str, Any] = {
                "cacheKey": cache_key,
                "confirmation": confirmation.model_dump(by_alias=True),
            }
            response = requests.post(
                f"{self._api_host}/models/v1/external/trt-compilation/timing-cache/private/confirm",
                json=payload,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return None

    @backoff.on_exception(
        backoff.fibo,
        exception=RetryError,
        max_tries=3,
        max_value=5,
    )
    def list_private_timing_cache_entries(
        self,
        page_size: Optional[int] = None,
        start_after: Optional[str] = None,
    ) -> PrivateTRTTimingCacheEntriesList:
        try:
            query: Dict[str, Any] = {}
            if page_size is not None:
                query["pageSize"] = page_size
            if start_after is not None:
                query["startAfter"] = start_after
            response = requests.get(
                f"{self._api_host}/models/v1/external/trt-compilation/timing-cache/private/list",
                params=query,
                headers=self._add_auth_headers(),
            )
        except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
            raise RetryError("Connectivity error")
        handle_response_errors(response=response)
        return PrivateTRTTimingCacheEntriesList.model_validate(response.json())  # type: ignore

    def _add_auth_headers(
        self, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
