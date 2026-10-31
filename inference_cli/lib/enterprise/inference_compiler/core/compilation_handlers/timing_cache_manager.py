import logging
import os.path
import tempfile
from typing import Any, Dict, Union

from inference_cli.lib.enterprise.inference_compiler.adapters.models_service import (
    ExternalPrivateTRTTimingCompilationEntryV1,
    ExternalPublicTRTTimingCompilationEntryV1,
    FileConfirmation,
    ModelsServiceClient,
)
from inference_cli.lib.enterprise.inference_compiler.errors import RequestError
from inference_cli.lib.enterprise.inference_compiler.utils.file_system import (
    calculate_local_file_md5,
    read_bytes,
    write_bytes,
)
from inference_cli.lib.enterprise.inference_compiler.utils.http import (
    upload_file_to_cloud,
)
from inference_models.utils.download import download_files_to_directory

logger = logging.getLogger("inference_cli.inference_compiler")

TIMING_CACHE_ROOT = "/tmp/timing-cache"


class TimingCacheManager:
    @classmethod
    def init(
        cls,
        models_service_client: ModelsServiceClient,
        compilation_features: Dict[str, Any],
    ) -> "TimingCacheManager":
        return cls(
            cache_root=TIMING_CACHE_ROOT,
            models_service_client=models_service_client,
            compilation_features=compilation_features,
        )

    def __init__(
        self,
        cache_root: str,
        models_service_client: ModelsServiceClient,
        compilation_features: Dict[str, Any],
    ):
        self._cache_root = cache_root
        self._models_service_client = models_service_client
        self._compilation_features = compilation_features
        self._should_not_populate_private_cache = False

    def get_cache_for_features(self) -> bytes:
        try:
            compilation_features_specs = self._attempt_getting_cache_entry()
            file_handle = compilation_features_specs.file_handle
            download_url = compilation_features_specs.download_url
            md5_hash = compilation_features_specs.md5_hash
            download_results = download_files_to_directory(
                target_dir=self._cache_root,
                files_specs=[(file_handle, download_url, md5_hash)],
            )
            logger.info(
                "TRT timing cache hit for compilation features: %s",
                self._compilation_features,
            )
            return read_bytes(download_results[file_handle])
        except RequestError as error:
            if error.status_code == 404:
                logger.info(
                    "TRT timing cache miss for compilation features: %s",
                    self._compilation_features,
                )
            else:
                self._should_not_populate_private_cache = True
                logger.warning(
                    "Could not retrieve TRT timing cache entry from RF API: status=%s, message=%s",
                    error.status_code,
                    error,
                )
            return b""
        except Exception:
            self._should_not_populate_private_cache = True
            logger.exception("Error retrieving TRT timing compilation cache")
            return b""

    def save_cache_for_features(self, cache: bytes) -> None:
        if self._should_not_populate_private_cache:
            return None
        try:
            registration_response = (
                self._models_service_client.register_private_trt_timing_cache(
                    compilation_features=self._compilation_features
                )
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                cache_entry_path = os.path.join(tmp_dir, "local-copy-of-cache-entry")
                write_bytes(path=cache_entry_path, content=cache)
                cache_entry_md5 = calculate_local_file_md5(file_path=cache_entry_path)
                upload_file_to_cloud(
                    file_path=cache_entry_path,
                    url=registration_response.upload_specs.signed_url_details.upload_url,
                    headers=registration_response.upload_specs.signed_url_details.extension_headers,
                )
                confirmation = FileConfirmation(
                    file_handle=registration_response.upload_specs.file_handle,
                    md5_hash=cache_entry_md5,
                )
                self._models_service_client.confirm_private_trt_timing_cache_upload(
                    cache_key=registration_response.cache_key,
                    confirmation=confirmation,
                )
                logger.info(
                    "TRT timing cache saved for compilation features: %s",
                    self._compilation_features,
                )
        except RequestError as error:
            if error.status_code == 409:
                return None
        except Exception:
            logger.exception("Error saving TRT timing compilation cache")

    def _attempt_getting_cache_entry(
        self,
    ) -> Union[
        ExternalPublicTRTTimingCompilationEntryV1,
        ExternalPrivateTRTTimingCompilationEntryV1,
    ]:
        try:
            features_specs = self._models_service_client.get_public_trt_timing_cache(
                compilation_features=self._compilation_features
            )
            self._should_not_populate_private_cache = True
            return features_specs  # type: ignore
        except RequestError as error:
            if error.status_code != 404:
                raise error
        return self._models_service_client.get_private_trt_timing_cache(  # type: ignore
            compilation_features=self._compilation_features
        )
