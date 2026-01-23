"""
GCP Logging Model Access Manager for inference-models integration.

This module provides a custom ModelAccessManager that logs ModelLoadedToDiskEvent
when models are downloaded through the inference-models library.
"""

import os
import time
from typing import Optional

from inference.core.gcp_logging.events import ModelLoadedToDiskEvent
from inference.core.gcp_logging.logger import gcp_logger
from inference.core.gcp_logging.context import get_gcp_context


class GCPLoggingModelAccessManager:
    """
    A ModelAccessManager that logs download events to GCP structured logging.

    This wraps the default LiberalModelAccessManager behavior and adds
    tracking of file downloads to emit ModelLoadedToDiskEvent when a model
    package is fully loaded.

    Usage:
        from inference.core.gcp_logging.access_manager import GCPLoggingModelAccessManager
        from inference_models import AutoModel

        model = AutoModel.from_pretrained(
            model_id,
            api_key=api_key,
            model_access_manager=GCPLoggingModelAccessManager(),
        )
    """

    def __init__(self):
        # Track download stats per model package
        self._download_start_time: Optional[float] = None
        self._total_bytes: int = 0
        self._artifact_count: int = 0
        self._current_model_id: Optional[str] = None
        self._current_package_id: Optional[str] = None

    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        pass

    def on_model_package_access_granted(self, access_identifiers) -> None:
        """Called when access to a model package is granted - start tracking."""
        self._download_start_time = time.time()
        self._total_bytes = 0
        self._artifact_count = 0
        self._current_model_id = access_identifiers.model_id
        self._current_package_id = access_identifiers.package_id

    def on_file_created(self, file_path: str, access_identifiers) -> None:
        """Called when a file is downloaded - track size."""
        try:
            if os.path.exists(file_path):
                self._total_bytes += os.path.getsize(file_path)
                self._artifact_count += 1
        except OSError:
            pass

    def on_file_renamed(
        self, old_path: str, new_path: str, access_identifiers
    ) -> None:
        pass

    def on_symlink_created(
        self, target_path: str, link_name: str, access_identifiers
    ) -> None:
        pass

    def on_symlink_deleted(self, link_name: str) -> None:
        pass

    def on_file_deleted(self, file_path: str) -> None:
        pass

    def on_directory_deleted(self, dir_path: str) -> None:
        pass

    def is_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> bool:
        return False

    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        return True

    def retrieve_model_instance(
        self,
        model_id: str,
        package_id: Optional[str],
        api_key: Optional[str],
        loading_parameter_digest: Optional[str],
    ):
        return None

    def on_model_loaded(
        self,
        model,
        access_identifiers,
        model_storage_path: str,
    ) -> None:
        """Called when model is fully loaded - log the disk event if files were downloaded."""
        if not gcp_logger.enabled:
            return

        # Only log if we actually downloaded files (not a cache hit)
        if self._artifact_count == 0:
            return

        download_duration_ms = 0.0
        if self._download_start_time is not None:
            download_duration_ms = (time.time() - self._download_start_time) * 1000

        # Try to get backend info from the model
        backend = "inference-models"
        if hasattr(model, "backend"):
            backend = f"inference-models/{model.backend}"

        ctx = get_gcp_context()
        gcp_logger.log_event(
            ModelLoadedToDiskEvent(
                request_id=ctx.request_id if ctx else None,
                model_id=access_identifiers.model_id,
                package_id=access_identifiers.package_id,
                backend=backend,
                download_bytes=self._total_bytes,
                download_duration_ms=download_duration_ms,
                artifact_count=self._artifact_count,
            ),
            sampled=False,  # Always log disk loads (low volume, high value)
        )

        # Reset tracking
        self._download_start_time = None
        self._total_bytes = 0
        self._artifact_count = 0

    def on_model_alias_discovered(self, alias: str, model_id: str) -> None:
        pass

    def on_model_dependency_discovered(
        self,
        base_model_id: str,
        base_model_package_id: Optional[str],
        dependent_model_id: str,
    ) -> None:
        pass
