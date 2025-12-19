import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional

from filelock import FileLock
from inference_exp.configuration import (
    AUTO_LOADER_CACHE_EXPIRATION_MINUTES,
    INFERENCE_HOME,
)
from inference_exp.logger import LOGGER, verbose_info
from inference_exp.models.auto_loaders.entities import (
    BackendType,
    ModelArchitecture,
    TaskType,
)
from inference_exp.utils.file_system import dump_json, read_json
from inference_exp.weights_providers.entities import ModelDependency
from pydantic import BaseModel, Field


class AutoResolutionCacheEntry(BaseModel):
    model_id: str
    model_package_id: str
    resolved_files: List[str]
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    model_dependencies: Optional[List[ModelDependency]] = Field(default=None)
    created_at: datetime


class AutoResolutionCache(ABC):

    @abstractmethod
    def register(
        self, auto_negotiation_hash: str, cache_entry: AutoResolutionCacheEntry
    ) -> None:
        pass

    @abstractmethod
    def retrieve(
        self, auto_negotiation_hash: str
    ) -> Optional[AutoResolutionCacheEntry]:
        pass

    @abstractmethod
    def invalidate(self, auto_negotiation_hash: str) -> None:
        pass


class BaseAutoLoadMetadataCache(AutoResolutionCache):

    def __init__(
        self,
        file_lock_acquire_timeout: int,
        verbose: bool = False,
        on_file_created: Optional[Callable[[str, str, str], None]] = None,
        on_file_deleted: Optional[Callable[[str], None]] = None,
    ):
        self._file_lock_acquire_timeout = file_lock_acquire_timeout
        self._verbose = verbose
        self._on_file_created = on_file_created
        self._on_file_deleted = on_file_deleted

    def register(
        self, auto_negotiation_hash: str, cache_entry: AutoResolutionCacheEntry
    ) -> None:
        path_for_cached_content = generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )
        target_file_dir, target_file_name = os.path.split(path_for_cached_content)
        lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
        content = cache_entry.model_dump(mode="json")
        with FileLock(lock_path, timeout=self._file_lock_acquire_timeout):
            dump_json(path=path_for_cached_content, content=content)
            if self._on_file_created:
                self._on_file_created(
                    path_for_cached_content,
                    cache_entry.model_id,
                    cache_entry.model_package_id,
                )

    def retrieve(
        self, auto_negotiation_hash: str
    ) -> Optional[AutoResolutionCacheEntry]:
        path_for_cached_content = generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )
        if not os.path.exists(path_for_cached_content):
            return None
        try:
            cache_content = read_json(path=path_for_cached_content)
            cache_entry = AutoResolutionCacheEntry.model_validate(cache_content)
            minutes_since_entry_created = (
                datetime.now() - cache_entry.created_at
            ).total_seconds() / 60
            if minutes_since_entry_created > AUTO_LOADER_CACHE_EXPIRATION_MINUTES:
                self.invalidate(auto_negotiation_hash=auto_negotiation_hash)
                verbose_info(
                    message=f"Auto-negotiation cache for hash: {auto_negotiation_hash} is expired - removed its content",
                    verbose_requested=self._verbose,
                )
                return None
            return cache_entry
        except Exception as error:
            LOGGER.warning(
                f"Encountered error {error} of type {type(error)} when attempted to load model using "
                f"auto-load cache. This may indicate corrupted cache of inference bug. Contact Roboflow submitting "
                f"issue under: https://github.com/roboflow/inference/issues/"
            )
            self.invalidate(auto_negotiation_hash=auto_negotiation_hash)

    def invalidate(self, auto_negotiation_hash: str) -> None:
        path_for_cached_content = generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )
        if not os.path.exists(path=path_for_cached_content):
            return None
        os.remove(path=path_for_cached_content)
        if self._on_file_deleted:
            self._on_file_deleted(path_for_cached_content)


def generate_auto_resolution_cache_path(auto_negotiation_hash: str) -> str:
    return os.path.join(
        INFERENCE_HOME, "auto-resolution-cache", f"{auto_negotiation_hash}.json"
    )
