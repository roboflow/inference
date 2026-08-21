import json
import os
import re
import stat
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from filelock import FileLock
from pydantic import BaseModel, Field

from inference_models.configuration import (
    AUTO_LOADER_CACHE_EXPIRATION_MINUTES,
    INFERENCE_HOME,
)
from inference_models.logger import LOGGER, verbose_info
from inference_models.models.auto_loaders.entities import (
    BackendType,
    ModelArchitecture,
    TaskType,
)
from inference_models.utils.file_system import read_json
from inference_models.weights_providers.entities import (
    ModelDependency,
    RecommendedParameters,
)


class AutoResolutionCacheEntry(BaseModel):
    model_id: str
    # Model id whose on-disk cache directory holds the package. Differs from
    # model_id for locally-discovered packages resolved under an alias.
    cache_model_id: Optional[str] = Field(default=None)
    # Provider-resolved identity for the model.  This must be explicit before
    # an API-key-independent cache lookup can safely attribute an alias.
    canonical_model_id: Optional[str] = Field(default=None)
    cache_attribution_version: Optional[int] = Field(default=None, strict=True)
    # One-way binding to the effective credential used for this resolution.
    # This lets metadata-only alias discovery avoid crossing API-key scopes.
    credential_hash: Optional[str] = Field(default=None)
    model_package_id: str
    resolved_files: List[str]
    model_architecture: Optional[ModelArchitecture]
    task_type: TaskType
    backend_type: Optional[BackendType]
    model_dependencies: Optional[List[ModelDependency]] = Field(default=None)
    created_at: datetime
    model_features: Optional[dict] = Field(default=None)
    recommended_parameters: Optional[RecommendedParameters] = Field(default=None)
    trusted_source: Optional[bool] = Field(default=None)


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

    def find_model_candidates(
        self,
        model_id: str,
        credential_hash: Optional[str] = None,
    ) -> List[Tuple[str, AutoResolutionCacheEntry]]:
        """Return entries for an exact requested identity and credential."""

        return []


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
        # Unset optional fields are omitted rather than written as nulls -
        # the tolerant reader defaults them, and retired fields (e.g. the
        # v1-era compatibility hashes) must not reappear in fresh entries.
        content = cache_entry.model_dump(mode="json", exclude_none=True)
        if os.path.islink(target_file_dir):
            LOGGER.warning(
                "Refusing to write auto-resolution metadata through a symlinked directory"
            )
            return None
        os.makedirs(target_file_dir, exist_ok=True)
        if (
            os.path.islink(target_file_dir)
            or os.path.islink(path_for_cached_content)
            or os.path.islink(lock_path)
        ):
            LOGGER.warning(
                "Refusing to write auto-resolution metadata through a symlink"
            )
            return None
        with FileLock(lock_path, timeout=self._file_lock_acquire_timeout):
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=target_file_dir,
                    prefix=f".{target_file_name}.",
                    suffix=".tmp",
                    delete=False,
                ) as file_handle:
                    temporary_path = file_handle.name
                    json.dump(content, file_handle)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                if os.path.islink(path_for_cached_content):
                    LOGGER.warning(
                        "Refusing to replace auto-resolution metadata through a symlink"
                    )
                    return None
                os.replace(temporary_path, path_for_cached_content)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except OSError:
                        pass
            if self._on_file_created:
                self._on_file_created(
                    path_for_cached_content,
                    cache_entry.canonical_model_id or cache_entry.model_id,
                    cache_entry.model_package_id,
                )

    def retrieve(
        self, auto_negotiation_hash: str
    ) -> Optional[AutoResolutionCacheEntry]:
        path_for_cached_content = generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )
        target_file_dir, target_file_name = os.path.split(path_for_cached_content)
        lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
        if os.path.islink(target_file_dir) or os.path.islink(path_for_cached_content):
            return None
        if os.path.islink(lock_path) or not os.path.exists(path_for_cached_content):
            return None
        with FileLock(lock_path, timeout=self._file_lock_acquire_timeout):
            if (
                os.path.islink(target_file_dir)
                or os.path.islink(path_for_cached_content)
                or not os.path.exists(path_for_cached_content)
            ):
                return None
            try:
                cache_content = read_json(path=path_for_cached_content)
                cache_entry = AutoResolutionCacheEntry.model_validate(cache_content)
                minutes_since_entry_created = (
                    datetime.now().timestamp() - cache_entry.created_at.timestamp()
                ) / 60
                if (
                    minutes_since_entry_created
                    > AUTO_LOADER_CACHE_EXPIRATION_MINUTES
                ):
                    self._invalidate_path_while_locked(
                        path_for_cached_content=path_for_cached_content
                    )
                    verbose_info(
                        message=f"Auto-negotiation cache for hash: {auto_negotiation_hash} is expired - removed its content",
                        verbose_requested=self._verbose,
                    )
                    return None
                return cache_entry
            except Exception as error:
                self._log_retrieval_error(error=error)
                self._invalidate_path_while_locked(
                    path_for_cached_content=path_for_cached_content
                )

    def find_model_candidates(
        self,
        model_id: str,
        credential_hash: Optional[str] = None,
    ) -> List[Tuple[str, AutoResolutionCacheEntry]]:
        cache_dir = os.path.abspath(
            os.path.join(INFERENCE_HOME, "auto-resolution-cache")
        )
        if os.path.islink(cache_dir) or not os.path.isdir(cache_dir):
            return []
        try:
            entries = sorted(os.listdir(cache_dir))
        except OSError:
            return []
        matches: List[Tuple[str, AutoResolutionCacheEntry]] = []
        for entry_name in entries:
            if not re.fullmatch(r"[0-9a-f]{64}\.json", entry_name):
                continue
            entry_path = os.path.join(cache_dir, entry_name)
            if os.path.islink(entry_path) or not os.path.isfile(entry_path):
                continue
            auto_negotiation_hash = entry_name[:-5]
            cache_entry = self.retrieve(auto_negotiation_hash=auto_negotiation_hash)
            if (
                cache_entry is not None
                and cache_entry.model_id == model_id
                and (
                    credential_hash is None
                    or cache_entry.credential_hash == credential_hash
                )
            ):
                matches.append((auto_negotiation_hash, cache_entry))
        return sorted(
            matches,
            key=lambda match: match[1].created_at.timestamp(),
            reverse=True,
        )

    def invalidate(self, auto_negotiation_hash: str) -> None:
        path_for_cached_content = generate_auto_resolution_cache_path(
            auto_negotiation_hash=auto_negotiation_hash
        )
        target_file_dir, target_file_name = os.path.split(path_for_cached_content)
        lock_path = os.path.join(target_file_dir, f".{target_file_name}.lock")
        if (
            os.path.islink(target_file_dir)
            or os.path.islink(path_for_cached_content)
            or os.path.islink(lock_path)
            or not os.path.exists(path=path_for_cached_content)
        ):
            return None
        with FileLock(lock_path, timeout=self._file_lock_acquire_timeout):
            self._invalidate_path_while_locked(
                path_for_cached_content=path_for_cached_content
            )

    def _invalidate_path_while_locked(
        self,
        path_for_cached_content: str,
    ) -> None:
        if os.path.islink(path_for_cached_content):
            return None
        try:
            os.remove(path=path_for_cached_content)
        except FileNotFoundError:
            return None
        if self._on_file_deleted:
            self._on_file_deleted(path_for_cached_content)

    @staticmethod
    def _log_retrieval_error(error: Exception) -> None:
        LOGGER.warning(
            f"Encountered error {error} of type {type(error)} when attempted to load model using "
            f"auto-load cache. This may indicate corrupted cache of inference bug. Contact Roboflow submitting "
            f"issue under: https://github.com/roboflow/inference/issues/"
        )
def generate_auto_resolution_cache_path(auto_negotiation_hash: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", auto_negotiation_hash):
        raise ValueError("Invalid auto-negotiation cache hash")
    return os.path.abspath(
        os.path.join(
            INFERENCE_HOME, "auto-resolution-cache", f"{auto_negotiation_hash}.json"
        )
    )
