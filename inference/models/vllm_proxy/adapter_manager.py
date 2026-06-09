"""Resolution + registration of Roboflow models against the vLLM sidecar.

Maps a Roboflow `model_id` to the name it is served under in vLLM:

- Base model ids (matching `VLLM_SERVED_BASE_VARIANT`) require no
  registration - vLLM already serves the base model.
- Fine-tunes resolve their model package via the Roboflow weights provider
  (the api_key is passed through so registry-side access control applies),
  download ONLY the adapter artifacts, run `patch_adapter`, and register the
  patched adapter with vLLM's dynamic LoRA endpoint.

Cache keys combine model_id + package_id + a content digest because package
ids are NOT unique per model version. Registered adapters are tracked with
LRU bookkeeping bounded by `VLLM_MAX_REGISTERED_ADAPTERS`.
"""

import hashlib
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

from inference.core import logger
from inference.models.vllm_proxy.adapter_patch import ADAPTER_CONFIG_FILE, patch_adapter
from inference.models.vllm_proxy.config import (
    get_model_cache_dir,
    get_vllm_dora_policy,
    get_vllm_max_registered_adapters,
    get_vllm_served_base_name,
    get_vllm_served_base_variant,
)
from inference.models.vllm_proxy.errors import NotServableOnVLLMError, VLLMProxyError
from inference.models.vllm_proxy.vllm_client import VLLMClient
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.core import get_model_from_provider

ADAPTERS_CACHE_SUBDIR = "vllm-adapters"
BASE_PACKAGE_DIR_PREFIX = "base/"
SUPPORTED_MODEL_ARCHITECTURE = "qwen3_5"


@dataclass
class AdapterRegistration:
    """Bookkeeping record for an adapter registered with vLLM."""

    served_name: str
    model_id: str
    package_id: str
    content_digest: str
    source_dir: str
    patched_dir: str


def sanitize_for_slug(value: str) -> str:
    """Lowercases and reduces a value to `[a-z0-9-]` for cache/served names."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.lower())
    slug = re.sub(r"[_-]{2,}", "-", slug)
    return slug.strip("-")


class AdapterManager:
    """Thread-safe, idempotent adapter registration with LRU eviction."""

    def __init__(self, client: Optional[VLLMClient] = None):
        self._client = client or VLLMClient()
        self._lock = threading.Lock()
        self._registered: "OrderedDict[str, AdapterRegistration]" = OrderedDict()

    @property
    def client(self) -> VLLMClient:
        return self._client

    def resolve_and_register(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        weights_provider_extra_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Resolves `model_id` to the name it is served under in vLLM.

        Base model ids return the served base name without registration.
        Fine-tunes are downloaded, patched and registered (idempotent: an
        already-registered adapter is just touched in the LRU order).

        Raises:
            NotServableOnVLLMError: If the model is not a qwen3_5 model whose
                base variant matches the vLLM deployment.
            AdapterNotServableError: If the adapter cannot be patched into a
                vLLM-servable form.
        """
        served_base_variant = get_vllm_served_base_variant()
        if model_id.lower() == served_base_variant.lower():
            return get_vllm_served_base_name()
        metadata = get_model_from_provider(
            model_id=model_id,
            provider="roboflow",
            api_key=api_key,
            weights_provider_extra_headers=weights_provider_extra_headers,
        )
        if not (metadata.model_architecture or "").startswith(
            SUPPORTED_MODEL_ARCHITECTURE
        ):
            raise NotServableOnVLLMError(
                f"Model {model_id} has architecture "
                f"{metadata.model_architecture!r} which is not servable by the "
                f"vLLM proxy (expected {SUPPORTED_MODEL_ARCHITECTURE!r})."
            )
        if metadata.model_variant != served_base_variant:
            raise NotServableOnVLLMError(
                f"Model {model_id} is based on variant "
                f"{metadata.model_variant!r}, but the vLLM deployment serves "
                f"{served_base_variant!r} (VLLM_SERVED_BASE_VARIANT)."
            )
        package = self._select_model_package(model_id=model_id, metadata=metadata)
        adapter_files = [
            artefact
            for artefact in package.package_artefacts
            if not artefact.file_handle.startswith(BASE_PACKAGE_DIR_PREFIX)
        ]
        if not any(
            artefact.file_handle == ADAPTER_CONFIG_FILE for artefact in adapter_files
        ):
            # No adapter artifacts - this is a base-model package of the
            # served variant.
            return get_vllm_served_base_name()
        content_digest = self._compute_content_digest(adapter_files=adapter_files)
        slug = self._build_slug(
            model_id=metadata.model_id,
            package_id=package.package_id,
            content_digest=content_digest,
        )
        with self._lock:
            if slug in self._registered:
                self._registered.move_to_end(slug)
                return slug
            registration = self._download_patch_and_load(
                slug=slug,
                model_id=metadata.model_id,
                package_id=package.package_id,
                content_digest=content_digest,
                adapter_files=adapter_files,
            )
            self._registered[slug] = registration
            self._evict_lru_adapters()
        return slug

    def get_registration(self, served_name: str) -> Optional[AdapterRegistration]:
        with self._lock:
            return self._registered.get(served_name)

    def _select_model_package(self, model_id: str, metadata):
        hf_packages = [
            package
            for package in metadata.model_packages
            if package.backend is BackendType.HF
        ]
        if not hf_packages:
            raise NotServableOnVLLMError(
                f"Model {model_id} exposes no HF model package - the vLLM "
                "proxy requires HF (PEFT adapter) packages."
            )
        return hf_packages[0]

    @staticmethod
    def _compute_content_digest(adapter_files) -> str:
        """Digest over the adapter artifact hashes (no download required).

        Package ids are not unique per model version, so the digest
        disambiguates cache entries when package content changes.
        """
        digest = hashlib.sha256()
        for artefact in sorted(adapter_files, key=lambda a: a.file_handle):
            digest.update(artefact.file_handle.encode("utf-8"))
            digest.update((artefact.md5_hash or artefact.download_url).encode("utf-8"))
        return digest.hexdigest()[:8]

    @staticmethod
    def _build_slug(model_id: str, package_id: str, content_digest: str) -> str:
        return (
            f"{sanitize_for_slug(model_id)}-{sanitize_for_slug(package_id)}"
            f"-{content_digest}"
        )

    def _download_patch_and_load(
        self,
        slug: str,
        model_id: str,
        package_id: str,
        content_digest: str,
        adapter_files,
    ) -> AdapterRegistration:
        adapter_cache_dir = os.path.join(
            get_model_cache_dir(), ADAPTERS_CACHE_SUBDIR, slug
        )
        source_dir = os.path.join(adapter_cache_dir, "src")
        patched_dir = os.path.join(adapter_cache_dir, "patched")
        os.makedirs(source_dir, exist_ok=True)
        download_files_to_directory(
            target_dir=source_dir,
            files_specs=[
                (artefact.file_handle, artefact.download_url, artefact.md5_hash)
                for artefact in adapter_files
            ],
            verbose=False,
        )
        patch_adapter(
            src_dir=source_dir,
            dst_dir=patched_dir,
            policy=get_vllm_dora_policy(),
        )
        self._client.load_lora_adapter(name=slug, path=patched_dir)
        logger.info(
            "Registered LoRA adapter %s (model_id=%s, package_id=%s) with vLLM",
            slug,
            model_id,
            package_id,
        )
        return AdapterRegistration(
            served_name=slug,
            model_id=model_id,
            package_id=package_id,
            content_digest=content_digest,
            source_dir=source_dir,
            patched_dir=patched_dir,
        )

    def _evict_lru_adapters(self) -> None:
        max_registered = get_vllm_max_registered_adapters()
        while len(self._registered) > max_registered:
            evicted_slug, _ = self._registered.popitem(last=False)
            try:
                self._client.unload_lora_adapter(name=evicted_slug)
                logger.info("Evicted LRU LoRA adapter %s from vLLM", evicted_slug)
            except VLLMProxyError as error:
                # The adapter may have been evicted on the vLLM side already -
                # bookkeeping stays consistent either way.
                logger.warning(
                    "Failed to unload LoRA adapter %s from vLLM: %s",
                    evicted_slug,
                    error,
                )


_ADAPTER_MANAGER: Optional[AdapterManager] = None
_ADAPTER_MANAGER_LOCK = threading.Lock()


def get_adapter_manager() -> AdapterManager:
    """Returns the process-wide AdapterManager singleton."""
    global _ADAPTER_MANAGER
    with _ADAPTER_MANAGER_LOCK:
        if _ADAPTER_MANAGER is None:
            _ADAPTER_MANAGER = AdapterManager()
        return _ADAPTER_MANAGER
