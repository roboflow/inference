"""Parent-side resolution of models that can share a base instance in one worker.

Runs in the MMP parent: it fetches model metadata via the same provider path
AutoModel uses, decides whether a head declares a supported shared base as a
dependency, resolves the base's concrete package (the same negotiation the worker
would run, stopping before any weight download), and derives the base_key used to
route heads to a shared-base owner. No weights are loaded here.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import UnauthorizedModelAccessError
from inference_models.models.auto_loaders.auto_negotiation import (
    negotiate_model_packages,
)
from inference_models.utils.hashing import hash_dict_content
from inference_models.weights_providers.core import get_model_from_provider
from inference_models.weights_providers.entities import ModelDependency, ModelMetadata

logger = logging.getLogger(__name__)

# Base architectures whose instance may be shared by heads.
SUPPORTED_SHARED_BASES = frozenset({"owlv2"})
# Public bases safe to share across tenants — base_key excludes api_key/access scope
# only for these. A supported-but-not-whitelisted base keeps per-tenant base_keys.
# ponytail: equal to SUPPORTED for v1; the day a non-public base becomes shareable,
# leaving it out of this set forces per-tenant keys automatically.
WHITELISTED_SHARED_BASES = frozenset({"owlv2"})

MetadataCacheKey = Tuple[str, str, str]


@dataclass(frozen=True)
class SharedBaseResolution:
    dep_name: str
    dep_model_id: str
    # Mirrors the head metadata dependency package id (None when unresolved). Goes
    # into SuppliedDependency for identity validation — NOT the resolved package.
    dep_metadata_package_id: Optional[str]
    # Concrete package the base negotiates to. Drives base_key + worker routing.
    resolved_package_id: str
    base_key: str


def _resolve_device(device: str) -> torch.device:
    return torch.device(device) if device else DEFAULT_DEVICE


def _fetch_metadata(
    model_id: str,
    api_key: str,
    provider: str,
    metadata_cache: Dict[MetadataCacheKey, ModelMetadata],
) -> ModelMetadata:
    key: MetadataCacheKey = (provider, model_id, api_key or "")
    if key not in metadata_cache:
        metadata_cache[key] = get_model_from_provider(
            provider=provider, model_id=model_id, api_key=api_key or None
        )
    return metadata_cache[key]


def derive_base_key(
    dep_model_id: str,
    resolved_package_id: str,
    device: torch.device,
    *,
    api_key: str,
    dep_model_architecture: Optional[str] = None,
) -> str:
    content = {
        "dep_model_id": dep_model_id,
        "resolved_package_id": resolved_package_id,
        "device": str(device),
        "backend": None,
        "quantization": None,
    }
    share_across_tenants = (
        dep_model_architecture in WHITELISTED_SHARED_BASES
        or dep_model_id in WHITELISTED_SHARED_BASES
    )
    if not share_across_tenants:
        content["api_key"] = api_key or ""
    return hash_dict_content(content=content)


def resolve_shared_base(
    model_id: str,
    api_key: str,
    device: str,
    metadata_cache: Dict[MetadataCacheKey, ModelMetadata],
    *,
    provider: str = "roboflow",
    supported_bases: frozenset = SUPPORTED_SHARED_BASES,
) -> Optional[SharedBaseResolution]:
    """Resolve whether ``model_id`` is a head that can share a supported base.

    Returns a SharedBaseResolution when shareable, None otherwise. Fail-open: any
    metadata/negotiation failure other than an auth/access denial returns None so the
    caller falls back to the normal subproc load; auth denials propagate (the normal
    load would hit them too).
    """
    resolved_device = _resolve_device(device)
    try:
        head_metadata = _fetch_metadata(model_id, api_key, provider, metadata_cache)
        dependency = None
        dep_metadata = None
        for candidate in head_metadata.model_dependencies or []:
            candidate_metadata = _fetch_metadata(
                candidate.model_id, api_key, provider, metadata_cache
            )
            if candidate_metadata.model_architecture in supported_bases:
                dependency = candidate
                dep_metadata = candidate_metadata
                break
        if dependency is None or dep_metadata is None:
            return None
        packages = negotiate_model_packages(
            model_architecture=dep_metadata.model_architecture,
            task_type=dep_metadata.task_type,
            model_packages=dep_metadata.model_packages,
            requested_model_package_id=dependency.model_package_id,
            device=resolved_device,
        )
        if not packages:
            return None
        resolved_package_id = packages[0].package_id
    except UnauthorizedModelAccessError:
        raise
    except Exception:
        logger.debug(
            "MMP: shared-base resolution failed for '%s' — falling back to normal load",
            model_id,
            exc_info=True,
        )
        return None
    base_key = derive_base_key(
        dependency.model_id,
        resolved_package_id,
        resolved_device,
        api_key=api_key,
        dep_model_architecture=dep_metadata.model_architecture,
    )
    return SharedBaseResolution(
        dep_name=dependency.name,
        dep_model_id=dependency.model_id,
        dep_metadata_package_id=dependency.model_package_id,
        resolved_package_id=resolved_package_id,
        base_key=base_key,
    )


def resolve_shared_base_model(
    model_id: str,
    api_key: str,
    device: str,
    metadata_cache: Dict[MetadataCacheKey, ModelMetadata],
    *,
    provider: str = "roboflow",
    supported_bases: frozenset = SUPPORTED_SHARED_BASES,
) -> Optional[SharedBaseResolution]:
    """Resolve a top-level base model for admin preload."""
    resolved_device = _resolve_device(device)
    try:
        metadata = _fetch_metadata(model_id, api_key, provider, metadata_cache)
        if metadata.model_architecture not in supported_bases:
            return None
        dep_name = "feature_extractor"
        packages = negotiate_model_packages(
            model_architecture=metadata.model_architecture,
            task_type=metadata.task_type,
            model_packages=metadata.model_packages,
            requested_model_package_id=None,
            device=resolved_device,
        )
        if len(packages) != 1:
            return None
        resolved_package_id = packages[0].package_id
    except UnauthorizedModelAccessError:
        raise
    except Exception:
        logger.debug(
            "MMP: shared-base preload resolution failed for '%s'",
            model_id,
            exc_info=True,
        )
        return None
    base_key = derive_base_key(
        metadata.model_id,
        resolved_package_id,
        resolved_device,
        api_key=api_key,
        dep_model_architecture=metadata.model_architecture,
    )
    return SharedBaseResolution(
        dep_name=dep_name,
        dep_model_id=metadata.model_id,
        dep_metadata_package_id=None,
        resolved_package_id=resolved_package_id,
        base_key=base_key,
    )
