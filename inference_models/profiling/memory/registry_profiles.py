"""Lookup task and package input profiles from ``registry_input_profiles.json``."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_REGISTRY_PROFILES_PATH = (
    Path(__file__).resolve().parent / "registry_input_profiles.json"
)


@lru_cache(maxsize=1)
def _load_registry_profiles_document() -> Dict[str, Any]:
    with _REGISTRY_PROFILES_PATH.open(encoding="utf-8") as handle:
        document = json.load(handle)

    return document


def find_registry_entry(
    *,
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    architecture: Optional[str] = None,
    task_type: Optional[str] = None,
    backend: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the first matching ``registry_entries`` row, if any."""
    document = _load_registry_profiles_document()
    entries: List[Dict[str, Any]] = document.get("registry_entries") or []

    for entry in entries:
        if module_name and entry.get("module_name") != module_name:
            continue
        if class_name and entry.get("class_name") != class_name:
            continue
        if architecture and entry.get("architecture") != architecture:
            continue
        if task_type and entry.get("task_type") != task_type:
            continue
        entry_backend = entry.get("backend")
        if backend:
            if backend == "torch":
                if entry_backend not in ("torch", "hugging-face", "torch-script"):
                    continue
            elif entry_backend != backend:
                continue

        return entry

    return None


def get_task_inference_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Return the declared task inference profile spec by name."""
    document = _load_registry_profiles_document()
    profiles: Dict[str, Any] = document.get("task_inference_profiles") or {}

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return None

    return profile


def resolve_registry_input_context(
    *,
    module_name: Optional[str] = None,
    class_name: Optional[str] = None,
    architecture: Optional[str] = None,
    task_type: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve registry entry and its task inference profile for metadata assembly."""
    entry = find_registry_entry(
        module_name=module_name,
        class_name=class_name,
        architecture=architecture,
        task_type=task_type,
        backend=backend,
    )
    if entry is None:
        return {}

    profile_name = entry.get("task_inference_profile")
    task_profile = (
        get_task_inference_profile(profile_name) if isinstance(profile_name, str) else None
    )

    return {
        "registry_entry": entry,
        "task_inference_profile": profile_name,
        "task_profile_spec": task_profile,
        "package_input_backend": entry.get("package_input_backend"),
        "torch_package_variant": entry.get("torch_package_variant"),
    }
