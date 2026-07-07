import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

VALUE_TEMPLATE = "{{value}}"
CATALOG_PATH = Path(__file__).with_name("camera_register_catalog.json")


@lru_cache(maxsize=1)
def _load_catalog() -> Dict[str, Any]:
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def get_catalog() -> Dict[str, Any]:
    return _load_catalog()


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def list_camera_families() -> List[str]:
    return list(_load_catalog()["cameraFamilies"].keys())


def get_register_definition(register_key: str) -> Optional[Dict[str, Any]]:
    return _load_catalog()["registers"].get(_normalize_key(register_key))


def get_register_binding(register_key: str, camera_family: str) -> Optional[Dict[str, Any]]:
    register = _normalize_key(register_key)
    family = _normalize_key(camera_family)
    if register == "manual" or not family:
        return None
    definition = get_register_definition(register)
    if not definition:
        return None
    bindings = definition.get("bindings") or {}
    return bindings.get(family)


def registers_for_camera_family(camera_family: str) -> List[str]:
    family = _normalize_key(camera_family)
    if not family or family not in list_camera_families():
        return ["manual"]

    supported = [
        register_key
        for register_key, definition in _load_catalog()["registers"].items()
        if register_key != "manual" and (definition.get("bindings") or {}).get(family)
    ]
    supported.append("manual")
    return supported


def get_register_labels_map() -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for register_key, definition in _load_catalog()["registers"].items():
        labels[register_key] = definition.get("label") or register_key
    return labels


REGISTER_LABELS = get_register_labels_map()


def _expand_value_template(value: Any, template_value: Any) -> Any:
    if value == VALUE_TEMPLATE:
        return template_value
    if isinstance(value, list):
        return [_expand_value_template(entry, template_value) for entry in value]
    if isinstance(value, dict):
        return {
            key: _expand_value_template(nested_value, template_value)
            for key, nested_value in value.items()
        }
    return value


def build_parameter_delta(
    register: str,
    value: Any,
    *,
    camera_family: str,
    manual_register_key: Optional[str] = None,
) -> Dict[str, Any]:
    register_key = _normalize_key(register or "manual")
    family = _normalize_key(camera_family)
    manual_key = str(manual_register_key or "").strip()

    if register_key == "manual":
        if not manual_key:
            raise ValueError("manual_register_key is required when register is manual")
        if not family:
            raise ValueError("camera_family is required for manual register writes")
        return {manual_key: value}

    if not family:
        raise ValueError("camera_family is required")

    binding = get_register_binding(register_key, family)
    if not binding:
        raise ValueError(
            f"Register '{register_key}' is not supported for camera family '{family}'"
        )

    parameter_delta = binding.get("parameterDelta")
    if not isinstance(parameter_delta, dict):
        raise ValueError(
            f"Register '{register_key}' has no parameterDelta for camera family '{family}'"
        )

    expanded = _expand_value_template(parameter_delta, value)
    if not isinstance(expanded, dict):
        raise ValueError(
            f"Register '{register_key}' produced invalid parameter delta for '{family}'"
        )
    return expanded
