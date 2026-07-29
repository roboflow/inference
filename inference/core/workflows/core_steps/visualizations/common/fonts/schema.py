"""JSON Schema helpers for approved font fields in workflow block manifests."""

from typing import Any, Dict, Literal

from inference.core.workflows.core_steps.visualizations.common.fonts.registry import (
    DEFAULT_FONT_FAMILY,
    FONTS_REGISTRY,
)


def get_font_family_identifiers() -> tuple[str, ...]:
    """Sorted stable identifiers for all approved fonts."""
    return tuple(sorted(FONTS_REGISTRY.keys()))


def get_font_family_display_names() -> tuple[str, ...]:
    """Sorted human-readable display names for all approved fonts."""
    return tuple(
        FONTS_REGISTRY[identifier].display_name
        for identifier in get_font_family_identifiers()
    )


# Must stay a Literal (inline JSON Schema enum): a Python Enum is emitted as a
# $ref under $defs, which the Workflow Builder renders as a text field, not a dropdown.
FontFamilyName = Literal.__getitem__(get_font_family_display_names())


def get_default_font_family_display_name() -> str:
    """Default font display name for new Rich Label blocks."""
    return FONTS_REGISTRY[DEFAULT_FONT_FAMILY].display_name


def coerce_font_family_input(value: str) -> str:
    """Accept legacy snake_case identifiers and normalize to display names."""
    if value in FONTS_REGISTRY:
        return FONTS_REGISTRY[value].display_name
    return value


def font_family_to_identifier(font_family: str) -> str:
    """Map a display name or legacy identifier to a registry identifier."""
    if font_family in FONTS_REGISTRY:
        return font_family

    for identifier, metadata in FONTS_REGISTRY.items():
        if metadata.display_name == font_family:
            return identifier

    supported = ", ".join(get_font_family_display_names())
    raise ValueError(
        f"Unknown font family: {font_family!r}. Supported font families: {supported}."
    )


def font_family_field_json_schema_extra() -> Dict[str, Any]:
    """Build json_schema_extra for a font_family manifest field."""
    return {
        "always_visible": True,
    }
