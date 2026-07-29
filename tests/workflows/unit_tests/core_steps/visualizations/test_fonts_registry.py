import pytest
from PIL import ImageFont

from inference.core.workflows.core_steps.visualizations.common.fonts import (
    ASSETS_DIR,
    DEFAULT_FONT_FAMILY,
    FONTS_REGISTRY,
    resolve_font_path,
    verify_font_checksum,
)


def test_default_font_family_is_registered() -> None:
    assert DEFAULT_FONT_FAMILY in FONTS_REGISTRY


def test_geist_mono_is_registered() -> None:
    # Geist Mono support is a hard product requirement
    assert "geist_mono" in FONTS_REGISTRY
    assert FONTS_REGISTRY["geist_mono"].display_name == "Geist Mono"


@pytest.mark.parametrize("font_family", sorted(FONTS_REGISTRY.keys()))
def test_provisioned_font_asset_is_valid(bundled_fonts, font_family: str) -> None:
    # when
    font_path = resolve_font_path(font_family)
    font = ImageFont.truetype(str(font_path), 16)
    license_path = ASSETS_DIR / font_family / "OFL.txt"

    # then - asset exists, matches the registry checksum, is loadable by
    # Pillow and ships with its license text (required by OFL-1.1)
    assert font_path.is_file()
    assert verify_font_checksum(font_family), (
        f"Provisioned font file for {font_family!r} does not match the sha256 "
        "recorded in FONTS_REGISTRY"
    )
    assert font.getname()[0] is not None
    assert license_path.is_file(), (
        f"Font {font_family!r} must ship with its license text "
        "(required by OFL-1.1 redistribution terms)"
    )
    assert "SIL OPEN FONT LICENSE" in license_path.read_text()


def test_resolve_font_path_accepts_display_name(bundled_fonts) -> None:
    assert resolve_font_path("Geist Mono") == resolve_font_path("geist_mono")


@pytest.mark.parametrize(
    "font_family",
    [
        "comic_sans",
        "../../../../etc/passwd",
        "/tmp/evil.ttf",
        "https://evil.com/font.ttf",
    ],
)
def test_resolve_font_path_raises_on_unregistered_identifier(font_family: str) -> None:
    # unknown identifiers, path traversal payloads, absolute paths and URLs
    # must all be rejected - only registry keys resolve
    with pytest.raises(ValueError) as error:
        _ = resolve_font_path(font_family)

    assert "Geist Mono" in str(
        error.value
    ), "Error message should list supported font families"


def test_registry_metadata_is_complete() -> None:
    for font_family, metadata in FONTS_REGISTRY.items():
        assert metadata.identifier == font_family
        assert metadata.source_url.startswith("https://")
        assert len(metadata.sha256) == 64
        assert metadata.license_url.startswith("https://")
        assert len(metadata.license_sha256) == 64
        assert metadata.license_name
        assert metadata.version
