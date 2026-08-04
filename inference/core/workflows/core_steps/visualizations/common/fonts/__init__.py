"""Resolution of approved fonts for rich text rendering in Workflows.

Workflow authors select fonts by display name (e.g. ``Geist Mono``); legacy
snake_case identifiers (e.g. ``geist_mono``) are also accepted. Names are
resolved to local font files internally. Arbitrary filesystem paths and
user-provided URLs are deliberately NOT supported - only fonts registered in
``registry.py`` can be used.

Resolution order for a registered identifier:

1. Font asset baked into the installation (``assets/<identifier>/``) -
   always the case in official Docker images and after ``make download_fonts``.
2. Font asset previously downloaded to the local cache directory
   (``$MODEL_CACHE_DIR/workflows/fonts/``), verified against the registry
   checksum.
3. Runtime download from the registry-pinned URL, verified against the
   registry checksum - only when ``ALLOW_WORKFLOWS_FONTS_DOWNLOAD`` is True.
4. A clear, actionable error.

See ``README.md`` in this package for licensing, attribution and
instructions for adding a new approved font.
"""

from pathlib import Path
from typing import Set

from filelock import FileLock

from inference.core.env import (
    ALLOW_WORKFLOWS_FONTS_DOWNLOAD,
    MODEL_CACHE_DIR,
    OFFLINE_MODE,
    SECURE_GATEWAY,
)
from inference.core.workflows.core_steps.visualizations.common.fonts.downloader import (
    FontDownloadError,
    compute_file_sha256,
    download_file_with_checksum,
)
from inference.core.workflows.core_steps.visualizations.common.fonts.registry import (
    DEFAULT_FONT_FAMILY,
    FONTS_REGISTRY,
    FontMetadata,
)
from inference.core.workflows.core_steps.visualizations.common.fonts.schema import (
    font_family_to_identifier,
)

ASSETS_DIR = Path(__file__).parent / "assets"
FONTS_CACHE_DIR = Path(MODEL_CACHE_DIR) / "workflows" / "fonts"
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 180

_VERIFIED_CACHED_FONTS: Set[Path] = set()


def resolve_font_path(font_family: str) -> Path:
    """Resolve a public font identifier to a local, verified font file path.

    This is the only supported way of obtaining font paths for rendering -
    workflow inputs never carry filesystem paths or URLs. When the asset is
    neither baked into the installation nor cached locally, it is downloaded
    from its registry-pinned URL and verified against the registry SHA-256
    checksum. Runtime downloads are skipped - failing fast with an actionable
    error instead - when ``ALLOW_WORKFLOWS_FONTS_DOWNLOAD=False``, when
    ``OFFLINE_MODE`` is enabled, or when the deployment sits behind a
    ``SECURE_GATEWAY`` (the fonts' upstream hosts are unreachable there).

    Args:
        font_family: Font display name (e.g. ``"Geist Mono"``) or legacy
            snake_case identifier (e.g. ``"geist_mono"``) declared in
            ``FONTS_REGISTRY``.

    Returns:
        Absolute path to a verified local font file.

    Raises:
        ValueError: If ``font_family`` is not a registered font.
        RuntimeError: If the font asset is missing locally and runtime
            downloads are disabled, or the download fails checksum
            verification.
    """
    font_identifier = font_family_to_identifier(font_family)
    metadata = FONTS_REGISTRY[font_identifier]
    packaged_path = ASSETS_DIR / metadata.identifier / metadata.file_name
    if packaged_path.is_file():
        return packaged_path

    cached_path = FONTS_CACHE_DIR / metadata.identifier / metadata.file_name
    if _cached_font_is_valid(cached_path, metadata=metadata):
        return cached_path

    downloads_disabled_reason = _downloads_disabled_reason()
    if downloads_disabled_reason:
        raise RuntimeError(
            f"Font {font_identifier!r} is not available locally and runtime font "
            f"downloads are disabled ({downloads_disabled_reason}). "
            "Bake the approved fonts into your installation by running "
            "`make download_fonts` (or `python build_scripts/download_fonts.py`) "
            "from the repository root."
        )

    downloaded_path = _download_font_to_cache(cached_path, metadata=metadata)

    return downloaded_path


def verify_font_checksum(font_family: str) -> bool:
    """Verify that a locally resolved font file matches its registry checksum.

    Intended for tests and packaging verification - not executed on the
    rendering hot path.

    Args:
        font_family: Stable font identifier declared in ``FONTS_REGISTRY``.

    Returns:
        True if the SHA-256 checksum of the resolved file matches the
        registry entry.
    """
    metadata = FONTS_REGISTRY[font_family]
    font_path = resolve_font_path(font_family)

    checksum_matches = compute_file_sha256(font_path) == metadata.sha256

    return checksum_matches


def _downloads_disabled_reason() -> str:
    # fail fast when the fonts' upstream hosts are unreachable by policy
    if not ALLOW_WORKFLOWS_FONTS_DOWNLOAD:
        return "ALLOW_WORKFLOWS_FONTS_DOWNLOAD=False"

    if OFFLINE_MODE:
        return "OFFLINE_MODE is enabled"

    if SECURE_GATEWAY:
        return (
            "SECURE_GATEWAY is configured and the fonts' upstream hosts "
            "are not reachable through the gateway"
        )

    return ""


def _cached_font_is_valid(cached_path: Path, *, metadata: FontMetadata) -> bool:
    # cache-dir files are re-verified on first use after every process start
    if not cached_path.is_file():
        return False

    if cached_path in _VERIFIED_CACHED_FONTS:
        return True

    if compute_file_sha256(cached_path) == metadata.sha256:
        _VERIFIED_CACHED_FONTS.add(cached_path)
        return True

    # corrupted cache entry - remove so the download path restores a verified copy
    cached_path.unlink(missing_ok=True)

    return False


def _download_font_to_cache(cached_path: Path, *, metadata: FontMetadata) -> Path:
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = f"{cached_path}.lock"

    with FileLock(lock_path, timeout=DOWNLOAD_LOCK_TIMEOUT_SECONDS):
        # another process may have completed the download while we waited
        if not _cached_font_is_valid(cached_path, metadata=metadata):
            download_file_with_checksum(
                metadata.source_url,
                destination=cached_path,
                expected_sha256=metadata.sha256,
            )

    _VERIFIED_CACHED_FONTS.add(cached_path)

    return cached_path
