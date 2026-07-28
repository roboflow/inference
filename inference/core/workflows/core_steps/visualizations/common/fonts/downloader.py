"""Checksum-verified download primitive for approved font assets.

Downloads are only ever performed for URLs recorded in ``registry.py`` -
callers must never pass user-provided URLs. The file is streamed to a
temporary path, verified against the expected SHA-256 checksum and only then
atomically moved to its destination, so a partially-written or tampered file
can never be observed at the destination path.

IMPORTANT: this module must stay stdlib-only. It is loaded standalone (by
file path, without importing the ``inference`` package) by
``build_scripts/download_fonts.py`` during Docker builds and local setup.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_FONT_FILE_BYTES = 20 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024


class FontDownloadError(RuntimeError):
    """Raised when an approved font asset cannot be downloaded and verified."""


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        file_path: Path of the file to hash.

    Returns:
        Hex-encoded SHA-256 digest of the file contents.
    """
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)

    file_digest = digest.hexdigest()

    return file_digest


def download_file_with_checksum(
    url: str,
    *,
    destination: Path,
    expected_sha256: str,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FONT_FILE_BYTES,
) -> Path:
    """Download a pinned asset and verify its SHA-256 checksum.

    The response is streamed to ``<destination>.part`` with a hard size cap,
    verified, and atomically renamed to ``destination``. On any failure the
    partial file is removed and ``FontDownloadError`` is raised - the
    destination path is never left holding unverified bytes.

    Args:
        url: HTTPS URL of the approved asset (must come from the registry).
        destination: Final path for the verified file.
        expected_sha256: Hex-encoded SHA-256 checksum the file must match.
        timeout_seconds: Connect/read timeout for the download.
        max_bytes: Hard cap on the downloaded size.

    Returns:
        The ``destination`` path, once the file is verified.

    Raises:
        FontDownloadError: On non-HTTPS URLs, network errors, oversized
            responses, or checksum mismatch.
    """
    if not url.lower().startswith("https://"):
        raise FontDownloadError(
            f"Refusing to download font asset over non-HTTPS URL: {url}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")
    digest = hashlib.sha256()
    downloaded_bytes = 0

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            with partial_path.open("wb") as partial_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break

                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        raise FontDownloadError(
                            f"Font asset from {url} exceeds the maximum allowed "
                            f"size of {max_bytes} bytes."
                        )

                    digest.update(chunk)
                    partial_file.write(chunk)
    except FontDownloadError:
        partial_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        partial_path.unlink(missing_ok=True)
        raise FontDownloadError(
            f"Could not download font asset from {url}: {error}"
        ) from error

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        partial_path.unlink(missing_ok=True)
        raise FontDownloadError(
            f"Checksum mismatch for font asset downloaded from {url}: "
            f"expected sha256={expected_sha256}, got sha256={actual_sha256}. "
            "The upstream file may have been replaced - refusing to use it. "
            "Please report the issue: https://github.com/roboflow/inference/issues"
        )

    shutil.move(str(partial_path), str(destination))

    return destination
