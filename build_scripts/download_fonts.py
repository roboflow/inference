"""Download the approved Workflows font assets into the source tree.

This is the single shared entry point for provisioning font assets across
Docker image builds, local development setup (``make download_fonts``) and
test asset preparation. Fonts and their license texts are downloaded from the
immutable URLs pinned in the fonts registry and verified against their
recorded SHA-256 checksums; the script fails hard on any mismatch.

The script is intentionally stdlib-only (argparse instead of click) so it can
run in any build environment before project dependencies are installed. It
loads the registry and downloader modules directly by file path to avoid
importing the heavyweight ``inference`` package.

Usage:
    python build_scripts/download_fonts.py
    python build_scripts/download_fonts.py --only geist_mono --only inter
    python build_scripts/download_fonts.py --target-dir /custom/assets/dir
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
FONTS_PACKAGE_DIR = (
    REPO_ROOT
    / "inference"
    / "core"
    / "workflows"
    / "core_steps"
    / "visualizations"
    / "common"
    / "fonts"
)
DEFAULT_TARGET_DIR = FONTS_PACKAGE_DIR / "assets"


def _load_module_by_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def download_fonts(target_dir: Path, *, only: list) -> int:
    """Download approved fonts and their licenses into ``target_dir``.

    Args:
        target_dir: Directory receiving ``<identifier>/<file_name>`` assets
            (plus ``<identifier>/OFL.txt`` license texts).
        only: Optional subset of font identifiers; empty means all fonts.

    Returns:
        Process exit code: 0 on success, 1 on any failure.
    """
    registry = _load_module_by_path(
        "workflows_fonts_registry", FONTS_PACKAGE_DIR / "registry.py"
    )
    downloader = _load_module_by_path(
        "workflows_fonts_downloader", FONTS_PACKAGE_DIR / "downloader.py"
    )

    selected_identifiers = only or sorted(registry.FONTS_REGISTRY.keys())
    unknown_identifiers = [
        identifier
        for identifier in selected_identifiers
        if identifier not in registry.FONTS_REGISTRY
    ]
    if unknown_identifiers:
        print(f"ERROR: unknown font identifiers: {', '.join(unknown_identifiers)}")
        return 1

    failures = []
    for identifier in selected_identifiers:
        metadata = registry.FONTS_REGISTRY[identifier]
        files_to_provision = [
            (
                metadata.source_url,
                target_dir / identifier / metadata.file_name,
                metadata.sha256,
            ),
            (
                metadata.license_url,
                target_dir / identifier / "OFL.txt",
                metadata.license_sha256,
            ),
        ]

        for source_url, destination, expected_sha256 in files_to_provision:
            if (
                destination.is_file()
                and downloader.compute_file_sha256(destination) == expected_sha256
            ):
                print(
                    f"[skip] {identifier}: {destination.name} already present and verified"
                )
                continue

            try:
                downloader.download_file_with_checksum(
                    source_url,
                    destination=destination,
                    expected_sha256=expected_sha256,
                )
                print(f"[ok]   {identifier}: downloaded and verified ({destination})")
            except Exception as error:
                failures.append(identifier)
                print(f"[FAIL] {identifier}: {error}")

    if failures:
        print(f"ERROR: failed to provision fonts: {', '.join(failures)}")
        return 1

    print(f"All requested fonts available in {target_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download approved Workflows font assets (checksum-verified).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=("Directory to place font assets in " f"(default: {DEFAULT_TARGET_DIR})"),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="FONT_IDENTIFIER",
        help="Limit download to the given font identifier (repeatable).",
    )
    arguments = parser.parse_args()

    exit_code = download_fonts(arguments.target_dir, only=arguments.only)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
