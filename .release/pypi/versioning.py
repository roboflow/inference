"""Normalize first-party RC metadata while preserving a post-release identity."""

import re

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def _final_version(version):
    parsed = Version(version)
    if parsed.local or parsed.dev is not None:
        raise ValueError(f"Unsupported release version: {version}")
    if parsed.pre and parsed.pre[0] != "rc":
        raise ValueError(f"Expected an RC: {version}")
    suffix = f".post{parsed.post}" if parsed.post is not None else ""
    return parsed.base_version + suffix


def _validate_inventory(
    inventory, *, context="dependency inventory", strict_packages=None
):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError(f"{context}: invalid dependency inventory {inventory!r}")

    for name, version in inventory.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or not isinstance(version, str)
            or not version
        ):
            raise ValueError(
                f"{context}: invalid inventory entry {name!r}: {version!r}"
            )

        if strict_packages is None or canonicalize_name(name) in strict_packages:
            try:
                Version(version)
            except InvalidVersion as error:
                raise ValueError(
                    f"{context}: invalid version for {name!r}: {version!r}"
                ) from error
