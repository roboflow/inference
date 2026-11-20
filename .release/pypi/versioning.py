"""Normalize first-party RC metadata while preserving a post-release identity."""

import re

from packaging.version import Version


def _final_version(version):
    parsed = Version(version)
    if parsed.local or parsed.dev is not None:
        raise ValueError(f"Unsupported release version: {version}")
    if parsed.pre and parsed.pre[0] != "rc":
        raise ValueError(f"Expected an RC: {version}")
    suffix = f".post{parsed.post}" if parsed.post is not None else ""
    return parsed.base_version + suffix


def _validate_inventory(inventory):
    if (
        not isinstance(inventory, dict)
        or not inventory
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or not isinstance(version, str)
            or not version
            for name, version in inventory.items()
        )
    ):
        raise ValueError("Invalid RC dependency inventory")

    for version in inventory.values():
        Version(version)
