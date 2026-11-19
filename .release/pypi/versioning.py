"""Normalize first-party RC metadata while preserving a post-release identity."""

from packaging.version import Version


def _final_version(version):
    parsed = Version(version)
    if parsed.local or parsed.dev is not None:
        raise ValueError(f"Unsupported release version: {version}")
    if parsed.pre and parsed.pre[0] != "rc":
        raise ValueError(f"Expected an RC: {version}")
    suffix = f".post{parsed.post}" if parsed.post is not None else ""
    return parsed.base_version + suffix
