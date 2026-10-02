"""Mask-resolution behaviour of the instance-segmentation block versions.

`mask_decode_mode` and `tradeoff_factor` are declared on every version's
manifest but were discarded by the server until recently. Now that they are
honoured, forwarding them from an older version would silently change the masks
an existing workflow produces, so only `@v5` forwards them. These tests pin
that split, and the manifest parity between a version and its tensor-native
sibling.
"""

import re
from pathlib import Path

import pytest


def _blocks_dir() -> Path:
    """Locate the block implementations relative to this test file.

    Returns:
        Directory holding the instance-segmentation block versions.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent
            / "workflows"
            / "roboflow_workflows"
            / "core_steps"
            / "models"
            / "roboflow"
            / "instance_segmentation"
        )
        if candidate.is_dir():
            return candidate
    raise RuntimeError("could not locate the instance-segmentation blocks")


BLOCKS_DIR = _blocks_dir()

PINNED_VERSIONS = ["v1", "v2", "v3", "v4"]
FORWARDING_VERSIONS = ["v5"]
ALL_VERSIONS = PINNED_VERSIONS + FORWARDING_VERSIONS


def _source(version: str, tensor: bool) -> str:
    """Read a block implementation's source.

    Args:
        version: Block version directory stem, such as ``"v5"``.
        tensor: Whether to read the tensor-native sibling.

    Returns:
        File contents.
    """
    suffix = "_tensor" if tensor else ""

    return (BLOCKS_DIR / f"{version}{suffix}.py").read_text()


def _manifest_fields(source: str) -> list:
    """Collect manifest field names declared on a block.

    Args:
        source: Block implementation source.

    Returns:
        Field names in declaration order.
    """
    start = source.index("class BlockManifest(")
    fields = []
    for line in source[start:].split("\n"):
        if line.startswith("class ") and "BlockManifest" not in line:
            break
        match = re.match(r"    ([a-z_]+): ", line)
        if match:
            fields.append(match.group(1))

    return fields


@pytest.mark.parametrize("version", ALL_VERSIONS)
@pytest.mark.parametrize("tensor", [False, True])
def test_version_declares_its_own_identifier(version: str, tensor: bool) -> None:
    # given
    source = _source(version, tensor)

    # then
    assert f'"version": "{version}"' in source
    assert f"instance_segmentation_model@{version}" in source
    assert f"RoboflowInstanceSegmentationModelBlock{version.upper()}" in source


@pytest.mark.parametrize("version", ALL_VERSIONS)
def test_manifest_parity_with_tensor_sibling(version: str) -> None:
    # given
    # the two files are independent implementations selected by an env var, so
    # a field present in one and not the other is invisible until runtime
    plain = _manifest_fields(_source(version, tensor=False))
    tensor = _manifest_fields(_source(version, tensor=True))

    # then
    assert plain == tensor


@pytest.mark.parametrize("version", PINNED_VERSIONS)
@pytest.mark.parametrize("tensor", [False, True])
def test_legacy_versions_pin_full_resolution(version: str, tensor: bool) -> None:
    # given
    # these versions must keep requesting full-resolution masks whatever the
    # caller set, because the fields never had an effect when they shipped
    source = _source(version, tensor)

    # then
    assert 'mask_decode_mode="accurate",' in source
    assert "tradeoff_factor=1.0," in source


@pytest.mark.parametrize("version", FORWARDING_VERSIONS)
@pytest.mark.parametrize("tensor", [False, True])
def test_v5_forwards_what_the_caller_set(version: str, tensor: bool) -> None:
    # given
    source = _source(version, tensor)

    # then
    assert "mask_decode_mode=mask_decode_mode," in source
    assert "tradeoff_factor=tradeoff_factor," in source
    assert 'mask_decode_mode="accurate",' not in source
