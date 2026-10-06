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
def test_legacy_versions_pin_local_conditionally_and_remote_unconditionally(
    version: str,
) -> None:
    # given
    # the legacy backend honours these fields today, so pinning there would
    # remove working behaviour. The pin is conditional on USE_INFERENCE_MODELS
    source = _source(version, tensor=False)

    # then
    assert source.count('"accurate" if USE_INFERENCE_MODELS else mask_decode_mode') == 1
    assert source.count("1.0 if USE_INFERENCE_MODELS else tradeoff_factor") == 1
    assert source.count('mask_decode_mode="accurate",') == 1


@pytest.mark.parametrize("version", PINNED_VERSIONS + FORWARDING_VERSIONS)
def test_tensor_siblings_pin_unconditionally(version: str) -> None:
    # given
    # the reason here is downstream rather than the backend: nothing under
    # roboflow_workflows/ reads mask_size, so a reduced grid is reinterpreted
    # as image-sized whichever backend produced it
    source = _source(version, tensor=True)

    # then
    assert source.count('mask_decode_mode="accurate",') == 2
    assert "USE_INFERENCE_MODELS" not in source


@pytest.mark.parametrize("version", FORWARDING_VERSIONS)
def test_v5_forwards_what_the_caller_set(version: str) -> None:
    # given
    source = _source(version, tensor=False)

    # then
    assert source.count("mask_decode_mode=mask_decode_mode,") == 4
    assert 'mask_decode_mode="accurate",' not in source


class TestPinningIsObservedAtTheCallSite:
    """Count pinned outbound call sites rather than grepping for a string.

    A version has exactly two places where the request leaves the block:
    `run_locally` and `run_remotely`. Asserting both are pinned catches a
    missed call site, which a substring search does not - the tensor-native
    local path was unpinned while the file still contained the pinned string.

    This is still source analysis. Exercising the block itself would be
    stronger and needs the WorkflowBlock init machinery.
    """

    @pytest.mark.parametrize("version", PINNED_VERSIONS)
    @pytest.mark.parametrize("tensor", [False, True])
    def test_every_outbound_call_site_is_pinned(
        self, version: str, tensor: bool
    ) -> None:
        # given
        # a version has exactly two places where the request leaves the block,
        # run_locally and run_remotely, and both must be pinned in some form:
        # unconditionally in the tensor siblings, conditionally elsewhere
        source = _source(version, tensor)
        outbound = [
            line
            for line in source.split("\n")
            if 'mask_decode_mode="accurate"' in line
            or '"accurate" if USE_INFERENCE_MODELS' in line
        ]

        # then
        assert len(outbound) == 2, (
            f"{version}{'_tensor' if tensor else ''} has {len(outbound)} pinned "
            "outbound call sites, expected 2 (local and remote)"
        )

    def test_v5_non_tensor_has_no_pinned_call_sites(self) -> None:
        # given
        source = _source("v5", tensor=False)

        # then
        assert 'mask_decode_mode="accurate",' not in source
        assert source.count("mask_decode_mode=mask_decode_mode,") == 4


@pytest.mark.parametrize("version", ALL_VERSIONS)
@pytest.mark.parametrize("local_backend", [True, False])
@pytest.mark.parametrize("remote_target", ["hosted", "local"])
def test_remote_request_preserves_versioned_mask_settings(
    monkeypatch, version, local_backend, remote_target
):
    from importlib import import_module
    from unittest.mock import MagicMock

    module = import_module(
        f"roboflow_workflows.core_steps.models.roboflow.instance_segmentation.{version}"
    )
    block_class = getattr(
        module, f"RoboflowInstanceSegmentationModelBlock{version.upper()}"
    )
    block = object.__new__(block_class)
    block._api_key = None
    block._post_process_result = MagicMock(return_value=[])
    client = MagicMock()
    monkeypatch.setattr(module, "InferenceHTTPClient", MagicMock(return_value=client))
    if version in PINNED_VERSIONS:
        monkeypatch.setattr(module, "USE_INFERENCE_MODELS", local_backend)
    monkeypatch.setattr(module, "WORKFLOWS_REMOTE_API_TARGET", remote_target)

    block.run_remotely(
        images=[],
        model_id="model/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.5,
        iou_threshold=0.5,
        max_detections=10,
        max_candidates=20,
        mask_decode_mode="fast",
        tradeoff_factor=0.0,
        disable_active_learning=True,
        active_learning_target_dataset=None,
    )

    config = client.configure.call_args.kwargs["inference_configuration"]
    assert config.mask_decode_mode == ("fast" if version == "v5" else "accurate")
    assert config.tradeoff_factor == (0.0 if version == "v5" else 1.0)
    assert config.response_mask_format == ("rle" if version in {"v4", "v5"} else None)


def test_v5_reduced_rle_output_can_be_decoded_on_original_image(monkeypatch):
    import numpy as np
    import supervision as sv
    from pycocotools import mask as mask_utils
    from roboflow_workflows.core_steps.models.roboflow.instance_segmentation import v5
    from roboflow_workflows.core_steps.visualizations.common.utils import (
        ensure_dense_masks,
    )
    from roboflow_workflows.execution_engine.constants import (
        RLE_MASK_KEY_IN_SV_DETECTIONS,
    )

    block = object.__new__(v5.RoboflowInstanceSegmentationModelBlockV5)
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:10, 8:14] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    prediction = {
        "image": {"height": 200, "width": 300},
        "predictions": [
            {
                "x": 110,
                "y": 75,
                "width": 60,
                "height": 50,
                "confidence": 0.9,
                "class_id": 0,
                "class": "car",
                "rle": rle,
            }
        ],
    }
    monkeypatch.setattr(
        v5,
        "attach_parents_coordinates_to_batch_of_sv_detections",
        lambda images, predictions: predictions,
    )

    output = block._post_process_result(
        images=[], predictions=[prediction], class_filter=None, model_id="model/1"
    )

    detections = output[0]["predictions"]
    encoded = detections.data[RLE_MASK_KEY_IN_SV_DETECTIONS][0]
    decoded = mask_utils.decode(encoded).astype(bool)
    assert decoded.shape == (200, 300)
    np.testing.assert_array_equal(decoded, detections.mask[0])
    assert decoded[50:100, 80:140].all()
    assert decoded.sum() == 50 * 60
    detections.mask = None
    detections = ensure_dense_masks(detections)
    annotated = sv.MaskAnnotator().annotate(
        scene=np.zeros((200, 300, 3), dtype=np.uint8), detections=detections
    )
    assert annotated.any()
    assert rle["size"] == [20, 30]
