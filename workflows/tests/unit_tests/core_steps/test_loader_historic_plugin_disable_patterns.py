"""WORKFLOW_DISABLED_BLOCK_PATTERNS keep matching the Roboflow-platform blocks
under every module path they have shipped from.

Inference 1.6.1-1.7.2 shipped these 9 blocks from the separate
`inference.roboflow_workflows_plugin` package. They are core blocks again, and
that package is no longer importable, but a pattern an operator wrote against
it must still disable the same blocks - as must the pre-1.6.1
`inference.core.workflows.*` spelling and the canonical `roboflow_workflows.*`
one.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest
from roboflow_workflows._compat_names import to_historic_plugin_module
from roboflow_workflows.core_steps import loader

# (canonical module, class name, pre-1.6.1 module, 1.6.1-1.7.2 plugin module)
# - one row per numpy block and per tensor sibling.
_SINKS = "roboflow_workflows.core_steps.sinks.roboflow"
_INTEGRATIONS = "roboflow_workflows.core_steps.integrations.roboflow"
_LEGACY_SINKS = "inference.core.workflows.core_steps.sinks.roboflow"
_LEGACY_INTEGRATIONS = "inference.core.workflows.core_steps.integrations.roboflow"
_PLUGIN_SINKS = "inference.roboflow_workflows_plugin.sinks"
_PLUGIN_INTEGRATIONS = "inference.roboflow_workflows_plugin.integrations"


def _sink(leaf: str, class_name: str) -> Tuple[str, str, str, str]:
    return (
        f"{_SINKS}.{leaf}",
        class_name,
        f"{_LEGACY_SINKS}.{leaf}",
        f"{_PLUGIN_SINKS}.{leaf}",
    )


def _integration(leaf: str, class_name: str) -> Tuple[str, str, str, str]:
    return (
        f"{_INTEGRATIONS}.{leaf}",
        class_name,
        f"{_LEGACY_INTEGRATIONS}.{leaf}",
        f"{_PLUGIN_INTEGRATIONS}.{leaf}",
    )


PLATFORM_BLOCK_MODULES = [
    _sink("dataset_upload.v1", "RoboflowDatasetUploadBlockV1"),
    _sink("dataset_upload.v1_tensor", "RoboflowDatasetUploadBlockV1"),
    _sink("dataset_upload.v2", "RoboflowDatasetUploadBlockV2"),
    _sink("dataset_upload.v2_tensor", "RoboflowDatasetUploadBlockV2"),
    _sink("custom_metadata.v1", "RoboflowCustomMetadataBlockV1"),
    _sink("custom_metadata.v1_tensor", "RoboflowCustomMetadataBlockV1"),
    _sink(
        "model_monitoring_inference_aggregator.v1",
        "ModelMonitoringInferenceAggregatorBlockV1",
    ),
    _sink(
        "model_monitoring_inference_aggregator.v1_tensor",
        "ModelMonitoringInferenceAggregatorBlockV1",
    ),
    _sink("vision_events.v1", "RoboflowVisionEventsBlockV1"),
    _sink("vision_events.v1_tensor", "RoboflowVisionEventsBlockV1"),
    _sink("vision_events_bundle.v1", "VisionEventBundleSinkBlockV1"),
    _sink("vision_events_bundle.v1_tensor", "VisionEventBundleSinkBlockV1"),
    _sink("asset_library_attributes.v1", "RoboflowAssetLibraryAttributesBlockV1"),
    _integration("visual_search.v1", "RoboflowVisualSearchBlockV1"),
    _integration(
        "visual_search_classifier.v1", "RoboflowVisualSearchClassifierBlockV1"
    ),
    _integration(
        "visual_search_classifier.v1_tensor", "RoboflowVisualSearchClassifierBlockV1"
    ),
]
_IDS = [
    row[0][len("roboflow_workflows.core_steps.") :] for row in PLATFORM_BLOCK_MODULES
]


def _block(module_name: str, class_name: str):
    return getattr(importlib.import_module(module_name), class_name)


def _filtered(block_class, patterns: List[str]) -> bool:
    with mock.patch.object(
        loader, "WORKFLOW_DISABLED_BLOCK_TYPES", []
    ), mock.patch.object(loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", patterns):
        return loader._should_filter_block(block_class)


@pytest.mark.parametrize("spelling", ["canonical", "pre_1_6_1", "plugin_1_7_2"])
@pytest.mark.parametrize(
    "canonical,class_name,legacy,plugin", PLATFORM_BLOCK_MODULES, ids=_IDS
)
def test_module_path_pattern_in_every_historic_spelling_disables_the_block(
    canonical: str, class_name: str, legacy: str, plugin: str, spelling: str
) -> None:
    # given
    block_class = _block(canonical, class_name)
    pattern = {"canonical": canonical, "pre_1_6_1": legacy, "plugin_1_7_2": plugin}[
        spelling
    ]

    # when / then
    assert _filtered(block_class, [pattern]) is True


@pytest.mark.parametrize(
    "canonical,class_name,legacy,plugin", PLATFORM_BLOCK_MODULES, ids=_IDS
)
def test_plugin_package_prefix_pattern_disables_every_platform_block(
    canonical: str, class_name: str, legacy: str, plugin: str
) -> None:
    # 1.7.2 matched patterns as case-insensitive substrings of the module path.
    block_class = _block(canonical, class_name)
    assert _filtered(block_class, ["Inference.Roboflow_Workflows_Plugin"]) is True


def test_plugin_sinks_prefix_disables_sinks_but_not_integrations() -> None:
    upload = _block(f"{_SINKS}.dataset_upload.v1", "RoboflowDatasetUploadBlockV1")
    search = _block(f"{_INTEGRATIONS}.visual_search.v1", "RoboflowVisualSearchBlockV1")
    patterns = ["inference.roboflow_workflows_plugin.sinks."]
    assert _filtered(upload, patterns) is True
    assert _filtered(search, patterns) is False


def test_plugin_path_of_another_block_does_not_disable_this_one() -> None:
    upload = _block(f"{_SINKS}.dataset_upload.v1", "RoboflowDatasetUploadBlockV1")
    patterns = [f"{_PLUGIN_SINKS}.custom_metadata"]
    assert _filtered(upload, patterns) is False


def test_plugin_path_pattern_does_not_disable_non_platform_blocks() -> None:
    webhook = _block(
        "roboflow_workflows.core_steps.sinks.webhook.v1", "WebhookSinkBlockV1"
    )
    patterns = ["inference.roboflow_workflows_plugin"]
    assert _filtered(webhook, patterns) is False


@pytest.mark.parametrize(
    "module_name",
    [
        "roboflow_workflows.core_steps.sinks.webhook.v1",
        "roboflow_workflows.core_steps.sinks.roboflow",
        "roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1",
        "inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1",
    ],
)
def test_to_historic_plugin_module_ignores_modules_the_plugin_never_held(
    module_name: str,
) -> None:
    assert to_historic_plugin_module(module_name) is None


CHILD = r"""
import dataclasses
import json
import os
from roboflow_workflows.configuration import (
    configure_process, default_configuration, resolve_image_tensor_device,
)

tensor_mode = os.environ["ENABLE_TENSOR_DATA_REPRESENTATION"] == "True"
base = default_configuration()
configure_process(dataclasses.replace(
    base,
    engine=dataclasses.replace(
        base.engine,
        disabled_block_types=[],
        disabled_block_patterns=json.loads(os.environ["CHILD_PATTERNS"]),
    ),
    tensor=dataclasses.replace(
        base.tensor, representation_enabled=tensor_mode,
        image_tensor_device=resolve_image_tensor_device(tensor_mode),
    ),
))
from roboflow_workflows.core_steps import loader

print(json.dumps({
    "flag": loader.ENABLE_TENSOR_DATA_REPRESENTATION,
    "blocks": sorted(f"{b.__module__}.{b.__name__}" for b in loader.load_blocks()),
}))
"""


_PACKAGE_ROOT = str(
    Path(importlib.import_module("roboflow_workflows").__file__).parents[1]
)


def _load(tensor_mode: bool, patterns: List[str], scratch: Path) -> dict:
    child_env = {**os.environ}
    # The child must import the very `roboflow_workflows` this process
    # tests - nothing else from any sibling checkout.
    child_env["PYTHONPATH"] = _PACKAGE_ROOT
    child_env["ENABLE_TENSOR_DATA_REPRESENTATION"] = "True" if tensor_mode else "False"
    child_env["USE_INFERENCE_MODELS"] = "True"
    child_env["SAM3_3D_OBJECTS_ENABLED"] = "False"
    child_env["CHILD_PATTERNS"] = json.dumps(patterns)
    child_env.pop("WORKFLOWS_PLUGINS", None)
    completed = subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=scratch,
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("tensor_mode", [False, True], ids=["numpy", "tensor"])
def test_load_blocks_drops_blocks_disabled_by_plugin_path_in_both_modes(
    tensor_mode: bool, tmp_path
) -> None:
    # given
    patterns = [
        "inference.roboflow_workflows_plugin.sinks.dataset_upload",
        "inference.roboflow_workflows_plugin.integrations.visual_search_classifier",
    ]

    # when
    baseline = _load(tensor_mode, [], tmp_path)
    filtered = _load(tensor_mode, patterns, tmp_path)

    # then
    assert baseline["flag"] is tensor_mode
    assert filtered["flag"] is tensor_mode
    removed = sorted(set(baseline["blocks"]) - set(filtered["blocks"]))
    suffix = "_tensor" if tensor_mode else ""
    assert removed == [
        f"{_INTEGRATIONS}.visual_search_classifier.v1{suffix}."
        "RoboflowVisualSearchClassifierBlockV1",
        f"{_SINKS}.dataset_upload.v1{suffix}.RoboflowDatasetUploadBlockV1",
        f"{_SINKS}.dataset_upload.v2{suffix}.RoboflowDatasetUploadBlockV2",
    ]
