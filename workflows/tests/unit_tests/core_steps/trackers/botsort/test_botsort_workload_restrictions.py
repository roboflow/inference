"""BoTSORT legacy (editor) restrictions match its SORT / OC-SORT / ByteTrack siblings.

Both BoTSORT manifests - numpy (``v1``) and tensor-native (``v1_tensor``) - are
checked explicitly, independent of the active tensor mode. Variant modules are
imported inside the tests, so collecting this file does not pull the tensor
trackers into unrelated suites.
"""

import importlib
from typing import Type

import pytest
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeInputMode,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    _get_restrictions,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import BlockSpecification
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    WorkflowBlockManifest,
)

from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions_discovery,
)

TRACKERS_PACKAGE = "roboflow_workflows.core_steps.trackers"
VARIANT_MODULES = ["v1", "v1_tensor"]
SIBLING_TRACKERS = {
    "sort": "SORTManifest",
    "ocsort": "OCSORTManifest",
    "bytetrack": "ByteTrackManifest",
}
STATEFUL_VIDEO_CODE = "stateful_video_state_resets_on_stateless_http"
STILL_IMAGE_CODE = "temporal_block_no_benefit_on_still_image"


def _botsort_manifest_class(variant: str) -> Type[WorkflowBlockManifest]:
    module = importlib.import_module(f"{TRACKERS_PACKAGE}.botsort.{variant}")

    return module.BoTSORTManifest


def _sibling_manifest_class(tracker: str, variant: str) -> Type[WorkflowBlockManifest]:
    module = importlib.import_module(f"{TRACKERS_PACKAGE}.{tracker}.{variant}")

    return getattr(module, SIBLING_TRACKERS[tracker])


def _editor_projection(variant: str) -> list:
    module = importlib.import_module(f"{TRACKERS_PACKAGE}.botsort.{variant}")
    block = BlockSpecification(
        block_source="workflows_core",
        identifier=f"{module.__name__}.BoTSORTBlockV1",
        block_class=module.BoTSORTBlockV1,
        manifest_class=module.BoTSORTManifest,
    )

    return _get_restrictions(block)


@pytest.mark.parametrize("variant", VARIANT_MODULES)
def test_botsort_legacy_restrictions_are_the_shared_stateful_video_pair(
    variant: str,
) -> None:
    # when
    legacy = _botsort_manifest_class(variant).get_restrictions()

    # then
    assert len(legacy) == 2
    assert legacy[0] is STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION
    assert legacy[1] is STILL_IMAGE_INPUT_SOFT_RESTRICTION
    assert [restriction.code for restriction in legacy] == [
        STATEFUL_VIDEO_CODE,
        STILL_IMAGE_CODE,
    ]


@pytest.mark.parametrize("variant", VARIANT_MODULES)
@pytest.mark.parametrize("tracker", sorted(SIBLING_TRACKERS))
def test_botsort_legacy_restrictions_match_sibling_tracker(
    variant: str,
    tracker: str,
) -> None:
    # when
    botsort_legacy = _botsort_manifest_class(variant).get_restrictions()
    sibling_legacy = _sibling_manifest_class(tracker, variant).get_restrictions()

    # then
    assert botsort_legacy == sibling_legacy


@pytest.mark.parametrize("variant", VARIANT_MODULES)
def test_botsort_editor_projection_carries_state_loss_and_still_image_caveats(
    variant: str,
) -> None:
    # when
    projection = _editor_projection(variant)

    # then
    assert projection == [
        {
            "severity": "soft",
            "note": STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.note,
            "applies_to_runtimes": ["hosted_serverless", "dedicated_deployment"],
            "applies_to_step_execution_modes": ["remote"],
            "applies_to_input_modes": ["video"],
        },
        {
            "severity": "soft",
            "note": STILL_IMAGE_INPUT_SOFT_RESTRICTION.note,
            "applies_to_input_modes": ["image"],
        },
    ]


@pytest.mark.parametrize("variant", VARIANT_MODULES)
def test_botsort_actual_restrictions_are_unchanged(variant: str) -> None:
    # given
    manifest = _botsort_manifest_class(variant).model_construct(
        type="roboflow_core/trackers_botsort@v1",
        name="tracker",
    )

    # when
    actual = declared_restrictions(manifest)
    discovery = portable_restrictions_discovery(manifest)

    # then
    assert actual == list(STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS)
    assert discovery.complete is True
    assert [item.code for item in discovery.items] == [
        STATEFUL_VIDEO_CODE,
        STILL_IMAGE_CODE,
    ]
    state_loss = actual[0]
    assert state_loss.applies_to_step_execution_modes is None
    assert set(state_loss.applies_to_runtimes) == {
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    }
    assert state_loss.applies_to_input_modes == [RuntimeInputMode.VIDEO]
