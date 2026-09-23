"""The portable view must not depend on this host's flag values.

`get_restrictions()` filters its declarations against the environment flags of
the machine it runs on - that behaviour is legacy and stays exactly as it was.
`get_actual_restrictions(ignore_environment_restrictions=True)` is the
opposite: it declares every branch unconditionally and carries the flag value
each branch applies to inside `applies_to_configuration` (projected onto
`when.configuration_equals` on the wire), so the answer is identical on a host
with the flag on and on a host with the flag off.

The declared items come back in the canonical discovery order - sorted by
`(code, severity, ...)`, not in the order the block happened to list them.
"""

from typing import Any, List, Type

import pytest
from roboflow_workflows.core_steps.models.foundation import lmm as lmm_package
from roboflow_workflows.core_steps.models.foundation import (
    moondream2 as moondream2_package,
)
from roboflow_workflows.core_steps.models.foundation import (
    segment_anything3 as sam3_package,
)
from roboflow_workflows.core_steps.models.foundation.lmm.v1 import (
    BlockManifest as LMMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.moondream2.v1 import (
    BlockManifest as Moondream2V1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.moondream2.v1_tensor import (
    BlockManifest as Moondream2V1TensorManifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3.v1 import (
    BlockManifest as SAM3V1Manifest,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    hosted_endpoint_disabled_by_flag,
)
from roboflow_workflows.execution_engine.entities.workload import (
    restriction_metadata_of,
)

from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions,
)

MOONDREAM2_MODULE = moondream2_package.v1
MOONDREAM2_TENSOR_MODULE = moondream2_package.v1_tensor
LMM_MODULE = lmm_package.v1
SAM3_MODULE = sam3_package.v1


def _manifest(manifest_class: Type, **kwargs: Any):
    payload = {"name": "step", "images": "$inputs.image"}
    payload.update(kwargs)
    literal = manifest_class.model_fields["type"].annotation.__args__[0]
    return manifest_class(type=literal, **payload)


@pytest.mark.parametrize(
    "module, manifest_class",
    [
        (MOONDREAM2_MODULE, Moondream2V1Manifest),
        (MOONDREAM2_TENSOR_MODULE, Moondream2V1TensorManifest),
    ],
)
def test_moondream2_portable_restrictions_ignore_the_host_flag(
    module: Any, manifest_class: Type, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    manifest = _manifest(manifest_class)
    expected = [
        hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED"),
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    ]

    # when - flag ON
    monkeypatch.setattr(module, "MOONDREAM2_ENABLED", True)
    portable_with_flag_on = declared_restrictions(manifest)
    legacy_with_flag_on = manifest_class.get_restrictions()

    # and - flag OFF
    monkeypatch.setattr(module, "MOONDREAM2_ENABLED", False)
    portable_with_flag_off = declared_restrictions(manifest)
    legacy_with_flag_off = manifest_class.get_restrictions()

    # then - the portable answer never moves ...
    assert portable_with_flag_on == expected
    assert portable_with_flag_off == expected
    # ... while the legacy answer still changes exactly as it did before
    assert len(legacy_with_flag_on) == 1
    assert len(legacy_with_flag_off) == 2
    assert "MOONDREAM2_ENABLED=False" in legacy_with_flag_off[1].note


def test_lmm_portable_restrictions_ignore_the_host_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - legacy list is EMPTY when the flag is on
    manifest = _manifest(LMMV1Manifest, prompt="describe", lmm_type="gpt_4v")
    expected = [hosted_endpoint_disabled_by_flag("LMM_ENABLED")]

    # when
    monkeypatch.setattr(LMM_MODULE, "LMM_ENABLED", True)
    portable_with_flag_on = declared_restrictions(manifest)
    legacy_with_flag_on = LMMV1Manifest.get_restrictions()
    monkeypatch.setattr(LMM_MODULE, "LMM_ENABLED", False)
    portable_with_flag_off = declared_restrictions(manifest)
    legacy_with_flag_off = LMMV1Manifest.get_restrictions()

    # then
    assert portable_with_flag_on == expected
    assert portable_with_flag_off == expected
    assert legacy_with_flag_on == []
    assert len(legacy_with_flag_off) == 1


def test_sam3_portable_restrictions_ignore_the_host_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    manifest = _manifest(SAM3V1Manifest)
    expected = [
        hosted_endpoint_disabled_by_flag("CORE_MODEL_SAM3_ENABLED"),
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    ]

    # when
    monkeypatch.setattr(SAM3_MODULE, "CORE_MODEL_SAM3_ENABLED", True)
    portable_with_flag_on = declared_restrictions(manifest)
    monkeypatch.setattr(SAM3_MODULE, "CORE_MODEL_SAM3_ENABLED", False)
    portable_with_flag_off = declared_restrictions(manifest)

    # then
    assert portable_with_flag_on == expected
    assert portable_with_flag_off == expected


def test_flag_condition_carries_the_false_value_it_applies_to() -> None:
    # given
    restriction = hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED")

    # then - the consumer decides, using ITS runtime configuration
    assert restriction.applies_to_configuration == {"MOONDREAM2_ENABLED": False}
    assert restriction.code == "hosted_endpoint_disabled_by_flag"
    # ... and the same value reaches the wire
    assert restriction_metadata_of(restriction).when.configuration_equals == {
        "MOONDREAM2_ENABLED": False
    }


def test_the_flag_branch_reaches_the_wire_through_the_public_hook() -> None:
    manifest = _manifest(Moondream2V1Manifest)

    published = portable_restrictions(manifest)

    assert [item.code for item in published] == [
        "hosted_endpoint_disabled_by_flag",
        "requires_gpu_for_local_execution",
    ]
    assert published[0].when.configuration_equals == {"MOONDREAM2_ENABLED": False}


def test_declarations_are_repeatable() -> None:
    # given
    manifest = _manifest(Moondream2V1Manifest)

    # then - no accumulation, no mutation of a shared list
    first_operations: List[Any] = manifest.discover_work_operations()
    second_operations: List[Any] = manifest.discover_work_operations()
    assert first_operations == second_operations
    first_operations.append("mutated")
    assert manifest.discover_work_operations() == second_operations

    first_restrictions = declared_restrictions(manifest)
    second_restrictions = declared_restrictions(manifest)
    assert first_restrictions == second_restrictions
    first_restrictions.clear()
    assert declared_restrictions(manifest) == second_restrictions
