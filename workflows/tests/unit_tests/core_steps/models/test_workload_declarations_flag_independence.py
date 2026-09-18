"""The portable restrictions must not depend on this host's flag values.

`get_restrictions()` filters its declarations against the environment flags of
the machine it runs on - that behaviour is legacy and stays exactly as it was.
`discover_portable_restrictions()` is the opposite: it declares every branch
unconditionally and carries the flag value each branch applies to inside
`when.configuration_equals`, so the answer is identical on a host with the flag
on and on a host with the flag off.
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
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED"),
    ]

    # when - flag ON
    monkeypatch.setattr(module, "MOONDREAM2_ENABLED", True)
    portable_with_flag_on = manifest.discover_portable_restrictions()
    legacy_with_flag_on = manifest_class.get_restrictions()

    # and - flag OFF
    monkeypatch.setattr(module, "MOONDREAM2_ENABLED", False)
    portable_with_flag_off = manifest.discover_portable_restrictions()
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
    portable_with_flag_on = manifest.discover_portable_restrictions()
    legacy_with_flag_on = LMMV1Manifest.get_restrictions()
    monkeypatch.setattr(LMM_MODULE, "LMM_ENABLED", False)
    portable_with_flag_off = manifest.discover_portable_restrictions()
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
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        hosted_endpoint_disabled_by_flag("CORE_MODEL_SAM3_ENABLED"),
    ]

    # when
    monkeypatch.setattr(SAM3_MODULE, "CORE_MODEL_SAM3_ENABLED", True)
    portable_with_flag_on = manifest.discover_portable_restrictions()
    monkeypatch.setattr(SAM3_MODULE, "CORE_MODEL_SAM3_ENABLED", False)
    portable_with_flag_off = manifest.discover_portable_restrictions()

    # then
    assert portable_with_flag_on == expected
    assert portable_with_flag_off == expected


def test_flag_condition_carries_the_false_value_it_applies_to() -> None:
    # given
    restriction = hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED")

    # then - the consumer decides, using ITS runtime configuration
    assert restriction.when.configuration_equals == {"MOONDREAM2_ENABLED": False}
    assert restriction.code == "hosted_endpoint_disabled_by_flag"


def test_declarations_are_repeatable() -> None:
    # given
    manifest = _manifest(Moondream2V1Manifest)

    # then - no accumulation, no mutation of a shared list
    first_operations: List[Any] = manifest.discover_work_operations()
    second_operations: List[Any] = manifest.discover_work_operations()
    assert first_operations == second_operations
    first_operations.append("mutated")
    assert manifest.discover_work_operations() == second_operations

    first_restrictions = manifest.discover_portable_restrictions()
    second_restrictions = manifest.discover_portable_restrictions()
    assert first_restrictions == second_restrictions
    first_restrictions.clear()
    assert manifest.discover_portable_restrictions() == second_restrictions
