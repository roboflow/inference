"""Workload declarations of the cache blocks.

The cache restriction is keyed on the RUNTIME only. It must not pick up a step
execution mode (the cache has no remote code path, so where model steps run is
irrelevant) nor an input mode (a still-image request degrades exactly like a
video frame).
"""

from roboflow_workflows.core_steps.cache.cache_get.v1 import (
    BlockManifest as CacheGetManifest,
)
from roboflow_workflows.core_steps.cache.cache_set.v1 import (
    BlockManifest as CacheSetManifest,
)
from roboflow_workflows.core_steps.cache.common import (
    IN_PROCESS_CACHE_HTTP_SOFT_RESTRICTION,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    Severity,
    WorkOperation,
    restriction_metadata_of,
)

from tests.unit_tests.workload_declaration_helpers import portable_restrictions


def _cache_get() -> CacheGetManifest:
    return CacheGetManifest(
        type="roboflow_core/cache_get@v1",
        name="cache_get",
        image="$inputs.image",
        key="counter",
    )


def _cache_set() -> CacheSetManifest:
    return CacheSetManifest(
        type="roboflow_core/cache_set@v1",
        name="cache_set",
        image="$inputs.image",
        key="counter",
        value="$steps.counter.count_in",
    )


def test_cache_blocks_declare_the_direction_of_the_access() -> None:
    assert _cache_get().discover_work_operations() == [WorkOperation.CACHE_READ]
    assert _cache_set().discover_work_operations() == [WorkOperation.CACHE_WRITE]


def test_cache_restriction_is_keyed_on_the_runtime_only() -> None:
    for manifest in (_cache_get(), _cache_set()):
        restrictions = portable_restrictions(manifest)
        assert restrictions == [
            restriction_metadata_of(IN_PROCESS_CACHE_HTTP_SOFT_RESTRICTION)
        ]
        condition = restrictions[0].when
        assert restrictions[0].code == "in_process_cache_not_shared_across_workers"
        assert restrictions[0].severity is Severity.SOFT
        assert set(condition.runtimes) == {
            Runtime.HOSTED_SERVERLESS,
            Runtime.DEDICATED_DEPLOYMENT,
        }
        assert condition.step_execution_modes is None
        assert condition.input_modes is None
        assert condition.configuration_equals == {}


def test_portable_and_legacy_cache_presets_describe_the_same_runtimes() -> None:
    """One preset, two views: the editor's note-carrying entity and the wire
    DTO derived from it. They cannot drift because the DTO is projected, but
    the axes are pinned here anyway."""
    legacy = IN_PROCESS_CACHE_HTTP_SOFT_RESTRICTION
    portable = restriction_metadata_of(legacy)

    assert set(legacy.applies_to_runtimes) == set(portable.when.runtimes)
    assert portable.severity is legacy.severity
    assert legacy.applies_to_step_execution_modes is None
    assert legacy.applies_to_input_modes is None
    assert portable.when.step_execution_modes is None
    assert portable.when.input_modes is None
    # the editor payload is unchanged: it never carried the code
    assert legacy.to_dict() == {
        "severity": "soft",
        "note": legacy.note,
        "applies_to_runtimes": ["hosted_serverless", "dedicated_deployment"],
    }
