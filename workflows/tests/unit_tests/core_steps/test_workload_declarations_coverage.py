"""Registry-wide census of the three declaration hooks.

This module is the coverage gate for the whole block library: it enumerates the
REAL registry (core blocks plus the enterprise plugin) instead of a hand-kept
list, so a block that is added without declarations, or a declaration that is
deleted, fails here rather than silently degrading a workload description into
"unknown".

Three things are checked, in increasing depth:

1. **Coverage** - every loaded manifest class DEFINES all three hooks itself
   (``hook in vars(manifest_class)``, not inherited from
   ``WorkflowBlockManifest``, whose defaults mean "unknown"). The third hook,
   ``discover_dependent_resources()``, has a NAMED exception list: a block whose
   resources are genuinely unknowable keeps the ``None`` default and must be
   registered in ``DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN`` with its reason.
2. **Numpy / tensor parity** - a block implemented twice (``vN.py`` and
   ``vN_tensor.py``) must declare the same work in both files. The check is
   static (``ast``), so it needs no second process and no tensor-mode switch.
3. **Truthfulness of the declarations themselves** - the hooks return values the
   ``Discovery`` convention accepts, the results do not depend on this host's
   environment flags, and no single operation is pasted onto every block.
"""

import ast
import collections
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionMetadata,
    WorkOperation,
    normalize_declaration,
)
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
    load_workflow_blocks,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest

WORK_OPERATIONS_HOOK = "discover_work_operations"
PORTABLE_RESTRICTIONS_HOOK = "discover_portable_restrictions"
DEPENDENT_RESOURCES_HOOK = "discover_dependent_resources"
# the two workload hooks - they share the Discovery return convention
HOOKS = (WORK_OPERATIONS_HOOK, PORTABLE_RESTRICTIONS_HOOK)
# everything a block declares about itself; all three must stay in step between
# the numpy and the tensor implementation
DECLARATION_HOOKS = HOOKS + (DEPENDENT_RESOURCES_HOOK,)
ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
TENSOR_SUFFIX = "_tensor"

# Blocks that deliberately keep the ``None`` (unknown) default for
# ``discover_dependent_resources()``. "Unknown" is not the same as "none", and
# declaring an empty list here would tell a caller that no model is involved.
# Every entry needs a reason, and the entry is rejected once the block starts
# declaring - this list cannot quietly rot into a suppression list.
DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN = {
    "roboflow_core/inner_workflow@v1": (
        "the dispatched child workflow may pull any model or project; the "
        "block itself pulls none, but claiming a known-empty set would hide "
        "the child's resources from a caller reading the manifest directly"
    ),
}


class _LoadedBlock:
    """The three facts the census needs about one registered block."""

    def __init__(self, block_type: str, manifest_class: type, source_file: Path):
        self.block_type = block_type
        self.manifest_class = manifest_class
        self.source_file = source_file

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<_LoadedBlock {self.block_type} {self.manifest_class.__name__}>"


@pytest.fixture(scope="module")
def loaded_blocks() -> List[_LoadedBlock]:
    """The real registry: core blocks plus the enterprise plugin.

    The plugin list is read from the environment at call time, so the
    environment is patched and every loader cache is cleared on the way in AND
    on the way out - a cached core-only registry must not leak into other
    modules.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
        blocks_loader.clear_caches()
        try:
            blocks = []
            for block in load_workflow_blocks():
                manifest_class = block.manifest_class
                identifiers = get_manifest_type_identifiers(
                    block_schema=manifest_class.model_json_schema(),
                    block_source=block.block_source,
                    block_identifier=block.identifier,
                )
                blocks.append(
                    _LoadedBlock(
                        block_type=identifiers[0],
                        manifest_class=manifest_class,
                        source_file=Path(inspect.getfile(manifest_class)).resolve(),
                    )
                )
            yield blocks
        finally:
            blocks_loader.clear_caches()


def test_registry_is_not_empty_and_includes_the_enterprise_plugin(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    # the census is only meaningful if the fixture really loaded a registry
    assert len(loaded_blocks) > 100, "block registry looks truncated"
    enterprise_blocks = [
        block
        for block in loaded_blocks
        if "enterprise_blocks" in block.source_file.as_posix()
    ]
    assert enterprise_blocks, "enterprise plugin was not loaded by the fixture"


def test_every_registered_manifest_declares_both_hooks(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """No hardcoded block count: the expectation IS the loaded registry."""
    missing: Dict[str, List[str]] = {hook: [] for hook in HOOKS}
    for block in loaded_blocks:
        for hook in HOOKS:
            # the base class defines both hooks returning None ("unknown"), so
            # inheritance is exactly what must NOT count as a declaration
            assert hasattr(WorkflowBlockManifest, hook)
            if hook not in vars(block.manifest_class):
                missing[hook].append(
                    f"{block.block_type} "
                    f"({block.manifest_class.__module__}.{block.manifest_class.__name__})"
                )
    report_lines = []
    for hook in HOOKS:
        if missing[hook]:
            report_lines.append(
                f"{hook}: {len(missing[hook])} of {len(loaded_blocks)} manifests "
                f"do not declare it:\n  " + "\n  ".join(sorted(missing[hook]))
            )
    assert not report_lines, "\n".join(report_lines)


def test_every_registered_manifest_declares_its_dependent_resources(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """``[]`` = "provably pulls no model, project or third-party model".

    A block that cannot know keeps the ``None`` default and is registered in
    ``DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN``; anything else is a gap.
    """
    missing = [
        f"{block.block_type} "
        f"({block.manifest_class.__module__}.{block.manifest_class.__name__})"
        for block in loaded_blocks
        if DEPENDENT_RESOURCES_HOOK not in vars(block.manifest_class)
        and block.block_type not in DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN
    ]
    assert not missing, (
        f"{len(missing)} of {len(loaded_blocks)} manifests neither declare "
        f"{DEPENDENT_RESOURCES_HOOK} nor are registered as intentionally "
        "unknown:\n  " + "\n  ".join(sorted(missing))
    )


def test_the_intentionally_unknown_list_cannot_rot(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """The exception list must stay an exception list, not a suppression list."""
    by_type = {block.block_type: block for block in loaded_blocks}
    stale = [
        block_type
        for block_type in DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN
        if block_type not in by_type
    ]
    assert not stale, f"no longer registered blocks: {sorted(stale)}"
    now_declaring = [
        block_type
        for block_type in DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN
        if DEPENDENT_RESOURCES_HOOK in vars(by_type[block_type].manifest_class)
    ]
    assert not now_declaring, (
        "these blocks now declare their resources and must be removed from "
        f"DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN: {sorted(now_declaring)}"
    )
    assert all(
        reason.strip() for reason in DEPENDENT_RESOURCES_INTENTIONALLY_UNKNOWN.values()
    ), "every exception needs a reason"


def test_an_empty_resource_declaration_is_inert_for_the_executable_path(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """``[]`` must behave exactly like ``None`` for the pre-load path.

    ``deduce_blocks_dependencies`` extends only truthy declarations, so turning
    an undeclared block into one that declares ``[]`` changes no runtime
    behaviour. This test pins that, because the whole T32 pass rests on it.
    """
    from roboflow_workflows.execution_engine.v1.compiler.utils import (
        deduce_blocks_dependencies,
    )

    declaring_empty = []
    for block in loaded_blocks:
        if DEPENDENT_RESOURCES_HOOK not in vars(block.manifest_class):
            continue
        try:
            declared = _manifest_instance(block).discover_dependent_resources()
        except Exception:  # noqa: BLE001
            # a model block whose resource identity comes from a field this
            # census leaves unset - not what this test is about
            continue
        if declared == []:
            declaring_empty.append(block)
    assert len(declaring_empty) > 50, len(declaring_empty)

    class _CompiledWorkflowStub:
        def __init__(self, steps):
            self.workflow_definition = type("_Definition", (), {"steps": steps})()

    steps = [_manifest_instance(block) for block in declaring_empty[:25]]
    assert deduce_blocks_dependencies(_CompiledWorkflowStub(steps)) == []


def test_no_block_falls_back_to_the_unknown_default(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """A declared hook must not simply re-return the base class default."""
    inherited = [
        block.block_type
        for block in loaded_blocks
        for hook in DECLARATION_HOOKS
        if vars(block.manifest_class).get(hook) is getattr(WorkflowBlockManifest, hook)
    ]
    assert not inherited, sorted(inherited)


def _normalised_hook_sources(path: Path) -> Dict[Tuple[str, str], str]:
    """``{(class_name, hook_name): whitespace-normalised source}`` for a file."""
    tree = ast.parse(path.read_text())
    sources: Dict[Tuple[str, str], str] = {}
    file_lines = path.read_text()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name in DECLARATION_HOOKS:
                segment = ast.get_source_segment(file_lines, member)
                sources[(node.name, member.name)] = re.sub(r"\s+", " ", segment).strip()
    return sources


def _imported_names(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text())
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _tensor_sibling(path: Path) -> Optional[Path]:
    if path.stem.endswith(TENSOR_SUFFIX):
        sibling = path.with_name(f"{path.stem[: -len(TENSOR_SUFFIX)]}.py")
    else:
        sibling = path.with_name(f"{path.stem}{TENSOR_SUFFIX}.py")
    return sibling if sibling.exists() else None


def test_numpy_and_tensor_implementations_declare_identical_hooks(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """A block implemented twice must not drift between the representations.

    Covers all three declaration hooks. Static check: whitespace is normalised,
    nothing else is. When the sibling does not define its own manifest class it
    must import it (then both representations share one class object and parity
    is structural).
    """
    checked = 0
    problems: List[str] = []
    for block in loaded_blocks:
        sibling = _tensor_sibling(block.source_file)
        if sibling is None:
            continue
        class_name = block.manifest_class.__name__
        own = _normalised_hook_sources(block.source_file)
        other = _normalised_hook_sources(sibling)
        if not any(key[0] == class_name for key in other):
            if class_name not in _imported_names(sibling):
                problems.append(
                    f"{block.block_type}: {sibling.name} neither defines nor "
                    f"imports {class_name}"
                )
            continue
        checked += 1
        for hook in DECLARATION_HOOKS:
            own_source = own.get((class_name, hook))
            other_source = other.get((class_name, hook))
            if own_source != other_source:
                problems.append(
                    f"{block.block_type}: {hook} differs between "
                    f"{block.source_file.name} and {sibling.name}\n"
                    f"    {block.source_file.name}: {own_source}\n"
                    f"    {sibling.name}: {other_source}"
                )
    assert not problems, "\n".join(problems)
    assert checked > 50, f"parity check covered only {checked} block pairs"


def _manifest_instance(block: _LoadedBlock) -> Any:
    """A field-complete-enough manifest instance to call the hooks on.

    ``model_construct`` skips validation on purpose - the census must not depend
    on inventing a valid value for every field of every block. Fields with
    defaults keep them; required fields are present as ``None`` so a hook that
    branches on a literal setting takes its fallback branch instead of raising
    ``AttributeError``. Exact per-setting declarations are pinned by the
    per-category tests, which build real, validated manifests.
    """
    values: Dict[str, Any] = {"type": block.block_type, "name": "workload_census_step"}
    for field_name, field in block.manifest_class.model_fields.items():
        if field_name in values:
            continue
        if field.is_required():
            values[field_name] = None
    return block.manifest_class.model_construct(**values)


def test_every_hook_returns_a_valid_declaration(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    failures: List[str] = []
    for block in loaded_blocks:
        manifest = _manifest_instance(block)
        try:
            operations = normalize_declaration(
                manifest.discover_work_operations(),
                f"work_operations_unknown:{block.block_type}",
            )
            restrictions = normalize_declaration(
                manifest.discover_portable_restrictions(),
                f"portable_restrictions_unknown:{block.block_type}",
            )
        except Exception as error:  # noqa: BLE001 - the message IS the report
            failures.append(f"{block.block_type}: {type(error).__name__}: {error}")
            continue
        for item in operations.items:
            if not isinstance(item, WorkOperation):
                failures.append(f"{block.block_type}: {item!r} is not a WorkOperation")
        for item in restrictions.items:
            if not isinstance(item, RestrictionMetadata):
                failures.append(
                    f"{block.block_type}: {item!r} is not a RestrictionMetadata"
                )
        if not operations.complete and not operations.unknown_reasons:
            failures.append(f"{block.block_type}: incomplete operations, no reason")
    assert not failures, "\n".join(failures)


def test_declared_operations_are_not_a_blanket_label(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """Every block returning the same set would satisfy coverage but say nothing."""
    declared_sets: Set[Tuple[str, ...]] = set()
    counts: Dict[WorkOperation, int] = {}
    for block in loaded_blocks:
        manifest = _manifest_instance(block)
        discovery = normalize_declaration(
            manifest.discover_work_operations(),
            f"work_operations_unknown:{block.block_type}",
        )
        declared_sets.add(tuple(sorted(item.value for item in discovery.items)))
        for item in set(discovery.items):
            counts[item] = counts.get(item, 0) + 1
    assert len(declared_sets) >= 20, sorted(declared_sets)
    total = len(loaded_blocks)
    pasted_everywhere = [
        operation.value for operation, count in counts.items() if count == total
    ]
    assert not pasted_everywhere, pasted_everywhere
    # and the vocabulary actually used is broad, not three labels reused
    assert len(counts) >= 15, sorted(operation.value for operation in counts)


FLAG_SENSITIVE_MODULES = (
    "roboflow_workflows.core_steps.sinks.local_file.v1",
    "roboflow_workflows.core_steps.secrets_providers.environment_secrets_store.v1",
    "roboflow_workflows.core_steps.models.roboflow.instance_segmentation.v1",
    "roboflow_workflows.core_steps.models.foundation.florence2.v1",
)

FLAG_NAMES = (
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES",
    "LMM_ENABLED",
    "FLORENCE2_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "DEPTH_ESTIMATION_ENABLED",
    "WORKFLOWS_STEP_EXECUTION_MODE",
    "API_KEY",
)


def _portable_restrictions_by_block(
    blocks: List[_LoadedBlock],
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        manifest = _manifest_instance(block)
        discovery = normalize_declaration(
            manifest.discover_portable_restrictions(),
            f"portable_restrictions_unknown:{block.block_type}",
        )
        result[block.block_type] = discovery.model_dump(mode="json")["items"]
    return result


def test_portable_restrictions_do_not_depend_on_this_host_flags(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """Flag-conditional declarations are conditions, not evaluated branches.

    ``get_restrictions()`` filters on the host's flags. The portable hook must
    not: it reports every applicable declaration with the flag pinned in
    ``configuration_equals``, so flipping the constants this process imported
    cannot change the answer.
    """
    before = _portable_restrictions_by_block(loaded_blocks)
    # the flag-sensitive blocks must actually be present, otherwise this test
    # would pass vacuously
    loaded_modules = {block.manifest_class.__module__ for block in loaded_blocks}
    covered = [module for module in FLAG_SENSITIVE_MODULES if module in loaded_modules]
    assert covered, FLAG_SENSITIVE_MODULES
    with pytest.MonkeyPatch.context() as monkeypatch:
        import roboflow_workflows.environment as workflows_environment

        modules = [workflows_environment] + [
            __import__(module, fromlist=["_"])
            for module in FLAG_SENSITIVE_MODULES
            if module in loaded_modules
        ]
        flipped = 0
        for module in modules:
            for flag in FLAG_NAMES:
                if not hasattr(module, flag):
                    continue
                current = getattr(module, flag)
                if isinstance(current, bool):
                    monkeypatch.setattr(module, flag, not current, raising=False)
                    flipped += 1
        assert flipped, "no boolean flag constant was flipped - test is vacuous"
        after = _portable_restrictions_by_block(loaded_blocks)
    differing = {
        block_type
        for block_type, restrictions in after.items()
        if restrictions != before[block_type]
    }
    assert not differing, sorted(differing)


def test_flag_conditional_declarations_pin_the_flag_in_the_condition(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """The local-file and environment-secrets blocks carry BOTH branches."""
    by_type = {block.block_type: block for block in loaded_blocks}
    local_file = by_type["roboflow_core/local_file_sink@v1"]
    restrictions = _manifest_instance(local_file).discover_portable_restrictions()
    by_code = {restriction.code: restriction for restriction in restrictions}
    assert set(by_code) == {
        "local_storage_access_disabled",
        "writes_to_deployment_volume_not_retrievable",
        "ephemeral_container_disk_loses_writes",
    }, sorted(by_code)
    assert by_code["local_storage_access_disabled"].when.configuration_equals == {
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": False
    }
    for enabled_branch_code in (
        "writes_to_deployment_volume_not_retrievable",
        "ephemeral_container_disk_loses_writes",
    ):
        assert by_code[enabled_branch_code].when.configuration_equals == {
            "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True
        }

    secrets = by_type["roboflow_core/environment_secrets_store@v1"]
    secret_restrictions = _manifest_instance(secrets).discover_portable_restrictions()
    assert [restriction.code for restriction in secret_restrictions] == [
        "environment_variable_access_disabled"
    ]
    assert secret_restrictions[0].when.configuration_equals == {
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": False
    }


def _restriction_axes(restriction: RestrictionMetadata) -> Tuple[Any, ...]:
    condition = restriction.when
    return (
        restriction.severity.value,
        tuple(item.value for item in condition.runtimes or ()),
        tuple(item.value for item in condition.step_execution_modes or ()),
        tuple(item.value for item in condition.input_modes or ()),
    )


def test_restriction_codes_are_authored_not_slugified(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """One code, one meaning.

    A code is reused only where the semantics are identical, so severity and the
    three condition axes must be the same everywhere the code appears.
    ``configuration_equals`` is deliberately NOT part of that identity: one code
    is parameterised by the flag it names (``hosted_endpoint_disabled_by_flag``
    is the same failure mode whichever ``*_ENABLED`` flag switches it off), and
    the two branches of a flag are distinct codes, not one code with two values.
    """
    axes_by_code: Dict[str, Set[Tuple[Any, ...]]] = {}
    configuration_keys_by_code: Dict[str, Set[Tuple[str, ...]]] = {}
    for block in loaded_blocks:
        manifest = _manifest_instance(block)
        discovery = normalize_declaration(
            manifest.discover_portable_restrictions(),
            f"portable_restrictions_unknown:{block.block_type}",
        )
        for restriction in discovery.items:
            axes_by_code.setdefault(restriction.code, set()).add(
                _restriction_axes(restriction)
            )
            configuration_keys_by_code.setdefault(restriction.code, set()).add(
                tuple(sorted(restriction.when.configuration_equals))
            )
    ambiguous = {
        code: sorted(axes) for code, axes in axes_by_code.items() if len(axes) > 1
    }
    assert not ambiguous, ambiguous
    # a code that is parameterised by a flag still pins exactly one flag per
    # declaration - never a mixed, partly-conditional meaning
    mixed_arity = {
        code: sorted(keys)
        for code, keys in configuration_keys_by_code.items()
        if len({len(entry) for entry in keys}) > 1
    }
    assert not mixed_arity, mixed_arity


def test_inner_workflow_reports_its_child_as_opaque(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    by_type = {block.block_type: block for block in loaded_blocks}
    manifest = _manifest_instance(by_type["roboflow_core/inner_workflow@v1"])
    operations = manifest.discover_work_operations()
    assert isinstance(operations, Discovery)
    assert operations.complete is False
    assert operations.items == [WorkOperation.EXTERNAL_REQUEST]
    assert operations.unknown_reasons == [
        "remote_dispatch_child_opaque:$steps.workload_census_step"
    ]
    restrictions = manifest.discover_portable_restrictions()
    assert isinstance(restrictions, Discovery)
    assert restrictions.complete is False
    assert restrictions.items == []
    assert restrictions.unknown_reasons == [
        "remote_dispatch_child_opaque:$steps.workload_census_step"
    ]


# ---------------------------------------------------------------------------
# Legacy / portable consistency (deep-review finding F4)
#
# `get_restrictions()` answers "what applies on THIS host", evaluating the flag
# constants its module imported. `discover_portable_restrictions()` answers
# "what applies on a target deployment", carrying the flag in the condition
# instead of evaluating it. The two must agree once the portable conditions are
# evaluated against the same flag values the legacy method sees: a migration
# that quietly drops or invents a caveat is a regression, not a refactor.
#
# Divergences are allowed only where they were recorded as judgment calls, and
# only by name. The registry below is kept tight by
# `test_the_legacy_divergence_registry_cannot_rot`.
# ---------------------------------------------------------------------------

# Portable entries with no legacy counterpart, keyed by (block type, code).
PORTABLE_WITHOUT_LEGACY = {
    (
        "roboflow_core/trackers_botsort@v1",
        "stateful_video_state_resets_on_stateless_http",
    ): (
        "BoT-SORT keeps per-video tracker state like its sibling trackers but "
        "never declared the caveat in get_restrictions(); the portable "
        "declaration fixes the legacy gap (DECISIONS D021)"
    ),
    (
        "roboflow_core/trackers_botsort@v1",
        "temporal_block_no_benefit_on_still_image",
    ): "second half of the same recorded addition (DECISIONS D021)",
    ("roboflow_core/cog_vlm@v1", "deprecated_block_always_raises"): (
        "run() raises FeatureDeprecatedError unconditionally; the legacy API "
        "has no way to say that, the portable one does (DECISIONS D020)"
    ),
    ("roboflow_core/gaze@v1", "deprecated_block_always_raises"): (
        "run() raises FeatureDeprecatedError unconditionally (DECISIONS D020)"
    ),
    ("roboflow_core/yolo_world_model@v1", "unsupported_in_tensor_representation"): (
        "the tensor sibling raises FeatureDeprecatedError; declared in both "
        "files and conditioned on ENABLE_TENSOR_DATA_REPRESENTATION, so it is "
        "inactive in numpy mode and active in tensor mode (DECISIONS D020)"
    ),
}

# Legacy entries with no active portable counterpart, keyed by
# (block type, (severity, runtimes, step_execution_modes, input_modes)).
LEGACY_WITHOUT_PORTABLE = {
    (
        "roboflow_core/s3_sink@v1",
        ("soft", ("dedicated_deployment", "hosted_serverless"), ("remote",), ()),
    ): (
        "get_restrictions() is a CLASSMETHOD and cannot see output_mode, so it "
        "declares the append-log caveat unconditionally; after Codex F003 the "
        "portable hook emits it only for output_mode='append_log'. This census "
        "builds a neutral instance (required fields None), so the two "
        "legitimately disagree here. The real per-mode behaviour is pinned by "
        "test_s3_append_caveat_follows_the_literal_output_mode in "
        "tests/unit_tests/core_steps/sinks/test_workload_declarations.py"
    ),
    (
        "roboflow_core/local_file_sink@v1",
        ("soft", ("dedicated_deployment",), (), ()),
    ): (
        "legacy emits 'files land on the deployment volume but are not "
        "retrievable' on BOTH branches of the storage flag; the portable "
        "writes_to_deployment_volume_not_retrievable pins the enabled branch, "
        "because with local storage disabled the block raises before writing "
        "anything (coordinator restriction registry)"
    ),
}

# Same divergence in the host plugin. That registry is not loadable from the
# workflows package (no `inference.*` imports here), so it cannot be checked in
# this module; `tests/inference/unit_tests/core/test_roboflow_plugin_workload_declarations.py`
# owns it. Listed so the decision is discoverable from one place.
HOST_PLUGIN_DIVERGENCES_CHECKED_ELSEWHERE = {
    (
        # NOTE: the package directory is `vision_events_bundle` (plural) but the
        # block's `type` literal is singular - do not "correct" it back.
        "roboflow_core/vision_event_bundle@v1",
        "writes_to_deployment_volume_not_retrievable",
    ): "same storage-flag reading as local_file_sink@v1",
}

FLAG_CONSTANTS_UNDER_TEST = ("ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",)


def _legacy_axes(restriction: Any) -> Tuple[Any, ...]:
    return (
        restriction.severity.value,
        tuple(sorted(item.value for item in (restriction.applies_to_runtimes or ()))),
        tuple(
            sorted(
                item.value
                for item in (restriction.applies_to_step_execution_modes or ())
            )
        ),
        tuple(
            sorted(item.value for item in (restriction.applies_to_input_modes or ()))
        ),
    )


def _portable_axes(restriction: RestrictionMetadata) -> Tuple[Any, ...]:
    condition = restriction.when
    return (
        restriction.severity.value,
        tuple(sorted(item.value for item in (condition.runtimes or ()))),
        tuple(sorted(item.value for item in (condition.step_execution_modes or ()))),
        tuple(sorted(item.value for item in (condition.input_modes or ()))),
    )


def _flag_value(block_module: Any, key: str) -> Any:
    """The value the LEGACY method would see for this configuration key.

    A block's `get_restrictions()` reads the constant its own module imported,
    so that module wins; `roboflow_workflows.environment` is the fallback for
    keys the block does not import itself.
    """
    import roboflow_workflows.environment as workflows_environment

    if hasattr(block_module, key):
        return getattr(block_module, key)
    if hasattr(workflows_environment, key):
        return getattr(workflows_environment, key)
    raise AssertionError(
        f"portable condition names {key}, which resolves neither in "
        f"{block_module.__name__} nor in roboflow_workflows.environment"
    )


def _condition_is_satisfied_here(
    restriction: RestrictionMetadata, block_module: Any
) -> bool:
    return all(
        _flag_value(block_module, key) == expected
        for key, expected in restriction.when.configuration_equals.items()
    )


def _portable_restrictions_of(block: _LoadedBlock) -> List[RestrictionMetadata]:
    declared = _manifest_instance(block).discover_portable_restrictions()
    return list(declared.items) if isinstance(declared, Discovery) else list(declared)


def _compare_legacy_and_portable(block: _LoadedBlock) -> Tuple[List[str], List[str]]:
    """``(legacy entries with no portable match, portable entries with no legacy match)``."""
    block_module = sys.modules[block.manifest_class.__module__]
    legacy_axes = collections.Counter(
        _legacy_axes(restriction)
        for restriction in block.manifest_class.get_restrictions()
    )
    active = [
        restriction
        for restriction in _portable_restrictions_of(block)
        if _condition_is_satisfied_here(restriction, block_module)
    ]
    portable_axes = collections.Counter(
        _portable_axes(restriction) for restriction in active
    )
    legacy_only = list((legacy_axes - portable_axes).elements())
    surplus = portable_axes - legacy_axes
    portable_only = [
        restriction
        for restriction in active
        if surplus.get(_portable_axes(restriction), 0) > 0
    ]
    unexplained_legacy = [
        f"{block.block_type}: legacy restriction {axes} has no active portable "
        "counterpart"
        for axes in legacy_only
        if (block.block_type, axes) not in LEGACY_WITHOUT_PORTABLE
    ]
    unexplained_portable = [
        f"{block.block_type}: portable restriction {restriction.code} "
        f"{_portable_axes(restriction)} applies here but legacy declares nothing "
        "like it"
        for restriction in portable_only
        if (block.block_type, restriction.code) not in PORTABLE_WITHOUT_LEGACY
    ]
    return unexplained_legacy, unexplained_portable


@pytest.mark.parametrize("local_storage_allowed", [True, False])
def test_portable_restrictions_agree_with_legacy_under_the_current_flags(
    loaded_blocks: List[_LoadedBlock], local_storage_allowed: bool
) -> None:
    """Evaluate the portable conditions against the flags legacy reads.

    The flag is flipped by patching the CONSTANTS, not the environment: the
    workflows package binds them once at import from the installed
    configuration, so an environment variable set afterwards changes nothing,
    while `get_restrictions()` genuinely reads the module constant this test
    patches (see the local-file sink).
    """
    problems: List[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith(
                "roboflow_workflows"
            ):
                continue
            for flag in FLAG_CONSTANTS_UNDER_TEST:
                if hasattr(module, flag):
                    monkeypatch.setattr(module, flag, local_storage_allowed)
        for block in loaded_blocks:
            unexplained_legacy, unexplained_portable = _compare_legacy_and_portable(
                block
            )
            problems.extend(unexplained_legacy)
            problems.extend(unexplained_portable)
    assert not problems, "\n".join(problems)


def test_the_legacy_divergence_registry_cannot_rot(
    loaded_blocks: List[_LoadedBlock],
) -> None:
    """Every recorded divergence must still be a real, reachable divergence."""
    by_type = {block.block_type: block for block in loaded_blocks}
    problems: List[str] = []
    for (block_type, code), reason in PORTABLE_WITHOUT_LEGACY.items():
        if not reason.strip():
            problems.append(f"{block_type}/{code}: no reason given")
        block = by_type.get(block_type)
        if block is None:
            problems.append(f"{block_type}: no longer in the registry")
            continue
        declared_codes = {
            restriction.code for restriction in _portable_restrictions_of(block)
        }
        if code not in declared_codes:
            problems.append(
                f"{block_type}: no longer declares {code}; remove the entry"
            )
    for (block_type, axes), reason in LEGACY_WITHOUT_PORTABLE.items():
        if not reason.strip():
            problems.append(f"{block_type}/{axes}: no reason given")
        block = by_type.get(block_type)
        if block is None:
            problems.append(f"{block_type}: no longer in the registry")
            continue
        legacy_axes = {
            _legacy_axes(restriction)
            for restriction in block.manifest_class.get_restrictions()
        }
        if axes not in legacy_axes:
            problems.append(
                f"{block_type}: legacy no longer declares {axes}; remove the entry"
            )
    assert all(
        reason.strip() for reason in HOST_PLUGIN_DIVERGENCES_CHECKED_ELSEWHERE.values()
    )
    assert not problems, "\n".join(problems)
