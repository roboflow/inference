"""Workload declarations of the 9 Roboflow-platform plugin blocks.

Every block registered by `inference.roboflow_workflows_plugin.loader` must
override ALL THREE declaration hooks explicitly - the two portable ones plus
`discover_dependent_resources` (D022). `None` (the base "unknown" answer) is
reserved for third-party plugins nobody audited. The census runs against the
REAL registry, so a new plugin block cannot silently pass as complete.

The numpy and tensor implementation files are independent copies of each other
(no delegation), so all three hook sources are compared text-for-text, and the
portable declarations' VALUES are compared across a fresh tensor-mode subprocess.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

# The plugin needs the SERVER's Workflows configuration installed before any
# `roboflow_workflows` module is imported; importing `inference.core.env` is
# what installs it. `# isort: split` keeps this import ahead of the rest.
import inference.core.env  # noqa: F401

# isort: split

from inference.core.workflows.prototypes.block import WorkflowBlockManifest
from inference.roboflow_workflows_plugin import loader as plugin_loader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
PLUGIN_ROOT = REPO_ROOT / "inference" / "roboflow_workflows_plugin"

# The two portable declaration hooks: plain data, no branching at all.
DECLARATION_HOOK_NAMES = (
    "discover_work_operations",
    "discover_portable_restrictions",
)
# D022: every registered manifest must ALSO carry an explicit, audited
# `discover_dependent_resources` override - the base `None` (unknown) is
# reserved for plugins nobody audited.
HOOK_NAMES = DECLARATION_HOOK_NAMES + ("discover_dependent_resources",)

# The 9 block types this plugin registers (block-coverage census, owner "host").
PLUGIN_BLOCK_TYPES = {
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
}

# D022 audit outcome: these four reference none of the three resource types.
AUDITED_EMPTY_RESOURCE_BLOCKS = {
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
}

# The five that already declared resources before T42, with the resource types
# their long-standing declarations produce.
DECLARED_RESOURCE_TYPES = {
    "roboflow_core/roboflow_dataset_upload@v1": ("roboflow_platform_project",),
    "roboflow_core/roboflow_dataset_upload@v2": ("roboflow_platform_project",),
    "roboflow_core/model_monitoring_inference_aggregator@v1": (
        "roboflow_platform_model",
    ),
    "roboflow_core/visual_search@v1": ("roboflow_platform_project",),
    "roboflow_core/visual_search_classifier@v1": ("roboflow_platform_project",),
}

# (numpy file, tensor sibling) pairs. `visual_search@v1` and
# `asset_library_attributes@v1` have no tensor sibling - the loader imports the
# same module in both modes.
SIBLING_PAIRS = [
    ("sinks/custom_metadata/v1.py", "sinks/custom_metadata/v1_tensor.py"),
    ("sinks/dataset_upload/v1.py", "sinks/dataset_upload/v1_tensor.py"),
    ("sinks/dataset_upload/v2.py", "sinks/dataset_upload/v2_tensor.py"),
    (
        "sinks/model_monitoring_inference_aggregator/v1.py",
        "sinks/model_monitoring_inference_aggregator/v1_tensor.py",
    ),
    ("sinks/vision_events/v1.py", "sinks/vision_events/v1_tensor.py"),
    ("sinks/vision_events_bundle/v1.py", "sinks/vision_events_bundle/v1_tensor.py"),
    (
        "integrations/visual_search_classifier/v1.py",
        "integrations/visual_search_classifier/v1_tensor.py",
    ),
]


def _manifest_of(block_class):
    return block_class.get_manifest()


def _type_of(manifest_cls) -> str:
    return manifest_cls.model_fields["type"].annotation.__args__[0]


def _declarations(manifest_cls) -> dict:
    # `model_construct()` skips field validation: the hooks are pure
    # declarations and read no manifest field, so no fixture data is needed and
    # no block is ever initialised.
    instance = manifest_cls.model_construct()
    return {
        "operations": [op.value for op in instance.discover_work_operations()],
        "restrictions": [
            restriction.model_dump(mode="json")
            for restriction in instance.discover_portable_restrictions()
        ],
    }


@pytest.fixture(scope="module")
def plugin_manifests():
    return {
        _type_of(_manifest_of(block)): _manifest_of(block)
        for block in plugin_loader.load_blocks()
    }


def test_census_covers_exactly_the_registered_plugin_blocks(plugin_manifests) -> None:
    assert set(plugin_manifests) == PLUGIN_BLOCK_TYPES


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_every_plugin_block_overrides_the_hook(plugin_manifests, hook_name) -> None:
    base_hook = getattr(WorkflowBlockManifest, hook_name)
    not_overridden = [
        block_type
        for block_type, manifest_cls in plugin_manifests.items()
        # `vars()`, not `getattr`: the override must be written ON the manifest
        # class, so an inherited one cannot stand in for an audit of this block.
        if hook_name not in vars(manifest_cls)
        or getattr(manifest_cls, hook_name) is base_hook
    ]
    assert not not_overridden, (
        f"{hook_name} is not declared by: {sorted(not_overridden)} - the base "
        "implementation returns None (unknown), which the census forbids for "
        "registered blocks"
    )


def test_every_plugin_block_declares_a_concrete_list(plugin_manifests) -> None:
    for block_type, manifest_cls in plugin_manifests.items():
        instance = manifest_cls.model_construct()
        operations = instance.discover_work_operations()
        restrictions = instance.discover_portable_restrictions()
        assert isinstance(operations, list), block_type
        assert isinstance(restrictions, list), block_type
        # A plain list means "complete declaration", the answer the census wants.
        assert operations is not None and restrictions is not None, block_type


def test_declarations_normalise_into_complete_discoveries(plugin_manifests) -> None:
    from roboflow_workflows.execution_engine.entities.workload import (
        normalize_declaration,
    )

    for block_type, manifest_cls in plugin_manifests.items():
        instance = manifest_cls.model_construct()
        operations = normalize_declaration(
            instance.discover_work_operations(), "unused_reason:$steps.x"
        )
        restrictions = normalize_declaration(
            instance.discover_portable_restrictions(), "unused_reason:$steps.x"
        )
        assert operations.complete, block_type
        assert operations.unknown_reasons == [], block_type
        assert restrictions.complete, block_type
        assert restrictions.unknown_reasons == [], block_type


def _construct(manifest_cls, **values):
    """Manifest instance without validation, carrying only the fields it has."""
    fields = {
        name: value
        for name, value in values.items()
        if name in manifest_cls.model_fields
    }
    return manifest_cls.model_construct(**fields)


def test_audited_blocks_declare_no_dependent_resource(plugin_manifests) -> None:
    """D022 audit: these four sinks call the Roboflow platform (or write to a
    local volume) but reference NONE of the three declarable resource types - no
    model, no project, no third-party model. A platform HTTP call by itself is
    not a dependent resource."""
    for block_type in sorted(AUDITED_EMPTY_RESOURCE_BLOCKS):
        instance = _construct(plugin_manifests[block_type])
        assert instance.discover_dependent_resources() == [], block_type


def test_pre_existing_resource_declarations_are_untouched(plugin_manifests) -> None:
    """The five blocks that already declared resources still declare exactly the
    same resource types - T42 only ADDED overrides, it changed none."""
    for block_type, expected in sorted(DECLARED_RESOURCE_TYPES.items()):
        instance = _construct(
            plugin_manifests[block_type],
            disable_sink=False,
            target_project="my-workspace/my-project",
            model_id="my-project/3",
        )
        resources = instance.discover_dependent_resources()
        assert (
            tuple(resource.resource_type.value for resource in resources) == expected
        ), block_type


def test_legacy_restriction_api_is_untouched(plugin_manifests) -> None:
    """The portable declaration is additive: `get_restrictions()` still answers
    with the human-readable `RuntimeRestriction` objects (notes included)."""
    aggregator = plugin_manifests[
        "roboflow_core/model_monitoring_inference_aggregator@v1"
    ]
    legacy = aggregator.get_restrictions()

    assert len(legacy) == 2
    assert any(
        "Aggregation buffers are stored in process memory" in restriction.note
        for restriction in legacy
    )
    # ... and the new entities carry no note at all
    portable = aggregator.model_construct().discover_portable_restrictions()
    assert all(not hasattr(restriction, "note") for restriction in portable)


def test_vision_event_bundle_declares_both_local_storage_flag_branches(
    plugin_manifests,
) -> None:
    """The exact expectation for the one block with a conditional legacy
    declaration. Both branches of `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE`
    are declared unconditionally, each pinned by `configuration_equals`, because
    the portable answer describes the TARGET deployment - it must not depend on
    the flag of the host answering the introspection call."""
    declarations = _declarations(
        plugin_manifests["roboflow_core/vision_event_bundle@v1"]
    )

    assert declarations["operations"] == ["image_encoding", "storage_write"]
    by_code = {
        restriction["code"]: restriction for restriction in declarations["restrictions"]
    }
    assert sorted(by_code) == [
        "cooldown_timer_resets_on_stateless_http",
        "ephemeral_container_disk_loses_writes",
        "local_storage_access_disabled",
        "writes_to_deployment_volume_not_retrievable",
    ]
    assert by_code["local_storage_access_disabled"]["severity"] == "hard"
    assert by_code["local_storage_access_disabled"]["when"]["configuration_equals"] == {
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": False
    }
    assert sorted(by_code["local_storage_access_disabled"]["when"]["runtimes"]) == [
        "dedicated_deployment",
        "hosted_serverless",
    ]
    assert by_code["ephemeral_container_disk_loses_writes"]["severity"] == "soft"
    assert by_code["ephemeral_container_disk_loses_writes"]["when"][
        "configuration_equals"
    ] == {"ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": True}
    assert by_code["ephemeral_container_disk_loses_writes"]["when"]["runtimes"] == [
        "hosted_serverless"
    ]
    assert by_code["writes_to_deployment_volume_not_retrievable"]["when"][
        "runtimes"
    ] == ["dedicated_deployment"]
    assert by_code["cooldown_timer_resets_on_stateless_http"]["when"][
        "step_execution_modes"
    ] == ["remote"]


def test_model_monitoring_aggregator_declaration_mirrors_its_legacy_axes(
    plugin_manifests,
) -> None:
    """The second exact expectation. The aggregation-buffer restriction keeps
    the legacy declaration's three axes verbatim; the still-image caveat reuses
    the shared preset."""
    declarations = _declarations(
        plugin_manifests["roboflow_core/model_monitoring_inference_aggregator@v1"]
    )

    assert declarations["operations"] == [
        "data_aggregation",
        "external_request",
        "temporal_buffering",
    ]
    by_code = {
        restriction["code"]: restriction for restriction in declarations["restrictions"]
    }
    assert sorted(by_code) == [
        "aggregation_buffer_resets_on_stateless_http",
        "temporal_block_no_benefit_on_still_image",
    ]
    buffer_condition = by_code["aggregation_buffer_resets_on_stateless_http"]["when"]
    assert by_code["aggregation_buffer_resets_on_stateless_http"]["severity"] == "soft"
    assert sorted(buffer_condition["runtimes"]) == [
        "dedicated_deployment",
        "hosted_serverless",
    ]
    assert buffer_condition["step_execution_modes"] == ["remote"]
    assert buffer_condition["input_modes"] == ["video"]
    assert buffer_condition["configuration_equals"] == {}
    still_image_condition = by_code["temporal_block_no_benefit_on_still_image"]["when"]
    assert still_image_condition["input_modes"] == ["image"]
    assert still_image_condition["runtimes"] is None


def test_no_hook_reads_an_environment_flag(plugin_manifests) -> None:
    """None of the three hooks may consult the host's configuration - the answer
    describes the TARGET deployment, not the server answering the call."""
    import inspect

    for block_type, manifest_cls in plugin_manifests.items():
        for hook_name in HOOK_NAMES:
            source = inspect.getsource(getattr(manifest_cls, hook_name))
            for forbidden in ("os.getenv", "os.environ", "ALLOW_WORKFLOW"):
                assert forbidden not in source, f"{block_type}.{hook_name}: {forbidden}"


def test_portable_declarations_do_not_branch_at_all(plugin_manifests) -> None:
    """The two portable declarations are plain data: no branch, so both flag
    branches of a conditional legacy restriction are always reported.
    (`discover_dependent_resources` is excluded on purpose - it may legitimately
    branch on a manifest FIELD, e.g. `disable_sink`.)"""
    import inspect

    for block_type, manifest_cls in plugin_manifests.items():
        for hook_name in DECLARATION_HOOK_NAMES:
            source = inspect.getsource(getattr(manifest_cls, hook_name))
            assert "if " not in source, f"{block_type}.{hook_name} branches"


def _hook_sources(path: pathlib.Path) -> dict:
    tree = ast.parse(path.read_text())
    manifests = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "WorkflowBlockManifest"
            for base in node.bases
        )
    ]
    assert len(manifests) == 1, f"{path}: {[m.name for m in manifests]}"
    source = path.read_text()
    found = {}
    for node in manifests[0].body:
        if isinstance(node, ast.FunctionDef) and node.name in HOOK_NAMES:
            found[node.name] = ast.get_source_segment(source, node)
    assert sorted(found) == sorted(HOOK_NAMES), f"{path}: {sorted(found)}"
    return found


@pytest.mark.parametrize("numpy_file,tensor_file", SIBLING_PAIRS)
def test_numpy_and_tensor_siblings_declare_identical_hooks(
    numpy_file, tensor_file
) -> None:
    numpy_hooks = _hook_sources(PLUGIN_ROOT / numpy_file)
    tensor_hooks = _hook_sources(PLUGIN_ROOT / tensor_file)

    assert numpy_hooks == tensor_hooks, (
        f"{numpy_file} and {tensor_file} declare different workload hooks - the "
        "tensor sibling is an independent copy and must state the same facts"
    )


DECLARATIONS_PROBE = """
import json

import inference.core.env as env_module
from inference.roboflow_workflows_plugin import loader as plugin_loader

declarations = {}
for block in plugin_loader.load_blocks():
    manifest_cls = block.get_manifest()
    identifier = manifest_cls.model_fields["type"].annotation.__args__[0]
    instance = manifest_cls.model_construct()
    declarations[identifier] = {
        "module": manifest_cls.__module__,
        "operations": [op.value for op in instance.discover_work_operations()],
        "restrictions": [
            restriction.model_dump(mode="json")
            for restriction in instance.discover_portable_restrictions()
        ],
    }

print(
    json.dumps(
        {
            "effective_tensor_mode": env_module.ENABLE_TENSOR_DATA_REPRESENTATION,
            "declarations": declarations,
        },
        sort_keys=True,
    )
)
"""


def _run_declarations_probe(tensor_mode: str) -> dict:
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        # Server-side the env var IS honoured, because `inference.core.env` is
        # imported before `roboflow_workflows` installs its configuration.
        "ENABLE_TENSOR_DATA_REPRESENTATION": tensor_mode,
        "PYTHONPATH": os.pathsep.join(
            [
                str(REPO_ROOT / "workflows"),
                str(REPO_ROOT / "inference_models"),
                str(REPO_ROOT),
            ]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", DECLARATIONS_PROBE],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_declarations_are_identical_in_both_representation_modes() -> None:
    """Fresh subprocesses: the tensor flag is frozen at first import, so the two
    modes cannot be compared inside one process."""
    numpy_mode = _run_declarations_probe("False")
    tensor_mode = _run_declarations_probe("True")

    assert numpy_mode["effective_tensor_mode"] is False
    assert tensor_mode["effective_tensor_mode"] is True
    assert set(numpy_mode["declarations"]) == PLUGIN_BLOCK_TYPES
    assert set(tensor_mode["declarations"]) == PLUGIN_BLOCK_TYPES

    swapped_modules = {
        block_type
        for block_type in PLUGIN_BLOCK_TYPES
        if numpy_mode["declarations"][block_type]["module"]
        != tensor_mode["declarations"][block_type]["module"]
    }
    # The 7 blocks with a tensor sibling really did swap implementation files,
    # so the comparison below is not vacuous.
    assert len(swapped_modules) == len(SIBLING_PAIRS)
    assert all(
        tensor_mode["declarations"][block_type]["module"].endswith("_tensor")
        for block_type in swapped_modules
    )

    for block_type in PLUGIN_BLOCK_TYPES:
        numpy_declaration = dict(numpy_mode["declarations"][block_type])
        tensor_declaration = dict(tensor_mode["declarations"][block_type])
        numpy_declaration.pop("module")
        tensor_declaration.pop("module")
        assert numpy_declaration == tensor_declaration, block_type


# ---------------------------------------------------------------------------
# Legacy <-> portable restriction axis equality (the host-plugin counterpart of
# `workflows/tests/unit_tests/core_steps/test_workload_declarations_coverage.py`,
# which cannot reach this plugin because the workflows package may not import
# `inference.*`).
#
# `get_restrictions()` answers "what applies on THIS host" - it evaluates the
# flags itself. `discover_portable_restrictions()` answers "what applies on a
# target deployment" - it carries the flag in `configuration_equals` instead.
# Once the portable conditions are evaluated against the same flag values the
# legacy method reads, the two must agree, entry for entry, on severity and on
# all three condition axes. A migration that quietly drops or invents a caveat
# is a regression, not a refactor.
# ---------------------------------------------------------------------------

# Portable entries with no legacy counterpart, keyed by (block type, code).
# Empty on purpose: this plugin's portable declarations were ported one-to-one.
PORTABLE_WITHOUT_LEGACY: dict = {}

# Legacy entries with no ACTIVE portable counterpart, keyed by
# (block type, (severity, runtimes, step_execution_modes, input_modes)).
LEGACY_WITHOUT_PORTABLE = {
    (
        "roboflow_core/vision_event_bundle@v1",
        ("soft", ("dedicated_deployment",), (), ()),
    ): (
        "legacy emits 'bundles land on the deployment volume but are not "
        "retrievable through the Roboflow API' on BOTH branches of the storage "
        "flag; the portable writes_to_deployment_volume_not_retrievable pins "
        "the ENABLED branch, because with local storage disabled the block "
        "raises before it writes anything. Same reading as local_file_sink@v1 "
        "(coordinator restriction registry; recorded in the workflows-package "
        "module as HOST_PLUGIN_DIVERGENCES_CHECKED_ELSEWHERE)"
    ),
}

# Flags a legacy `get_restrictions()` in this plugin evaluates. Flipped by
# patching the CONSTANT each block module imported - an environment variable set
# after import changes nothing.
FLAG_CONSTANTS_UNDER_TEST = ("ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",)


def _legacy_axes(restriction) -> tuple:
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


def _portable_axes(restriction) -> tuple:
    condition = restriction.when
    return (
        restriction.severity.value,
        tuple(sorted(item.value for item in (condition.runtimes or ()))),
        tuple(sorted(item.value for item in (condition.step_execution_modes or ()))),
        tuple(sorted(item.value for item in (condition.input_modes or ()))),
    )


def _flag_value(block_module, key: str):
    """The value the LEGACY method would see for this configuration key.

    A block's `get_restrictions()` reads the constant its own module imported,
    so that module wins; `inference.core.env` is the fallback for keys the block
    does not import itself.
    """
    if hasattr(block_module, key):
        return getattr(block_module, key)
    if hasattr(inference.core.env, key):
        return getattr(inference.core.env, key)
    raise AssertionError(
        f"portable condition names {key}, which resolves neither in "
        f"{block_module.__name__} nor in inference.core.env"
    )


def _condition_is_satisfied_here(restriction, block_module) -> bool:
    return all(
        _flag_value(block_module, key) == expected
        for key, expected in restriction.when.configuration_equals.items()
    )


def _portable_restrictions_of(manifest_cls) -> list:
    from roboflow_workflows.execution_engine.entities.workload import Discovery

    declared = _construct(manifest_cls).discover_portable_restrictions()
    return list(declared.items) if isinstance(declared, Discovery) else list(declared)


def _raw_divergence(block_type: str, manifest_cls) -> tuple:
    """`(legacy axes with no active portable match, portable entries with no legacy match)`.

    The allow-list is NOT applied here, so the rot test can see what the
    comparison really finds.
    """
    import collections

    block_module = sys.modules[manifest_cls.__module__]
    legacy_axes = collections.Counter(
        _legacy_axes(restriction) for restriction in manifest_cls.get_restrictions()
    )
    active = [
        restriction
        for restriction in _portable_restrictions_of(manifest_cls)
        if _condition_is_satisfied_here(restriction, block_module)
    ]
    portable_axes = collections.Counter(
        _portable_axes(restriction) for restriction in active
    )
    surplus = portable_axes - legacy_axes
    legacy_only = list((legacy_axes - portable_axes).elements())
    portable_only = [
        restriction
        for restriction in active
        if surplus.get(_portable_axes(restriction), 0) > 0
    ]
    return legacy_only, portable_only


def _patch_storage_flag(monkeypatch, value: bool) -> int:
    """Patch the constant in every plugin module that imported it, plus the env
    module the fallback reads. Returns how many modules were patched, so a test
    can prove the patch was not a no-op."""
    patched = 0
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "") if module is not None else ""
        if not (
            name.startswith("inference.roboflow_workflows_plugin")
            or name == "inference.core.env"
        ):
            continue
        for flag in FLAG_CONSTANTS_UNDER_TEST:
            if hasattr(module, flag):
                monkeypatch.setattr(module, flag, value)
                patched += 1
    return patched


@pytest.mark.parametrize("local_storage_allowed", [True, False])
def test_portable_restrictions_agree_with_legacy_under_the_current_flags(
    plugin_manifests, local_storage_allowed
) -> None:
    problems = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        patched = _patch_storage_flag(monkeypatch, local_storage_allowed)
        # The bundle block's module and inference.core.env both hold it.
        assert patched >= 2, "the storage flag patch reached nothing"
        for block_type, manifest_cls in plugin_manifests.items():
            legacy_only, portable_only = _raw_divergence(block_type, manifest_cls)
            problems.extend(
                f"{block_type}: legacy restriction {axes} has no active portable "
                f"counterpart (ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE="
                f"{local_storage_allowed})"
                for axes in legacy_only
                if (block_type, axes) not in LEGACY_WITHOUT_PORTABLE
            )
            problems.extend(
                f"{block_type}: portable restriction {restriction.code} "
                f"{_portable_axes(restriction)} applies here but legacy declares "
                f"nothing like it (ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE="
                f"{local_storage_allowed})"
                for restriction in portable_only
                if (block_type, restriction.code) not in PORTABLE_WITHOUT_LEGACY
            )
    assert not problems, "\n".join(problems)


def test_the_legacy_divergence_registry_cannot_rot(plugin_manifests) -> None:
    """Every allow-listed divergence must still be a real, reachable one.

    An entry that no longer corresponds to an actual mismatch under EITHER flag
    value is stale and must be deleted - otherwise the allow-list would silently
    start excusing a future regression.
    """
    problems = []
    observed = {True: set(), False: set()}
    with pytest.MonkeyPatch.context() as monkeypatch:
        for value in (True, False):
            _patch_storage_flag(monkeypatch, value)
            for block_type, manifest_cls in plugin_manifests.items():
                legacy_only, _ = _raw_divergence(block_type, manifest_cls)
                observed[value].update((block_type, axes) for axes in legacy_only)

    for key, reason in LEGACY_WITHOUT_PORTABLE.items():
        block_type, axes = key
        if not reason.strip():
            problems.append(f"{block_type}/{axes}: no reason given")
        if block_type not in plugin_manifests:
            problems.append(f"{block_type}: no longer registered by the plugin")
            continue
        legacy_declared = {
            _legacy_axes(restriction)
            for restriction in plugin_manifests[block_type].get_restrictions()
        }
        if axes not in legacy_declared:
            problems.append(
                f"{block_type}: legacy no longer declares {axes}; remove the entry"
            )
        if key not in observed[True] and key not in observed[False]:
            problems.append(
                f"{block_type}/{axes}: no longer diverges under either value of "
                "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE; remove the entry"
            )

    for block_type, code in PORTABLE_WITHOUT_LEGACY:
        if block_type not in plugin_manifests:
            problems.append(f"{block_type}: no longer registered by the plugin")
            continue
        declared_codes = {
            restriction.code
            for restriction in _portable_restrictions_of(plugin_manifests[block_type])
        }
        if code not in declared_codes:
            problems.append(
                f"{block_type}: no longer declares {code}; remove the entry"
            )

    assert not problems, "\n".join(problems)


def test_the_recorded_divergence_is_flag_specific(plugin_manifests) -> None:
    """The one allow-listed divergence exists ONLY with local storage disabled.

    With the flag enabled the bundle block's legacy and portable declarations
    agree exactly, so the allow-list must not be load-bearing there. This is
    what makes the entry above a narrow, dated exception rather than a blanket
    exemption for the block.
    """
    bundle = plugin_manifests["roboflow_core/vision_event_bundle@v1"]
    divergence_key = (
        "roboflow_core/vision_event_bundle@v1",
        ("soft", ("dedicated_deployment",), (), ()),
    )
    assert divergence_key in LEGACY_WITHOUT_PORTABLE

    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_storage_flag(monkeypatch, True)
        legacy_only, portable_only = _raw_divergence(
            "roboflow_core/vision_event_bundle@v1", bundle
        )
        assert legacy_only == [], (
            "with local storage ENABLED the bundle block must match legacy "
            f"exactly, but {legacy_only} diverged"
        )
        assert portable_only == []

        _patch_storage_flag(monkeypatch, False)
        legacy_only, portable_only = _raw_divergence(
            "roboflow_core/vision_event_bundle@v1", bundle
        )
        assert legacy_only == [divergence_key[1]]
        assert portable_only == []
