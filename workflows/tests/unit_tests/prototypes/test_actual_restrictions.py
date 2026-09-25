"""The public restriction API: ``get_actual_restrictions()``.

What is pinned here:

* ``RuntimeRestriction`` stays constructor-compatible and its ``to_dict()``
  payload (the editor's) is unchanged by the two new fields;
* the projection onto the portable ``RestrictionMetadata`` DTO;
* the default body - whatever ``self.get_restrictions()`` answers is wrapped as
  incomplete, whether that is a legacy override, an explicit ``[]`` or the
  inherited ``[]`` - and that a plugin may override the public method outright,
  wrap its own legacy list as complete, or extend ``super()``;
* the flag: ``True`` keeps every conditional declaration untouched, ``False``
  evaluates ONLY the configuration predicates against this process, drops what
  definitively does not apply, and reports what it cannot evaluate;
* that the values come from the installed ``WorkflowsConfiguration``, not from
  ``os.environ``;
* that every entry is rebuilt from its validated projection (enum members,
  lists, a configuration map; note, authored axis order and ``None`` kept), and
  that a malformed field or a failing host evaluation becomes a sanitised
  ``declaration_failed`` problem.
"""

import importlib
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List, Literal, Union

import pytest
from roboflow_workflows import configuration as configuration_module
from roboflow_workflows import environment
from roboflow_workflows.configuration import (
    EngineConfiguration,
    ModelsConfiguration,
    TensorConfiguration,
    WorkflowsConfiguration,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    UNSUPPORTED_IN_TENSOR_REPRESENTATION,
    hosted_endpoint_disabled_by_flag,
)
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblemCode,
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
    complete_discovery,
    incomplete_discovery,
    restriction_metadata_of,
    unresolved_selector_problem,
)
from roboflow_workflows.execution_engine.introspection import restriction_environment
from roboflow_workflows.execution_engine.introspection.restriction_environment import (
    EVALUABLE_CONFIGURATION_KEYS,
    ConfigurationMatch,
    evaluate_configuration_condition,
)
from roboflow_workflows.prototypes import block as block_module
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    WorkflowBlockManifest,
    actual_restrictions_of,
    is_workflow_selector,
)

TENSOR_FLAG = "ENABLE_TENSOR_DATA_REPRESENTATION"
CUSTOM_PYTHON_FLAG = "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS"
CUSTOM_PYTHON_MODE = "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE"


# ---------------------------------------------------------------------------
# Manifests used by the tests
# ---------------------------------------------------------------------------


class _UndeclaredManifest(WorkflowBlockManifest):
    type: Literal["test/undeclared@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class _DeclaringManifest(WorkflowBlockManifest):
    type: Literal["test/declaring@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=[
                STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
                STILL_IMAGE_INPUT_SOFT_RESTRICTION,
            ],
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class _ConditionalManifest(WorkflowBlockManifest):
    """A built-in-style block whose caveat is pinned to a configuration flag."""

    type: Literal["test/conditional@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=[
                UNSUPPORTED_IN_TENSOR_REPRESENTATION,
                REQUIRES_GPU_FOR_LOCAL_EXECUTION,
            ],
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class _InstanceRefinedManifest(WorkflowBlockManifest):
    """Refines its declaration from its own literal setting, and refuses to
    guess when that setting is a selector."""

    type: Literal["test/refined@v1"]
    fire_and_forget: Union[bool, str] = True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        declared: Union[List[RuntimeRestriction], Discovery[RuntimeRestriction]]
        if is_workflow_selector(self.fire_and_forget):
            declared = incomplete_discovery(
                items=[],
                reasons=[
                    unresolved_selector_problem(
                        node_id=f"$steps.{self.name}",
                        declaration="restrictions",
                        field="fire_and_forget",
                        selector=self.fire_and_forget,
                    )
                ],
            )
        elif self.fire_and_forget:
            declared = [STILL_IMAGE_INPUT_SOFT_RESTRICTION]
        else:
            declared = []
        return actual_restrictions_of(
            declared=declared,
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class _LegacyOnlyManifest(WorkflowBlockManifest):
    """A plugin from before the portable hooks: only ``get_restrictions()``."""

    type: Literal["test/legacy@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                Severity.SOFT,
                "Legacy note, possibly filtered against the host that answered.",
                [Runtime.HOSTED_SERVERLESS],
            ),
            RuntimeRestriction(
                Severity.HARD,
                "A legacy entry that already carries its own code.",
                [Runtime.HOSTED_SERVERLESS],
                code="a_legacy_specific_code",
            ),
        ]


class _LegacyIntermediateManifest(_LegacyOnlyManifest):
    """An ordinary subclass in between, adding nothing of its own."""

    type: Literal["test/legacy-intermediate@v1"]


class _LegacySubclassManifest(_LegacyIntermediateManifest):
    """The concrete block two levels down, with its own legacy list."""

    type: Literal["test/legacy-subclass@v1"]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [REQUIRES_GPU_FOR_LOCAL_EXECUTION]


class _LegacyEmptyManifest(WorkflowBlockManifest):
    """Overrides ``get_restrictions()`` and returns an empty list."""

    type: Literal["test/legacy-empty@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return []


class _LegacyConditionalManifest(WorkflowBlockManifest):
    """A legacy getter whose entry pins a configuration flag."""

    type: Literal["test/legacy-conditional@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [UNSUPPORTED_IN_TENSOR_REPRESENTATION]


class _CompleteFromLegacyManifest(WorkflowBlockManifest):
    """A block whose class-level declaration IS complete and environment
    independent, so it wraps its own legacy list explicitly instead of taking
    the fallback."""

    type: Literal["test/complete-legacy@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [STILL_IMAGE_INPUT_SOFT_RESTRICTION]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=list(self.get_restrictions()),
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class _GarbageManifest(WorkflowBlockManifest):
    """Declares something that is not a ``RuntimeRestriction`` at all."""

    type: Literal["test/garbage@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return actual_restrictions_of(
            declared=["not a restriction"],
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class _ExtendingPluginManifest(_DeclaringManifest):
    """A plugin that subclasses a built-in and extends what ``super()`` declares."""

    type: Literal["test/extending@v1"]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        inherited = super().get_actual_restrictions(
            ignore_environment_restrictions=ignore_environment_restrictions
        )
        return Discovery[RuntimeRestriction](
            items=list(inherited.items) + [REQUIRES_GPU_FOR_LOCAL_EXECUTION],
            complete=inherited.complete,
            unknown_reasons=list(inherited.unknown_reasons),
        )


def _manifest(manifest_type, **kwargs):
    payload = {
        "type": manifest_type.model_fields["type"].annotation.__args__[0],
        "name": "step",
    }
    payload.update(kwargs)
    return manifest_type.model_validate(payload)


@contextmanager
def installed_configuration(candidate: WorkflowsConfiguration):
    """Install a real ``WorkflowsConfiguration`` and rebind ``environment``.

    This is the ACTUAL source the evaluator reads, so a flag test that goes
    through here proves more than patching a module constant would.
    """
    original = configuration_module.get_configuration()
    configuration_module.reset_configuration()
    configuration_module.configure_process(candidate)
    importlib.reload(environment)
    try:
        yield
    finally:
        configuration_module.reset_configuration()
        configuration_module.configure_process(original)
        importlib.reload(environment)


# ---------------------------------------------------------------------------
# The entity
# ---------------------------------------------------------------------------


def test_runtime_restriction_keeps_its_positional_constructor() -> None:
    positional = RuntimeRestriction(
        Severity.HARD,
        "Raises RuntimeError.",
        [Runtime.HOSTED_SERVERLESS],
        [StepExecutionMode.REMOTE],
        [RuntimeInputMode.VIDEO],
    )
    named = RuntimeRestriction(
        severity=Severity.HARD,
        note="Raises RuntimeError.",
        applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
        applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
        applies_to_input_modes=[RuntimeInputMode.VIDEO],
    )

    assert positional == named
    assert positional.code == "generic_restriction"
    assert positional.applies_to_configuration is None


def test_to_dict_carries_neither_the_code_nor_the_configuration() -> None:
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="Raises RuntimeError.",
        applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
        code="a_specific_code",
        applies_to_configuration={TENSOR_FLAG: True},
    )

    assert restriction.to_dict() == {
        "severity": "hard",
        "note": "Raises RuntimeError.",
        "applies_to_runtimes": ["hosted_serverless"],
    }
    assert RuntimeRestriction(Severity.SOFT, "Bare.").to_dict() == {
        "severity": "soft",
        "note": "Bare.",
    }


def test_restriction_is_frozen() -> None:
    restriction = RuntimeRestriction(Severity.SOFT, "Bare.")

    with pytest.raises(FrozenInstanceError):
        restriction.code = "other"


def test_projection_onto_the_portable_dto() -> None:
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="Human explanation that the wire form does not carry.",
        applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
        applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
        applies_to_input_modes=[RuntimeInputMode.VIDEO],
        code="a_specific_code",
        applies_to_configuration={TENSOR_FLAG: True},
    )

    metadata = restriction_metadata_of(restriction)

    assert metadata == RestrictionMetadata(
        code="a_specific_code",
        severity=Severity.HARD,
        when=RestrictionCondition(
            runtimes=[Runtime.HOSTED_SERVERLESS],
            step_execution_modes=[StepExecutionMode.REMOTE],
            input_modes=[RuntimeInputMode.VIDEO],
            configuration_equals={TENSOR_FLAG: True},
        ),
    )
    assert "note" not in metadata.model_dump(mode="json")


def test_projection_copies_the_configuration_map() -> None:
    declared: Dict[str, Any] = {TENSOR_FLAG: True}
    restriction = RuntimeRestriction(
        severity=Severity.HARD, note="n", code="c", applies_to_configuration=declared
    )

    metadata = restriction_metadata_of(restriction)
    metadata.when.configuration_equals["injected"] = True

    assert declared == {TENSOR_FLAG: True}
    assert restriction.applies_to_configuration == {TENSOR_FLAG: True}


def test_generic_restrictions_with_different_notes_are_not_merged() -> None:
    first = RuntimeRestriction(Severity.SOFT, "One failure mode.")
    second = RuntimeRestriction(Severity.SOFT, "A different failure mode.")

    discovery = Discovery[RuntimeRestriction](
        items=[first, second, first], complete=True, unknown_reasons=[]
    )

    assert discovery.items == [second, first]
    assert {item.code for item in discovery.items} == {"generic_restriction"}


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_an_undeclared_manifest_is_unknown_not_empty() -> None:
    discovery = _manifest(_UndeclaredManifest).get_actual_restrictions()

    assert discovery.items == []
    assert discovery.complete is False
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    ]
    # the fallback answered through the inherited `get_restrictions()`, which
    # is what the `source` detail names
    assert discovery.unknown_reasons[0].details == {
        "node_id": "$steps.step",
        "declaration": "restrictions",
        "source": "get_restrictions",
    }


def test_a_declaring_manifest_is_complete() -> None:
    discovery = _manifest(_DeclaringManifest).get_actual_restrictions(
        ignore_environment_restrictions=True
    )

    assert discovery == complete_discovery(
        [STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION, STILL_IMAGE_INPUT_SOFT_RESTRICTION]
    )


def test_the_public_hook_is_an_instance_method() -> None:
    assert not isinstance(
        WorkflowBlockManifest.__dict__["get_actual_restrictions"], classmethod
    )
    assert isinstance(WorkflowBlockManifest.__dict__["get_restrictions"], classmethod)


@pytest.mark.parametrize("ignore_environment", [True, False])
def test_a_legacy_plugin_keeps_its_items_and_codes_but_is_never_complete(
    ignore_environment: bool,
) -> None:
    expected = _LegacyOnlyManifest.get_restrictions()

    discovery = _manifest(_LegacyOnlyManifest).get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment
    )

    assert len(discovery.items) == len(expected)
    for restriction in expected:
        assert restriction in discovery.items
    assert sorted(item.code for item in discovery.items) == [
        "a_legacy_specific_code",
        "generic_restriction",
    ]
    assert discovery.complete is False
    assert discovery.unknown_reasons[0].code is (
        DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    )
    assert discovery.unknown_reasons[0].details["source"] == "get_restrictions"


@pytest.mark.parametrize("ignore_environment", [True, False])
@pytest.mark.parametrize(
    "manifest_type",
    [_UndeclaredManifest, _LegacyEmptyManifest],
    ids=["inherited_empty", "explicit_empty"],
)
def test_an_empty_legacy_answer_is_not_proof_of_absence(
    manifest_type, ignore_environment: bool
) -> None:
    """The fallback cannot certify absence: an override returning ``[]`` and
    the inherited ``[]`` are both kept incomplete, under either flag value."""
    discovery = _manifest(manifest_type).get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment
    )

    assert discovery.items == []
    assert discovery.complete is False
    assert discovery.unknown_reasons[0].details["source"] == "get_restrictions"


@pytest.mark.parametrize("ignore_environment", [True, False])
@pytest.mark.parametrize(
    "manifest_type, expected",
    [
        (_UndeclaredManifest, []),
        (_LegacyEmptyManifest, []),
        (_LegacyOnlyManifest, _LegacyOnlyManifest.get_restrictions()),
    ],
    ids=["inherited_empty", "explicit_empty", "legacy_list"],
)
def test_a_manifest_without_a_name_reports_an_undefined_node_id(
    manifest_type, expected: List[RuntimeRestriction], ignore_environment: bool
) -> None:
    """A manifest built without `name` still answers: the fallback names the
    step `$steps.<undefined>` instead of raising."""
    discovery = manifest_type.model_construct().get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment
    )

    assert len(discovery.items) == len(expected)
    for restriction in expected:
        assert restriction in discovery.items
    assert discovery.complete is False
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    ]
    assert discovery.unknown_reasons[0].details == {
        "node_id": "$steps.<undefined>",
        "declaration": "restrictions",
        "source": "get_restrictions",
    }


@pytest.mark.parametrize(
    "manifest_type, expected",
    [
        (_LegacyIntermediateManifest, _LegacyOnlyManifest.get_restrictions()),
        (_LegacySubclassManifest, [REQUIRES_GPU_FOR_LOCAL_EXECUTION]),
    ],
    ids=["inherits_through_an_intermediate_base", "overrides_two_levels_down"],
)
def test_the_fallback_follows_ordinary_classmethod_dispatch(
    manifest_type, expected: List[RuntimeRestriction]
) -> None:
    """``self.get_restrictions()`` resolves the way Python resolves any
    classmethod, so a subclass of a subclass answers with its own list and an
    intermediate base that declares nothing keeps its parent's."""
    discovery = _manifest(manifest_type).get_actual_restrictions(
        ignore_environment_restrictions=True
    )

    assert len(discovery.items) == len(expected)
    for restriction in expected:
        assert restriction in discovery.items
    assert discovery.complete is False


def test_the_fallback_still_evaluates_configuration_predicates_for_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag keeps its ordinary meaning on the fallback path: with ``False``
    a legacy entry whose configuration predicate does not hold here is dropped,
    with ``True`` it is kept. The result stays incomplete either way."""
    manifest = _manifest(_LegacyConditionalManifest)

    monkeypatch.setattr(environment, TENSOR_FLAG, False)
    numpy_host = manifest.get_actual_restrictions()
    portable = manifest.get_actual_restrictions(ignore_environment_restrictions=True)
    monkeypatch.setattr(environment, TENSOR_FLAG, True)
    tensor_host = manifest.get_actual_restrictions()

    assert numpy_host.items == []
    assert portable.items == [UNSUPPORTED_IN_TENSOR_REPRESENTATION]
    assert tensor_host.items == [UNSUPPORTED_IN_TENSOR_REPRESENTATION]
    for discovery in (numpy_host, portable, tensor_host):
        assert discovery.complete is False


@pytest.mark.parametrize("ignore_environment", [True, False])
def test_a_block_may_wrap_its_own_legacy_list_as_complete(
    ignore_environment: bool,
) -> None:
    """A block whose class declaration is environment independent may opt out
    of the fallback and state completeness itself."""
    discovery = _manifest(_CompleteFromLegacyManifest).get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment
    )

    assert discovery == complete_discovery([STILL_IMAGE_INPUT_SOFT_RESTRICTION])


def test_a_plugin_may_extend_what_super_declares() -> None:
    """The extension pattern: a plugin subclasses a built-in, calls
    ``super().get_actual_restrictions(...)`` and adds its own entry. The flag is
    passed on, so the portable and the host view stay consistent."""
    manifest = _manifest(_ExtendingPluginManifest)

    portable = manifest.get_actual_restrictions(ignore_environment_restrictions=True)

    assert portable == complete_discovery(
        [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
            REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        ]
    )
    assert manifest.get_actual_restrictions().complete is True


@pytest.mark.parametrize(
    "restriction, label",
    [
        (RuntimeRestriction(Severity.HARD, "blank code", code=""), "blank code"),
        (
            RuntimeRestriction(Severity.HARD, "invalid code", code="Not A Code"),
            "code that is not an identifier",
        ),
        (
            RuntimeRestriction(Severity.HARD, "empty axis", applies_to_runtimes=[]),
            "an axis declared as an empty list",
        ),
        (
            RuntimeRestriction(
                Severity.HARD,
                "blank configuration key",
                code="a_code",
                applies_to_configuration={"   ": True},
            ),
            "a blank configuration key",
        ),
    ],
    ids=["blank_code", "invalid_code", "empty_axis", "blank_configuration_key"],
)
@pytest.mark.parametrize("ignore_environment", [True, False])
def test_a_declaration_the_portable_contract_cannot_express_is_a_failure(
    restriction: RuntimeRestriction, label: str, ignore_environment: bool
) -> None:
    """A well-typed `RuntimeRestriction` whose SEMANTIC fields are invalid must
    not be published as a complete declaration the wire DTO cannot carry."""

    class _InvalidManifest(WorkflowBlockManifest):
        type: Literal["test/invalid@v1"]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="output")]

        def get_actual_restrictions(
            self, *, ignore_environment_restrictions: bool = False
        ) -> Discovery[RuntimeRestriction]:
            return actual_restrictions_of(
                declared=[restriction],
                node_id=f"$steps.{self.name}",
                ignore_environment_restrictions=ignore_environment_restrictions,
            )

    discovery = _manifest(_InvalidManifest).get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment
    )

    assert discovery.items == [], label
    assert discovery.complete is False, label
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_FAILED
    ], label
    assert "Not A Code" not in discovery.model_dump_json()


def test_the_legacy_entity_constructor_stays_permissive() -> None:
    """The validation lives at the NEW API boundary only: the dataclass and the
    editor payload accept exactly what they always did."""
    permissive = RuntimeRestriction(Severity.HARD, "blank code", code="")

    assert permissive.code == ""
    assert permissive.to_dict() == {"severity": "hard", "note": "blank code"}
    assert RuntimeRestriction(
        Severity.SOFT, "empty axis", applies_to_runtimes=[]
    ).to_dict() == {"severity": "soft", "note": "empty axis", "applies_to_runtimes": []}


def test_an_invalid_declaration_is_reported_without_its_contents() -> None:
    """A declaration that is not a ``RuntimeRestriction`` becomes a sanitised
    ``declaration_failed`` problem; what the block declared is never echoed.
    A hook that RAISES is sanitised by the workload builder, which is where
    every declaration hook is called from - see the builder tests."""
    discovery = _manifest(_GarbageManifest).get_actual_restrictions()

    assert discovery.items == []
    assert discovery.complete is False
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_FAILED
    ]
    assert "not a restriction" not in discovery.model_dump_json()


# ---------------------------------------------------------------------------
# Instance refinement
# ---------------------------------------------------------------------------


def test_a_literal_setting_selects_the_declaration() -> None:
    on = _manifest(_InstanceRefinedManifest, fire_and_forget=True)
    off = _manifest(_InstanceRefinedManifest, fire_and_forget=False)

    assert on.get_actual_restrictions() == complete_discovery(
        [STILL_IMAGE_INPUT_SOFT_RESTRICTION]
    )
    assert off.get_actual_restrictions() == complete_discovery([])


def test_a_selector_setting_stays_incomplete_and_is_never_resolved() -> None:
    manifest = _manifest(_InstanceRefinedManifest, fire_and_forget="$inputs.wait")

    discovery = manifest.get_actual_restrictions()

    assert discovery.items == []
    assert discovery.complete is False
    assert discovery.unknown_reasons == [
        unresolved_selector_problem(
            node_id="$steps.step",
            declaration="restrictions",
            field="fire_and_forget",
            selector="$inputs.wait",
        )
    ]


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def test_true_keeps_every_conditional_declaration_whatever_this_host_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_ConditionalManifest)

    answers = []
    for tensor_enabled in (False, True):
        monkeypatch.setattr(environment, TENSOR_FLAG, tensor_enabled)
        answers.append(
            manifest.get_actual_restrictions(ignore_environment_restrictions=True)
        )

    assert answers[0] == answers[1]
    assert answers[0] == complete_discovery(
        [UNSUPPORTED_IN_TENSOR_REPRESENTATION, REQUIRES_GPU_FOR_LOCAL_EXECUTION]
    )
    # the condition is retained, not stripped
    tensor_entry = next(
        item for item in answers[0].items if item.code.startswith("unsupported")
    )
    assert tensor_entry.applies_to_configuration == {TENSOR_FLAG: True}


def test_false_drops_only_what_definitively_does_not_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_ConditionalManifest)

    monkeypatch.setattr(environment, TENSOR_FLAG, False)
    numpy_host = manifest.get_actual_restrictions()
    monkeypatch.setattr(environment, TENSOR_FLAG, True)
    tensor_host = manifest.get_actual_restrictions()

    # REQUIRES_GPU_FOR_LOCAL_EXECUTION has no configuration predicate at all,
    # so it survives both branches - the runtime axis is never evaluated.
    assert numpy_host == complete_discovery([REQUIRES_GPU_FOR_LOCAL_EXECUTION])
    assert tensor_host == complete_discovery(
        [UNSUPPORTED_IN_TENSOR_REPRESENTATION, REQUIRES_GPU_FOR_LOCAL_EXECUTION]
    )


def test_the_host_view_reads_the_installed_configuration() -> None:
    """The flag-on / flag-off branches through the ACTUAL configuration source."""
    manifest = _manifest(_ConditionalManifest)

    with installed_configuration(
        WorkflowsConfiguration(tensor=TensorConfiguration(representation_enabled=False))
    ):
        numpy_host = manifest.get_actual_restrictions()
    with installed_configuration(
        WorkflowsConfiguration(tensor=TensorConfiguration(representation_enabled=True))
    ):
        tensor_host = manifest.get_actual_restrictions()

    assert [item.code for item in numpy_host.items] == [
        "requires_gpu_for_local_execution"
    ]
    assert [item.code for item in tensor_host.items] == [
        "requires_gpu_for_local_execution",
        "unsupported_in_tensor_representation",
    ]


def test_multiple_predicates_are_anded() -> None:
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="Custom python is refused.",
        code="custom_python_execution_disabled",
        applies_to_configuration={
            CUSTOM_PYTHON_FLAG: False,
            CUSTOM_PYTHON_MODE: "local",
        },
    )

    with installed_configuration(
        WorkflowsConfiguration(
            engine=EngineConfiguration(
                allow_custom_python_execution=False,
                custom_python_execution_mode="local",
            )
        )
    ):
        assert evaluate_configuration_condition(restriction=restriction) == (
            ConfigurationMatch.MATCHES,
            [],
        )
    with installed_configuration(
        WorkflowsConfiguration(
            engine=EngineConfiguration(
                allow_custom_python_execution=False,
                custom_python_execution_mode="modal",
            )
        )
    ):
        # one predicate fails, so the restriction cannot apply here
        assert evaluate_configuration_condition(restriction=restriction) == (
            ConfigurationMatch.INACTIVE,
            [],
        )


def test_an_unknown_configuration_key_keeps_the_entry_and_reports_it() -> None:
    class _UnknownKeyManifest(WorkflowBlockManifest):
        type: Literal["test/unknown-key@v1"]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="output")]

        def get_actual_restrictions(
            self, *, ignore_environment_restrictions: bool = False
        ) -> Discovery[RuntimeRestriction]:
            return actual_restrictions_of(
                declared=[
                    RuntimeRestriction(
                        severity=Severity.HARD,
                        note="Depends on a flag this package cannot read.",
                        code="depends_on_unknown_flag",
                        applies_to_configuration={
                            "A_FLAG_THIS_PACKAGE_DOES_NOT_KNOW": True
                        },
                    )
                ],
                node_id=f"$steps.{self.name}",
                ignore_environment_restrictions=ignore_environment_restrictions,
            )

    manifest = _manifest(_UnknownKeyManifest)

    host_view = manifest.get_actual_restrictions()

    assert [item.code for item in host_view.items] == ["depends_on_unknown_flag"]
    assert host_view.complete is False
    problem = host_view.unknown_reasons[0]
    assert problem.code is DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    assert problem.details["configuration_keys"] == [
        "A_FLAG_THIS_PACKAGE_DOES_NOT_KNOW"
    ]
    # the portable view makes no claim about this host at all
    assert (
        manifest.get_actual_restrictions(ignore_environment_restrictions=True).complete
        is True
    )


def test_a_failing_predicate_beats_an_unknown_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, TENSOR_FLAG, False)
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="Two predicates, one of them unreadable.",
        code="mixed_predicates",
        applies_to_configuration={TENSOR_FLAG: True, "UNKNOWN_FLAG": True},
    )

    assert evaluate_configuration_condition(restriction=restriction) == (
        ConfigurationMatch.INACTIVE,
        [],
    )


def test_a_boolean_never_matches_a_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, TENSOR_FLAG, 1)
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="n",
        code="c",
        applies_to_configuration={TENSOR_FLAG: True},
    )

    verdict, _ = evaluate_configuration_condition(restriction=restriction)

    assert verdict is ConfigurationMatch.INACTIVE


def test_the_host_view_does_not_mutate_the_shared_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, TENSOR_FLAG, True)
    before = restriction_metadata_of(UNSUPPORTED_IN_TENSOR_REPRESENTATION)

    discovery = _manifest(_ConditionalManifest).get_actual_restrictions()

    assert UNSUPPORTED_IN_TENSOR_REPRESENTATION.applies_to_configuration == {
        TENSOR_FLAG: True
    }
    assert restriction_metadata_of(UNSUPPORTED_IN_TENSOR_REPRESENTATION) == before
    assert discovery.items[1] == UNSUPPORTED_IN_TENSOR_REPRESENTATION


# ---------------------------------------------------------------------------
# The configuration allow-list
# ---------------------------------------------------------------------------


def test_every_evaluable_key_exists_in_the_environment_module() -> None:
    missing = sorted(
        key for key in EVALUABLE_CONFIGURATION_KEYS if not hasattr(environment, key)
    )

    assert missing == []


def test_the_flags_the_builtin_presets_pin_are_all_evaluable() -> None:
    pinned = set()
    for restriction in (
        UNSUPPORTED_IN_TENSOR_REPRESENTATION,
        hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED"),
        hosted_endpoint_disabled_by_flag("LMM_ENABLED"),
    ):
        pinned.update(restriction.applies_to_configuration or {})

    assert pinned <= EVALUABLE_CONFIGURATION_KEYS


def test_a_model_flag_branch_comes_from_the_models_configuration() -> None:
    restriction = hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED")

    with installed_configuration(
        WorkflowsConfiguration(models=ModelsConfiguration(moondream2_enabled=True))
    ):
        enabled, _ = evaluate_configuration_condition(restriction=restriction)
    with installed_configuration(
        WorkflowsConfiguration(models=ModelsConfiguration(moondream2_enabled=False))
    ):
        disabled, _ = evaluate_configuration_condition(restriction=restriction)

    assert enabled is ConfigurationMatch.INACTIVE
    assert disabled is ConfigurationMatch.MATCHES


def test_a_block_may_declare_its_restrictions_unknown() -> None:
    class _OptionalManifest(WorkflowBlockManifest):
        type: Literal["test/optional@v1"]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return [OutputDefinition(name="output")]

        def get_actual_restrictions(
            self, *, ignore_environment_restrictions: bool = False
        ) -> Discovery[RuntimeRestriction]:
            return actual_restrictions_of(
                declared=None,
                node_id=f"$steps.{self.name}",
                ignore_environment_restrictions=ignore_environment_restrictions,
            )

    discovery = _manifest(_OptionalManifest).get_actual_restrictions()

    assert discovery.items == []
    assert discovery.complete is False
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_UNAVAILABLE
    ]


# ---------------------------------------------------------------------------
# Entries are rebuilt from the validated projection
# ---------------------------------------------------------------------------

SECRET = "sk-live-do-not-leak-0123456789"
NODE_ID = "$steps.step"


class _SecretBearingValue:
    """Not a JSON value; its text form carries a secret."""

    def __repr__(self) -> str:
        return SECRET

    def __str__(self) -> str:
        return SECRET


def _accepted_but_unsafe() -> RuntimeRestriction:
    # The dataclass accepts a sequence of pairs; the projection reads it as the
    # map {"X": True}.
    return RuntimeRestriction(
        code="example_restriction",
        severity=Severity.HARD,
        note="review",
        applies_to_configuration=[("X", True)],
    )


def _canonical_example() -> RuntimeRestriction:
    return RuntimeRestriction(
        code="example_restriction",
        severity=Severity.HARD,
        note="review",
        applies_to_configuration={"X": True},
    )


def _assert_sanitised_failure(discovery: Discovery[RuntimeRestriction]) -> None:
    assert discovery.items == []
    assert discovery.complete is False
    assert [reason.code for reason in discovery.unknown_reasons] == [
        DiscoveryProblemCode.DECLARATION_FAILED
    ]
    payload = discovery.model_dump_json()
    assert SECRET not in payload
    assert "Traceback" not in payload


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(lambda: [_accepted_but_unsafe()], id="list"),
        pytest.param(lambda: complete_discovery([_accepted_but_unsafe()]), id="typed"),
    ],
)
def test_the_portable_view_carries_the_canonical_configuration(declared) -> None:
    discovery = actual_restrictions_of(
        declared=declared(),
        node_id=NODE_ID,
        ignore_environment_restrictions=True,
    )

    assert discovery == complete_discovery([_canonical_example()])
    assert type(discovery.items[0].applies_to_configuration) is dict


def test_the_host_view_evaluates_the_canonical_configuration() -> None:
    """The reported crash: the raw pair list reached the host evaluator."""
    discovery = actual_restrictions_of(
        declared=complete_discovery([_accepted_but_unsafe()]),
        node_id=NODE_ID,
        ignore_environment_restrictions=False,
    )

    # "X" is not an evaluable key: the entry is kept and the view is incomplete
    assert discovery.items == [_canonical_example()]
    assert discovery.complete is False
    assert [
        (reason.code, reason.details["configuration_keys"])
        for reason in discovery.unknown_reasons
    ] == [(DiscoveryProblemCode.DECLARATION_UNAVAILABLE, ["X"])]


def test_an_inactive_predicate_drops_a_rebuilt_entry_beside_an_unknown_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, TENSOR_FLAG, False)
    restriction = RuntimeRestriction(
        severity=Severity.HARD,
        note="Two predicates, one of them unreadable.",
        code="mixed_predicates",
        applies_to_configuration=[(TENSOR_FLAG, True), ("UNKNOWN_FLAG", True)],
    )

    host_view = actual_restrictions_of(
        declared=[restriction], node_id=NODE_ID, ignore_environment_restrictions=False
    )
    portable_view = actual_restrictions_of(
        declared=[restriction], node_id=NODE_ID, ignore_environment_restrictions=True
    )

    assert host_view == complete_discovery([])
    assert portable_view.complete is True
    assert portable_view.items[0].applies_to_configuration == {
        TENSOR_FLAG: True,
        "UNKNOWN_FLAG": True,
    }


@pytest.mark.parametrize("ignore_environment", [True, False])
def test_axis_values_are_rebuilt_as_enum_members_in_authored_order(
    ignore_environment: bool,
) -> None:
    authored = RuntimeRestriction(
        severity=Severity.SOFT,
        note="Strings and tuples are accepted by the dataclass.",
        applies_to_runtimes=("self_hosted_gpu", Runtime.DEDICATED_DEPLOYMENT),
        applies_to_step_execution_modes=("remote",),
        applies_to_input_modes=["video"],
        code="typed_axes",
    )

    [rebuilt] = actual_restrictions_of(
        declared=[authored],
        node_id=NODE_ID,
        ignore_environment_restrictions=ignore_environment,
    ).items

    assert rebuilt.note == authored.note
    assert rebuilt.code == "typed_axes"
    assert rebuilt.severity is Severity.SOFT
    assert type(rebuilt.applies_to_runtimes) is list
    assert [type(member) for member in rebuilt.applies_to_runtimes] == [
        Runtime,
        Runtime,
    ]
    assert rebuilt.applies_to_runtimes == [
        Runtime.SELF_HOSTED_GPU,
        Runtime.DEDICATED_DEPLOYMENT,
    ]
    assert rebuilt.applies_to_step_execution_modes == [StepExecutionMode.REMOTE]
    assert [type(member) for member in rebuilt.applies_to_input_modes] == [
        RuntimeInputMode
    ]
    assert rebuilt.applies_to_configuration is None


@pytest.mark.parametrize(
    "axis, condition_field, member",
    [
        ("applies_to_runtimes", "runtimes", Runtime.HOSTED_SERVERLESS),
        (
            "applies_to_step_execution_modes",
            "step_execution_modes",
            StepExecutionMode.REMOTE,
        ),
        ("applies_to_input_modes", "input_modes", RuntimeInputMode.VIDEO),
    ],
    ids=["runtimes", "step_execution_modes", "input_modes"],
)
@pytest.mark.parametrize("ignore_environment", [True, False])
def test_a_one_shot_axis_iterable_is_read_once(
    axis: str, condition_field: str, member: Any, ignore_environment: bool
) -> None:
    """Reported: the projection consumed the iterator, and the rebuild then
    published a complete entry with an empty axis."""
    authored = RuntimeRestriction(
        Severity.HARD, "test", code="one_shot_axis", **{axis: iter([member])}
    )

    discovery = actual_restrictions_of(
        declared=[authored],
        node_id=NODE_ID,
        ignore_environment_restrictions=ignore_environment,
    )

    assert discovery.complete is True
    [rebuilt] = discovery.items
    rebuilt_axis = getattr(rebuilt, axis)
    assert type(rebuilt_axis) is list
    assert rebuilt_axis == [member]
    assert type(rebuilt_axis[0]) is type(member)
    # the published entry can itself be projected
    assert getattr(restriction_metadata_of(rebuilt).when, condition_field) == [member]


@pytest.mark.parametrize(
    "configuration",
    [None, {}, {TENSOR_FLAG: True}],
    ids=["none", "empty_map", "map"],
)
def test_none_and_a_configuration_map_stay_distinct(configuration) -> None:
    authored = RuntimeRestriction(
        Severity.HARD,
        "note",
        code="a_code",
        applies_to_configuration=configuration,
    )

    [rebuilt] = actual_restrictions_of(
        declared=[authored], node_id=NODE_ID, ignore_environment_restrictions=True
    ).items

    assert rebuilt == authored
    assert rebuilt.applies_to_configuration == configuration
    if configuration is not None:
        assert rebuilt.applies_to_configuration is not configuration


def test_rebuilding_shared_presets_neither_mutates_nor_aliases_them() -> None:
    tensor_payload = UNSUPPORTED_IN_TENSOR_REPRESENTATION.to_dict()
    video_payload = STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.to_dict()

    discovery = actual_restrictions_of(
        declared=[
            UNSUPPORTED_IN_TENSOR_REPRESENTATION,
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
        ],
        node_id=NODE_ID,
        ignore_environment_restrictions=True,
    )
    by_code = {item.code: item for item in discovery.items}
    tensor_entry = by_code[UNSUPPORTED_IN_TENSOR_REPRESENTATION.code]
    video_entry = by_code[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.code]

    # equal to the presets, authored axis order and notes included
    assert tensor_entry == UNSUPPORTED_IN_TENSOR_REPRESENTATION
    assert video_entry == STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION
    # but holding its own containers
    assert tensor_entry is not UNSUPPORTED_IN_TENSOR_REPRESENTATION
    assert (
        tensor_entry.applies_to_configuration
        is not UNSUPPORTED_IN_TENSOR_REPRESENTATION.applies_to_configuration
    )
    assert (
        video_entry.applies_to_runtimes
        is not STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.applies_to_runtimes
    )
    # and the legacy editor payload is unchanged
    assert tensor_entry.to_dict() == tensor_payload
    assert video_entry.to_dict() == video_payload
    assert UNSUPPORTED_IN_TENSOR_REPRESENTATION.to_dict() == tensor_payload
    assert STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION.to_dict() == video_payload


@pytest.mark.parametrize("ignore_environment", [True, False])
def test_rebuilding_keeps_notes_and_the_declared_incompleteness(
    ignore_environment: bool,
) -> None:
    reason = unresolved_selector_problem(
        node_id=NODE_ID,
        declaration="restrictions",
        field="fire_and_forget",
        selector="$inputs.fire_and_forget",
    )
    first = RuntimeRestriction(
        Severity.SOFT,
        "Track ids reset between requests.",
        (Runtime.HOSTED_SERVERLESS,),
    )
    second = RuntimeRestriction(
        Severity.SOFT,
        "Cooldown does not throttle.",
        (Runtime.HOSTED_SERVERLESS,),
    )

    discovery = actual_restrictions_of(
        declared=incomplete_discovery([first, second], [reason]),
        node_id=NODE_ID,
        ignore_environment_restrictions=ignore_environment,
    )

    assert sorted(item.note for item in discovery.items) == [
        "Cooldown does not throttle.",
        "Track ids reset between requests.",
    ]
    assert {item.code for item in discovery.items} == {"generic_restriction"}
    assert all(
        item.applies_to_runtimes == [Runtime.HOSTED_SERVERLESS]
        for item in discovery.items
    )
    assert discovery.complete is False
    assert discovery.unknown_reasons == [reason]


@pytest.mark.parametrize(
    "restriction",
    [
        pytest.param(
            RuntimeRestriction(Severity.HARD, None, code="a_code"),
            id="note_is_none",
        ),
        pytest.param(
            RuntimeRestriction(Severity.HARD, 42, code="a_code"),
            id="note_is_not_a_string",
        ),
        pytest.param(
            RuntimeRestriction(Severity.HARD, "n", code=[SECRET]),
            id="code_is_not_a_string",
        ),
        pytest.param(
            RuntimeRestriction(Severity.HARD, "n", applies_to_runtimes=[SECRET]),
            id="unknown_runtime",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD,
                "n",
                applies_to_runtimes={Runtime.HOSTED_SERVERLESS: SECRET},
            ),
            id="runtimes_axis_is_a_mapping",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD, "n", applies_to_runtimes="hosted_serverless"
            ),
            id="runtimes_axis_is_a_string",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD,
                "n",
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS] * 2,
            ),
            id="duplicated_runtime",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD, "n", applies_to_step_execution_modes=[SECRET]
            ),
            id="unknown_step_execution_mode",
        ),
        pytest.param(
            RuntimeRestriction(Severity.HARD, "n", applies_to_input_modes=(SECRET,)),
            id="unknown_input_mode",
        ),
        pytest.param(
            RuntimeRestriction(Severity.HARD, "n", applies_to_configuration=SECRET),
            id="configuration_is_not_a_map",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD, "n", applies_to_configuration={7: SECRET}
            ),
            id="configuration_key_is_not_a_string",
        ),
        pytest.param(
            RuntimeRestriction(
                Severity.HARD,
                "n",
                applies_to_configuration={"X": _SecretBearingValue()},
            ),
            id="configuration_value_is_not_json",
        ),
    ],
)
@pytest.mark.parametrize("ignore_environment", [True, False])
def test_a_malformed_field_is_reported_without_its_contents(
    restriction: RuntimeRestriction, ignore_environment: bool
) -> None:
    discovery = actual_restrictions_of(
        declared=[restriction],
        node_id=NODE_ID,
        ignore_environment_restrictions=ignore_environment,
    )

    _assert_sanitised_failure(discovery=discovery)


def test_a_host_evaluation_failure_is_reported_without_its_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def leaking_read(key: str) -> Any:
        raise RuntimeError(f"cannot read {key}: password={SECRET}")

    monkeypatch.setattr(
        restriction_environment, "read_configuration_value", leaking_read
    )

    host_view = actual_restrictions_of(
        declared=[UNSUPPORTED_IN_TENSOR_REPRESENTATION],
        node_id=NODE_ID,
        ignore_environment_restrictions=False,
    )
    portable_view = actual_restrictions_of(
        declared=[UNSUPPORTED_IN_TENSOR_REPRESENTATION],
        node_id=NODE_ID,
        ignore_environment_restrictions=True,
    )

    _assert_sanitised_failure(discovery=host_view)
    # the portable view never evaluates this host, so it is unaffected
    assert portable_view == complete_discovery([UNSUPPORTED_IN_TENSOR_REPRESENTATION])


def test_a_host_evaluation_failure_through_the_legacy_fallback_is_sanitised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def leaking_evaluation(restriction: RuntimeRestriction) -> Any:
        raise ValueError(SECRET)

    monkeypatch.setattr(
        block_module, "evaluate_configuration_condition", leaking_evaluation
    )

    discovery = _manifest(_LegacyConditionalManifest).get_actual_restrictions()

    _assert_sanitised_failure(discovery=discovery)
