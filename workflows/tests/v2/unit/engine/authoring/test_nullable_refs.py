"""``Ref(..., allow_none=True)``: declaration, published metadata, payload checks.

Synthetic blocks only. Omitting the flag, or passing ``False``, must publish
exactly what the engine published before the flag existed: the pinned digest
below is ``contract_digest`` of the same probe block computed on the frozen
``ecosystem-e3-query-r1`` engine. ``True`` appears only at the opted-in
selector leaf, changes the block's contract on purpose and lets a ``None``
payload pass that leaf's own check; other payloads and the author's
validators are checked as before.
"""

import json
from typing import Any, Dict, Optional

import pytest
from pydantic import Field, StrictStr, model_validator
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine import (
    Catalogue,
    Implementation,
    ImplementationContribution,
    contract_digest,
)
from roboflow_workflows.v2.engine._selectors import SelectorMarker, selector_arm
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    StepRef,
    spec_of,
)
from roboflow_workflows.v2.engine.errors import (
    ContractError,
    DeclarationError,
    ResolvedParameterError,
)

R1_PROBE_DIGEST = (
    "v1:sha256:9d2968d98f779371b295ee50f01af63ad2b0b2e0d1057c15c987ef7568a251ee"
)
"""``contract_digest`` of ``probe_block()`` on the frozen r1 engine (no flag)."""


def probe_block(**flag: Any) -> type:
    """The block fingerprinted on r1; ``flag`` goes to each of its Refs."""

    class Probe(Block):
        type = "nullable_fingerprint/probe@v1"
        outputs = {"out": Output(STRING_KIND, source="whole")}

        class Params(BlockParams):
            whole: Optional[StrictStr] | Ref(
                STRING_KIND, batch="if_varying", **flag  # noqa: F821
            )
            listed: list[StrictStr | Ref(STRING_KIND, **flag)] = []
            mapped: dict[str, StrictStr | Ref(STRING_KIND, **flag)] = {}

        def run(self, *, whole: Any, listed: Any, mapped: Any) -> Any:
            return [{"out": value} for value in whole]

    return Probe


class Nullable(Block):
    """Opts in at a whole field, a list leaf, a dict leaf and an aliased field."""

    type = "nullable/declared@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        whole: Ref(STRING_KIND, allow_none=True)
        listed: list[Ref(STRING_KIND, allow_none=True)] = []
        mapped: dict[str, Ref(STRING_KIND, allow_none=True)] = {}
        aliased: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None, alias="package"
        )
        strict: Optional[StrictStr] | Ref(STRING_KIND) = None
        group: Group(STRING_KIND) = None

    def run(self, **arguments: Any) -> dict:
        return {"out": arguments}


class GuardedNone(Block):
    """Allows a selected None, but its own validator rejects it for 'strict'."""

    type = "nullable/guarded@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        package: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = None
        policy: str = "lenient"

        @model_validator(mode="after")
        def _strict_needs_a_package(self) -> "GuardedNone.Params":
            if self.policy == "strict" and self.package is None:
                raise ValueError("a strict policy needs a package")

            return self

    def run(self, *, package: Any, policy: str) -> dict:
        return {"out": package}


def _published(spec: Any) -> str:
    published = json.dumps(
        [spec.describe(), spec.params_schema()], sort_keys=True, default=repr
    )

    return published


# --- default metadata is unchanged --------------------------------------------


@pytest.mark.parametrize("flag", [{}, {"allow_none": False}], ids=["omitted", "false"])
def test_the_default_keeps_the_r1_contract_digest(flag: Dict[str, Any]) -> None:
    spec = spec_of(probe_block(**flag))

    assert contract_digest(spec) == R1_PROBE_DIGEST
    assert "allow_none" not in _published(spec)


def test_omitted_and_false_publish_identical_descriptions_and_schemas() -> None:
    omitted = spec_of(probe_block())
    explicit = spec_of(probe_block(allow_none=False))

    assert _published(omitted) == _published(explicit)
    for name in omitted.fields:
        assert omitted.fields[name].describe() == explicit.fields[name].describe()


def test_true_changes_the_contract_digest() -> None:
    assert contract_digest(probe_block(allow_none=True)) != R1_PROBE_DIGEST


# --- True is published only where it was declared ------------------------------


def test_true_is_published_only_at_the_opted_in_selector_leaves() -> None:
    spec = spec_of(Nullable)
    fields = {name: field.describe() for name, field in spec.fields.items()}
    properties = spec.params_schema()["properties"]

    assert fields["whole"]["whole_selector"]["allow_none"] is True
    assert fields["listed"]["leaf_selector"]["allow_none"] is True
    assert fields["mapped"]["leaf_selector"]["allow_none"] is True
    assert fields["aliased"]["whole_selector"]["allow_none"] is True
    assert fields["aliased"]["input_path"] == ["package"]
    for name in ("strict", "group"):
        assert "allow_none" not in json.dumps(fields[name]), name
    assert "allow_none" not in json.dumps(properties["strict"])
    assert "allow_none" not in json.dumps(properties["group"])
    assert '"allow_none": true' in json.dumps(properties["package"])
    assert '"allow_none": true' in json.dumps(properties["listed"])


@pytest.mark.parametrize("value", [1, "yes", None, 0.0])
def test_a_non_bool_allow_none_is_a_declaration_error(value: Any) -> None:
    with pytest.raises(DeclarationError, match="allow_none must be a bool"):
        Ref(STRING_KIND, allow_none=value)


def test_group_and_step_ref_have_no_allow_none() -> None:
    with pytest.raises(TypeError):
        Group(STRING_KIND, allow_none=True)
    assert selector_arm(StepRef).marker.allow_none is False
    assert spec_of(Nullable).fields["group"].whole.allow_none is False


@pytest.mark.parametrize("role", ["group", "step"])
def test_markers_reject_true_on_roles_without_a_none_payload(role: str) -> None:
    with pytest.raises(DeclarationError, match="applies only to Ref selectors"):
        SelectorMarker(role=role, kinds=(STRING_KIND,), allow_none=True)


def test_a_marker_rejects_a_non_bool_flag() -> None:
    with pytest.raises(DeclarationError, match="allow_none must be a bool"):
        SelectorMarker(role="item", kinds=(STRING_KIND,), allow_none="true")


# --- payload checks ------------------------------------------------------------


def test_an_opted_in_marker_accepts_none_and_still_checks_other_payloads() -> None:
    nullable = SelectorMarker(role="item", kinds=(STRING_KIND,), allow_none=True)
    strict = SelectorMarker(role="item", kinds=(STRING_KIND,))

    nullable.check_payload(None)
    nullable.check_payload("trt")
    with pytest.raises(ContractError):
        strict.check_payload(None)
    with pytest.raises(ContractError):
        nullable.check_payload(7)


@pytest.mark.parametrize(
    "field, position",
    [("whole", ()), ("listed", (0,)), ("mapped", ("k",)), ("aliased", ())],
)
def test_resolved_none_passes_at_every_opted_in_position(
    field: str, position: tuple
) -> None:
    spec = spec_of(Nullable)

    assert spec.validate_resolved_value(field, None, position=position) is None
    with pytest.raises(ResolvedParameterError) as error:
        spec.validate_resolved_value(field, 7, position=position)
    assert error.value.field_path == (field,) + position


def test_per_invocation_validation_accepts_none_and_rejects_wrong_kinds() -> None:
    spec = spec_of(Nullable)
    params = spec.validate_params(
        {
            "whole": "$inputs.a",
            "listed": ["$inputs.b"],
            "mapped": {"k": "$inputs.c"},
            "package": "$inputs.d",
            "strict": "$inputs.e",
        }
    )
    arguments = {
        "whole": None,
        "listed": [None],
        "mapped": {"k": None},
        "aliased": None,
        "strict": "text",
        "group": None,
    }

    spec.validate_resolved_arguments(params, arguments)
    # The Optional literal arm does not admit a wrong selected kind.
    with pytest.raises(ResolvedParameterError) as error:
        spec.validate_resolved_arguments(params, {**arguments, "aliased": 3})
    assert error.value.field_path == ("aliased",)
    with pytest.raises(ResolvedParameterError) as error:
        spec.validate_resolved_arguments(params, {**arguments, "listed": [3]})
    assert error.value.field_path == ("listed", 0)


def test_the_authors_validators_still_see_and_may_reject_a_selected_none() -> None:
    spec = spec_of(GuardedNone)
    lenient = spec.validate_params({"package": "$inputs.p"})
    strict = spec.validate_params({"package": "$inputs.p", "policy": "strict"})

    spec.validate_resolved_arguments(lenient, {"package": None, "policy": "lenient"})
    spec.validate_resolved_arguments(strict, {"package": "trt", "policy": "strict"})
    with pytest.raises(ResolvedParameterError, match="strict policy needs a package"):
        spec.validate_resolved_arguments(strict, {"package": None, "policy": "strict"})


# --- contributions ---------------------------------------------------------------


class _Served(Implementation):
    name = "served"

    def run(self, *, whole: Any, listed: Any, mapped: Any) -> Any:
        return [{"out": value} for value in whole]


def _contribution(block: type) -> ImplementationContribution:
    contribution = ImplementationContribution(
        block_type=block.type,
        contract=contract_digest(block),
        implementations=(_Served,),
        provider="example-provider",
        version="1.0.0",
    )

    return contribution


def test_a_default_contract_contribution_still_loads_against_false() -> None:
    pinned = _contribution(probe_block())

    assert pinned.contract == R1_PROBE_DIGEST
    assert (
        Catalogue([probe_block(allow_none=False)]).contribution_problem(pinned) is None
    )


def test_a_contribution_pinned_before_the_opt_in_is_rejected() -> None:
    pinned = _contribution(probe_block())
    opted_in = probe_block(allow_none=True)

    problem = Catalogue([opted_in]).contribution_problem(pinned)

    assert problem is not None and "was built for contract" in problem
    assert Catalogue([opted_in]).contribution_problem(_contribution(opted_in)) is None
