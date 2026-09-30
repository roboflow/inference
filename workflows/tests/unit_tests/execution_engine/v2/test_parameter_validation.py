"""Parameter validation policy (decision 018) on declarations and compiled runs.

Literals are validated once, when the step definition is validated. Selected
values are checked per logical invocation by their ``Ref``/``Group`` kinds,
the constraints shared by the whole field (outer ``Field``) and the author's
field/model validators. Bounds written on a literal alternative only
constrain literals. Selected payloads keep their identity.
"""

from typing import Annotated, Dict, Optional

import pytest
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ParamsValidationError,
    ResolvedParameterError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)


class Emit(Block):
    """Upstream producer of a fresh dictionary payload per invocation."""

    type = "pv/emit@v1"
    outputs = {"payload": Output(DICTIONARY_KIND), "number": Output(FLOAT_KIND)}

    class Params(BlockParams):
        seed: Ref(FLOAT_KIND)

    def __init__(self):
        self.payloads = []

    def run(self, *, seed) -> dict:
        payload = {"seed": seed, "blocked": seed < 0}
        self.payloads.append(payload)

        return {"payload": payload, "number": seed}


class Record(Block):
    """Mixed literal/selector fields with shared bounds and custom validators."""

    type = "pv/record@v1"
    outputs = {"seen": Output()}

    class Params(BlockParams):
        payload: str | Ref(DICTIONARY_KIND) = "none"
        opacity: float | Ref(FLOAT_KIND) = Field(default=0.0, ge=0, le=1)
        literal_bounded: Annotated[float, Field(ge=0, le=1)] | Ref(FLOAT_KIND) = 0.0
        even: int | Ref(INTEGER_KIND) = Field(
            default=0, validation_alias=AliasChoices("even", "even_number")
        )
        low: int | Ref(INTEGER_KIND) = 0
        high: int | Ref(INTEGER_KIND) = 0

        @field_validator("even")
        @classmethod
        def must_be_even(cls, value):
            if isinstance(value, int) and value % 2:
                raise ValueError("must be even")
            return value

        @model_validator(mode="after")
        def ordered(self):
            if isinstance(self.low, int) and isinstance(self.high, int):
                if self.low > self.high:
                    raise ValueError("low must not exceed high")
            return self

    def __init__(self):
        self.calls = []

    def run(self, **arguments) -> dict:
        self.calls.append(arguments)

        return {"seen": arguments["payload"]}


class Columns(Block):
    """Compound field with a checking validator and a normalized literal."""

    type = "pv/columns@v1"
    outputs = {"columns": Output()}

    class Params(BlockParams):
        columns: Dict[str, str | float | Ref(batch="if_varying")] = Field(min_length=1)
        factor: int = 3

        @field_validator("columns")
        @classmethod
        def no_blocked_payload(cls, value):
            if any(
                isinstance(item, dict) and item.get("blocked")
                for item in value.values()
            ):
                raise ValueError("a blocked payload is not allowed")
            return value

        @field_validator("factor")
        @classmethod
        def doubled(cls, value):
            return value * 2

    def __init__(self):
        self.calls = []

    def run(self, *, columns, factor):
        self.calls.append({"columns": columns, "factor": factor})
        if any(isinstance(value, Batch) for value in columns.values()):
            return [{"columns": dict(columns)} for _ in range(len(columns["value"]))]

        return {"columns": columns}


class Positive(Block):
    """Group reducer whose validator checks the whole group."""

    type = "pv/positive@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Group(FLOAT_KIND)

        @field_validator("values")
        @classmethod
        def positive_children(cls, value):
            if isinstance(value, Batch) and any(child <= 0 for child in value):
                raise ValueError("children must be positive")
            return value

    def __init__(self):
        self.calls = []

    def run(self, *, values) -> dict:
        self.calls.append(values)

        return {"total": sum(values)}


class ScaleMany(Block):
    """Batch-delivering block with a shared bound on each logical value."""

    type = "pv/scale_many@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always") = Field(ge=0)

    def __init__(self):
        self.calls = []

    def run(self, *, values) -> list:
        self.calls.append(values)

        return [{"scaled": value * 2} for value in values]


class Replacing(Block):
    """A validator that replaces the selected payload (not allowed at run time)."""

    type = "pv/replacing@v1"
    outputs = {"seen": Output()}

    class Params(BlockParams):
        payload: Ref(DICTIONARY_KIND)

        @field_validator("payload")
        @classmethod
        def copy_payload(cls, value):
            return dict(value) if isinstance(value, dict) else value

    def run(self, *, payload) -> dict:
        return {"seen": payload}


def require_id(value):
    """Shared-position validator that fails with a lookup bug, not a ValueError."""
    if isinstance(value, dict):
        value["id"]
    return value


class Fragile(Block):
    """Validators that raise arbitrary exceptions on selected payloads."""

    type = "pv/fragile@v1"
    outputs = {"seen": Output()}

    class Params(BlockParams):
        payload: Ref(DICTIONARY_KIND)
        checked: Annotated[Ref(DICTIONARY_KIND), AfterValidator(require_id)] = None
        audit: bool = False

        @field_validator("payload")
        @classmethod
        def lookup(cls, value):
            if isinstance(value, dict) and value.get("service_down"):
                raise RuntimeError("validation service failed")
            return require_id(value)

        @model_validator(mode="after")
        def audited(self):
            if self.audit and isinstance(self.payload, dict):
                self.payload["audit_id"]
            return self

    def __init__(self):
        self.calls = []

    def run(self, **arguments) -> dict:
        self.calls.append(arguments)

        return {"seen": arguments["payload"]}


CATALOGUE = Catalogue(
    [Emit, Record, Columns, Positive, ScaleMany, Replacing, Fragile],
    kinds=(FLOAT_KIND, INTEGER_KIND, DICTIONARY_KIND, STRING_KIND),
)


def workflow(inputs, steps, outputs=()):
    definition = {
        "version": "2.0",
        "inputs": list(inputs),
        "steps": list(steps),
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs
        ],
    }

    return definition


def batch(name, kind="float", dimensionality=1):
    return {
        "type": "WorkflowBatchInput",
        "name": name,
        "kind": [kind],
        "dimensionality": dimensionality,
    }


def parameter(name, kind="*"):
    return {"type": "WorkflowParameter", "name": name, "kind": [kind]}


def session_for(inputs, steps, outputs=()):
    plan = compile_workflow(workflow(inputs, steps, outputs), catalogue=CATALOGUE)
    session = plan.create_session()

    return session


def failure(session, inputs) -> StepExecutionError:
    with pytest.raises(StepExecutionError) as error:
        session.run(inputs)

    return error.value


# Declarations -----------------------------------------------------------------


def test_outer_field_constraints_are_declarable_on_selector_fields() -> None:
    spec = spec_of(Record)

    selected = spec.validate_params({"opacity": "$inputs.opacity"})
    with pytest.raises(ParamsValidationError) as error:
        spec.validate_params({"opacity": 1.5})

    assert selected.opacity == "$inputs.opacity"
    assert error.value.field_path == ("opacity",)
    assert "less than or equal to 1" in str(error.value)


def test_definition_errors_name_the_written_field_without_union_branches() -> None:
    spec = spec_of(Record)

    with pytest.raises(ParamsValidationError) as error:
        spec.validate_params({"even_number": []})

    assert error.value.field_path == ("even_number",)
    assert "constrained-str" not in str(error.value)
    assert "selector like $inputs" in str(error.value)


def test_compound_definition_errors_name_the_leaf() -> None:
    spec = spec_of(Columns)

    with pytest.raises(ParamsValidationError) as error:
        spec.validate_params({"columns": {"good": "text", "bad": [1]}})

    assert error.value.field_path == ("columns", "bad")


def test_missing_alias_path_field_is_reported_at_its_nested_keys() -> None:
    class Nested(BlockParams):
        score: Ref(FLOAT_KIND) = Field(validation_alias=AliasPath("reading", "score"))

    spec = spec_of(
        type(
            "Reading",
            (Block,),
            {
                "type": "pv/reading@v1",
                "Params": Nested,
                "run": lambda self, *, score: {},
            },
        )
    )

    with pytest.raises(ParamsValidationError) as error:
        spec.validate_params({"score": "$inputs.number"})

    assert error.value.field_path == ("reading", "score")
    assert "reading.score: Field required" in str(error.value)
    assert "; score: Extra inputs are not permitted" in str(error.value)


def test_literal_only_bounds_do_not_constrain_selected_values() -> None:
    spec = spec_of(Record)
    params = spec.validate_params({"literal_bounded": "$inputs.x"})

    with pytest.raises(ParamsValidationError):
        spec.validate_params({"literal_bounded": 1.5})
    spec.validate_resolved_arguments(params, _record_arguments(literal_bounded=1.5))


def test_validate_resolved_arguments_reports_structured_paths() -> None:
    spec = spec_of(Record)
    params = spec.validate_params(
        {"opacity": "$inputs.o", "low": "$inputs.l", "high": "$inputs.h"}
    )

    with pytest.raises(ResolvedParameterError) as field_error:
        spec.validate_resolved_arguments(params, _record_arguments(opacity=2.0))
    with pytest.raises(ResolvedParameterError) as model_error:
        spec.validate_resolved_arguments(params, _record_arguments(low=5, high=2))

    assert field_error.value.field_path == ("opacity",)
    assert model_error.value.field_path == ()
    assert "low must not exceed high" in str(model_error.value)


def test_optional_fields_apply_shared_constraints_to_non_null_values() -> None:
    class OptionalBound(Block):
        type = "pv/optional_bound@v1"

        class Params(BlockParams):
            columns: Optional[int] | Ref(INTEGER_KIND) = Field(default=None, ge=1)

        def run(self, *, columns) -> dict:
            return {}

    spec = spec_of(OptionalBound)

    assert spec.validate_params({"columns": None}).columns is None
    with pytest.raises(ParamsValidationError, match="greater than or equal to 1"):
        spec.validate_params({"columns": 0})


def test_shared_constraints_on_container_leaves_apply_to_both_forms() -> None:
    class Weights(Block):
        type = "pv/weights@v1"

        class Params(BlockParams):
            weights: Dict[str, Annotated[float | Ref(FLOAT_KIND), Field(ge=0)]]

        def run(self, *, weights) -> dict:
            return {}

    spec = spec_of(Weights)
    params = spec.validate_params({"weights": {"a": 1.0, "b": "$inputs.w"}})

    with pytest.raises(ParamsValidationError) as literal_error:
        spec.validate_params({"weights": {"a": -1.0}})
    with pytest.raises(ResolvedParameterError) as selected_error:
        spec.validate_resolved_arguments(params, {"weights": {"a": 1.0, "b": -2.0}})

    assert literal_error.value.field_path == ("weights", "a")
    assert selected_error.value.field_path == ("weights", "b")


def _record_arguments(**overrides):
    arguments = {
        "payload": "none",
        "opacity": 0.0,
        "literal_bounded": 0.0,
        "even": 0,
        "low": 0,
        "high": 0,
    }
    arguments.update(overrides)

    return arguments


# Compiled runs ----------------------------------------------------------------


def test_selected_dictionary_is_accepted_by_identity_from_step_and_parameter() -> None:
    session = session_for(
        [batch("seeds"), parameter("config", "dictionary")],
        [
            {"type": "pv/emit@v1", "name": "emit", "seed": "$inputs.seeds"},
            {
                "type": "pv/record@v1",
                "name": "upstream",
                "payload": "$steps.emit.payload",
            },
            {"type": "pv/record@v1", "name": "parameter", "payload": "$inputs.config"},
        ],
        [("seen", "$steps.upstream.seen")],
    )
    config = {"mode": "fast"}

    session.run({"seeds": [1.0, 2.0], "config": config})

    produced = session.instances[("emit",)].payloads
    received = [call["payload"] for call in session.instances[("upstream",)].calls]
    assert [id(item) for item in received] == [id(item) for item in produced]
    assert all(
        call["payload"] is config for call in session.instances[("parameter",)].calls
    )


def test_selected_value_of_the_wrong_kind_cannot_use_the_literal_branch() -> None:
    session = session_for(
        [parameter("text")],
        [{"type": "pv/record@v1", "name": "record", "payload": "$inputs.text"}],
    )

    error = failure(session, {"text": "plain"})

    assert isinstance(error.__cause__, ResolvedParameterError)
    assert error.__cause__.field_path == ("payload",)
    assert session.instances[("record",)].calls == []


@pytest.mark.parametrize(
    "source",
    ["parameter", "step"],
)
def test_shared_bounds_and_validators_reject_selected_values_from_any_origin(
    source,
) -> None:
    steps = [{"type": "pv/emit@v1", "name": "emit", "seed": "$inputs.seeds"}]
    selector = "$inputs.bad" if source == "parameter" else "$steps.emit.number"
    steps.append({"type": "pv/record@v1", "name": "record", "opacity": selector})
    session = session_for([batch("seeds"), parameter("bad", "float")], steps)

    error = failure(session, {"seeds": [0.5, 3.0], "bad": 3.0})

    assert error.__cause__.field_path == ("opacity",)
    assert session.instances[("record",)].calls == []


def test_aliased_field_validator_checks_selected_values() -> None:
    session = session_for(
        [parameter("number", "integer")],
        [{"type": "pv/record@v1", "name": "record", "even_number": "$inputs.number"}],
    )

    session.run({"number": 4})
    error = failure(session, {"number": 3})

    assert session.instances[("record",)].calls[0]["even"] == 4
    assert error.__cause__.field_path == ("even",)
    assert "must be even" in str(error)


def test_model_validator_sees_the_complete_resolved_invocation() -> None:
    session = session_for(
        [batch("lows", "integer"), parameter("high", "integer")],
        [
            {
                "type": "pv/record@v1",
                "name": "record",
                "low": "$inputs.lows",
                "high": "$inputs.high",
            }
        ],
    )

    error = failure(session, {"lows": [1, 9], "high": 5})

    assert error.index == (1,)
    assert error.__cause__.field_path == ()
    assert session.instances[("record",)].calls == []


def test_compound_leaves_keep_identity_and_literals_are_not_renormalized() -> None:
    session = session_for(
        [batch("seeds")],
        [
            {"type": "pv/emit@v1", "name": "emit", "seed": "$inputs.seeds"},
            {
                "type": "pv/columns@v1",
                "name": "columns",
                "columns": {"label": "fixed", "value": "$steps.emit.payload"},
                "factor": 3,
            },
        ],
    )

    session.run({"seeds": [1.0]})
    session.run({"seeds": [2.0]})

    produced = session.instances[("emit",)].payloads
    first, second = [call["columns"] for call in session.instances[("columns",)].calls]
    assert isinstance(first["value"], Batch)
    assert first["value"][0] is produced[0]
    assert second["value"][0] is produced[1]
    assert first["label"] == second["label"] == "fixed"
    assert [call["factor"] for call in session.instances[("columns",)].calls] == [6, 6]


def test_compound_field_validator_runs_on_resolved_invocations() -> None:
    session = session_for(
        [batch("seeds")],
        [
            {"type": "pv/emit@v1", "name": "emit", "seed": "$inputs.seeds"},
            {
                "type": "pv/columns@v1",
                "name": "columns",
                "columns": {"value": "$steps.emit.payload"},
            },
        ],
    )

    error = failure(session, {"seeds": [1.0, -1.0]})

    assert error.index == (1,)
    assert error.__cause__.field_path == ("columns",)
    assert "blocked payload" in str(error)
    assert session.instances[("columns",)].calls == []


def test_group_validator_sees_the_batch_with_its_indices() -> None:
    session = session_for(
        [batch("groups", dimensionality=2)],
        [{"type": "pv/positive@v1", "name": "sum", "values": "$inputs.groups"}],
        [("total", "$steps.sum.total")],
    )

    rows = session.run({"groups": [[1.0, 2.0], [3.0]]}).rows()
    error = failure(session, {"groups": [[1.0], [2.0, -1.0]]})

    assert rows == [{"total": 3.0}, {"total": 3.0}]
    assert session.instances[("sum",)].calls[0].indices == ((0, 0), (0, 1))
    assert error.index == (1,)
    assert error.__cause__.field_path == ("values",)


def test_batch_delivering_step_is_checked_per_logical_value_before_any_call() -> None:
    session = session_for(
        [batch("values")],
        [{"type": "pv/scale_many@v1", "name": "scale", "values": "$inputs.values"}],
        [("scaled", "$steps.scale.scaled")],
    )

    rows = session.run({"values": [1.0, 2.0]}).rows()
    calls_before = len(session.instances[("scale",)].calls)
    error = failure(session, {"values": [1.0, -2.0]})

    assert rows == [{"scaled": 2.0}, {"scaled": 4.0}]
    assert calls_before == 1
    assert len(session.instances[("scale",)].calls) == 1
    assert error.index == (1,)
    assert error.__cause__.field_path == ("values",)


def test_validator_replacing_a_selected_payload_is_rejected() -> None:
    session = session_for(
        [batch("seeds")],
        [
            {"type": "pv/emit@v1", "name": "emit", "seed": "$inputs.seeds"},
            {
                "type": "pv/replacing@v1",
                "name": "replace",
                "payload": "$steps.emit.payload",
            },
        ],
    )

    error = failure(session, {"seeds": [1.0]})

    assert error.__cause__.field_path == ("payload",)
    assert "replaced the selected payload" in str(error)


@pytest.mark.parametrize(
    "bad, audit, raised, message",
    [
        ({"id": 1, "service_down": True}, False, RuntimeError, "RuntimeError"),
        ({}, False, KeyError, "KeyError: 'id'"),
        ({"id": 2}, True, KeyError, "KeyError: 'audit_id'"),
    ],
)
def test_unexpected_validator_errors_are_step_errors_with_their_cause(
    bad, audit, raised, message
) -> None:
    errors = []
    definition = workflow(
        [batch("payloads", "dictionary")],
        [
            {
                "type": "pv/fragile@v1",
                "name": "fragile",
                "payload": "$inputs.payloads",
                "audit": audit,
            }
        ],
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session(error_handler=errors.append)

    with pytest.raises(StepExecutionError) as error:
        session.run({"payloads": [{"id": 1, "audit_id": 1}, bad]})

    assert error.value.step_path == ("fragile",)
    assert error.value.index == (1,)
    assert errors == [error.value]
    assert error.value.__cause__.field_path == ("payload",)
    assert message in str(error.value)
    assert type(error.value.__cause__.__cause__) is raised
    assert session.instances[("fragile",)].calls == []


def test_unexpected_error_of_a_shared_position_validator_is_located() -> None:
    spec = spec_of(Fragile)

    with pytest.raises(ResolvedParameterError, match="KeyError: 'id'") as error:
        spec.validate_resolved_value("checked", {})

    assert error.value.field_path == ("checked",)
    assert type(error.value.__cause__) is KeyError
    assert spec.validate_resolved_value("checked", {"id": 3}) == {"id": 3}


@pytest.mark.parametrize(
    "bad, audit, raised",
    [
        ({"id": 1, "service_down": True}, False, RuntimeError),
        ({}, False, KeyError),
        ({"id": 2}, True, KeyError),
    ],
)
def test_unexpected_literal_validator_errors_keep_the_step_and_cause(
    bad, audit, raised
) -> None:
    class LiteralFragile(Fragile):
        type = "pv/literal_fragile@v1"

        class Params(Fragile.Params):
            payload: dict | Ref(DICTIONARY_KIND)

    definition = workflow(
        [],
        [
            {
                "type": LiteralFragile.type,
                "name": "fragile",
                "payload": bad,
                "audit": audit,
            }
        ],
    )

    with pytest.raises(ParamsValidationError) as error:
        compile_workflow(definition, catalogue=Catalogue([LiteralFragile]))

    assert error.value.step_path == ("fragile",)
    assert type(error.value.__cause__) is raised
    assert raised.__name__ in str(error.value)


def test_runtime_projection_preserves_frozen_author_model_and_validators() -> None:
    class FrozenRecord(Record):
        type = "pv/frozen-record@v1"

        class Params(Record.Params):
            model_config = ConfigDict(frozen=True)

    original_config = dict(FrozenRecord.Params.model_config)
    declared = spec_of(FrozenRecord).validate_params({"even": "$inputs.even"})
    with pytest.raises(ValidationError, match="frozen"):
        declared.even = 2

    plan = compile_workflow(
        workflow(
            [parameter("even", "integer")],
            [{"type": FrozenRecord.type, "name": "record", "even": "$inputs.even"}],
        ),
        catalogue=Catalogue([FrozenRecord]),
    )
    session = plan.create_session()
    session.run({"even": 2})
    error = failure(session, {"even": 3})

    assert "must be even" in str(error)
    assert FrozenRecord.Params.model_config == original_config
    assert FrozenRecord.Params.model_config["frozen"] is True
