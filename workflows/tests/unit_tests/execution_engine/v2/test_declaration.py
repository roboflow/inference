"""Tests of class-owned V2 block declarations."""

from typing import Annotated, Any, Dict, List, Mapping, Optional, Tuple

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
    parse_selector,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    ParamsValidationError,
    ResolvedParameterError,
    SelectorError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Kind,
)

UnitFloat = Annotated[float, Field(ge=0, le=1)]


class ScoreAndRecord(Block):
    """Score a value against a threshold and record labels."""

    type = "example/score_and_record@v2"
    aliases = ("ScoreAndRecord",)
    outputs = {"score": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND, INTEGER_KIND)
        threshold: float | Ref(FLOAT_KIND) = Field(default=0.5, ge=0, le=1)
        labels: Dict[str, str | Ref(STRING_KIND)] = Field(default_factory=dict)
        tags: List[str] = Field(default_factory=list)
        note: Optional[str] = None

    def __init__(self, *, audit: list):
        self._audit = audit
        self.calls = 0

    def run(self, *, value, threshold, labels, tags, note) -> dict:
        self.calls += 1
        self._audit.append((self.calls, labels))

        return {"score": value - threshold}


class Mosaic(Block):
    """Combine one group of children into one value per parent."""

    type = "example/mosaic@v2"
    outputs = {
        "count": Output(INTEGER_KIND),
        "ranks": Output(INTEGER_KIND, preserve="images"),
    }

    class Params(BlockParams):
        images: Group(FLOAT_KIND)
        tile_size: int | Ref(INTEGER_KIND) = 64

    def run(self, *, images, tile_size) -> dict:
        return {"count": len(images), "ranks": images}


class Tile(Block):
    """Split one value into children plus a per-parent summary."""

    type = "example/tile@v2"
    outputs = {
        "tiles": Output(FLOAT_KIND, expand="tile"),
        "regions": Output(FLOAT_KIND, expand="region", stationary=True),
        "summary": Output(),
    }

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        rows: int | Ref(INTEGER_KIND) = 2

    def run(self, *, value, rows) -> dict:
        return {}


class InvertMany(Block):
    """Vectorized block receiving a Batch of all invocations."""

    type = "example/invert_many@v2"
    outputs = {"inverted": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")
        offset: float = 0.0

    def run(self, *, values, offset) -> list:
        return [{"inverted": -value + offset} for value in values]


class ScaleMixed(Block):
    """Vectorized block accepting a scalar-or-batch factor (V1 mixed mode)."""

    type = "example/scale_mixed@v2"
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Ref(FLOAT_KIND, batch="always")
        factor: float | Ref(FLOAT_KIND, batch="if_varying") = 1.0

    def run(self, *, values, factor) -> list:
        return [{"scaled": value} for value in values]


class Counter(Block):
    """Stateful block: counts its own invocations."""

    type = "example/counter@v2"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref()

    def __init__(self):
        self._count = 0

    def run(self, *, value) -> dict:
        self._count += 1

        return {"count": self._count}


class ContinueIfPositive(Block):
    """Control-only block routing positive values."""

    type = "example/continue_if_positive@v2"

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        next_steps: List[StepRef]

    def run(self, *, value, next_steps):
        if value > 0:
            return Select(next_steps)

        return Stop()


class SwitchCase(Block):
    """Control block with independent named routes."""

    type = "example/switch@v2"

    class Params(BlockParams):
        value: Ref(STRING_KIND)
        routes: Dict[str, StepRef]
        default: Optional[StepRef] = None

    def run(self, *, value, routes, default):
        return Select(routes.get(value, default) or ())


class ConstantSource(Block):
    """Input-free source."""

    type = "example/constant@v2"
    outputs = {"value": Output(STRING_KIND)}

    class Params(BlockParams):
        text: str = "hello"

    def run(self, *, text) -> dict:
        return {"value": text}


class AlertSink(Block):
    """Output-free side-effect block accepting missing values."""

    type = "example/alert_sink@v2"
    accepts_empty = True

    class Params(BlockParams):
        payload: str | Ref() = "constant alert"

    def run(self, *, payload) -> dict:
        return {}


class PropertyExtractor(Block):
    """Outputs named by literal configuration (V1 get_actual_outputs)."""

    type = "example/properties@v2"
    output_fields = ("names",)

    class Params(BlockParams):
        data: Ref()
        names: List[str]

    @classmethod
    def describe_outputs(cls, params) -> Mapping[str, Output]:
        return {name: Output() for name in params.names}

    def run(self, *, data, names) -> dict:
        return {name: data.get(name) for name in names}


class Annotate(Block):
    """Declares that it modifies its image payload in place."""

    type = "example/annotate@v2"
    mutates = ("image",)
    outputs = {"image": Output()}
    engine_compatibility = ">=2.0,<3"
    metadata = {"section": "visualization"}

    class Params(BlockParams):
        image: Ref()
        copy_image: bool | Ref() = True

    def run(self, *, image, copy_image) -> dict:
        return {"image": image}

    @classmethod
    def discover_dependent_resources(cls, params):
        return [DependentResource(resource_type="model", identifier="demo-1")]

    @classmethod
    def discover_work_operations(cls, params):
        return [WorkOperation.VISUALIZATION]

    @classmethod
    def discover_restrictions(cls, params):
        raise RuntimeError("secret detail")


class AbstractBase(Block):
    """No own type: an abstract helper base."""

    def run(self, **kwargs) -> dict:
        return {}


class ConcreteFromBase(AbstractBase):
    type = "example/concrete@v2"


def test_ordinary_block_spec_exposes_identity_fields_outputs_and_resources() -> None:
    spec = spec_of(ScoreAndRecord)

    assert spec.type == "example/score_and_record@v2"
    assert spec.identities == ("example/score_and_record@v2", "ScoreAndRecord")
    assert spec.description == "Score a value against a threshold and record labels."
    assert list(spec.fields) == ["value", "threshold", "labels", "tags", "note"]
    assert spec.fields["value"].required and not spec.fields["value"].literal_allowed
    assert (
        spec.fields["threshold"].has_default
        and spec.fields["threshold"].literal_allowed
    )
    assert spec.fields["labels"].container == "dict"
    assert spec.fields["tags"].accepts_selectors is False
    assert spec.outputs["score"].kind_names == ("float",)
    assert [resource.name for resource in spec.resources] == ["audit"]
    assert spec.resources[0].required is True
    assert spec.is_control is False and spec.accepts_batches is False


def test_same_field_accepts_literal_default_input_and_step_selectors() -> None:
    spec = spec_of(ScoreAndRecord)

    default = spec.validate_params({"value": "$inputs.value"})
    literal = spec.validate_params({"value": "$inputs.value", "threshold": 0.4})
    from_input = spec.validate_params(
        {"value": "$inputs.value", "threshold": "$inputs.t"}
    )
    from_step = spec.validate_params(
        {"value": "$steps.a.b", "threshold": "$steps.calibration.value"}
    )

    assert default.threshold == 0.5
    assert literal.threshold == 0.4
    assert [use.selector for use in spec.find_selectors(from_input)] == [
        "$inputs.value",
        "$inputs.t",
    ]
    assert [use.field_path for use in spec.find_selectors(from_step)] == [
        ("value",),
        ("threshold",),
    ]


def test_compound_selectors_are_found_only_at_declared_positions() -> None:
    spec = spec_of(ScoreAndRecord)

    params = spec.validate_params(
        {
            "value": "$inputs.value",
            "labels": {"source": "camera", "class": "$steps.classify.label"},
            "tags": ["$inputs.looks_like_a_selector"],
        }
    )
    uses = spec.find_selectors(params)

    assert [(use.field_path, use.selector) for use in uses] == [
        (("value",), "$inputs.value"),
        (("labels", "class"), "$steps.classify.label"),
    ]
    assert params.tags == ["$inputs.looks_like_a_selector"]


def test_defaults_and_explicit_null_remain_distinct() -> None:
    spec = spec_of(ScoreAndRecord)

    defaulted = spec.validate_params({"value": "$inputs.value"})
    explicit = spec.validate_params({"value": "$inputs.value", "note": None})

    assert "note" not in defaulted.model_fields_set
    assert "note" in explicit.model_fields_set
    assert explicit.note is None


@pytest.mark.parametrize(
    "raw, error_type, fragment",
    [
        ({"value": 5}, ParamsValidationError, "value"),
        ({}, ParamsValidationError, "value"),
        ({"value": "$inputs.v", "threshold": 1.5}, ParamsValidationError, "threshold"),
        ({"value": "$inputs.v", "unknown": 1}, ParamsValidationError, "unknown"),
        (
            {"value": "$inputs.v", "labels": {"a": "$steps.only_step"}},
            SelectorError,
            "labels.a",
        ),
    ],
)
def test_invalid_parameters_fail_with_field_context(raw, error_type, fragment) -> None:
    spec = spec_of(ScoreAndRecord)

    with pytest.raises(error_type) as error:
        spec.validate_params(raw, step_path=("child", "score"))

    assert fragment in str(error.value)
    assert "$steps.child/score" in str(error.value)
    assert error.value.step_path == ("child", "score")


def test_type_and_name_keys_of_a_step_are_not_parameters() -> None:
    spec = spec_of(ScoreAndRecord)

    params = spec.validate_params({"type": "x", "name": "s", "value": "$inputs.v"})

    assert params.value == "$inputs.v"


def test_resolved_values_are_checked_by_kind_and_shared_constraints() -> None:
    spec = spec_of(ScoreAndRecord)
    labels = "dog"

    assert spec.validate_resolved_value("threshold", 1) == 1
    assert spec.validate_resolved_value("labels", labels, position=("class",)) is labels
    with pytest.raises(ContractError, match="threshold"):
        spec.validate_resolved_value("threshold", 1.5)
    with pytest.raises(ResolvedParameterError) as error:
        spec.validate_resolved_value("value", object)

    assert error.value.field_path == ("value",)


def test_grouped_block_declares_group_field_and_preserved_output() -> None:
    spec = spec_of(Mosaic)

    assert spec.fields["images"].role == "group"
    assert spec.outputs["count"].transform == "same"
    assert spec.outputs["ranks"].transform == "preserve"
    assert spec.outputs["ranks"].preserve == "images"


def test_outputs_declare_independent_layouts() -> None:
    spec = spec_of(Tile)

    assert spec.outputs["tiles"].transform == "expand"
    assert spec.outputs["regions"].stationary is True
    assert spec.outputs["summary"].transform == "same"
    assert spec.outputs["summary"].kind_names == ("*",)


def test_batch_capabilities_are_declared_per_selector() -> None:
    always = spec_of(InvertMany)
    mixed = spec_of(ScaleMixed)

    assert always.accepts_batches is True and mixed.accepts_batches is True
    assert always.fields["values"].whole.batch == "always"
    assert mixed.fields["factor"].whole.batch == "if_varying"
    assert spec_of(ScoreAndRecord).fields["value"].whole.batch == "never"


def test_stateful_block_keeps_instance_state() -> None:
    counter = Counter()

    results = [counter.run(value=1), counter.run(value=2)]

    assert results == [{"count": 1}, {"count": 2}]
    assert spec_of(Counter).resources == ()


def test_control_blocks_route_with_select_and_stop() -> None:
    block = ContinueIfPositive()
    spec = spec_of(ContinueIfPositive)

    selected = block.run(value=1.0, next_steps=["$steps.a", "$steps.b"])
    stopped = block.run(value=-1.0, next_steps=["$steps.a"])

    assert spec.is_control is True
    assert spec.fields["next_steps"].leaves.role == "step"
    assert selected == Select(["$steps.a", "$steps.b"])
    assert stopped.targets == ()
    assert isinstance(stopped, Stop)


def test_control_targets_in_dicts_and_optional_fields() -> None:
    spec = spec_of(SwitchCase)

    params = spec.validate_params(
        {
            "value": "$inputs.label",
            "routes": {"cat": "$steps.cats"},
            "default": "$steps.other",
        }
    )
    uses = spec.find_selectors(params)

    assert [(use.field_path, use.marker.role) for use in uses] == [
        (("value",), "item"),
        (("routes", "cat"), "step"),
        (("default",), "step"),
    ]
    with pytest.raises(SelectorError, match="routes.cat"):
        spec.validate_params(
            {"value": "$inputs.label", "routes": {"cat": "$steps.a.b"}}
        )


def test_select_rejects_non_step_targets() -> None:
    with pytest.raises(ContractError, match="step selectors"):
        Select(["$inputs.x"])


def test_input_free_source_and_output_free_sink_are_valid() -> None:
    source = spec_of(ConstantSource)
    sink = spec_of(AlertSink)

    assert source.find_selectors(source.validate_params({})) == ()
    assert sink.outputs == {}
    assert sink.accepts_empty is True
    assert source.accepts_empty is False


def test_configured_outputs_come_from_literal_fields() -> None:
    spec = spec_of(PropertyExtractor)

    params = spec.validate_params({"data": "$inputs.data", "names": ["a", "b"]})

    assert spec.configured_outputs is True
    assert spec.output_fields == ("names",)
    assert list(spec.resolve_outputs(params)) == ["a", "b"]


def test_configured_output_names_are_validated() -> None:
    spec = spec_of(PropertyExtractor)

    params = spec.validate_params({"data": "$inputs.data", "names": ["not valid"]})

    with pytest.raises(DeclarationError, match="letters, digits"):
        spec.resolve_outputs(params)


def test_mutation_compatibility_metadata_and_workload() -> None:
    spec = spec_of(Annotate)
    params = spec.validate_params({"image": "$inputs.image"})

    workload = spec.describe_workload(params, node_id="$steps.annotate")

    assert spec.mutates == ("image",)
    assert spec.engine_compatibility == ">=2.0,<3"
    assert spec.metadata["section"] == "visualization"
    assert workload.dependencies.complete is True
    assert workload.dependencies.items[0].identifier == "demo-1"
    assert workload.operations.items == [WorkOperation.VISUALIZATION]
    assert workload.restrictions.complete is False
    assert "secret detail" not in workload.restrictions.model_dump_json()


def test_undeclared_workload_is_unknown_not_absent() -> None:
    spec = spec_of(ScoreAndRecord)
    params = spec.validate_params({"value": "$inputs.value"})

    workload = spec.describe_workload(params, node_id="$steps.score")

    assert workload.dependencies.complete is False
    assert (
        workload.dependencies.unknown_reasons[0].code.value == "declaration_unavailable"
    )


def test_params_schema_publishes_selector_metadata() -> None:
    schema = spec_of(ScoreAndRecord).params_schema()

    threshold_options = schema["properties"]["threshold"]["anyOf"]
    selector = next(option for option in threshold_options if "selector" in option)

    assert selector["selector"] == {
        "role": "item",
        "kinds": ["float"],
        "batch": "never",
    }
    assert schema["properties"]["threshold"]["default"] == 0.5


def test_describe_is_json_friendly_and_does_not_construct() -> None:
    description = spec_of(ScoreAndRecord).describe()

    assert description["type"] == "example/score_and_record@v2"
    assert description["resources"] == [
        {"name": "audit", "required": True, "annotation": "list"}
    ]
    assert description["fields"]["labels"]["leaf_selector"]["kinds"] == ["string"]


def test_abstract_base_is_not_a_spec_but_concrete_subclass_is() -> None:
    with pytest.raises(DeclarationError, match="abstract"):
        spec_of(AbstractBase)

    assert spec_of(ConcreteFromBase).type == "example/concrete@v2"
    with pytest.raises(DeclarationError, match="Block subclass"):
        spec_of(object)


def test_parse_selector() -> None:
    assert parse_selector("$inputs.image").target == "input"
    assert parse_selector("$steps.crop.crops").output == "crops"
    assert parse_selector("$steps.crop.*").output == "*"
    assert parse_selector("$steps.crop").target == "step"
    with pytest.raises(SelectorError):
        parse_selector("$step.crop")


def _declare(body: Dict[str, Any], *, params: Optional[type] = None) -> type:
    namespace = {"type": "example/invalid@v2", "run": lambda self, **kwargs: {}}
    if params is not None:
        namespace["Params"] = params
    namespace.update(body)

    return type("Invalid", (Block,), namespace)


def _params(**annotations: Any) -> type:
    return type("Params", (BlockParams,), {"__annotations__": annotations})


class NestedModel(BaseModel):
    reference: Ref()


@pytest.mark.parametrize(
    "body, params, fragment",
    [
        ({"type": ""}, None, "type must be"),
        ({"aliases": ("example/invalid@v2",)}, None, "repeats the canonical"),
        ({"aliases": "single"}, None, "aliases must be a tuple"),
        ({}, _params(name=str), "reserved"),
        (
            {},
            _params(value=Ref(FLOAT_KIND) | Ref(STRING_KIND)),
            "selector alternatives",
        ),
        ({}, _params(value=List[List[Ref()]]), "deeper"),
        ({}, _params(value=Dict[str, List[Ref()]]), "deeper"),
        ({}, _params(value=NestedModel), "deeper"),
        (
            {},
            _params(value=List[str | Ref()] | Dict[str, Ref()]),
            "two different containers",
        ),
        (
            {"outputs": {"out": Output()}},
            _params(next_steps=List[StepRef]),
            "control blocks",
        ),
        (
            {"outputs": {"out": Output(preserve="value")}},
            _params(value=Ref()),
            "not a Group",
        ),
        ({"outputs": {"bad name": Output()}}, None, "letters, digits"),
        ({"outputs": {"out": "float"}}, None, "must be an Output"),
        ({"outputs": ["out"]}, None, "must be a mapping"),
        ({"mutates": ("missing",)}, None, "unknown field"),
        ({"mutates": ("flag",)}, _params(flag=bool), "accepts no data selector"),
        ({"accepts_empty": "yes"}, None, "accepts_empty"),
        ({"engine_compatibility": "not a spec"}, None, "valid specifier"),
        ({"output_fields": ("names",)}, _params(names=List[str]), "without overriding"),
        ({"metadata": ["x"]}, None, "metadata must be a mapping"),
        ({"Params": dict}, None, "subclass of BlockParams"),
    ],
)
def test_invalid_declarations_fail_when_the_class_is_created(
    body, params, fragment
) -> None:
    with pytest.raises(DeclarationError, match=fragment):
        _declare(body, params=params)


def test_configured_outputs_require_literal_output_fields() -> None:
    def describe(cls, params):
        return {}

    with pytest.raises(DeclarationError, match="declares no output_fields"):
        _declare(
            {"describe_outputs": classmethod(describe)}, params=_params(names=List[str])
        )
    with pytest.raises(DeclarationError, match="must be literal-only"):
        _declare(
            {"describe_outputs": classmethod(describe), "output_fields": ("names",)},
            params=_params(names=List[str] | Ref(LIST_OF_VALUES_KIND)),
        )


def test_params_must_keep_rejecting_unknown_fields() -> None:
    class Permissive(BlockParams):
        model_config = {"extra": "allow"}

    with pytest.raises(DeclarationError, match="extra='forbid'"):
        _declare({}, params=Permissive)


def test_run_signature_must_match_params() -> None:
    with pytest.raises(DeclarationError, match="does not accept Params field"):
        _declare({"run": lambda self, *, other=None: {}}, params=_params(value=Ref()))
    with pytest.raises(DeclarationError, match="not a Params field"):
        _declare({"run": lambda self, *, required_extra: {}})
    with pytest.raises(DeclarationError, match="passable by keyword"):
        _declare({"run": lambda self, *args: {}})
    with pytest.raises(DeclarationError, match="does not implement run"):
        type("NoRun", (Block,), {"type": "example/no_run@v2"})


def test_constructor_must_declare_resources_by_name() -> None:
    def init(self, *resources):
        pass

    with pytest.raises(DeclarationError, match="named keyword"):
        _declare({"__init__": init})


def test_kinds_must_be_kind_objects_with_unique_names() -> None:
    other_float = Kind(name="float")

    with pytest.raises(DeclarationError, match="Kind objects"):
        Ref("float")
    with pytest.raises(DeclarationError, match="two different kinds"):
        _declare(
            {"outputs": {"out": Output(other_float)}},
            params=_params(value=Ref(FLOAT_KIND)),
        )


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"expand": "a", "preserve": "b"}, "both expand and preserve"),
        ({"expand": "not valid"}, "letters, digits"),
        ({"stationary": True}, "requires expand"),
    ],
)
def test_invalid_output_declarations(arguments, fragment) -> None:
    with pytest.raises(DeclarationError, match=fragment):
        Output(**arguments)


def test_unknown_batch_mode_is_rejected() -> None:
    with pytest.raises(DeclarationError, match="batch must be"):
        Ref(batch="sometimes")


DETECTION_KIND = Kind(name="object_detection_prediction")
IMAGE_KIND = Kind(name="image")


class DetectionsConsensus(Block):
    """V1 consensus shape: a list of batch selectors arrives as list[Batch]."""

    type = "example/detections_consensus@v2"
    outputs = {"predictions": Output(DETECTION_KIND)}

    class Params(BlockParams):
        predictions_batches: List[Ref(DETECTION_KIND, batch="always")] = Field(
            min_length=1
        )
        required_objects: int | Dict[str, int] | Ref(INTEGER_KIND) = 1

    def run(self, *, predictions_batches, required_objects) -> list:
        return [{"predictions": merged} for merged in zip(*predictions_batches)]


class CsvFormatter(Block):
    """V1 CSV shape: dict leaves are scalar or batch depending on each binding."""

    type = "example/csv_formatter@v2"
    outputs = {"csv_content": Output(STRING_KIND)}

    class Params(BlockParams):
        columns_data: Dict[str, str | float | bool | None | Ref(batch="if_varying")]

    def run(self, *, columns_data):
        return {"csv_content": str(columns_data)}


class NamedCropsStitch(Block):
    """Compound group: each dict value is a group of children of one parent."""

    type = "example/named_crops_stitch@v2"
    outputs = {
        "summary": Output(STRING_KIND, source="reference_image"),
        "labelled": Output(DETECTION_KIND, preserve="crops"),
    }

    class Params(BlockParams):
        reference_image: Ref(IMAGE_KIND)
        crops: Dict[str, Group(DETECTION_KIND)]

    def run(self, *, reference_image, crops) -> dict:
        return {"summary": ",".join(crops), "labelled": next(iter(crops.values()))}


class StitchAndTranslate(Block):
    """One block, two layouts: parent-level stitched and child-level translated."""

    type = "example/stitch_and_translate@v2"
    outputs = {
        "stitched": Output(DETECTION_KIND, source="reference_image"),
        "translated": Output(DETECTION_KIND, preserve="predictions"),
    }

    class Params(BlockParams):
        reference_image: Ref(IMAGE_KIND)
        predictions: Group(DETECTION_KIND)

    def run(self, *, reference_image, predictions) -> dict:
        return {"stitched": list(predictions), "translated": predictions}


class JsonParser(Block):
    """Configured outputs use selector segments, not Python identifiers."""

    type = "example/json_parser@v2"
    output_fields = ("expected_fields",)

    class Params(BlockParams):
        raw_json: str | Ref(STRING_KIND)
        expected_fields: List[str]

    @classmethod
    def describe_outputs(cls, params) -> Mapping[str, Output]:
        outputs = {name: Output() for name in params.expected_fields}
        outputs["error_status"] = Output(BOOLEAN_KIND)

        return outputs

    def run(self, *, raw_json, expected_fields) -> dict:
        return {}


class FirstNonEmpty(Block):
    """Alternative branches at one level; missing branches arrive as None."""

    type = "example/first_non_empty@v2"
    accepts_empty = True
    outputs = {"output": Output()}

    class Params(BlockParams):
        data: List[Ref()] = Field(min_length=1)
        default: Any = None

    def run(self, *, data, default) -> dict:
        found = next((item for item in data if item is not None), default)

        return {"output": found}


def test_consensus_list_leaves_are_batch_delivered_and_keep_positions() -> None:
    spec = spec_of(DetectionsConsensus)

    params = spec.validate_params(
        {"predictions_batches": ["$steps.a.predictions", "$steps.b.predictions"]}
    )
    uses = spec.find_selectors(params)

    assert spec.fields["predictions_batches"].container == "list"
    assert [(use.field_path, use.marker.role, use.marker.batch) for use in uses] == [
        (("predictions_batches", 0), "item", "always"),
        (("predictions_batches", 1), "item", "always"),
    ]
    assert spec.accepts_batches is True
    with pytest.raises(ParamsValidationError, match="predictions_batches"):
        spec.validate_params({"predictions_batches": []})


def test_consensus_required_objects_accepts_int_dict_or_selector() -> None:
    spec = spec_of(DetectionsConsensus)
    base = {"predictions_batches": ["$steps.a.predictions"]}

    as_int = spec.validate_params({**base, "required_objects": 2})
    as_dict = spec.validate_params({**base, "required_objects": {"car": 2}})
    as_selector = spec.validate_params({**base, "required_objects": "$inputs.n"})

    assert as_int.required_objects == 2
    assert as_dict.required_objects == {"car": 2}
    assert spec.find_selectors(as_selector)[-1].field_path == ("required_objects",)


def test_csv_dict_keeps_literal_leaves_and_marks_mixed_selector_leaves() -> None:
    spec = spec_of(CsvFormatter)

    params = spec.validate_params(
        {"columns_data": {"camera": "north", "flag": True, "count": "$steps.det.count"}}
    )
    uses = spec.find_selectors(params)

    assert [(use.field_path, use.marker.batch) for use in uses] == [
        (("columns_data", "count"), "if_varying")
    ]
    assert params.columns_data["camera"] == "north"
    assert spec.validate_resolved_value("columns_data", 3.5, position=("count",)) == 3.5


def test_compound_group_leaves_and_dual_layout_outputs() -> None:
    spec = spec_of(NamedCropsStitch)

    params = spec.validate_params(
        {
            "reference_image": "$inputs.image",
            "crops": {"cars": "$steps.cars.crops", "people": "$steps.people.crops"},
        }
    )
    uses = spec.find_selectors(params)

    assert [(use.field_path, use.marker.role) for use in uses] == [
        (("reference_image",), "item"),
        (("crops", "cars"), "group"),
        (("crops", "people"), "group"),
    ]
    assert spec.outputs["summary"].source == "reference_image"
    assert spec.outputs["labelled"].preserve == "crops"
    assert spec.outputs["labelled"].source == "crops"


def test_group_input_does_not_force_every_output_to_collapse() -> None:
    spec = spec_of(StitchAndTranslate)

    assert spec.outputs["stitched"].transform == "same"
    assert spec.outputs["stitched"].source == "reference_image"
    assert spec.outputs["translated"].transform == "preserve"
    assert spec.outputs["translated"].context_policy == "common_or_none"


def test_configured_output_names_may_be_hyphenated_or_numeric() -> None:
    spec = spec_of(JsonParser)

    params = spec.validate_params(
        {"raw_json": "$steps.llm.output", "expected_fields": ["class-name", "2026"]}
    )
    outputs = spec.resolve_outputs(params)

    assert list(outputs) == ["class-name", "2026", "error_status"]
    assert parse_selector("$steps.parse.class-name").output == "class-name"
    assert parse_selector("$steps.parse.2026").output == "2026"
    assert parse_selector("$inputs.camera-1").name == "camera-1"


def test_first_non_empty_list_of_alternatives() -> None:
    spec = spec_of(FirstNonEmpty)

    params = spec.validate_params(
        {
            "data": ["$steps.left.value", "$steps.right.value"],
            "default": "$steps.fake.value",
        }
    )

    assert [use.field_path for use in spec.find_selectors(params)] == [
        ("data", 0),
        ("data", 1),
    ]
    assert params.default == "$steps.fake.value"
    assert FirstNonEmpty().run(data=[None, 7], default=0) == {"output": 7}


@pytest.mark.parametrize(
    "text",
    [
        "$inputs.image\n",
        "$steps.crop.crops ",
        "$steps.crop.crops.extra",
        "$steps.a/b.c",
    ],
)
def test_selectors_must_match_completely(text) -> None:
    spec = spec_of(ScoreAndRecord)

    with pytest.raises(SelectorError):
        parse_selector(text)
    with pytest.raises((ParamsValidationError, SelectorError)):
        spec.validate_params({"value": text})
    with pytest.raises(SelectorError):
        spec.validate_params({"value": "$inputs.v", "labels": {"a": text}})


def test_control_targets_must_match_completely() -> None:
    with pytest.raises(ContractError, match="step selectors"):
        Select("$steps.a\n")


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"source": "not a field"}, "must name a Params field"),
        ({"context_policy": "first"}, "context_policy"),
        ({"preserve": "crops", "source": "reference_image"}, "cannot differ"),
    ],
)
def test_invalid_output_context_declarations(arguments, fragment) -> None:
    with pytest.raises(DeclarationError, match=fragment):
        Output(**arguments)


def test_output_source_must_be_a_data_field() -> None:
    with pytest.raises(DeclarationError, match="accepts no data selector"):
        _declare(
            {"outputs": {"out": Output(source="flag")}},
            params=_params(flag=bool, value=Ref()),
        )


# Accepted input keys and schema properties (decision 025) ---------------------
MIXED_FLOAT = float | Ref(FLOAT_KIND)


def _aliased(config: Optional[Dict[str, Any]] = None, **fields: Any) -> Any:
    namespace: Dict[str, Any] = {
        "__annotations__": {
            name: annotation for name, (annotation, _) in fields.items()
        }
    }
    namespace.update({name: info for name, (_, info) in fields.items()})
    if config is not None:
        namespace["model_config"] = ConfigDict(**config)
    params = type("Params", (BlockParams,), namespace)

    return spec_of(_declare({}, params=params))


def _nested(path: Tuple[Any, ...], value: Any) -> Any:
    for segment in reversed(path):
        if isinstance(segment, int):
            value = [None] * segment + [value]
        else:
            value = {segment: value}

    return value


@pytest.mark.parametrize(
    "info, config, input_paths, schema_property",
    [
        (Field(0.5, alias="cutoff"), None, [("cutoff",)], "cutoff"),
        (
            Field(0.5, alias="cutoff", validation_alias="limit"),
            None,
            [("limit",)],
            "limit",
        ),
        (
            Field(0.5, alias="cutoff"),
            {"validate_by_name": True},
            [("cutoff",), ("max_value",)],
            "cutoff",
        ),
        (
            Field(0.5, alias="cutoff"),
            {"populate_by_name": True},
            [("cutoff",), ("max_value",)],
            "cutoff",
        ),
        (
            Field(0.5, alias="cutoff"),
            {"populate_by_name": True, "validate_by_name": False},
            [("cutoff",)],
            "cutoff",
        ),
        (
            Field(0.5, alias="cutoff"),
            {"validate_by_alias": False, "validate_by_name": True},
            [("max_value",)],
            "max_value",
        ),
        (
            Field(0.5),
            {"alias_generator": to_camel, "validate_by_name": True},
            [("maxValue",), ("max_value",)],
            "maxValue",
        ),
        (
            Field(0.5, validation_alias=AliasPath("limits", "upper")),
            None,
            [("limits", "upper")],
            "max_value",
        ),
        (
            Field(0.5, validation_alias=AliasChoices(AliasPath("limits", 0), "upper")),
            None,
            [("limits", 0), ("upper",)],
            "upper",
        ),
        (Field(0.5), None, [("max_value",)], "max_value"),
    ],
)
def test_fields_publish_the_input_paths_pydantic_accepts(
    info, config, input_paths, schema_property
) -> None:
    spec = _aliased(config, max_value=(MIXED_FLOAT, info))
    field = spec.fields["max_value"]

    assert field.input_paths == tuple(input_paths)
    assert field.schema_property == schema_property
    assert schema_property in spec.params_schema()["properties"]
    assert field.describe()["input_paths"] == [list(path) for path in input_paths]
    for path in field.input_paths:
        params = spec.validate_params(_nested(path, "$inputs.limit"))
        assert params.max_value == "$inputs.limit"
        assert spec.find_selectors(params)[0].field_path == ("max_value",)
    if ("max_value",) not in field.input_paths:
        with pytest.raises(ParamsValidationError, match="Extra inputs"):
            spec.validate_params({"max_value": 0.25})


def test_input_path_is_the_first_location_made_of_object_keys() -> None:
    choices = _aliased(
        max_value=(
            MIXED_FLOAT,
            Field(0.5, validation_alias=AliasChoices(AliasPath("limits", 0), "upper")),
        )
    )
    indexed = _aliased(
        max_value=(MIXED_FLOAT, Field(0.5, validation_alias=AliasPath("limits", 0)))
    )

    assert choices.fields["max_value"].input_path == ("upper",)
    assert indexed.fields["max_value"].input_path is None
    assert indexed.fields["max_value"].describe()["input_path"] is None
    assert indexed.validate_params({"limits": [0.25]}).max_value == 0.25


def test_reserved_step_keys_are_not_advertised_but_alternatives_stay_usable() -> None:
    choices = _aliased(
        label=(str, Field("x", validation_alias=AliasChoices("name", "label_text")))
    )
    only_reserved = _aliased(label=(str, Field("x", alias="name")))

    assert choices.fields["label"].input_paths == (("label_text",),)
    assert choices.validate_params({"name": "step", "label_text": "y"}).label == "y"
    assert only_reserved.fields["label"].input_paths == ()
    assert only_reserved.fields["label"].input_path is None
    assert only_reserved.validate_params({"name": "step"}).label == "x"


def test_fields_may_share_input_paths_and_schema_properties() -> None:
    shared_key = _aliased(low=(float, Field(0.0, alias="high")), high=(float, 1.0))
    shared_property = _aliased(
        low=(float, Field(0.0, validation_alias=AliasPath("bounds", "l"))),
        high=(float, Field(1.0, validation_alias="low")),
    )

    params = shared_key.validate_params({"high": 2.0})
    assert shared_key.fields["low"].input_paths == (("high",),)
    assert shared_key.fields["high"].input_paths == (("high",),)
    assert (params.low, params.high) == (2.0, 2.0)
    assert shared_property.fields["low"].schema_property == "low"
    assert shared_property.fields["high"].schema_property == "low"
    assert set(shared_property.params_schema()["properties"]) == {"low"}
